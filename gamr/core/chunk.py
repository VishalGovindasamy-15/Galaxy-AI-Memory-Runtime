"""
gamr/core/chunk.py
==================
ResourceChunk — the atomic unit of GAMR.

ARCHITECTURE v1 (FROZEN)
─────────────────────────
Every piece of data flowing through the GAMR pipeline is a ResourceChunk.
The runtime knows nothing about layers, attention, or tokens.
It only knows chunks, their state, and their cost metadata.

Chunk hierarchy:
    ResourceChunk  (base)
    ├── WeightChunk       (model parameters)
    ├── ActivationChunk   (forward pass activations, for grad checkpointing)
    ├── KVCacheChunk      (key/value attention cache)
    ├── GradientChunk     (gradients during training) [future_ideas.md]
    └── OptimizerChunk    (Adam state during training) [future_ideas.md]

Design choices:
- Metadata is ALWAYS present, even when data is None (chunk on disk)
- Thread-safe state transitions via lock + Event
- reuse_probability drives cache replacement (better than LRU)
- All timing stored as exponential moving averages (EMA)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


# ─── State Machine ────────────────────────────────────────────────────────────

class ChunkState(Enum):
    """Tracks where a chunk's data currently lives."""
    ON_DISK      = "on_disk"       # Data on SSD, not in memory
    LOADING      = "loading"       # SSD → RAM in progress
    IN_RAM       = "in_ram"        # Data in CPU RAM
    TRANSFERRING = "transferring"  # RAM → VRAM in progress
    IN_VRAM      = "in_vram"       # Data in GPU VRAM, ready for compute
    COMPUTING    = "computing"     # GPU kernel actively using this chunk
    EVICTING     = "evicting"      # Being removed from VRAM back to RAM/disk


class ResourceType(Enum):
    """What kind of tensor data this chunk holds."""
    WEIGHT       = auto()   # Model parameters (weights)
    ACTIVATION   = auto()   # Forward pass activations
    KV_CACHE     = auto()   # Attention key/value cache
    GRADIENT     = auto()   # Gradient tensors (training)
    OPTIMIZER    = auto()   # Optimizer state (training)


# ─── Metadata ─────────────────────────────────────────────────────────────────

@dataclass
class ChunkMetadata:
    """
    Persistent performance record for one ResourceChunk.

    Updated on every access using exponential moving average (EMA).
    Used by the Adaptive Cost Model to estimate future costs.

    This is the AI equivalent of CPU cache metadata — actual measured
    costs drive eviction and scheduling decisions, not naive LRU.
    """

    # Identity
    chunk_id:           str
    resource_type:      ResourceType
    tensor_name:        str

    # Size
    size_bytes:         int     # Size on disk (may be compressed)
    size_bytes_loaded:  int     # Size in RAM/VRAM (decompressed, FP16)

    # Geometry — slice of the parent tensor
    row_start:          int = 0
    row_end:            int = -1    # -1 = end of tensor
    col_start:          int = 0
    col_end:            int = -1

    # ── Performance history (EMA, updated on each access) ──────────────
    transfer_time_ms:   float = 0.0     # Full SSD → RAM → VRAM time
    compute_time_ms:    float = 0.0     # GPU kernel time
    ssd_to_ram_ms:      float = 0.0     # SSD → RAM only
    ram_to_vram_ms:     float = 0.0     # RAM → VRAM only

    # EMA smoothing factor (0.0 = ignore new, 1.0 = only new)
    ema_alpha:          float = 0.2

    # ── Scheduling hints ───────────────────────────────────────────────
    reuse_probability:  float = 0.5     # 0.0 = evict freely, 1.0 = keep forever
    compression_ratio:  float = 1.0     # compressed / raw (< 1.0 = compressible)
    access_frequency:   int   = 0       # Total compute calls on this chunk
    last_access_step:   int   = -1      # Execution step that last used this

    # ── Counters ───────────────────────────────────────────────────────
    load_count:         int = 0         # Times loaded from SSD
    eviction_count:     int = 0         # Times evicted from VRAM
    stall_count:        int = 0         # Times GPU stalled waiting for this

    def _ema(self, current: float, measured: float) -> float:
        if current == 0.0:
            return measured
        return (1 - self.ema_alpha) * current + self.ema_alpha * measured

    def record_transfer(self, ssd_ram_ms: float, ram_vram_ms: float) -> None:
        self.ssd_to_ram_ms    = self._ema(self.ssd_to_ram_ms, ssd_ram_ms)
        self.ram_to_vram_ms   = self._ema(self.ram_to_vram_ms, ram_vram_ms)
        self.transfer_time_ms = self._ema(self.transfer_time_ms, ssd_ram_ms + ram_vram_ms)
        self.load_count += 1

    def record_compute(self, kernel_ms: float) -> None:
        self.compute_time_ms = self._ema(self.compute_time_ms, kernel_ms)
        self.access_frequency += 1

    def record_stall(self) -> None:
        self.stall_count += 1

    @property
    def stall_risk_ms(self) -> float:
        """Expected GPU stall for this chunk. Positive = GPU will wait."""
        if self.compute_time_ms == 0.0:
            return float('inf')
        return max(0.0, self.transfer_time_ms - self.compute_time_ms)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id":          self.chunk_id,
            "resource_type":     self.resource_type.name,
            "size_mb":           round(self.size_mb, 3),
            "transfer_time_ms":  round(self.transfer_time_ms, 4),
            "compute_time_ms":   round(self.compute_time_ms, 4),
            "stall_risk_ms":     round(self.stall_risk_ms, 4),
            "reuse_probability": round(self.reuse_probability, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "access_frequency":  self.access_frequency,
            "load_count":        self.load_count,
            "eviction_count":    self.eviction_count,
            "stall_count":       self.stall_count,
        }


# ─── Base ResourceChunk ───────────────────────────────────────────────────────

class ResourceChunk:
    """
    The atomic unit of GAMR.

    Represents one contiguous slice of any tensor type (weight, activation,
    KV cache, etc.) at any point in the memory hierarchy.

    Thread-safe: state transitions are protected by a lock.
    Compute threads can wait on _ready_event until chunk is IN_VRAM.
    """

    def __init__(self, metadata: ChunkMetadata):
        self.metadata   = metadata
        self._data: Optional[Any] = None          # torch.Tensor when loaded
        self._state     = ChunkState.ON_DISK
        self._lock      = threading.Lock()
        self._ready     = threading.Event()       # Set when IN_VRAM

    # ── State management ──────────────────────────────────────────────────────

    @property
    def state(self) -> ChunkState:
        return self._state

    @state.setter
    def state(self, new_state: ChunkState) -> None:
        with self._lock:
            self._state = new_state
            if new_state == ChunkState.IN_VRAM:
                self._ready.set()
            elif new_state in (ChunkState.ON_DISK, ChunkState.EVICTING):
                self._ready.clear()

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until chunk is IN_VRAM. Returns False on timeout."""
        return self._ready.wait(timeout=timeout)

    def is_ready(self) -> bool:
        return self._state == ChunkState.IN_VRAM

    # ── Data management ───────────────────────────────────────────────────────

    @property
    def data(self) -> Optional[Any]:
        return self._data

    @data.setter
    def data(self, tensor: Any) -> None:
        self._data = tensor

    def free(self) -> None:
        """Release data. Chunk returns to ON_DISK."""
        self._data = None
        self.state = ChunkState.ON_DISK
        self.metadata.eviction_count += 1

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def resource_type(self) -> ResourceType:
        return self.metadata.resource_type

    @property
    def size_bytes(self) -> int:
        return self.metadata.size_bytes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.chunk_id!r}, "
            f"type={self.resource_type.name}, "
            f"state={self._state.value}, "
            f"size={self.metadata.size_mb:.1f}MB)"
        )


# ─── Concrete Chunk Types ─────────────────────────────────────────────────────

class WeightChunk(ResourceChunk):
    """
    A slice of model weight parameters.
    Most common chunk type in inference.
    """

    def __init__(self, metadata: ChunkMetadata,
                 layer_name: str = "", matrix_name: str = ""):
        super().__init__(metadata)
        self.layer_name  = layer_name    # e.g. "transformer.layer.22"
        self.matrix_name = matrix_name  # e.g. "ffn.w1"


class ActivationChunk(ResourceChunk):
    """
    A slice of forward-pass activations.
    Used for gradient checkpointing (stored in RAM, recomputed if needed).
    In-scope for Phase 3+, but not Phase 1.
    """
    pass


class KVCacheChunk(ResourceChunk):
    """
    A slice of the key/value attention cache.
    Can be streamed in/out for very long contexts.
    In-scope for Phase 4+.
    """
    pass
