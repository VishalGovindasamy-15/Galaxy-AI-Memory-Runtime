"""
gamr/core/block.py
==================
Block abstraction — the fundamental unit of GAMR scheduling.

A WeightBlock is a contiguous slice of a weight tensor stored on disk.
It carries full metadata about its own performance history, enabling
the Cost Model and scheduler to make intelligent decisions.

The runtime never loads an entire tensor — only blocks.

Design Notes:
- Block size is tunable (Phase 1 experiment determines optimal sizes)
- Metadata is always present, even when data is None (block on disk)
- reuse_probability drives cache replacement (better than LRU)
- compression_ratio guides whether to compress cold blocks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import threading


class BlockState(Enum):
    """Tracks where a block's data currently lives."""
    ON_DISK     = "on_disk"      # Data is on SSD, not in memory
    LOADING     = "loading"      # SSD → RAM transfer in progress
    IN_RAM      = "in_ram"       # Data is in CPU RAM
    TRANSFERRING = "transferring" # RAM → VRAM transfer in progress
    IN_VRAM     = "in_vram"      # Data is in GPU VRAM, ready to compute
    COMPUTING   = "computing"    # GPU kernel is actively using this block
    EVICTING    = "evicting"     # Being removed from VRAM


@dataclass
class BlockMetadata:
    """
    Persistent performance record for a single weight block.

    Updated on every access. Used by the Cost Model to estimate
    future costs and make intelligent scheduling decisions.

    This is the AI equivalent of CPU cache metadata — instead of
    treating all cache lines equally, we use actual performance
    data to decide what to keep and what to evict.
    """

    # Identity
    block_id:           str             # Unique ID, e.g. "layer_22.ffn.w1.rows_0_512"
    tensor_name:        str             # Parent tensor name
    size_bytes:         int             # Size on disk (compressed or raw)
    size_bytes_loaded:  int             # Size when decompressed in RAM/VRAM

    # Geometry — where in the parent tensor this block lives
    row_start:          int
    row_end:            int
    col_start:          int             # 0 if full row slice
    col_end:            int             # -1 if full row slice

    # Performance history (exponential moving average, updated on each access)
    transfer_time_ms:   float = 0.0    # Avg SSD → RAM → VRAM time (measured)
    compute_time_ms:    float = 0.0    # Avg GPU kernel time (measured)
    ssd_to_ram_ms:      float = 0.0    # SSD → RAM only
    ram_to_vram_ms:     float = 0.0    # RAM → VRAM only

    # Scheduling hints (updated by the Learning Scheduler)
    reuse_probability:  float = 0.5    # 0.0 = safe to evict, 1.0 = keep permanently
    compression_ratio:  float = 1.0    # compressed_size / raw_size (< 1.0 = compressible)
    access_frequency:   int   = 0      # Total number of times this block was computed
    last_access_token:  int   = -1     # Which inference step last used this block

    # State tracking
    load_count:         int   = 0      # How many times was it loaded from SSD?
    eviction_count:     int   = 0      # How many times was it evicted from VRAM?
    stall_count:        int   = 0      # How many times did the GPU stall waiting for this?

    def update_transfer_time(self, measured_ms: float, alpha: float = 0.2) -> None:
        """Exponential moving average update for transfer time."""
        if self.transfer_time_ms == 0.0:
            self.transfer_time_ms = measured_ms
        else:
            self.transfer_time_ms = (1 - alpha) * self.transfer_time_ms + alpha * measured_ms

    def update_compute_time(self, measured_ms: float, alpha: float = 0.2) -> None:
        """Exponential moving average update for compute time."""
        if self.compute_time_ms == 0.0:
            self.compute_time_ms = measured_ms
        else:
            self.compute_time_ms = (1 - alpha) * self.compute_time_ms + alpha * measured_ms

    @property
    def stall_risk(self) -> float:
        """
        Expected GPU stall time for this block.
        Positive = GPU will wait. Zero = perfect overlap.
        """
        if self.compute_time_ms == 0.0:
            return float('inf')  # Unknown — assume worst case
        return max(0.0, self.transfer_time_ms - self.compute_time_ms)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/research output."""
        return {
            "block_id":          self.block_id,
            "size_mb":           round(self.size_mb, 3),
            "transfer_time_ms":  round(self.transfer_time_ms, 4),
            "compute_time_ms":   round(self.compute_time_ms, 4),
            "stall_risk_ms":     round(self.stall_risk, 4),
            "reuse_probability": round(self.reuse_probability, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "access_frequency":  self.access_frequency,
            "load_count":        self.load_count,
            "eviction_count":    self.eviction_count,
            "stall_count":       self.stall_count,
        }


class WeightBlock:
    """
    A contiguous slice of a weight tensor.

    This is the atomic unit of GAMR. The scheduler, cost model, and
    all pipelines operate on WeightBlocks, never on full tensors.

    The block always carries its metadata (which lives in RAM).
    The actual tensor data (.data) is None when the block is on disk.
    """

    def __init__(self, metadata: BlockMetadata):
        self.metadata = metadata
        self._data: Optional[Any] = None        # torch.Tensor when loaded
        self._state: BlockState = BlockState.ON_DISK
        self._lock = threading.Lock()
        self._ready_event = threading.Event()   # Set when block is IN_VRAM

    @property
    def state(self) -> BlockState:
        return self._state

    @state.setter
    def state(self, new_state: BlockState) -> None:
        with self._lock:
            self._state = new_state
            if new_state == BlockState.IN_VRAM:
                self._ready_event.set()
            elif new_state in (BlockState.ON_DISK, BlockState.EVICTING):
                self._ready_event.clear()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, tensor) -> None:
        self._data = tensor

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Block calling thread until this block is IN_VRAM. Returns True on success."""
        return self._ready_event.wait(timeout=timeout)

    def is_ready(self) -> bool:
        """True if data is in VRAM and ready for GPU kernel."""
        return self._state == BlockState.IN_VRAM

    def free_data(self) -> None:
        """Release tensor memory. Block returns to ON_DISK state."""
        self._data = None
        self.state = BlockState.ON_DISK
        self.metadata.eviction_count += 1

    def __repr__(self) -> str:
        return (
            f"WeightBlock(id={self.metadata.block_id!r}, "
            f"state={self._state.value}, "
            f"size={self.metadata.size_mb:.1f}MB)"
        )
