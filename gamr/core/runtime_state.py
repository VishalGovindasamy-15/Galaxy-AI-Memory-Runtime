"""
gamr/core/runtime_state.py
===========================
RuntimeState — Single source of truth for the GAMR runtime.

ARCHITECTURE v1 (FROZEN)
─────────────────────────
All hardware metrics, queue depths, and execution progress live here.

Events UPDATE RuntimeState.
Schedulers READ RuntimeState.
No scheduler accesses hardware directly.

This clean separation means:
  - Schedulers are testable without real hardware
  - Multiple schedulers share one consistent view
  - The System Predictor can snapshot state history
  - The Adaptive Cost Model uses state for queue delay estimates

Thread-safety: all fields protected by RLock.
All reads/writes go through properties or update() method.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class RuntimeSnapshot:
    """
    Immutable snapshot of RuntimeState at one point in time.
    Used by the System Predictor for time-series analysis.
    """
    timestamp_s:            float

    # GPU
    gpu_utilization_pct:    float
    gpu_memory_used_bytes:  int
    gpu_memory_free_bytes:  int
    gpu_temperature_c:      float
    gpu_power_watts:        float
    cuda_occupancy_pct:     float

    # Memory
    vram_queue_depth:       int
    ram_queue_depth:        int
    ram_used_bytes:         int
    ram_free_bytes:         int

    # I/O
    ssd_throughput_gbps:    float
    pcie_throughput_gbps:   float
    ssd_queue_depth:        int

    # Execution
    completed_ops:          int
    total_ops:              int
    stall_count:            int
    last_stall_ms:          float

    @property
    def gpu_idle_pct(self) -> float:
        return max(0.0, 100.0 - self.gpu_utilization_pct)

    @property
    def ram_usage_pct(self) -> float:
        total = self.ram_used_bytes + self.ram_free_bytes
        return (self.ram_used_bytes / total * 100.0) if total > 0 else 0.0

    @property
    def vram_usage_pct(self) -> float:
        total = self.gpu_memory_used_bytes + self.gpu_memory_free_bytes
        return (self.gpu_memory_used_bytes / total * 100.0) if total > 0 else 0.0


class RuntimeState:
    """
    Global runtime state for GAMR.

    This is the single source of truth. Every scheduler, cost model,
    and predictor reads from here. Only the event handlers write here.

    Historical snapshots are kept for the System Predictor.
    """

    # How many snapshots to keep in history
    MAX_HISTORY = 200

    def __init__(self):
        self._lock = threading.RLock()

        # ── GPU ──────────────────────────────────────────────────
        self.gpu_utilization_pct:    float = 0.0
        self.gpu_memory_used_bytes:  int   = 0
        self.gpu_memory_free_bytes:  int   = 0
        self.gpu_temperature_c:      float = 0.0
        self.gpu_power_watts:        float = 0.0
        self.cuda_occupancy_pct:     float = 0.0

        # ── Memory queues ────────────────────────────────────────
        self.vram_queue_depth:       int   = 0
        self.ram_queue_depth:        int   = 0
        self.ram_used_bytes:         int   = 0
        self.ram_free_bytes:         int   = 0

        # ── I/O ──────────────────────────────────────────────────
        self.ssd_throughput_gbps:    float = 0.0
        self.pcie_throughput_gbps:   float = 0.0
        self.ssd_queue_depth:        int   = 0

        # ── Execution progress ───────────────────────────────────
        self.total_ops:              int   = 0
        self.completed_ops:          int   = 0
        self.current_step:           int   = 0

        # ── Stall tracking ───────────────────────────────────────
        self.stall_count_total:      int   = 0
        self.last_stall_time_ms:     float = 0.0
        self.stall_history_ms:       List[float] = []   # Last 20 stall durations

        # ── Scheduler config (set by feedback controller) ────────
        self.active_cuda_streams:    int   = 2
        self.target_block_size_mb:   float = 16.0
        self.prefetch_depth:         int   = 4
        self.compression_enabled:    bool  = False

        # ── Snapshot history (for System Predictor) ──────────────
        self._history:               List[RuntimeSnapshot] = []
        self._last_snapshot_time:    float = 0.0

    # ── Update interface ──────────────────────────────────────────────────────

    def update_gpu(self, utilization_pct: float, memory_used: int, memory_free: int,
                   temperature_c: float = 0.0, power_watts: float = 0.0,
                   occupancy_pct: float = 0.0) -> None:
        with self._lock:
            self.gpu_utilization_pct   = utilization_pct
            self.gpu_memory_used_bytes = memory_used
            self.gpu_memory_free_bytes = memory_free
            self.gpu_temperature_c     = temperature_c
            self.gpu_power_watts       = power_watts
            self.cuda_occupancy_pct    = occupancy_pct

    def update_memory(self, vram_q: int, ram_q: int,
                      ram_used: int, ram_free: int) -> None:
        with self._lock:
            self.vram_queue_depth = vram_q
            self.ram_queue_depth  = ram_q
            self.ram_used_bytes   = ram_used
            self.ram_free_bytes   = ram_free

    def update_io(self, ssd_gbps: float, pcie_gbps: float,
                  ssd_q: int = 0) -> None:
        with self._lock:
            self.ssd_throughput_gbps  = ssd_gbps
            self.pcie_throughput_gbps = pcie_gbps
            self.ssd_queue_depth      = ssd_q

    def record_stall(self, stall_ms: float) -> None:
        with self._lock:
            self.stall_count_total += 1
            self.last_stall_time_ms = stall_ms
            self.stall_history_ms.append(stall_ms)
            if len(self.stall_history_ms) > 20:
                self.stall_history_ms = self.stall_history_ms[-20:]

    def advance_step(self) -> None:
        with self._lock:
            self.completed_ops += 1
            self.current_step  += 1

    def set_scheduler_config(self, block_size_mb: float = None,
                             prefetch_depth: int = None,
                             cuda_streams: int = None) -> None:
        """Called by the feedback controller to update scheduling parameters."""
        with self._lock:
            if block_size_mb  is not None: self.target_block_size_mb = block_size_mb
            if prefetch_depth is not None: self.prefetch_depth = prefetch_depth
            if cuda_streams   is not None: self.active_cuda_streams = cuda_streams

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> RuntimeSnapshot:
        """Create an immutable snapshot of current state for the Predictor."""
        with self._lock:
            snap = RuntimeSnapshot(
                timestamp_s            = time.perf_counter(),
                gpu_utilization_pct    = self.gpu_utilization_pct,
                gpu_memory_used_bytes  = self.gpu_memory_used_bytes,
                gpu_memory_free_bytes  = self.gpu_memory_free_bytes,
                gpu_temperature_c      = self.gpu_temperature_c,
                gpu_power_watts        = self.gpu_power_watts,
                cuda_occupancy_pct     = self.cuda_occupancy_pct,
                vram_queue_depth       = self.vram_queue_depth,
                ram_queue_depth        = self.ram_queue_depth,
                ram_used_bytes         = self.ram_used_bytes,
                ram_free_bytes         = self.ram_free_bytes,
                ssd_throughput_gbps    = self.ssd_throughput_gbps,
                pcie_throughput_gbps   = self.pcie_throughput_gbps,
                ssd_queue_depth        = self.ssd_queue_depth,
                completed_ops          = self.completed_ops,
                total_ops              = self.total_ops,
                stall_count            = self.stall_count_total,
                last_stall_ms          = self.last_stall_time_ms,
            )
            self._history.append(snap)
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]
            self._last_snapshot_time = snap.timestamp_s
        return snap

    @property
    def history(self) -> List[RuntimeSnapshot]:
        with self._lock:
            return list(self._history)

    @property
    def recent_history(self) -> List[RuntimeSnapshot]:
        """Last 10 snapshots — for the System Predictor."""
        with self._lock:
            return list(self._history[-10:])

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def gpu_idle_pct(self) -> float:
        return max(0.0, 100.0 - self.gpu_utilization_pct)

    @property
    def avg_stall_ms(self) -> float:
        with self._lock:
            h = self.stall_history_ms
            return sum(h) / len(h) if h else 0.0

    @property
    def vram_free_bytes(self) -> int:
        return self.gpu_memory_free_bytes

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "gpu_util_pct":       round(self.gpu_utilization_pct, 1),
                "gpu_idle_pct":       round(self.gpu_idle_pct, 1),
                "vram_free_gb":       round(self.gpu_memory_free_bytes / 1e9, 2),
                "ram_free_gb":        round(self.ram_free_bytes / 1e9, 2),
                "ssd_gbps":           round(self.ssd_throughput_gbps, 3),
                "pcie_gbps":          round(self.pcie_throughput_gbps, 3),
                "vram_queue":         self.vram_queue_depth,
                "ram_queue":          self.ram_queue_depth,
                "stall_count":        self.stall_count_total,
                "avg_stall_ms":       round(self.avg_stall_ms, 3),
                "completed_ops":      self.completed_ops,
                "total_ops":          self.total_ops,
                "block_size_mb":      self.target_block_size_mb,
                "prefetch_depth":     self.prefetch_depth,
            }

    def __repr__(self) -> str:
        return (
            f"RuntimeState("
            f"gpu={self.gpu_utilization_pct:.1f}%, "
            f"vram_q={self.vram_queue_depth}, "
            f"ssd={self.ssd_throughput_gbps:.2f}GB/s, "
            f"stalls={self.stall_count_total})"
        )
