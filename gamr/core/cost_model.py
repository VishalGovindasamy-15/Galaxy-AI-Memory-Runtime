"""
gamr/core/cost_model.py
=======================
The Cost Model — GAMR's decision engine.

Before every scheduling decision, the Cost Model estimates the cost
of each possible action and the scheduler picks the minimum.

This is how modern database query optimizers and compilers work.
We bring the same discipline to AI execution scheduling.

Instead of:
    if GPU_idle:
        increase_prefetch()    ← heuristic, fragile

We do:
    costs = cost_model.estimate_all(possible_actions)
    action = min(costs, key=lambda x: x.total)
    execute(action)

Cost Components
───────────────
  transfer_time   = block.size / measured_bandwidth
  compute_time    = block.flops / measured_gpu_throughput
  queue_delay     = queue.depth × avg_block_time
  stall_risk      = max(0, transfer_time - compute_time)
  eviction_cost   = block.reuse_prob × reload_cost

References
──────────
  - Database query optimization: Selinger et al. 1979
  - Compiler instruction scheduling: Bernstein & Rodeh 1991
  - Applied here to AI tensor scheduling (novel application)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import threading


@dataclass
class HardwareSpeeds:
    """
    Measured hardware performance numbers.
    Populated by the Hardware Profiler (Phase 1 experiments).
    Updated continuously during execution.
    """
    ssd_read_gbps:      float = 3.0    # Conservative default for DRAM-less NVMe
    pcie_gbps:          float = 7.0    # PCIe Gen4 x4 effective
    vram_bandwidth_gbps: float = 224.0  # RTX 3050 Laptop
    gpu_tflops_fp16:    float = 10.0   # RTX 3050 Laptop FP16

    # Moving average window size (number of recent measurements to average)
    window_size:        int   = 20

    def __repr__(self) -> str:
        return (
            f"HardwareSpeeds("
            f"ssd={self.ssd_read_gbps:.2f}GB/s, "
            f"pcie={self.pcie_gbps:.2f}GB/s, "
            f"vram={self.vram_bandwidth_gbps:.1f}GB/s, "
            f"gpu={self.gpu_tflops_fp16:.1f}TFLOPS)"
        )


@dataclass
class Cost:
    """
    Estimated cost breakdown for a single scheduling action.

    All times are in milliseconds.
    Lower total = better action.
    """
    transfer_time_ms:   float = 0.0    # Time to move block to VRAM
    compute_time_ms:    float = 0.0    # Time to compute on this block
    queue_delay_ms:     float = 0.0    # Expected wait in queue
    stall_risk_ms:      float = 0.0    # Expected GPU idle if we do this now
    eviction_cost_ms:   float = 0.0    # Cost of evicting another block to make room

    @property
    def total(self) -> float:
        """
        Total cost. The scheduler minimizes this.

        Stall risk is weighted 2× because GPU idle is the most expensive
        outcome — it defeats the entire purpose of streaming.
        """
        return (
            self.transfer_time_ms
            + self.queue_delay_ms
            + 2.0 * self.stall_risk_ms
            + self.eviction_cost_ms
        )

    @property
    def overlap_ratio(self) -> float:
        """
        How well can I/O overlap with compute for this action?
        0.0 = no overlap (stalls)
        1.0 = perfect overlap (GPU never waits)
        """
        if self.transfer_time_ms == 0:
            return 1.0
        if self.compute_time_ms == 0:
            return 0.0
        return min(1.0, self.compute_time_ms / self.transfer_time_ms)

    def __repr__(self) -> str:
        return (
            f"Cost(total={self.total:.3f}ms, "
            f"transfer={self.transfer_time_ms:.3f}ms, "
            f"compute={self.compute_time_ms:.3f}ms, "
            f"stall_risk={self.stall_risk_ms:.3f}ms, "
            f"overlap={self.overlap_ratio:.2%})"
        )


class CostModel:
    """
    Estimates the cost of scheduling actions based on:
    1. Block metadata (measured transfer and compute times)
    2. Current hardware speeds (continuously updated)
    3. Current runtime state (queue depth, memory pressure)

    The cost model is the brain of the scheduler. It converts
    heuristic rules into principled cost estimates.

    Thread-safe: hardware_speeds are updated from the profiler thread
    and read from the scheduler thread.
    """

    def __init__(self, hardware_speeds: Optional[HardwareSpeeds] = None):
        self.hw = hardware_speeds or HardwareSpeeds()
        self._lock = threading.RLock()

        # Runtime state (updated by schedulers)
        self._vram_queue_depth: int = 0
        self._ram_queue_depth: int = 0
        self._avg_block_time_ms: float = 5.0  # Initial estimate

        # History for updating the model
        self._recent_transfer_times: List[float] = []
        self._recent_compute_times: List[float] = []

    def estimate_transfer_time(self, size_bytes: int, has_history: bool = False,
                                 historical_ms: float = 0.0) -> float:
        """
        Estimate how long it will take to move `size_bytes` to VRAM.

        If the block has been loaded before (historical_ms > 0), use that.
        Otherwise, derive from hardware speeds.

        SSD → RAM → VRAM is sequential, so the bottleneck is the slower stage.
        """
        if has_history and historical_ms > 0.0:
            # Trust measured data over theoretical estimates
            return historical_ms

        with self._lock:
            ssd_time_ms = (size_bytes / (self.hw.ssd_read_gbps * 1e9)) * 1000
            pcie_time_ms = (size_bytes / (self.hw.pcie_gbps * 1e9)) * 1000

        # Sequential pipeline: both happen, bottleneck dominates
        # Add small constant for CUDA API overhead (~0.1ms)
        return ssd_time_ms + pcie_time_ms + 0.1

    def estimate_compute_time(self, size_bytes: int, has_history: bool = False,
                                historical_ms: float = 0.0) -> float:
        """
        Estimate GPU compute time for a weight block of `size_bytes`.

        For matrix multiply (GEMM), compute time scales with FLOPs.
        FLOPs ≈ 2 × M × N × K for a (M×K) × (K×N) multiply.
        For a square block of size S bytes (FP16 = 2 bytes/element):
            elements = S / 2
            side ≈ sqrt(elements)
            FLOPs ≈ 2 × side^3

        This is an approximation — real workloads vary.
        Phase 1 experiments will calibrate this.
        """
        if has_history and historical_ms > 0.0:
            return historical_ms

        import math
        elements = size_bytes / 2  # FP16 = 2 bytes
        side = math.sqrt(elements)
        flops = 2 * (side ** 3)

        with self._lock:
            gpu_flops_per_ms = self.hw.gpu_tflops_fp16 * 1e12 / 1000
            compute_time_ms = flops / gpu_flops_per_ms

        # Add CUDA kernel launch overhead (~0.05ms)
        return compute_time_ms + 0.05

    def estimate_stall_risk(self, transfer_ms: float, compute_ms: float) -> float:
        """
        How long will the GPU stall waiting for this block?

        stall = max(0, transfer_time - compute_time)

        If transfer_time < compute_time: GPU never waits (perfect overlap).
        If transfer_time > compute_time: GPU waits (transfer_time - compute_time) ms.
        """
        return max(0.0, transfer_ms - compute_ms)

    def estimate_queue_delay(self, queue_depth: int) -> float:
        """
        Expected delay due to blocks ahead in the VRAM queue.
        """
        with self._lock:
            return queue_depth * self._avg_block_time_ms

    def estimate_eviction_cost(self, reuse_probability: float,
                                transfer_ms: float) -> float:
        """
        Cost of evicting a block = probability it will be needed again
        multiplied by the cost to reload it.

        If reuse_probability = 0.0 → free eviction, no future cost.
        If reuse_probability = 1.0 → very expensive to evict.
        """
        return reuse_probability * transfer_ms

    def estimate(self, block_metadata, queue_depth: int = 0,
                 eviction_candidate_reuse: float = 0.0) -> Cost:
        """
        Full cost estimate for loading and computing a specific block.

        Args:
            block_metadata: BlockMetadata instance
            queue_depth: Current VRAM queue depth
            eviction_candidate_reuse: reuse_probability of the block that
                                      would be evicted to make room
        """
        has_history = block_metadata.transfer_time_ms > 0.0

        transfer_ms = self.estimate_transfer_time(
            block_metadata.size_bytes,
            has_history=has_history,
            historical_ms=block_metadata.transfer_time_ms
        )

        compute_ms = self.estimate_compute_time(
            block_metadata.size_bytes_loaded,
            has_history=(block_metadata.compute_time_ms > 0.0),
            historical_ms=block_metadata.compute_time_ms
        )

        stall_ms = self.estimate_stall_risk(transfer_ms, compute_ms)
        queue_delay_ms = self.estimate_queue_delay(queue_depth)
        eviction_ms = self.estimate_eviction_cost(
            eviction_candidate_reuse, transfer_ms
        )

        return Cost(
            transfer_time_ms=transfer_ms,
            compute_time_ms=compute_ms,
            queue_delay_ms=queue_delay_ms,
            stall_risk_ms=stall_ms,
            eviction_cost_ms=eviction_ms,
        )

    def choose_best(self, candidates: List[Any],
                     queue_depth: int = 0) -> Optional[Any]:
        """
        Given a list of blocks, return the one with minimum estimated cost.
        This is the core decision function called by schedulers.
        """
        if not candidates:
            return None

        best = None
        best_cost = None

        for candidate in candidates:
            cost = self.estimate(candidate.metadata, queue_depth=queue_depth)
            if best_cost is None or cost.total < best_cost.total:
                best = candidate
                best_cost = cost

        return best

    def update_hardware_speeds(self, **kwargs) -> None:
        """
        Update hardware speed measurements from the profiler.
        Called after every hardware measurement.

        Args:
            ssd_read_gbps: New SSD throughput measurement
            pcie_gbps: New PCIe throughput measurement
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.hw, key):
                    setattr(self.hw, key, value)

    def record_actual_times(self, transfer_ms: float, compute_ms: float) -> None:
        """
        Record actual measured times to improve future estimates.
        Updates the moving average for block_time.
        """
        with self._lock:
            self._recent_transfer_times.append(transfer_ms)
            self._recent_compute_times.append(compute_ms)

            window = self.hw.window_size
            if len(self._recent_compute_times) > window:
                self._recent_compute_times = self._recent_compute_times[-window:]
            if len(self._recent_transfer_times) > window:
                self._recent_transfer_times = self._recent_transfer_times[-window:]

            # Update average block time (used for queue delay estimates)
            all_times = self._recent_transfer_times + self._recent_compute_times
            if all_times:
                self._avg_block_time_ms = sum(all_times) / len(all_times)

    def summary(self) -> Dict[str, Any]:
        """Return current model state for logging."""
        with self._lock:
            n_transfer = len(self._recent_transfer_times)
            n_compute = len(self._recent_compute_times)
            return {
                "hardware": repr(self.hw),
                "avg_block_time_ms": round(self._avg_block_time_ms, 3),
                "transfer_samples": n_transfer,
                "compute_samples": n_compute,
                "avg_transfer_ms": round(
                    sum(self._recent_transfer_times) / n_transfer, 3
                ) if n_transfer else 0.0,
                "avg_compute_ms": round(
                    sum(self._recent_compute_times) / n_compute, 3
                ) if n_compute else 0.0,
            }
