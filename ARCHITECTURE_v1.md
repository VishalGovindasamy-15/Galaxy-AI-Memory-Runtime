# GAMR Architecture v1 — FROZEN
## Hierarchical Adaptive Memory Runtime

> **Status: FROZEN as of 2026-06-28**
>
> This document describes the finalized architecture of GAMR.
> No new components, abstractions, or design changes may be added to this file.
>
> **Rule**: Every future change must originate from a measured experiment result.
> If it doesn't come from data, it goes to `future_ideas.md`, not here.

---

## Core Philosophy

GAMR treats AI execution as a **dynamic systems optimization problem**.

Not memory management. Not inference serving. Not a model loader.

An execution runtime that:
1. Receives computation as a dependency graph
2. Manages all hardware resources (SSD, RAM, VRAM, GPU, CPU threads)
3. Reacts to hardware events via an event queue
4. Maintains a global runtime state
5. Uses an adaptive cost model to make scheduling decisions
6. Predicts system behavior (not model logic) to prevent stalls

---

## Layer Diagram

```
╔══════════════════════════════════════════════════════╗
║                  AI Model (External)                 ║
║   LLM / Diffusion / Vision / Speech / Any Model     ║
╚══════════════════════╦═══════════════════════════════╝
                       ║  Compute Graph (DAG)
╔══════════════════════╩═══════════════════════════════╗
║              Graph Compiler                          ║
║   Converts model execution graph into a sequence     ║
║   of ComputeOperations with resource dependencies    ║
╚══════════════════════╦═══════════════════════════════╝
                       ║  ComputeOperations
╔══════════════════════╩═══════════════════════════════╗
║                  GAMR Runtime Core                   ║
║                                                      ║
║   ┌─────────────────────────────────────────────┐   ║
║   │              Event Queue                    │   ║
║   │  SSD_COMPLETE / GPU_IDLE / RAM_PRESSURE /   │   ║
║   │  KERNEL_DONE / STALL_DETECTED / ...         │   ║
║   └──────────────────┬──────────────────────────┘   ║
║                      │                              ║
║   ┌──────────────────▼──────────────────────────┐   ║
║   │            Runtime State                    │   ║
║   │  gpu_utilization, queue_depth, ssd_speed,   │   ║
║   │  ram_usage, temperature, power_limit,       │   ║
║   │  cuda_occupancy, stall_history              │   ║
║   └──────────────────┬──────────────────────────┘   ║
║                      │                              ║
║   ┌──────────────────▼──────────────────────────┐   ║
║   │          Adaptive Cost Model                │   ║
║   │  Estimates: transfer_cost, compute_cost,    │   ║
║   │  queue_delay, stall_risk, eviction_cost     │   ║
║   │  Learns from: per-chunk measurement history │   ║
║   └──────┬───────────────────────┬──────────────┘   ║
║          │                       │                  ║
║   ┌──────▼──────┐    ┌───────────▼───────────────┐  ║
║   │  System     │    │  Resource Allocation      │  ║
║   │  Predictor  │    │  Engine                   │  ║
║   │  (MPC-style)│    │  How much VRAM/RAM/        │  ║
║   │  Predicts:  │    │  CUDA streams/CPU threads │  ║
║   │  starvation │    │  /SSD queue depth to      │  ║
║   │  occupancy  │    │  allocate right now?      │  ║
║   │  throughput │    └───────────┬───────────────┘  ║
║   └──────┬──────┘                │                  ║
║          └───────────┬───────────┘                  ║
║                      │                              ║
╚══════════════════════╬══════════════════════════════╝
                       ║  Scheduling Decisions
╔══════════════════════╩═══════════════════════════════╗
║                  Scheduler Layer                     ║
║                                                      ║
║  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ║
║  │   Storage   │  │    Memory    │  │  Compute   │  ║
║  │  Scheduler  │  │  Scheduler   │  │ Scheduler  │  ║
║  │  SSD→RAM    │  │  RAM→VRAM    │  │   CUDA     │  ║
║  │  Prefetch   │  │  Eviction    │  │  Streams   │  ║
║  └─────────────┘  └──────────────┘  └────────────┘  ║
║                                                      ║
║  ┌─────────────────────────────────────────────────┐ ║
║  │         Compression Scheduler                   │ ║
║  │  Decides: compress cold chunks to INT4/INT8     │ ║
║  │  Based on: reuse_probability + memory_pressure  │ ║
║  └─────────────────────────────────────────────────┘ ║
╚══════════════════════╦══════════════════════════════╝
                       ║
╔══════════════════════╩═══════════════════════════════╗
║              Hardware Layer                          ║
║                                                      ║
║   SSD  ↔  RAM  ↔  VRAM  ↔  GPU Tensor Cores        ║
╚══════════════════════════════════════════════════════╝
```

---

## Data Model

### ResourceChunk (base class)

The atomic unit of GAMR. Every piece of data that flows through the pipeline is a ResourceChunk.

```python
class ResourceChunk:
    chunk_id:      str           # Globally unique
    resource_type: ResourceType  # WEIGHT / ACTIVATION / KV_CACHE / GRADIENT / OPTIMIZER
    size_bytes:    int
    state:         ChunkState    # ON_DISK / IN_RAM / IN_VRAM / COMPUTING
    metadata:      ChunkMetadata # Performance history
```

**Subtypes** (all inherit ResourceChunk):
- `WeightChunk` — model parameter slices
- `ActivationChunk` — forward pass activations (for gradient checkpointing)
- `KVCacheChunk` — key/value attention cache
- `GradientChunk` — gradient tensors during training
- `OptimizerChunk` — Adam momentum/variance (training only)

### ChunkMetadata

```python
class ChunkMetadata:
    # Performance history (exponential moving average)
    transfer_time_ms:  float   # Measured SSD→RAM→VRAM time
    compute_time_ms:   float   # Measured GPU kernel time
    ssd_to_ram_ms:     float
    ram_to_vram_ms:    float

    # Scheduling hints
    reuse_probability: float   # 0.0 = evict freely, 1.0 = keep permanently
    compression_ratio: float   # compressed/raw size
    access_frequency:  int
    last_access_step:  int

    # Stall history
    stall_count:       int     # How often GPU waited for this chunk
```

### ExecutionGraph (DAG)

```python
class ExecutionGraph:
    nodes:  List[ComputeOperation]  # Each operation is a node
    edges:  Dict[str, List[str]]    # op_id → [dependent_op_ids]

    def get_ready_ops(self) -> List[ComputeOperation]:
        """Operations whose input dependencies are all IN_VRAM."""
        ...

    def get_critical_path(self) -> List[ComputeOperation]:
        """Longest dependency chain — must be scheduled first."""
        ...
```

### ComputeOperation

```python
class ComputeOperation:
    op_id:          str
    input_chunks:   List[ChunkRef]    # Must be in VRAM before execution
    output_chunks:  List[ChunkRef]    # Written to VRAM after execution
    kernel_fn:      Callable          # GPU kernel (runtime-agnostic)
    priority:       int               # From critical path analysis
    estimated_cost: Cost              # From CostModel
```

**The runtime knows nothing beyond this.** No layers, no attention, no tokens.

---

## Runtime State (Single Source of Truth)

```python
class RuntimeState:
    # Hardware utilization (updated every ~50ms by profiler)
    gpu_utilization_pct:    float
    gpu_memory_used_bytes:  int
    gpu_temperature_c:      float
    gpu_power_watts:        float
    cuda_occupancy_pct:     float

    # Memory state
    vram_free_bytes:        int
    ram_free_bytes:         int
    vram_queue_depth:       int
    ram_queue_depth:        int

    # I/O state
    ssd_throughput_gbps:    float
    pcie_throughput_gbps:   float
    ssd_queue_depth:        int

    # Scheduler state
    active_cuda_streams:    int
    pending_transfers:      int
    stall_count_total:      int
    last_stall_time_ms:     float

    # Execution progress
    total_ops:              int
    completed_ops:          int
    current_step:           int
```

Events update RuntimeState. Schedulers read RuntimeState. No scheduler accesses hardware directly.

---

## Adaptive Cost Model

Unlike a static formula, the Adaptive Cost Model **learns** from every measurement.

```
Static Formula:
    cost = size / bandwidth + overhead

Adaptive Formula:
    cost = weighted_average(
        static_estimate,
        historical_average,
        recent_measurements    ← highest weight
    )

If recent_measurements diverge from historical:
    → Emit SSD_THROUGHPUT_DROP event
    → Increase block size (better sequential efficiency)
    → Increase prefetch depth
```

The cost model also tracks **per-chunk history**. Block 421 that usually takes 0.72ms but today took 1.40ms → the model detects SSD degradation, not just a noisy measurement.

---

## System Predictor (MPC-Style)

Predicts **system performance**, not model execution.

```
Input signals (from RuntimeState):
    - Current SSD speed
    - Current queue depth
    - Current GPU occupancy
    - Current RAM usage
    - Recent stall frequency

Predictions (next N seconds):
    - Probability of GPU starvation
    - Expected throughput
    - Expected queue depth at T+1s
    - Memory pressure trajectory

Control actions (if starvation predicted):
    - Increase prefetch depth NOW (before stall happens)
    - Evict low-priority chunks NOW (before RAM fills)
    - Compress cold chunks NOW (before compression adds latency at critical moment)
```

This is **Model Predictive Control** applied to AI runtime management. Proactive, not reactive.

---

## Three Parallel Tracks

| Track | Name | Produces |
|---|---|---|
| A | Systems Model | Performance model, cost model math, queue theory, paper sections 3–5 |
| B | Experiments | Measurements, benchmark results, plots, tables |
| C | Engineering | Runtime code, tests, API, documentation |

All three tracks start simultaneously. Track B measurements feed Track A theory. Track A theory guides Track C implementation.

---

## Comparison Progression (Strictly Ordered)

No external benchmark comparison until the internal baseline sequence is complete:

```
V0: Full model in RAM/VRAM (theoretical ceiling — not our work)
    ↓
V1: Naive streaming (Load → Wait → Compute → Wait → Unload)
    ↓ Measure: GPU idle time, tokens/sec
V2: Pipelined streaming (Double buffer, fixed block size)
    ↓ Measure: Δ GPU idle vs V1
V3: GAMR Adaptive (Event-driven, cost model, self-tuning)
    ↓ Measure: Δ GPU idle vs V2
V4: External comparison (AirLLM, FlexGen, llama.cpp offloading)
    ↓ Measure: GAMR vs production systems
```

**We do not claim V4 wins until we have V4 data.**

---

## Research Questions (No Promises)

**RQ1**: For which block sizes does `compute_time ≥ transfer_time` on DRAM-less NVMe + RTX 3050?

**RQ2**: What fraction of GPU idle time does pipelining eliminate vs naive streaming (V1 → V2)?

**RQ3**: Does adaptive block sizing + cost-model scheduling further reduce idle time vs fixed-block pipelining (V2 → V3)?

**RQ4**: Can the Adaptive Cost Model predict SSD throughput drops before they cause GPU stalls?

**RQ5**: What is the maximum runnable model size on 6 GB VRAM with acceptable throughput degradation?

**RQ6**: Does the System Predictor (MPC) reduce stall frequency vs the reactive feedback controller?

---

## What Is NOT In v1

The following ideas are real but unvalidated. They go to `future_ideas.md`:

- Multi-GPU streaming
- Training support (gradients, optimizer states)
- Mixture-of-Experts expert selection
- Cross-machine distributed streaming
- Neural network-based scheduler
- GGUF format support (deferred until core runtime is validated)

---

## Freeze Declaration

This architecture is complete and frozen.

**The next action is not design. It is measurement.**

Run `experiments/exp01_ssd_bandwidth.py` and get real numbers.
Everything built after this point must be justified by experiment results.
