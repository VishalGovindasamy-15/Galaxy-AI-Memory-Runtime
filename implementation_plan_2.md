# Galaxy AI Memory Runtime (GAMR)
## Revised Research & Product Implementation Plan

> **Revision note**: This version incorporates a complete architectural redesign based on code review. All premature performance claims removed. Architecture is now event-driven. Runtime is fully model-agnostic (knows nothing about transformers or attention). A Cost Model is added as a core first-class component. Three parallel tracks replace the single linear sequence.

---

## Vision

Build **GAMR** — an adaptive, block-streaming execution runtime that treats `SSD → RAM → VRAM → GPU` as one unified, self-optimizing pipeline.

Not another LLM library. Not another offloading hack.

**An Operating System for AI Memory.**

```
User writes:

    runtime = GAMR()
    runtime.load(tensor_graph)
    runtime.execute()

GAMR automatically decides:
    - Block size
    - SSD read strategy
    - RAM queue depth
    - VRAM eviction policy
    - CUDA stream scheduling
    - Compression level
    - Pipeline depth
```

The user never touches hardware settings.

---

## Paper Identity

| For | Name |
|---|---|
| **Project / GitHub** | Galaxy AI Memory Runtime (GAMR) |
| **Academic Paper** | **HAMR: Hierarchical Adaptive Memory Runtime for Large AI Models** |

"HAMR" tells a reader exactly what the work is:
- **Hierarchical** — SSD → RAM → VRAM
- **Adaptive** — self-tuning
- **Memory** — the problem domain
- **Runtime** — the contribution type

---

## What Changed From v1 (And Why)

| v1 Plan | v2 (This Plan) | Reason |
|---|---|---|
| GGUF loader first | Plain `.pt` tensors first | Isolate runtime bugs from format bugs |
| Compare to AirLLM immediately | Compare Naive → Pipeline → Adaptive first | Need to attribute each improvement |
| Fixed performance targets (80–90%) | Research questions only | Numbers before experiments = fiction |
| Hierarchical scheduler | Event-driven architecture | Events are cleaner, more extensible |
| Runtime knows transformers | Runtime knows only Tensors/Blocks | Model-agnostic by design |
| No cost model | Cost model is a core component | Enables principled scheduling decisions |
| No block metadata | Block carries full metadata history | Enables intelligent cache replacement |
| One linear track | Three parallel tracks (Theory/Exp/Eng) | Real research lab structure |

---

## The Three Parallel Tracks

```
┌────────────────────┬────────────────────┬────────────────────┐
│   Track A          │   Track B          │   Track C          │
│   THEORY           │   EXPERIMENTS      │   ENGINEERING      │
├────────────────────┼────────────────────┼────────────────────┤
│ Mathematical model │ Hardware timing    │ Runtime code       │
│ Cost model         │ Benchmark suite    │ API design         │
│ Scheduler proofs   │ Adaptive algorithm │ Test suite         │
│ Complexity bounds  │ Comparisons        │ Documentation      │
│ Paper writing      │ Plots & tables     │ Package structure  │
└────────────────────┴────────────────────┴────────────────────┘
                            ↓
              All three merge into the final paper
```

Each track runs simultaneously. Every experiment informs the theory. Every theory guides the engineering.

---

## Hardware Context

| Component | Spec | Impact on Design |
|---|---|---|
| GPU | RTX 3050 Laptop, 6 GB VRAM | Critical constraint; forces streaming |
| CPU | i5-13450HX, 10c/16t | Excellent for async I/O threads |
| RAM | 16 GB | ~8 GB safely usable as pipeline buffer |
| SSD 1 | Micron 2550 NVMe (DRAM-less) | DRAM-less = prefer large sequential reads |
| SSD 2 | Sandisk PC SN740 NVMe (DRAM-less) | Same — avoid small random I/O |
| PCIe | Gen4 x4 (laptop) | ~8 GB/s practical limit |

> [!IMPORTANT]
> **DRAM-less SSDs** have no internal cache. Small (1–4 MB) random reads will perform significantly worse than on cached SSDs. Our block size tuner must account for this — favor 8–64 MB sequential reads.

---

## Project File Structure

```
Galaxy-AI-Memory-Runtime/
│
├── idea.txt                        ← Original conversation (never delete)
├── README.md
├── requirements.txt
├── setup.py
│
├── gamr/                           ← The runtime package
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── block.py                ← Block + BlockMetadata
│   │   ├── event.py                ← Event types + EventQueue
│   │   ├── cost_model.py           ← Cost estimator
│   │   ├── pipeline.py             ← 3-tier memory pipeline
│   │   └── runtime.py              ← GAMR main class
│   │
│   ├── schedulers/
│   │   ├── base.py                 ← BaseScheduler interface
│   │   ├── storage.py              ← SSD → RAM
│   │   ├── memory.py               ← RAM → VRAM + eviction
│   │   ├── compute.py              ← CUDA streams
│   │   ├── feedback.py             ← Feedback controller
│   │   └── learner.py              ← History-based optimizer
│   │
│   ├── io/
│   │   ├── tensor_loader.py        ← Load plain .pt tensors (Phase 1–3)
│   │   └── safetensors_loader.py   ← HuggingFace format (Phase 4+)
│   │
│   └── profiler/
│       ├── hardware.py             ← Measure SSD/PCIe/VRAM/GPU
│       └── block_profiler.py       ← Per-block timing history
│
├── experiments/
│   ├── exp01_ssd_bandwidth.py      ← Raw SSD speed for block sizes
│   ├── exp02_pcie_bandwidth.py     ← RAM→VRAM transfer speed
│   ├── exp03_gpu_compute.py        ← GPU compute time per block size
│   ├── exp04_overlap_test.py       ← Can I/O hide behind compute?
│   ├── exp05_naive_baseline.py     ← Load → Compute → Unload (V1)
│   ├── exp06_pipelined.py          ← Pipelined streaming (V2)
│   ├── exp07_adaptive.py           ← Adaptive scheduling (V3)
│   └── results/                    ← JSON + CSV results
│
├── benchmarks/
│   ├── bench_naive.py
│   ├── bench_pipelined.py
│   ├── bench_adaptive.py
│   └── bench_airllm.py             ← Only after V3 is complete
│
├── research/
│   ├── theory/
│   │   ├── cost_model.md           ← Mathematical derivations
│   │   ├── scheduler_analysis.md
│   │   └── complexity_bounds.md
│   ├── paper/
│   │   ├── hamr_paper.tex
│   │   ├── figures/
│   │   └── tables/
│   └── notes/
│
├── tests/
│   ├── test_block.py
│   ├── test_event_queue.py
│   ├── test_cost_model.py
│   └── test_pipeline.py
│
└── docs/
    ├── architecture.md
    ├── event_system.md
    ├── cost_model.md
    └── api.md
```

---

## Core Architecture: Event-Driven Runtime

> [!IMPORTANT]
> This is the single biggest architectural change from v1. The scheduler does NOT poll. It **reacts to events**.

### Event Types

```python
class EventType(Enum):
    # I/O Events
    SSD_READ_COMPLETE    = "ssd_read_complete"
    RAM_TO_VRAM_COMPLETE = "ram_to_vram_complete"

    # Compute Events
    KERNEL_LAUNCHED      = "kernel_launched"
    KERNEL_FINISHED      = "kernel_finished"

    # Resource Events
    GPU_IDLE             = "gpu_idle"
    GPU_STALLED          = "gpu_stalled"
    RAM_PRESSURE_HIGH    = "ram_pressure_high"
    VRAM_PRESSURE_HIGH   = "vram_pressure_high"

    # Storage Events
    SSD_THROUGHPUT_DROP  = "ssd_throughput_drop"
    SSD_THROUGHPUT_SPIKE = "ssd_throughput_spike"

    # Scheduler Events
    BLOCK_EVICT_REQUEST  = "block_evict_request"
    PREFETCH_TRIGGER     = "prefetch_trigger"
```

### Runtime Flow

```
            ┌──────────────────────────┐
            │       GAMR Runtime       │
            │                          │
            │   ┌──────────────────┐   │
            │   │   Event Queue    │   │
            │   │                  │   │
            │   │  SSD_COMPLETE    │   │
            │   │  GPU_IDLE        │   │
            │   │  RAM_FULL        │   │
            │   │  KERNEL_DONE     │   │
            │   └────────┬─────────┘   │
            │            │             │
            │   ┌────────▼─────────┐   │
            │   │   Dispatcher     │   │
            │   │  (routes events  │   │
            │   │   to schedulers) │   │
            │   └────────┬─────────┘   │
            │            │             │
            │   ┌────────▼─────────┐   │
            │   │   Cost Model     │   │
            │   │  (estimates cost │   │
            │   │   of each action)│   │
            │   └────────┬─────────┘   │
            │            │             │
            │   ┌────────▼─────────┐   │
            │   │  Scheduler picks │   │
            │   │  lowest-cost     │   │
            │   │  action          │   │
            │   └────────┬─────────┘   │
            │            │             │
            │     Execute action       │
            │     → emits new events   │
            └──────────────────────────┘
```

Everything reacts. Nothing polls. New events cascade naturally.

---

## The Cost Model (New Core Component)

> Before every scheduling decision, estimate the cost of each possible action. Choose the lowest.

This is how **database query optimizers** and **modern compilers** work. We bring that discipline to AI execution.

### Cost Estimation

```python
class CostModel:
    def estimate_action(self, action: SchedulerAction) -> Cost:
        return Cost(
            transfer_time  = self.estimate_transfer(action.block),
            compute_time   = self.estimate_compute(action.block),
            queue_delay    = self.estimate_queue_delay(),
            stall_risk     = self.estimate_stall_probability(),
            eviction_cost  = self.estimate_eviction_cost(action.block),
            total          = self._combine()
        )

    def choose_best(self, actions: List[SchedulerAction]) -> SchedulerAction:
        costs = [(a, self.estimate_action(a)) for a in actions]
        return min(costs, key=lambda x: x[1].total)[0]
```

### Cost Components

| Component | Formula | Source |
|---|---|---|
| Transfer time | `block.size_bytes / measured_bandwidth` | Block metadata |
| Compute time | `block.flops / measured_gpu_tflops` | Block metadata |
| Queue delay | `queue.current_depth * avg_block_time` | Runtime state |
| Stall risk | `max(0, transfer_time - compute_time)` | Measured delta |
| Eviction cost | `block.reuse_probability * reload_cost` | Block metadata |

---

## Block Metadata (New)

Every block carries a full history of its own performance. The scheduler learns from this.

```python
@dataclass
class BlockMetadata:
    # Identity
    block_id:           str       # "tensor_A.row_0_to_512"
    size_bytes:         int       # Exact size on disk

    # Performance history (updated every access)
    transfer_time_ms:   float     # SSD → RAM → VRAM measured time
    compute_time_ms:    float     # GPU kernel time for this block
    last_access_token:  int       # Which token last used it

    # Scheduling hints
    reuse_probability:  float     # 0.0 → evict freely, 1.0 → keep in VRAM
    compression_ratio:  float     # Actual compressed/uncompressed ratio
    access_frequency:   int       # How often this block is used

@dataclass
class WeightBlock:
    # Data
    data:       Optional[torch.Tensor]   # None if not loaded
    state:      BlockState               # ON_DISK / IN_RAM / IN_VRAM / COMPUTING

    # Metadata (always present, even when data is None)
    metadata:   BlockMetadata
```

This is the AI equivalent of **CPU cache replacement policy**. Instead of treating all blocks equally (LRU), we make intelligent decisions based on actual cost.

---

## Runtime is Model-Agnostic

> [!IMPORTANT]
> **The runtime knows nothing about transformers, attention, layers, or tokens.** It only knows three things: Tensors, Blocks, and ComputeRequests.

### What the runtime sees

```python
# This is all GAMR knows about the computation
@dataclass
class ComputeRequest:
    input_tensors:   List[BlockRef]       # "I need these blocks in VRAM"
    kernel_fn:       Callable             # "Run this function on them"
    output_tensors:  List[BlockRef]       # "Store result here"
    priority:        int                  # Scheduling priority
```

### What the AI model adapter does (separate layer)

```
┌─────────────────────────────────┐
│  AI Model Adapter               │
│  (knows about transformers)     │
│                                 │
│  Converts:                      │
│    Attention Layer    →  ComputeRequest  │
│    FFN Block          →  ComputeRequest  │
│    LayerNorm          →  ComputeRequest  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  GAMR Runtime                   │
│  (knows nothing about AI)       │
│                                 │
│  Handles:                       │
│    Tensor → Block → Schedule    │
│    Event → Cost → Action        │
└─────────────────────────────────┘
```

This means GAMR works for:
- LLMs (transformer blocks → ComputeRequests)
- Diffusion models (U-Net blocks → ComputeRequests)
- Vision Transformers (patch attention → ComputeRequests)
- Speech models (encoder/decoder → ComputeRequests)

---

## Comparison Progression (Not AirLLM First)

> [!IMPORTANT]
> We compare against our own previous versions first. Only after V3 is proven do we compare against external systems.

```
V1: Naive Streaming
────────────────────────────
Load block → Wait → Compute → Wait → Unload → Repeat

V2: Pipelined Streaming
────────────────────────────
Load N+1 while computing N (double buffering)
Fixed block size, no adaptation

V3: Adaptive Streaming (GAMR)
────────────────────────────
Event-driven, cost-model-guided, self-tuning

External Comparison (only after V3)
────────────────────────────
vs. AirLLM
vs. FlexGen
vs. llama.cpp offloading
```

**Research question for each step:**
- V1 → V2: "How much does pipelining reduce GPU idle time?"
- V2 → V3: "How much does adaptation improve over fixed-block pipelining?"
- V3 → External: "How does HAMR compare to production systems?"

---

## Research Questions (Not Performance Promises)

Replace all specific numbers with honest research questions:

> **RQ1**: For which block sizes does `compute_time >= transfer_time` on DRAM-less NVMe + RTX 3050?

> **RQ2**: What fraction of GPU idle time can pipelining eliminate compared to naive streaming?

> **RQ3**: Does adaptive block sizing outperform a fixed optimal block size found by experiment?

> **RQ4**: Can the Cost Model make better eviction decisions than LRU for transformer weight blocks?

> **RQ5**: What is the maximum model size runnable on 6 GB VRAM with acceptable throughput degradation?

> **RQ6**: Does HAMR's GPU utilization approach that of FlexGen or AirLLM, and under what conditions?

---

## Phase Breakdown

### Phase 1 — Foundation (Track B leads, Track A documents, Track C scaffolds)

**Goal**: Answer RQ1. Establish the exact hardware performance envelope before writing runtime code.

**Experiments (Track B)**:

```
exp01_ssd_bandwidth.py
  → Read blocks of size [1, 2, 4, 8, 16, 32, 64, 128] MB from SSD
  → Measure real throughput (not hdparm theoretical)
  → 20 trials each, report mean ± std

exp02_pcie_bandwidth.py
  → Transfer pinned CPU tensors → CUDA
  → Same block sizes
  → Measure DMA throughput

exp03_gpu_compute.py
  → Simulate transformer weight-matrix multiply (GEMM)
  → Same block sizes
  → Measure kernel time per block

exp04_overlap_test.py
  → Simultaneously: SSD read + VRAM copy + GPU compute
  → Measure actual vs theoretical time
  → Find where I/O hides behind compute
```

**Theory (Track A)**:
- Write the mathematical model: `stall_time = max(0, T_transfer - T_compute)`
- Derive minimum block size for zero stall given measured hardware numbers

**Engineering (Track C)**:
- Set up repo structure
- Create `gamr/core/block.py` and `gamr/core/event.py` stubs
- Set up `requirements.txt`, `setup.py`, CI
- Initialize git branches

---

### Phase 2 — Naive Baseline (V1)

**Goal**: Build the simplest correct streaming runtime. Establish the baseline all future versions beat.

```
V1 Runtime:
  For each block in execution order:
      read_from_ssd(block)        # Blocking
      copy_to_vram(block)         # Blocking
      gpu_compute(block)          # Execute
      free_vram(block)            # Blocking
```

**Key output**: `GPU_idle_time_V1` — the number we will spend the rest of the project reducing.

---

### Phase 3 — Pipelined Baseline (V2)

**Goal**: Double buffering. Load block N+1 while computing block N.

```
V2 Runtime:
  prefetch_thread: reads ahead N+1
  compute_thread:  executes current block
  synchronize via events
```

**Key output**: `GPU_idle_time_V2 < GPU_idle_time_V1` — quantify the improvement.

---

### Phase 4 — GAMR Adaptive Runtime (V3)

**Goal**: Event-driven + cost-model guided + adaptive block sizing.

- Implement full event queue
- Implement cost model
- Implement all 5 schedulers
- Block metadata collection starts here
- Adaptive block size + pipeline depth

**Key output**: `GPU_idle_time_V3 < GPU_idle_time_V2` — again quantify.

---

### Phase 5 — External Benchmarks & Paper

**Goal**: Compare V3 (GAMR) against AirLLM, FlexGen, llama.cpp offloading.

- Run the same workload on all systems
- Collect: tokens/sec, GPU utilization %, peak RAM usage, SSD bandwidth used
- Write HAMR paper using all collected results from Track B
- Publish to arXiv, target MLSys/NeurIPS

---

## Git Strategy

```
main
├── track/a-theory
│   └── phase/01-hardware-model
├── track/b-experiments
│   ├── phase/01-timing
│   └── phase/02-baseline
└── track/c-engineering
    ├── phase/01-scaffolding
    └── phase/02-naive-runtime
```

Commit format:
```
[TrackX][PhaseN] Brief title

What was done, why, and what was measured.
Result summary (no promises, just data).
```

---

## Immediate Next Actions (Day 1)

1. **Track C**: Set up full repo structure, install PyTorch + CUDA, create package stubs, push
2. **Track B**: Write and run `exp01_ssd_bandwidth.py` — get real SSD numbers
3. **Track A**: Write `research/theory/cost_model.md` — the mathematical foundation

These three happen in parallel, committed to their respective branches.

---

## Open Questions (Answered by User Feedback)

| Question | Decision |
|---|---|
| GGUF or PyTorch first? | **Plain PyTorch `.pt` tensors** — isolate runtime from format complexity |
| Package or research repo? | **Research repo first**, package structure added in Phase 4 |
| Training now or later? | **Inference only** for Phases 1–4, training deferred to future work |
| Paper audience? | **Academic** (MLSys/NeurIPS) using HAMR name; GAMR is the project/product name |
| Compare to AirLLM when? | **Only after V3 (GAMR adaptive) is working** — not before |
