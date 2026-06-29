# GAMR / HAMR — Final Research Plan
## Architecture: **LOCKED** | Version: 1.3 — Final

> **Status**: Architecture is permanently locked. No additions, no redesigns.
> Changes allowed only from: measured experiments, correctness issues, or profiled bottlenecks.
>
> **Rule**: No code written without explicit permission.
> Architecture changes only from: measured experiments, correctness issues, or profiled bottlenecks.
> Everything else goes to `future_ideas.md`.

---

## The Three Formal Hypotheses

These are the scientific foundation of the project. Every experiment either supports, refutes, or refines one of them.

### H1 — Stochastic Latency Hiding
> *"For some block sizes and prefetch depths achievable on our hardware, the probability that transfer time exceeds the available GPU overlap window is low enough that a prefetch scheduler can bound GPU stall frequency to an acceptable rate."*

Formally:
```
P(stall | B, D) = P(T_transfer(B) > W_overlap(B, D))

where:
  W_overlap(B, D) = total useful GPU work available while D blocks transfer
                   (NOT a single GEMM — includes multiple layers, KV-cache,
                    attention, scheduling overhead, sync, etc.)

H1 confirmed  ↔  ∃(B, D) such that P(stall | B, D) < ε

where ε is determined experimentally (e.g. 5% stall rate = P95 stall bound).
```

**Key insight from Exp01**: Transfer time is stochastic, not deterministic.
**Key insight from Exp03**: H1 is NOT supported for isolated single-token compute
kernels (T_compute ≈ 0.85ms vs T_transfer ≈ 95ms for 128MB blocks). However,
this does not mean H1 is false in general. The correct comparison is against
W_overlap (the full overlap window), not a single kernel.

**If TRUE**: HAMR can schedule prefetch depth D to target P(stall) < ε.
**If marginal**: HAMR reduces stall frequency but cannot eliminate it.
**If FALSE**: Document gap; the compute-transfer mismatch itself is a publishable finding.

---

### H2 — Adaptive Advantage

### H4 — Distribution-Aware Scheduling (Created by Exp01)
> *"A scheduler that uses the full latency distribution (mean, P95, variance) to make prefetch decisions outperforms a mean-based scheduler under stochastic storage latency."*

**Why this exists**: Exp01 showed P95 latency is 2× the mean for 32MB reads.
A mean-based scheduler will stall 5% of the time. A P95-aware scheduler can pre-absorb these spikes.

**Tested in**: Phase 5 (V3) vs Phase 4 (V2), with H4 specifically comparing risk-aware vs mean-only cost functions.

---
> *"An adaptive scheduler (HAMR V3) reduces GPU idle time compared to a fixed-block pipelined scheduler (V2) across multiple hardware configurations and model sizes."*

**Tested in**: Phase 5 (V3) vs Phase 4 (V2).
**Measured by**: GPU idle %, tokens/sec, stall frequency.

---

### H3 — Simulator Fidelity
> *"The HAMR Simulator predicts real runtime behavior with measurable, quantified accuracy across all three runtime versions (V1, V2, V3)."*

**Tested in**: Phase 6 (Simulation vs Reality).
**Metrics** (threshold determined AFTER experiments, not before):
- **MAPE** — Mean Absolute Percentage Error (prediction accuracy)
- **R²** — Pearson correlation (prediction direction)
- **RMSE** — Root Mean Square Error (absolute timing error)

> [!NOTE]
> We do NOT hardcode an error threshold (e.g. 15%) before we have data.
> After Phase 6 measurements we determine what error level is acceptable given the use case.
> The threshold is a result, not an assumption.

**If prediction is strong**: The simulator is a validated design tool — publishable independently.
**If prediction is weak**: We analyze where the model breaks down. Still scientifically valuable.

---

## Final Roadmap

```
Phase 0 — Freeze Architecture                    ✅ COMPLETE
     ↓
Phase 1 — Measure Hardware (5 experiments)       ← WE ARE HERE
     ↓
Phase 1.5 — Mathematical Performance Model
     ↓
Phase 2 — HAMR Simulator
     ↓
Phase 3 — Naive Runtime (V1)
     ↓
Phase 4 — Pipelined Runtime (V2)
     ↓
Phase 5 — Adaptive Runtime (V3 = HAMR)
     ↓
Phase 6 — Simulation vs Reality Validation
     ↓
Phase 7 — One Excellent Paper
```

---

## Phase 0 — Freeze ✅ COMPLETE

Files locked:
- `ARCHITECTURE_v1.md` — **permanently locked**, do not modify
- `future_ideas.md` — add unvalidated ideas here, not to architecture
- `gamr/core/chunk.py`, `event.py`, `cost_model.py`, `runtime_state.py` — scaffolded

### Correct Comparison Framing

The objective is NOT to outperform a fully VRAM-resident model. That is physically impossible given PCIe bandwidth limits.

The correct research claim is:

> *"How close can HAMR get to full-VRAM performance while running models far larger than VRAM?"*

| Scenario | Role |
|---|---|
| Full model in VRAM | Upper performance bound (not our target) |
| Naive streaming (V1) | Internal baseline |
| Pipelined streaming (V2) | Intermediate baseline |
| HAMR Adaptive (V3) | Best streaming result |
| AirLLM / FlexGen | External comparison |

### Uncertainty Throughout

All measured quantities must be stored as **mean ± σ**, not as point estimates.

```
7 GB/s           ← not this
7.02 ± 0.18 GB/s ← this
```

This enables the Phase 2 simulator to run **Monte Carlo sweeps**:
```
1000 simulation runs
  → sample bandwidth from N(7.02, 0.18²)
  → run scheduler
  → collect GPU idle distribution

Result: GPU Idle = 6.1% [95% CI: 5.8–6.4%]
```
That is a publishable result. A single number is not.

---

## Phase 1 — Hardware Measurement

**Goal**: Replace all assumptions in the cost model with real measured numbers.

### Research Notebook Format (Required for every experiment)

```
Experiment ID:     exp01 / exp02 / ...
Date:              YYYY-MM-DD
Objective:         One sentence
Hypothesis:        Which of H1/H2/H3 does this address?
Hardware:          Exact machine state (GPU temp, RAM free, etc.)
Independent var:   What we vary
Dependent var:     What we measure
Trials:            N (minimum 20 for timing)
Raw data:          JSON file
Statistical summary: mean ± std, 95% CI, min, max
Observations:      What we actually saw
Conclusions:       What this means for the design
Next action:       What experiment or code change follows from this
```

Every experiment result is a document, not just a number.

---

### Experiment 1 — `exp01_ssd_bandwidth.py`

**Addresses**: H1 (transfer time, first stage)

```
Objective:        Measure SSD read throughput and latency for block sizes 1–128 MB
Independent var:  Block size (MB), access pattern (sequential / random)
Dependent var:    Throughput (GB/s ± σ), Latency (ms ± σ), full distribution
Output:           raw.json, summary.json, notebook.md, plots/, hardware_snapshot.json
```

**Required before running (7-point checklist)**:

| # | Issue | Severity | Fixed? |
|---|---|---|---|
| 1 | Cache drop: 3-level strategy (sudo → warn → O_DIRECT) | Critical | ☐ |
| 2 | Random read mode with seed-aligned offsets (seed=42) | High | ☐ |
| 3 | Research notebook metadata in output (git commit, env, hypothesis) | Medium | ☐ |
| 4 | Warm-up: 5 runs (not 3), excluded from all statistics | Medium | ☐ |
| 5 | Full statistics: mean, median, std, variance, 95% CI, P5, P95, min, max | High | ☐ |
| 6 | Structured output: raw.json + summary.json + notebook.md + hardware_snapshot.json | Medium | ☐ |
| 7 | Thermal snapshot: SSD/GPU/CPU temperature before and after | Medium | ☐ |

---

### Experiment 2 — `exp02_pcie_bandwidth.py`

**Addresses**: H1 (transfer time, second stage)

```
Objective:       Measure RAM → VRAM transfer speed
Independent var: Block size (MB), memory type (pinned vs pageable)
Dependent var:   Throughput (GB/s), latency (ms)
Output:          JSON + CSV + plot
```

---

### Experiment 3 — `exp03_gpu_compute.py`

**Addresses**: H1 (compute time measurement — the other side of the inequality)

```
Objective:       Measure GPU GEMM compute time for transformer-shaped weight blocks
Independent var: Block size (MB), hidden dimension
Dependent var:   Kernel time (ms), achieved TFLOPS
Special:         Cross-reference vs exp01 to produce H1 verdict table
Output:          JSON + cross-reference table
```

---

### Experiment 4 — `exp04_pipeline_overlap.py`

**Addresses**: H1 (measures W_overlap directly — the sustained pipeline experiment)

```
Objective:       Measure GPU utilization under sustained pipelined execution at
                 varying prefetch depths. This is the only experiment that can
                 properly evaluate H1, because it measures W_overlap, not isolated kernels.
Independent var: Prefetch depth (1, 2, 4, 8, 16), block size (MB)
Dependent var:   GPU idle %, tokens/sec, stall count, sustained throughput
Design:          Simulate a HAMR-like pipeline:
                   - SSD read thread fills a queue of D blocks
                   - GPU compute thread consumes from the queue
                   - Measure: time GPU spends waiting for data vs computing
Output:          JSON + pipeline efficiency curve + stall histogram
```

---

### Experiment 5 — `exp05_kernel_overhead.py` ← NEW

**Addresses**: H1 (runtime execution cost beyond just GEMM)

```
Objective:       Measure VRAM → GPU → VRAM round-trip costs that are NOT
                 captured by the GEMM kernel time alone
Measurements:
  - CUDA kernel launch latency (time from dispatch to first instruction)
  - CUDA synchronize() cost
  - torch.cuda.empty_cache() cost
  - Memory allocator overhead (alloc + free in VRAM)
  - CUDA stream creation overhead
  - Stream context switch cost
Why:            These overheads will be part of every compute operation in
                the real runtime. If they're significant vs block time,
                they must be included in the cost model.
Output:          JSON + overhead breakdown table
```

---

## Phase 1.5 — Stochastic Performance Model

**Goal**: Derive stochastic models for every hardware component. The simulator samples from these distributions. All parameters are distributions, not constants.

> **Research direction shift (from Exp01)**: SSD transfer time has P95 ≈ 2× Mean.
> All performance models must capture this. Constants are forbidden.

### Models to Derive (Track A — Systems Model)

**SSD Transfer Model**
```
T_ssd(B) = B / bandwidth_ssd + latency_ssd
where:
  B             = block size (bytes)
  bandwidth_ssd = measured peak from exp01 (GB/s)
  latency_ssd   = measured per-request latency overhead
```

**PCIe Transfer Model**
```
T_pcie(B) = B / bandwidth_pcie + latency_pcie
where values from exp02
```

**Full Transfer Pipeline Model**
```
T_transfer(B) = T_ssd(B) + T_pcie(B)
(sequential pipeline — both stages must complete)
```

**GPU Compute Model**
```
T_compute(B, H) = f(B, H) / tflops_gpu + overhead_kernel
where:
  f(B, H) = 2 × batch × H × (B / (2 × H))   [GEMM FLOPs]
  overhead_kernel = measured from exp05
```

**Stall Model**
```
M(B)       = T_compute(B, H) − T_transfer(B)
T_stall(B) = max(0, −M(B))       [stall when M < 0]

H1 confirmed  ↔  ∃B such that M(B) > σ_transfer(B)
H1 marginal   ↔  ∃B such that 0 < M(B) ≤ σ_transfer(B)
H1 refuted    ↔  ∀B, M(B) ≤ 0
```

**Queue Delay Model** *(Initial Approximation — will be refined by Phase 1 data)*
```
T_queue(D) ≈ D × T_transfer(B)
where D = prefetch depth

Simplified bound: GPU avoids stalls if D ≥ ceil(T_transfer / T_compute)

This approximation assumes:
  - Infinite RAM and VRAM (no eviction pressure)
  - No queue management overhead
  - No scheduling delay
  - No synchronization cost

The real optimal D = f(T_transfer, T_compute, queue_overhead,
                       memory_capacity, scheduler_latency)

Phase 1 measurements + Phase 2 simulator will produce a better model.
Do not treat this equation as accurate until validated.
```

**GPU Execution Model** *(replaces "GPU Compute Model" — broader scope)*
```
T_execution(B) = T_compute(B)     [GEMM kernel time]
               + T_overhead        [kernel launch + sync + alloc + free]

where:
  T_compute(B, H) = 2 × batch × H × (B / (2H)) / tflops_gpu
  T_overhead      = measured from exp05 (kernel launch + sync + VRAM alloc)

Transformers are NOT only GEMM. They include:
  Softmax, LayerNorm, Attention, memory reads,
  kernel launches, synchronization.

For Phase 1.5: T_execution uses T_compute + T_overhead as measured.
For Phase 3+: each operation type can plug in its own T_compute.
```

**Pipeline Overlap Model**
```
T_total(N) ≈ N × max(T_execution(B), T_transfer(B))
Efficiency  = T_execution(B) / max(T_execution(B), T_transfer(B))

(Approximation — ignores queue overhead; validated in Phase 6)
```

### Deliverable

`research/theory/performance_model.md` — a document with all derived equations, parameterized by Phase 1 measurements, with a prediction table for GPU idle % at each block size.

This document feeds directly into Phase 7 paper sections 3–5.

---

## Phase 2 — HAMR Simulator

**Goal**: A discrete-event simulator that evaluates scheduling algorithms using Phase 1.5 equations.

### What it is NOT

- Not a neural network simulator
- Not a transformer emulator
- Not an LLM inference tool

### What it IS

A tool that answers: *"Given these hardware speeds and this scheduling algorithm, what is the predicted GPU idle %?"*

```python
# The entire simulator API
hw_profile = HardwareProfile.from_json("experiments/results/")
model_graph = SyntheticExecutionGraph(n_ops=1000, chunk_size_mb=16)

sim = HAMRSimulator(hardware=hw_profile, graph=model_graph)

result = sim.run(scheduler=NaiveScheduler())
print(result.gpu_idle_pct)        # → 34.2%

result = sim.run(scheduler=PipelinedScheduler(depth=4))
print(result.gpu_idle_pct)        # → 8.1%

result = sim.run(scheduler=AdaptiveScheduler())
print(result.gpu_idle_pct)        # → 5.8%
```

### Sweep capability (the key research tool)
```python
# Test 1,000 configurations in seconds — no hardware needed
results = sim.sweep(
    block_sizes=[1, 2, 4, 8, 16, 32, 64],
    depths=[2, 4, 8, 16, 32],
    schedulers=[NaiveScheduler, PipelinedScheduler, AdaptiveScheduler],
)
# → Produces a 3D heatmap: block_size × depth × GPU_idle
```

---

## Phases 3–5 — Runtime Implementation

| Version | Name | Key feature | Compared to |
|---|---|---|---|
| V1 | Naive | Load → Wait → Compute → Wait → Unload | Simulator V1 prediction |
| V2 | Pipelined | Double buffer, fixed block size | Simulator V2 + V1 actual |
| V3 | Adaptive (HAMR) | Event-driven, cost model, self-tuning | Simulator V3 + V2 actual |

Each version measured against simulator prediction → validates H3.

---

## Phase 6 — Simulation vs Reality

**Goal**: Validate H3. Quantify simulator fidelity.

| Metric | Sim V1 | V1 Real | Sim V2 | V2 Real | Sim V3 | V3 Real |
|---|---|---|---|---|---|---|
| GPU idle % | ? | ? | ? | ? | ? | ? |
| Stall count | ? | ? | ? | ? | ? | ? |
| Throughput | ? | ? | ? | ? | ? | ? |

Then: HAMR V3 vs AirLLM, FlexGen, llama.cpp.
**External comparison only happens here.**

---

## Phase 7 — One Excellent Paper

> **Plan one paper. Split later only if the data supports it.**

Title: *"HAMR: Hierarchical Adaptive Memory Runtime for Large AI Models"*

If the simulator results are strong enough to stand alone, split to:
- *"HAMR-Sim: A Simulator for Hierarchical AI Memory Scheduling"*

If the cost model demonstrates learning behavior that's independently interesting, split to:
- *"Adaptive Cost Modeling for AI Execution Runtimes"*

**Decision point**: After Phase 6 data is complete.

Target: arXiv preprint → MLSys 2027 or NeurIPS Systems Track 2027.

---

## What Changes the Architecture

Only three things are allowed to change `ARCHITECTURE_v1.md`:

1. **A measured experiment** contradicts a design assumption
2. **A correctness issue** — the design cannot support a required feature
3. **A reproducible profiling bottleneck** — identified by profiler, not intuition

Everything else → `future_ideas.md`.

---

## Immediate Next Step

**Review `exp01_ssd_bandwidth.py` together.**

This is the most important file in Phase 1. The numbers it produces will:
- Feed the SSD Transfer Model in Phase 1.5
- Calibrate the Simulator in Phase 2
- Either confirm or challenge H1
- Determine what block sizes the runtime uses

---

## Project Status

```
Architecture      ████████████████████  100%  LOCKED
Research Planning ████████████████████  100%  LOCKED
Theory            ████████████████░░░░   75%  In progress
Experiments       ████████████████░░░░   75%  exp01-03 complete, exp04-05 pending
Runtime           ░░░░░░░░░░░░░░░░░░░░    0%  Blocked on experiments + Phase 1.5
Paper             ░░░░░░░░░░░░░░░░░░░░    0%  Blocked on runtime
```

**Next action**: Run exp04_pipeline_overlap.py (sustained prefetch depth) to properly evaluate H1 with W_overlap.
