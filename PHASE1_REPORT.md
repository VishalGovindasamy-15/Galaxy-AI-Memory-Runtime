# GAMR / HAMR — Phase 1 Research Report
## Hardware Characterization Complete

**Date**: June 28, 2026  
**Hardware**: Pop!_OS 24.04 | RTX 3050 6GB Laptop | DRAM-less NVMe SSD  
**Status**: Phase 1 experiments complete. Ready for Phase 1.5 (Stochastic Performance Model).

---

## Executive Summary

Phase 1 measured three hardware components that form the HAMR execution pipeline: **SSD → RAM → VRAM → GPU**. The key finding is that the system is **transfer-bound, not compute-bound**, and that transfer latency is **stochastic, not deterministic**. This reshapes HAMR from a "bandwidth-hiding" runtime into a **risk-aware scheduling system**.

---

## Experiments Conducted

| Experiment | What it measures | Trials | Status |
|---|---|---|---|
| exp01 | SSD read bandwidth + latency (sequential/random, 1-128MB) | 20/size | ✅ Complete |
| exp01b | SSD repeatability (32MB, 100 trials) | 100 | ✅ Complete |
| exp02 | PCIe RAM→VRAM bandwidth (pinned/pageable, 1-128MB) | 20/size | ✅ Complete |
| exp02b | PCIe 4MB anomaly investigation (500 trials) | 500 | ✅ Complete |
| exp03 | GPU compute time (GEMM, FFN, Attention, LayerNorm, Softmax) | 50/op | ✅ Complete |
| exp04 | Sustained pipeline overlap (prefetch depth 1-16) | 40/config | ⚠️ Cache-contaminated |

---

## Key Findings

### Finding 1: SSD is Slow AND Stochastic

| Metric | 32MB Block |
|---|---|
| Mean latency | 25.30 ms |
| P95 latency | 51.15 ms |
| CV (σ/μ) | ~0.50 |
| Effective bandwidth | ~1.5–2.2 GB/s |

The SSD P95 latency is **2× the mean**. This is the most important finding of Phase 1.

**Implication**: The cost model cannot use constants. Every SSD parameter must be a distribution.

### Finding 2: PCIe is Fast AND Predictable

| Metric | 32MB Pinned |
|---|---|
| Mean latency | 3.31 ms |
| P95 latency | 3.64 ms |
| CV (σ/μ) | 0.08 |
| Effective bandwidth | ~10-11 GB/s (saturated) |

PCIe CV ≈ 0.08 vs SSD CV ≈ 0.50. PCIe is nearly deterministic.

**Implication**: Risk management belongs at the SSD→RAM boundary. Once data reaches RAM, the rest of the pipeline is predictable. Risk ≈ SSD Risk.

### Finding 3: GPU Compute is ~100× Faster Than Transfer

| Block | GEMM (ms) | Transfer Mean (ms) | Ratio |
|---|---|---|---|
| 1 MB | 0.017 | 1.296 | 76× |
| 32 MB | 0.228 | 24.950 | 109× |
| 128 MB | 0.848 | 94.953 | 112× |

**Implication**: H1 is NOT supported for isolated single-token GEMM kernels. However, this does not mean H1 is false in general — the correct comparison is against W_overlap (full overlap window), not a single kernel.

### Finding 4: 4MB PCIe Has a Fat Tail

exp02b confirmed rare spikes at 4MB pinned: 7/500 trials (1.4%) exceeded 1.5ms, with P99 = 2.37ms.

**Implication**: Use P99 (not mean) for 4MB PCIe in the model.

### Finding 5: Exp04 Pipeline Results Are Cache-Contaminated

Exp04 showed bimodal GPU idle: some configs at ~2-4% (cache-warm), others at ~95-98% (cache-cold). The inconsistency indicates OS page cache contamination, not a real scheduling effect.

**Implication**: Exp04 needs methodological fix (enforce O_DIRECT with aligned buffers). However, the "cached" results (~2-4% idle) preview what HAMR can achieve if prefetch succeeds.

---

## Hardware Baseline (For Phase 1.5 Equations)

All values below are **measured, not assumed**.

| Parameter | Value | Source |
|---|---|---|
| SSD seq bandwidth | ~1.5 GB/s | exp01 |
| SSD random bandwidth | ~1.7-2.5 GB/s | exp01 |
| SSD 32MB mean latency | 25.30 ms | exp01b |
| SSD 32MB P95 latency | 51.15 ms | exp01b |
| SSD latency CV | ~0.50 | exp01/exp01b |
| `dd iflag=direct` bandwidth | 2.2 GB/s | dd verification |
| PCIe pinned bandwidth (sat.) | ~10-11 GB/s | exp02 |
| PCIe 32MB mean latency | 3.31 ms | exp02 |
| PCIe latency CV | ~0.08 | exp02 |
| GPU GEMM 32MB | 0.228 ms | exp03 |
| GPU FFN 32MB | 0.450 ms | exp03 |
| GPU peak achieved TFLOPS | ~0.16 (for tested workload) | exp03 |
| NVMe temp (stable) | 62.85°C | exp01 snapshot |

---

## Hypothesis Status After Phase 1

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1** — Stochastic Latency Hiding | **Not yet evaluable** | Isolated GEMM is 100× faster than transfer. H1 requires W_overlap measurement (full overlap window across multiple layers). Exp04 was contaminated. |
| **H2** — Adaptive Advantage | **Not yet tested** | Requires V2 vs V3 runtime comparison (Phase 4-5). |
| **H3** — Simulator Fidelity | **Not yet tested** | Requires simulator (Phase 2) + runtime (Phase 3-5). |
| **H4** — Distribution-Aware Scheduling | **Supported by data** | SSD P95 = 2× mean. A mean-based scheduler will stall ~5% of the time. Distribution-aware scheduling is justified by the measured variance. |

---

## Research Direction Shift

**Before Phase 1**: HAMR was about *hiding bandwidth*.

**After Phase 1**: HAMR is about *managing uncertainty in hierarchical memory systems*.

The central objective changed from:
```
Minimize average transfer time
```
to:
```
Minimize P(stall) — the probability of a GPU stall event
```

This is a deeper and more original research direction.

---

## Next Steps

1. **Phase 1.5**: Build the Stochastic Performance Model using measured distributions (not constants)
2. **Phase 2**: Build the HAMR Simulator sampling from these distributions
3. The simulator should answer: *"What prefetch depth D and block size B minimize P(stall)?"*

---

## File Manifest

| File | Purpose |
|---|---|
| `experiments/exp01_ssd_bandwidth.py` | SSD bandwidth benchmark |
| `experiments/exp01b_repeatability.py` | SSD tail latency (100 trials) |
| `experiments/exp01_analysis.py` | Plot generation for exp01 |
| `experiments/exp02_pcie_bandwidth.py` | PCIe bandwidth distribution |
| `experiments/exp02b_pcie_4mb_rerun.py` | PCIe 4MB anomaly (500 trials) |
| `experiments/exp03_gpu_compute.py` | GPU compute benchmarks |
| `experiments/exp04_pipeline_overlap.py` | Pipeline overlap (needs fix) |
| `experiments/results/exp01_*/` | Raw data, plots, notebooks |
| `experiments/results/exp02_*/` | Raw data, plots, notebooks |
| `experiments/results/exp03_*/` | Raw data, plots, notebooks |
| `experiments/results/exp04_*/` | Raw data, plots, notebooks |
| `ARCHITECTURE_v1.md` | Frozen system architecture |
