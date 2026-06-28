# Experiment 03 — GPU Execution Time

**Date:** 20260628_190700
**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
**Hypothesis:** H1, H4
**Git Commit:** 4dbc6e06e1ed967d58c81a721e1b7f0f9068ad3f

## H1 Empirical Evaluation

| Block | T_compute | T_xfer(mean) | T_xfer(P95) | M(P95) | Verdict |
|---|---|---|---|---|---|
| 1MB | 0.017ms | 1.296ms | 2.37ms | -2.353ms | ✗ STALL |
| 2MB | 0.021ms | 1.956ms | 3.494ms | -3.473ms | ✗ STALL |
| 4MB | 0.036ms | 4.215ms | 6.262ms | -6.226ms | ✗ STALL |
| 8MB | 0.071ms | 7.731ms | 13.078ms | -13.007ms | ✗ STALL |
| 16MB | 0.127ms | 19.376ms | 37.072ms | -36.945ms | ✗ STALL |
| 32MB | 0.228ms | 24.95ms | 54.458ms | -54.23ms | ✗ STALL |
| 64MB | 0.44ms | 51.284ms | 95.121ms | -94.681ms | ✗ STALL |
| 128MB | 0.848ms | 94.953ms | 167.034ms | -166.186ms | ✗ STALL |

## Observations

**Observation 1: Transfer dominates isolated compute on this hardware.**
For single-token GEMM, a 128MB weight block is computed in **0.848ms**. Transferring that same block from SSD to VRAM takes **94.953ms (Mean)** and **167.034ms (P95)**. The compute-to-transfer ratio is approximately 1:112.

**Observation 2: H1 is not supported for isolated single-token compute kernels.**
For all block sizes tested, `T_compute` (single GEMM) is orders of magnitude smaller than `T_transfer`. However, this does NOT mean H1 is false in general. A real inference runtime overlaps transfer with many consecutive layers, KV-cache operations, scheduler overhead, CUDA stream synchronization, and potentially multiple tokens. The correct formulation of H1 should use the full overlap window `W_overlap`, not a single kernel:
```
P(T_transfer > W_overlap)
where W_overlap = sum of all useful GPU work available before the next weight block is needed
```

**Observation 3: Achieved TFLOPS (~0.06–0.16) does not represent GPU capability.**
The RTX 3050 FP16 peak is ~8.9 TFLOPS. Our benchmarks achieved only 1-2% of peak. This is because the tested kernels are too small to saturate the GPU (small matrix dimensions, launch overhead dominating, insufficient occupancy). The correct interpretation is: "the benchmark achieved ~0.16 TFLOPS for this workload" — not "the GPU can only do 0.16 TFLOPS."

**Observation 4: Attention, LayerNorm, and Softmax are negligible in absolute timing.**
At hidden dim corresponding to 32MB parameters:
- LayerNorm: 0.0155 ms
- Softmax: 0.0089 ms
- Attention block: 0.0736 ms
- FFN block (two GEMMs + GELU): 0.4495 ms

## Conclusions

1. On this hardware, for isolated single-token compute kernels, storage and transfer latency dominate GEMM execution time by ~100×.
2. H1 cannot be evaluated from single-kernel microbenchmarks alone. The overlap window `W_overlap` (not a single GEMM) is the correct quantity to compare against transfer time.
3. A sustained pipeline experiment measuring actual overlap at varying prefetch depths is required before drawing broader conclusions about H1.
4. The stochastic model should use measured distributions, not the microbenchmark TFLOPS values, as inputs.

## Next Action
1. Run exp04_pipeline_overlap to measure W_overlap directly at prefetch depths 1, 2, 4, 8, 16.
2. Only after exp04 can H1 be properly evaluated.
3. Then define Phase 1.5 Stochastic Performance Model and move to Phase 2 (Simulator).

