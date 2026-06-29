# Experiment 02 — PCIe Bandwidth Distribution

**Date:** 20260628_185613
**Objective:** Measure PCIe RAM→VRAM transfer time as a stochastic distribution
**Hypothesis:** H1, H4
**Git Commit:** 154826c8dfe9649a4f4384650d5c62bc72343509
**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU

**Key Question:** Is PCIe more predictable (lower CV) than the SSD?

**SSD Reference (from exp01b):** 32MB block: Mean=25.3ms, P95=51.1ms, CV≈0.5

## Observations

**Observation 1: PCIe is 7-8× faster AND far more predictable than the SSD.**
At 32MB: SSD Mean=25.3ms P95=51.1ms CV=0.50 vs PCIe-Pinned Mean=3.31ms P95=3.64ms CV=0.08.
PCIe behaves almost deterministically. SSD is highly stochastic.

**Observation 2: Risk is dominated by SSD, not PCIe.**
SSD Risk >> PCIe Risk. Therefore: Risk ≈ SSD Risk.
Architectural implication: once data reaches RAM, the PCIe stage can be treated as near-constant in the scheduler.
The adaptive logic should concentrate entirely on the SSD → RAM stage.

**Observation 3: PCIe saturates at ~10-11 GB/s for blocks ≥ 32MB.**
Making blocks larger will not improve PCIe throughput. Block size decisions should be driven by SSD behavior and GPU compute overlap, not PCIe saturation.

**Observation 4: Pinned memory matters most for small blocks.**
1MB: Pinned=3.88 GB/s vs Pageable=2.45 GB/s (58% faster).
128MB: Pinned=11.15 GB/s vs Pageable=9.57 GB/s (16% faster).
HAMR should selectively pin only blocks in the active prefetch window, not all memory. Pinned memory is a limited resource.

**Observation 5: 4MB pinned block has anomalous CV=0.77 (requires investigation).**
One trial took ~3.25ms while most took ~0.6ms. This is a single outlier, likely a CUDA runtime or OS event.
Do NOT incorporate into performance model until confirmed by exp02b (100-trial rerun at 4MB pinned).

## Conclusions

The PCIe link (RAM→VRAM) is nearly deterministic on this hardware (CV≈0.08).
This simplifies HAMR's design: the adaptive/risk-aware scheduling intelligence belongs entirely at the SSD→RAM boundary.
The PCIe stage can be modeled as a constant (or near-constant) in Phase 1.5.

The central inequality for H1 now simplifies:
  P(stall) ≈ P(T_SSD > T_compute)  [PCIe term is small and stable, nearly absorbed]

## Next Action

1. Run exp02b to investigate the 4MB pinned anomaly before it enters the model.
2. Proceed to exp03 (GPU compute) to measure T_compute for GEMM, Attention, LayerNorm, Softmax, and full transformer block.
3. After exp03: compute P(stall | B) = P(T_SSD(B) + T_PCIe(B) > T_compute(B)) for the first time.
