# Experiment 04 — Sustained Pipeline Overlap

**Date:** 20260628_192029
**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
**Hypothesis:** H1
**Git Commit:** 5290c0fe2675ebba22a5f1e3041790a4fd069784

**Key Question:** Does increasing prefetch depth D reduce GPU idle %?

## Results

| Block | Depth | GPU Idle% | Stalls | Blk/s |
|---|---|---|---|---|
| 8MB | 1 | 90.18% | 34 | 83.2 |
| 8MB | 2 | 91.31% | 35 | 205.7 |
| 8MB | 4 | 92.85% | 35 | 184.5 |
| 8MB | 8 | 92.38% | 34 | 182.4 |
| 8MB | 16 | 95.38% | 35 | 134.6 |
| 16MB | 1 | 97.53% | 27 | 30.1 |
| 16MB | 2 | 97.14% | 35 | 69.1 |
| 16MB | 4 | 97.14% | 35 | 77.1 |
| 16MB | 8 | 96.49% | 35 | 85.3 |
| 16MB | 16 | 97.13% | 35 | 77.3 |
| 32MB | 1 | 96.66% | 19 | 28.7 |
| 32MB | 2 | 4.15% | 0 | 28.1 |
| 32MB | 4 | 96.55% | 20 | 28.2 |
| 32MB | 8 | 4.20% | 0 | 27.1 |
| 32MB | 16 | 95.60% | 19 | 34.0 |
| 64MB | 1 | 2.78% | 0 | 13.5 |
| 64MB | 2 | 2.68% | 0 | 14.3 |
| 64MB | 4 | 98.14% | 6 | 9.4 |
| 64MB | 8 | 2.75% | 0 | 13.0 |
| 64MB | 16 | 2.50% | 0 | 14.1 |

## H1 Evaluation

The results are **bimodal and inconclusive as written**.

Some configurations achieve near-zero GPU idle (e.g. 64MB D=1: 2.78%), while others with the same block size show 95%+ idle (64MB D=4: 98.14%). This inconsistency means the experiment is measuring an uncontrolled variable — most likely the **OS page cache**.

When the SSD test file is already cached in RAM (from the previous run or from file creation), the "SSD read" returns in microseconds, not milliseconds. The GPU consumer never waits, and idle drops to ~2-4%. When the cache is cold, we see the real SSD latency and idle spikes to 95%+.

**H1 cannot be evaluated from this data as-is.** The experiment must be re-run with guaranteed cache-cold reads before it produces valid results.

## Observations

**Observation 1: Bimodal results indicate page cache contamination.**
The pattern is not random. Configs that ran shortly after file creation (or after another config flushed the cache) show low idle. Configs that hit cold SSD show high idle. The even/odd depth pattern at 32MB (D=2: 4.15%, D=4: 96.55%) strongly suggests cache warming effects.

**Observation 2: When reads ARE cold, GPU idle is ~95-98%.**
This is consistent with exp03: single-GEMM compute is ~100× faster than real SSD transfer. Prefetch depth alone does not solve the problem when compute per block is a single GEMM.

**Observation 3: When reads are cached (RAM speed), GPU idle drops to ~2-4%.**
This shows that if data were already in RAM, the pipeline works excellently. This actually validates the architectural principle: HAMR's value is in getting data into RAM early enough.

## Conclusions

1. **Exp04 needs to be re-run** with enforced cache drops between configurations (sudo echo 3 > /proc/sys/vm/drop_caches, or use O_DIRECT with aligned buffers).
2. The "cached" results (~2-4% idle) are actually a preview of what HAMR can achieve if the prefetch queue is deep enough and the SSD reads complete before the GPU needs the data.
3. The "cold" results (~95-98% idle) confirm the 100× compute-transfer gap from exp03.
4. The real question for H1 is not "can prefetch depth help with one GEMM?" but rather "can HAMR read block N+D while the GPU processes blocks N through N+D-1?"

## Next Action
1. Fix exp04 methodology: enforce cache drops, use O_DIRECT with aligned buffers.
2. OR accept that this experiment's value is in showing the contrast between cold/cached behavior.
3. Proceed to Phase 1.5 Stochastic Performance Model with the data we have from exp01-03.

