# GAMR / HAMR — Stochastic Performance Model
## Phase 1.5 | Derived from Phase 1 Hardware Measurements

**Date**: June 29, 2026  
**Status**: COMPLETE — ready to feed into Phase 2 (Simulator)  
**Source data**: exp01–exp04, fit by `fit_distributions.py`

---

## 1. Overview

Phase 1 established that every hardware component in the HAMR pipeline has a measurable, empirical latency distribution. This document replaces all assumptions in the cost model with fitted equations and measured parameters.

The central scheduling condition HAMR must satisfy:

```
P(T_SSD(B) > W_overlap(B, D)) < ε

where:
  B         = block size (MB)
  D         = prefetch depth (number of blocks read ahead)
  W_overlap = total GPU compute available while D blocks transfer from SSD
  ε         = target stall probability (default: 5%)
```

---

## 2. SSD Transfer Model

**Distribution**: Lognormal — fits measured data well (KS test p > 0.05 for most sizes)

```
T_SSD(B) ~ Lognormal(μ_B, σ_B)

P(T_SSD > w) = 1 - Φ((ln(w) - μ_B) / σ_B)

where Φ is the standard normal CDF.
```

### Fitted Parameters (from exp01 + exp01b)

| Block | μ | σ | Mean (ms) | P95 (ms) | P99 (ms) | CV |
|---|---|---|---|---|---|---|
| 1 MB | -0.409 | 0.403 | 0.72 | 1.30 | 1.65 | 0.42 |
| 2 MB | -0.046 | 0.227 | 0.98 | 1.24 | 1.37 | 0.23 |
| 4 MB | 0.889 | 0.320 | 2.56 | 3.64 | 4.17 | 0.33 |
| 8 MB | 1.138 | 0.320 | 3.28 | 5.52 | 6.32 | 0.33 |
| 16 MB | 2.133 | 0.345 | 8.95 | 13.23 | 15.43 | 0.36 |
| **32 MB** | **3.119** | **0.474** | **25.30** | **51.15** | **65.90** | **0.50** |
| 64 MB | 3.823 | 0.347 | 48.59 | 76.83 | 89.82 | 0.36 |
| 128 MB | 4.278 | 0.399 | 78.08 | 138.04 | 166.30 | 0.42 |

> **Note on 32MB**: KS p=0.0007 — the lognormal is a reasonable but imperfect fit. The measured distribution may be bimodal (SLC-cache vs exhausted states). Use empirical P95/P99 directly for risk calculations.

### Key finding

The SSD is not a constant. It is a stochastic process. The coefficient of variation (CV ≈ 0.3–0.5) means a mean-based scheduler will be surprised regularly. **The simulator must sample from this distribution, not use the mean.**

---

## 3. PCIe Transfer Model (RAM → VRAM)

**Distribution**: Near-deterministic (CV ≈ 0.024–0.08 for most sizes). Model as linear.

```
T_PCIe(B) ≈ B_MB × 0.094 ms/MB + 0.341 ms

Effective bandwidth: ~10.68 GB/s (pinned memory, saturated for B ≥ 32MB)
Fit RMSE: 0.125 ms
```

### Risk simplification (from Phase 1 data)

```
SSD CV ≈ 0.50      (highly stochastic)
PCIe CV ≈ 0.08    (near-deterministic)

Therefore: Risk ≈ SSD Risk

T_total = T_SSD + T_PCIe ≈ T_SSD + constant
```

The simulator uses the full two-stage model but the scheduling intelligence focuses on the SSD stage.

### Special case: 4MB pinned

4MB pinned PCIe has a fat tail: P99 = 2.37ms vs mean = 0.58ms (1.4% spike rate).  
Use P99 when computing worst-case transfer bounds for this block size.

---

## 4. GPU Execution Model

**Source**: exp03 (CUDA Event timing, 50 trials each)

```
T_one_layer(B) ≈ T_FFN(B) + T_Attention + 2 × T_LayerNorm

Note: T_one_layer is a LOWER BOUND on W_overlap.
      Real inference includes CUDA overhead, memory ops, scheduler latency.
      Use as a conservative baseline for the simulator.
```

### Measured Values

| Block | T_GEMM (ms) | T_FFN (ms) | T_one_layer (ms) |
|---|---|---|---|
| 1 MB | 0.0168 | 0.0259 | 0.1306 |
| 2 MB | 0.0211 | 0.0383 | 0.1430 |
| 4 MB | 0.0364 | 0.0669 | 0.1717 |
| 8 MB | 0.0714 | 0.1365 | 0.2412 |
| 16 MB | 0.1271 | 0.2473 | 0.3520 |
| **32 MB** | **0.2278** | **0.4495** | **0.5542** |
| 64 MB | 0.4400 | 0.8683 | 0.9730 |
| 128 MB | 0.8483 | 1.6877 | 1.7924 |

### Shared ops (measured once at 32MB-equivalent hidden dim)

| Operation | Time (ms) |
|---|---|
| Attention (seq=512) | 0.0736 |
| LayerNorm | 0.0155 |
| Softmax | 0.0089 |

---

## 5. RAM Residency State Model

Phase 1 (exp04) demonstrated that the dominant performance variable is whether a block is **RAM-resident** before GPU demand. The simulator must track this explicitly.

```
Block State Machine:
  SSD  →  RAM_COLD  →  RAM_HOT  →  VRAM  →  GPU
          (reading)   (resident)   (xfer)  (compute)

Transition times:
  SSD  → RAM_COLD:  T_SSD(B) ~ Lognormal(μ_B, σ_B)   [stochastic]
  RAM_COLD → RAM_HOT: 0ms  (instantaneous on read completion)
  RAM_HOT → VRAM:   T_PCIe(B) ≈ B_MB × 0.094 + 0.341 ms  [near-constant]
  VRAM → GPU:       0ms  (already in compute unit)
  GPU → complete:   T_one_layer(B)                         [measured]

Stall condition:
  GPU demands block N but block N is still in RAM_COLD state
  Stall duration = remaining T_SSD for block N
```

**Scheduler objective**: Maximise fraction of blocks that are `RAM_HOT` when the GPU needs them.

---

## 6. P(stall | B, D) Prediction Table

W_overlap(B, D) = D × T_one_layer(B) [lower bound — single-token]

| D | 1MB | 2MB | 4MB | 8MB | 16MB | 32MB | 64MB | 128MB |
|---|---|---|---|---|---|---|---|---|
| 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 99% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 4 | 73% | 99% | 100% | 100% | 100% | 100% | 100% | 100% |
| 8 | 13% | 21% | 96% | 93% | 100% | 100% | 100% | 100% |
| 16 | **0.2%✓** | **0.0%✓** | 35% | 25% | 88% | 98% | 100% | 99% |
| 32 | **0.0%✓** | **0.0%✓** | **0.5%✓** | **0.2%✓** | 20% | 70% | 87% | 72% |
| 64 | ✓ | ✓ | ✓ | ✓ | **0.2%✓** | 17% | 19% | 12% |
| 128 | ✓ | ✓ | ✓ | ✓ | ✓ | **0.8%✓** | **0.2%✓** | **0.2%✓** |

✓ = meets 5% stall target

---

## 7. Minimum Required Prefetch Depth D*

To achieve P(stall) < 5% using the lower-bound W_overlap:

| Block | D* | T_SSD P95 | T_layer | Feasibility |
|---|---|---|---|---|
| 1 MB | 10 | 1.30 ms | 0.131 ms | ✅ < 32 layers |
| 2 MB | 10 | 1.24 ms | 0.143 ms | ✅ < 32 layers |
| 4 MB | 24 | 3.64 ms | 0.172 ms | ✅ < 32 layers |
| 8 MB | 22 | 5.52 ms | 0.241 ms | ✅ < 32 layers |
| 16 MB | 43 | 13.23 ms | 0.352 ms | ⚠️ > 32 layers |
| 32 MB | 89 | 51.15 ms | 0.554 ms | ⚠️ >> 32 layers |
| 64 MB | 84 | 76.83 ms | 0.973 ms | ⚠️ >> 32 layers |
| 128 MB | 78 | 138.04 ms | 1.792 ms | ⚠️ >> 32 layers |

> **Critical interpretation**: D* here uses a **lower bound** on W_overlap (one layer's compute). Real W_overlap will be larger because:
> - Multiple tokens processed per block (batching)
> - CUDA overhead and memory ops add to the overlap window
> - The SSD is reading block N+D while the GPU processes layers N through N+D-1
>
> The simulator (Phase 2) will compute a more accurate W_overlap by simulating full pipeline execution. D* values above 32 do NOT necessarily mean H1 is false — they mean H1 requires either batching, careful block-size selection (1-8MB), or deeper pipeline modelling.

---

## 8. Key Equations for the Simulator

```python
import numpy as np

# SSD: sample transfer time for block of size B_mb
def sample_T_ssd(B_mb: int, mu_table: dict, sigma_table: dict) -> float:
    mu = mu_table[B_mb]
    sigma = sigma_table[B_mb]
    return float(np.random.lognormal(mu, sigma))

# PCIe: deterministic transfer time
def T_pcie(B_mb: int) -> float:
    return B_mb * 0.094 + 0.341   # ms

# GPU: one-layer compute time
def T_compute(B_mb: int, t_layer_table: dict) -> float:
    return t_layer_table[B_mb]    # ms

# P(stall): analytical
from scipy import stats
def p_stall(B_mb: int, D: int, mu_B: float, sigma_B: float,
            t_layer: float) -> float:
    w = D * t_layer
    if w <= 0: return 1.0
    z = (np.log(w) - mu_B) / sigma_B
    return float(1 - stats.norm.cdf(z))
```

---

## 9. What Phase 2 (Simulator) Inherits From This Model

1. **SSD sampling function**: `sample_T_ssd(B_mb)` → lognormal draw
2. **PCIe model**: `T_pcie(B_mb)` → deterministic linear
3. **GPU model**: `T_compute(B_mb)` → measured lookup table
4. **Block state machine**: `SSD → RAM_COLD → RAM_HOT → VRAM → GPU`
5. **P(stall) calculator**: analytical or Monte Carlo from samples
6. **Target**: sweep (B, D) space to find configurations achieving `P(stall) < ε`

All parameters are stored in `performance_model_params.json` for the simulator to load directly.
