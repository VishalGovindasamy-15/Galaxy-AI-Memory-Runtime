"""
research/theory/fit_distributions.py
======================================
Phase 1.5 — Stochastic Performance Model: Distribution Fitting

PURPOSE
───────
Reads raw Phase 1 experiment data and:
  1. Fits probability distributions to SSD latency (lognormal)
  2. Fits a linear bandwidth model to PCIe transfer times
  3. Tabulates GPU compute times from exp03
  4. Computes P(stall | B, D) for a range of prefetch depths
  5. Computes the minimum prefetch depth D* required to meet a target stall rate
  6. Writes fitted parameters to performance_model_params.json
  7. Generates prediction plots

OUTPUTS
───────
  research/theory/performance_model_params.json   ← simulator reads this
  research/theory/plots/ssd_distribution_fits.png
  research/theory/plots/pstall_vs_depth.png
  research/theory/plots/required_depth.png
"""

import json
import math
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
RESULTS = BASE / "experiments" / "results"
OUT_DIR = Path(__file__).parent
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ─── Load Phase 1 raw data ────────────────────────────────────────────────────

def load_ssd_data():
    """Load exp01 raw SSD read times per block size and mode."""
    raw_path = RESULTS / "exp01_20260628_134323" / "raw.json"
    with open(raw_path) as f:
        data = json.load(f)
    # {(block_size_mb, mode): [times_ms]}
    by_size_mode = {}
    for r in data["results"]:
        key = (r["block_size_mb"], r["mode"])
        by_size_mode[key] = r["raw_times_ms"]
    return by_size_mode


def load_ssd_repeatability():
    """Load exp01b: 100-trial 32MB random SSD reads."""
    raw_path = RESULTS / "exp01b_20260628_184912" / "raw_1b.json"
    with open(raw_path) as f:
        data = json.load(f)
    return data["raw_times_ms"]


def load_pcie_data():
    """Load exp02 raw PCIe transfer times per block size and mode."""
    raw_path = RESULTS / "exp02_20260628_185613" / "raw.json"
    with open(raw_path) as f:
        data = json.load(f)
    by_size_mode = {}
    for r in data["results"]:
        key = (r["block_size_mb"], r["mode"])
        by_size_mode[key] = r["raw_times_ms"]
    return by_size_mode


def load_gpu_data():
    """Load exp03 GEMM and FFN compute times per block size."""
    raw_path = RESULTS / "exp03_20260628_190700" / "raw.json"
    with open(raw_path) as f:
        data = json.load(f)
    gemm = {int(k): v for k, v in data["gemm_results"].items()}
    ffn  = {int(k): v for k, v in data["ffn_results"].items()}
    other = data["other_ops"]
    return gemm, ffn, other


# ─── SSD Distribution Fitting ─────────────────────────────────────────────────

def fit_lognormal(times_ms: list) -> dict:
    """Fit a lognormal distribution to latency samples."""
    arr = np.array(times_ms)
    # Method of moments: match mean and variance
    mu_raw = np.mean(arr)
    var_raw = np.var(arr, ddof=1)
    # Lognormal params: σ² = ln(1 + CV²), μ = ln(E[X]) - σ²/2
    cv2 = var_raw / (mu_raw ** 2)
    sigma2 = math.log(1 + cv2)
    mu = math.log(mu_raw) - sigma2 / 2
    sigma = math.sqrt(sigma2)

    # Goodness of fit: KS test
    ks_stat, ks_p = stats.kstest(arr, "lognorm",
                                  args=(sigma, 0, math.exp(mu)))

    return {
        "distribution": "lognormal",
        "mu": round(mu, 5),
        "sigma": round(sigma, 5),
        "mean_ms": round(mu_raw, 4),
        "std_ms": round(math.sqrt(var_raw), 4),
        "cv": round(math.sqrt(cv2), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "ks_stat": round(ks_stat, 4),
        "ks_p_value": round(ks_p, 4),
        "n_samples": len(times_ms),
    }


def p_stall_lognormal(mu: float, sigma: float, w_overlap_ms: float) -> float:
    """P(T_SSD > W_overlap) for lognormal T_SSD."""
    if w_overlap_ms <= 0:
        return 1.0
    # P(X > w) = 1 - CDF(w) = 1 - Phi((ln(w) - mu) / sigma)
    z = (math.log(w_overlap_ms) - mu) / sigma
    return float(1 - stats.norm.cdf(z))


def min_depth_for_target(mu: float, sigma: float,
                          t_per_block_ms: float, epsilon: float = 0.05) -> int:
    """
    Minimum prefetch depth D such that P(T_SSD > D * t_per_block) < epsilon.
    Binary search on D.
    """
    for D in range(1, 2000):
        w = D * t_per_block_ms
        p = p_stall_lognormal(mu, sigma, w)
        if p < epsilon:
            return D
    return -1  # not achievable in range


# ─── PCIe Linear Model ────────────────────────────────────────────────────────

def fit_pcie_linear(pcie_data: dict) -> dict:
    """
    Fit T_pcie(B) = B / bandwidth + latency_offset
    using pinned memory data (more relevant for HAMR prefetch path).
    """
    sizes_mb = []
    means_ms = []
    for (bs, mode), times in pcie_data.items():
        if mode == "pinned":
            arr = np.array(times)
            sizes_mb.append(bs)
            means_ms.append(float(np.mean(arr)))

    sizes_mb = np.array(sizes_mb)
    means_ms = np.array(means_ms)
    sizes_bytes = sizes_mb * 1024 * 1024

    # Fit: T_ms = (B_bytes / bw_bytes_per_sec) * 1000 + offset_ms
    # → T_ms = (1000 / bw_GB_per_sec) * B_MB + offset_ms
    # Linear regression: T_ms = slope * B_MB + intercept
    coeffs = np.polyfit(sizes_mb, means_ms, 1)
    slope_ms_per_mb = coeffs[0]
    intercept_ms    = coeffs[1]

    bw_gb_per_s = 1000.0 / (slope_ms_per_mb * 1024)  # convert MB→GB
    residuals = means_ms - np.polyval(coeffs, sizes_mb)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # CV per block size
    cvs = {}
    for (bs, mode), times in pcie_data.items():
        if mode == "pinned":
            arr = np.array(times)
            cvs[bs] = round(float(np.std(arr, ddof=1) / np.mean(arr)), 4)

    return {
        "model": "linear",
        "slope_ms_per_mb": round(float(slope_ms_per_mb), 6),
        "intercept_ms":    round(float(intercept_ms), 4),
        "bandwidth_gb_per_s": round(bw_gb_per_s, 3),
        "fit_rmse_ms": round(rmse, 4),
        "cv_per_block": cvs,
        "note_4mb": "4MB pinned has fat tail: P99=2.37ms. Use P99 for risk model."
    }


# ─── GPU Compute Model ────────────────────────────────────────────────────────

def build_gpu_model(gemm_data: dict, ffn_data: dict, other_ops: dict) -> dict:
    """
    Build a lookup table of GPU execution times per block size.
    T_one_layer ≈ T_ffn + T_attention + T_layernorm (for one transformer layer)
    """
    layer_model = {}
    t_attention = other_ops["attention"]["mean_ms"]
    t_layernorm = other_ops["layernorm"]["mean_ms"]

    for bs in sorted(gemm_data.keys()):
        t_gemm = gemm_data[bs]["mean_ms"]
        t_ffn  = ffn_data[bs]["mean_ms"]
        # One transformer layer involves: 3 projections (Q,K,V), O projection,
        # 2 FFN projections, 2 LayerNorms, 1 Attention = simplify to FFN + attn + 2×LN
        t_layer = t_ffn + t_attention + 2 * t_layernorm
        layer_model[bs] = {
            "t_gemm_ms": round(t_gemm, 5),
            "t_ffn_ms": round(t_ffn, 5),
            "t_attention_ms": round(t_attention, 5),
            "t_layernorm_ms": round(t_layernorm, 5),
            "t_one_layer_ms": round(t_layer, 5),
            "note": "t_one_layer is a lower bound on W_overlap for 1 complete layer"
        }
    return layer_model


# ─── P(stall) Prediction Tables ──────────────────────────────────────────────

def compute_pstall_table(ssd_fits: dict, gpu_model: dict,
                          depths: list, epsilon: float = 0.05) -> list:
    """
    For each (block_size, prefetch_depth) compute P(stall).
    W_overlap(B, D) = D × T_one_layer(B)
    """
    rows = []
    for bs in sorted(gpu_model.keys()):
        if bs not in ssd_fits:
            continue
        fit = ssd_fits[bs]
        mu, sigma = fit["mu"], fit["sigma"]
        t_layer = gpu_model[bs]["t_one_layer_ms"]

        for D in depths:
            w_overlap = D * t_layer
            p_s = p_stall_lognormal(mu, sigma, w_overlap)
            rows.append({
                "block_mb": bs,
                "prefetch_depth": D,
                "w_overlap_ms": round(w_overlap, 4),
                "p_stall": round(p_s, 4),
                "stall_pct": round(p_s * 100, 2),
                "meets_target": p_s < epsilon,
            })
    return rows


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_ssd_fits(ssd_fits: dict, ssd_raw: dict):
    """Plot measured histogram vs fitted lognormal for each block size."""
    block_sizes = sorted(ssd_fits.keys())
    n = len(block_sizes)
    cols = 4
    rows_p = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_p, cols, figsize=(4 * cols, 3.5 * rows_p))
    fig.suptitle("SSD Latency: Measured vs Fitted Lognormal Distribution", fontsize=13)
    axes = axes.flatten()

    for i, bs in enumerate(block_sizes):
        ax = axes[i]
        fit = ssd_fits[bs]
        raw = ssd_raw.get((bs, "random"), ssd_raw.get((bs, "sequential"), []))
        if not raw:
            continue
        arr = np.array(raw)
        ax.hist(arr, bins=15, density=True, alpha=0.6, color="steelblue",
                edgecolor="black", label="Measured")

        # Fitted lognormal PDF
        x = np.linspace(arr.min() * 0.5, arr.max() * 1.5, 200)
        pdf = stats.lognorm.pdf(x, fit["sigma"], 0, math.exp(fit["mu"]))
        ax.plot(x, pdf, "r-", linewidth=2, label=f"Lognormal (KS p={fit['ks_p_value']:.2f})")
        ax.axvline(fit["p95_ms"], color="orange", linestyle="--", linewidth=1,
                   label=f"P95={fit['p95_ms']:.1f}ms")
        ax.set_title(f"{bs} MB")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ssd_distribution_fits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ssd_distribution_fits.png")


def plot_pstall_curves(pstall_table: list, depths: list):
    """P(stall) vs prefetch depth for each block size."""
    block_sizes = sorted(set(r["block_mb"] for r in pstall_table))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("P(stall) vs Prefetch Depth D\n[W_overlap = D × T_one_layer]", fontsize=13)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(block_sizes)))

    for i, bs in enumerate(block_sizes):
        bs_rows = [r for r in pstall_table if r["block_mb"] == bs]
        ds = [r["prefetch_depth"] for r in bs_rows]
        ps = [r["stall_pct"] for r in bs_rows]
        ax1.plot(ds, ps, "-o", label=f"{bs}MB", color=colors[i])
        ax2.semilogy(ds, [max(p, 0.001) for p in ps], "-o", label=f"{bs}MB",
                     color=colors[i])

    for ax in [ax1, ax2]:
        ax.axhline(5, linestyle="--", color="red", linewidth=1, label="ε=5% target")
        ax.set_xlabel("Prefetch Depth D (layers)")
        ax.set_ylabel("P(stall) %")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax1.set_title("Linear Scale")
    ax2.set_title("Log Scale")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pstall_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pstall_vs_depth.png")


def plot_required_depth(ssd_fits: dict, gpu_model: dict):
    """Required prefetch depth D* to achieve P(stall) < 5%."""
    block_sizes = sorted(set(ssd_fits.keys()) & set(gpu_model.keys()))
    d_star = []
    for bs in block_sizes:
        fit = ssd_fits[bs]
        t_layer = gpu_model[bs]["t_one_layer_ms"]
        d = min_depth_for_target(fit["mu"], fit["sigma"], t_layer, epsilon=0.05)
        d_star.append(d)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"{b}MB" for b in block_sizes], d_star, color="coral",
                  edgecolor="black")
    ax.axhline(32, linestyle="--", color="blue", linewidth=1.5,
               label="32 layers (typical 7B model depth)")
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Required Prefetch Depth D*")
    ax.set_title("Min Prefetch Depth to Achieve P(stall) < 5%\n[Based on T_one_layer lower bound]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar, d in zip(bars, d_star):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(d), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "required_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: required_depth.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("GAMR Phase 1.5 — Stochastic Performance Model: Distribution Fitting")
    print("=" * 65)

    print("\n[1] Loading Phase 1 raw data...")
    ssd_raw      = load_ssd_data()
    ssd_rep      = load_ssd_repeatability()
    pcie_raw     = load_pcie_data()
    gemm, ffn, other = load_gpu_data()

    # ── SSD Model ──────────────────────────────────────────────────────────
    print("\n[2] Fitting lognormal distributions to SSD latency...")
    ssd_fits = {}
    block_sizes = sorted(set(bs for (bs, _) in ssd_raw.keys()))

    for bs in block_sizes:
        # Use "random" mode as the relevant access pattern for HAMR
        times = ssd_raw.get((bs, "random"), ssd_raw.get((bs, "sequential"), []))
        if times:
            fit = fit_lognormal(times)
            ssd_fits[bs] = fit

    # Supplement 32MB with the higher-quality exp01b data (100 trials)
    if ssd_rep:
        ssd_fits[32] = fit_lognormal(ssd_rep)
        ssd_fits[32]["source"] = "exp01b (100 trials, higher quality)"

    print(f"\n  {'Block':>8} | {'μ':>7} | {'σ':>7} | {'Mean(ms)':>10} | {'P95(ms)':>9} | {'CV':>7} | {'KS p':>8}")
    print("  " + "-" * 68)
    for bs, fit in sorted(ssd_fits.items()):
        print(f"  {bs:>6}MB | {fit['mu']:>7.4f} | {fit['sigma']:>7.4f} | "
              f"{fit['mean_ms']:>10.3f} | {fit['p95_ms']:>9.3f} | "
              f"{fit['cv']:>7.4f} | {fit['ks_p_value']:>8.4f}")

    # ── PCIe Model ─────────────────────────────────────────────────────────
    print("\n[3] Fitting linear model to PCIe transfer times...")
    pcie_model = fit_pcie_linear(pcie_raw)
    print(f"  Bandwidth:  {pcie_model['bandwidth_gb_per_s']:.2f} GB/s (from slope)")
    print(f"  Intercept:  {pcie_model['intercept_ms']:.3f} ms")
    print(f"  Fit RMSE:   {pcie_model['fit_rmse_ms']:.3f} ms")
    print(f"  CV range:   {min(pcie_model['cv_per_block'].values()):.3f} – "
          f"{max(pcie_model['cv_per_block'].values()):.3f}  (near-deterministic)")

    # ── GPU Model ──────────────────────────────────────────────────────────
    print("\n[4] Building GPU execution model from exp03...")
    gpu_model = build_gpu_model(gemm, ffn, other)
    print(f"\n  {'Block':>8} | {'T_GEMM(ms)':>12} | {'T_FFN(ms)':>11} | {'T_layer(ms)':>12}")
    print("  " + "-" * 54)
    for bs, m in sorted(gpu_model.items()):
        print(f"  {bs:>6}MB | {m['t_gemm_ms']:>12.5f} | {m['t_ffn_ms']:>11.5f} | "
              f"{m['t_one_layer_ms']:>12.5f}")

    # ── P(stall) Table ─────────────────────────────────────────────────────
    print("\n[5] Computing P(stall | B, D) for D = 1, 2, 4, 8, 16, 32, 64, 128...")
    depths = [1, 2, 4, 8, 16, 32, 64, 128]
    pstall_table = compute_pstall_table(ssd_fits, gpu_model, depths)

    print(f"\n  P(stall) % — W_overlap = D × T_one_layer(B)")
    print(f"  ε target = 5%  (values at or below ε marked ✓)")
    header_bs = sorted(set(r["block_mb"] for r in pstall_table))
    print(f"\n  {'D':>4} | " + " | ".join(f"{b:>6}MB" for b in header_bs))
    print("  " + "-" * (8 + len(header_bs) * 11))
    for D in depths:
        row_vals = []
        for bs in header_bs:
            r = next(r for r in pstall_table if r["block_mb"] == bs and r["prefetch_depth"] == D)
            mark = "✓" if r["meets_target"] else " "
            row_vals.append(f"{r['stall_pct']:>6.1f}%{mark}")
        print(f"  {D:>4} | " + " | ".join(row_vals))

    # ── D* Table ───────────────────────────────────────────────────────────
    print("\n[6] Minimum D* to achieve P(stall) < 5%:")
    print(f"\n  {'Block':>8} | {'D*':>6} | {'T_SSD P95':>12} | {'T_layer':>10} | Note")
    print("  " + "-" * 65)
    for bs in sorted(set(ssd_fits.keys()) & set(gpu_model.keys())):
        fit = ssd_fits[bs]
        t_layer = gpu_model[bs]["t_one_layer_ms"]
        d = min_depth_for_target(fit["mu"], fit["sigma"], t_layer, epsilon=0.05)
        note = "achievable (< 32 layers)" if d <= 32 else f"needs {d} layers deep"
        print(f"  {bs:>6}MB | {d:>6} | {fit['p95_ms']:>10.2f}ms | "
              f"{t_layer:>8.4f}ms | {note}")

    # ── Generate Plots ─────────────────────────────────────────────────────
    print("\n[7] Generating plots...")
    plot_ssd_fits(ssd_fits, ssd_raw)
    plot_pstall_curves(pstall_table, depths)
    plot_required_depth(ssd_fits, gpu_model)

    # ── Save Parameters ────────────────────────────────────────────────────
    params = {
        "version": "1.0",
        "phase": "1.5",
        "description": "Stochastic Performance Model parameters fitted from Phase 1 data",
        "ssd_model": {
            "type": "lognormal_per_block_size",
            "parameters": {str(k): v for k, v in ssd_fits.items()},
            "note": "T_SSD(B) ~ Lognormal(mu_B, sigma_B). Use random-mode data."
        },
        "pcie_model": pcie_model,
        "gpu_model": {
            "type": "lookup_table",
            "parameters": {str(k): v for k, v in gpu_model.items()},
            "other_ops": {
                "attention_ms": other["attention"]["mean_ms"],
                "layernorm_ms": other["layernorm"]["mean_ms"],
                "softmax_ms": other["softmax"]["mean_ms"],
                "ffn_32mb_ms": other["ffn_block"]["mean_ms"],
            },
            "note": "T_one_layer is a LOWER BOUND on W_overlap. Real W_overlap includes all layers processed while one block transfers."
        },
        "pstall_table": pstall_table,
        "epsilon_target": 0.05,
    }

    out_path = OUT_DIR / "performance_model_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved parameters to: {out_path}")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print("\n✓ Phase 1.5 distribution fitting complete.")


if __name__ == "__main__":
    main()
