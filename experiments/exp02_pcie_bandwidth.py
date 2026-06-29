"""
experiments/exp02_pcie_bandwidth.py
====================================
Phase 1, Experiment 2 — PCIe Bandwidth Distribution (RAM → VRAM)

VERSION: 1.0.0
HYPOTHESIS: H1, H4

PURPOSE
───────
Measure the distribution of CPU RAM → GPU VRAM transfer times for blocks
of different sizes. Follows the exact same scientific philosophy as exp01:

Don't ask "How fast is PCIe?"
Ask "How predictable is PCIe?"

KEY COMPARISON
──────────────
After this experiment we will have:

  SSD   P95 latency  (stochastic, from exp01)
  PCIe  P95 latency  (this experiment)

If PCIe is far more deterministic (low variance, P95 ≈ Mean), then:
  → The SSD is the dominant source of scheduling uncertainty.
  → The scheduler should focus risk management on the SSD stage.

If PCIe is also stochastic:
  → Both stages contribute uncertainty.
  → The scheduler needs a two-stage risk model.

METHODOLOGY
───────────
1. Pinned vs Pageable: Pin_memory() bypasses OS virtual memory, reducing jitter.
2. 5 warmup trials excluded from stats.
3. Full statistics: mean, median, std, P5, P95, P99, 95% CI.
4. Structured output: raw.json, summary.json, notebook.md, hardware_snapshot.json.
5. Thermal snapshot before and after.
6. Histograms for visual distribution comparison.
"""

import os
import sys
import json
import time
import math
import subprocess
import statistics
import argparse
import ctypes
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BLOCK_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_TRIALS = 20
WARMUP_TRIALS = 5
SEED = 42


# ─── Hardware Snapshot ────────────────────────────────────────────────────────

def get_thermal_snapshot():
    snap = {"timestamp": datetime.now().isoformat()}
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            snap["cpu_temp_c"] = float(f.read().strip()) / 1000.0
    except Exception:
        snap["cpu_temp_c"] = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.free",
             "--format=csv,noheader"]).decode().strip().split(",")
        snap["gpu_temp_c"] = float(out[0].strip())
        snap["gpu_mem_used_mb"] = float(out[1].strip().split()[0])
        snap["gpu_mem_free_mb"] = float(out[2].strip().split()[0])
    except Exception:
        snap["gpu_temp_c"] = None
        snap["gpu_mem_used_mb"] = None
        snap["gpu_mem_free_mb"] = None
    return snap


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# ─── Statistics ───────────────────────────────────────────────────────────────

def full_stats(raw_times_ms: list, block_size_bytes: int) -> dict:
    n = len(raw_times_ms)
    arr = np.array(raw_times_ms)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci_95 = 1.96 * std_ms / math.sqrt(n) if n > 0 else 0.0
    mean_gbps = (block_size_bytes / (mean_ms / 1000.0)) / 1e9 if mean_ms > 0 else 0
    std_gbps = (std_ms / mean_ms) * mean_gbps if mean_ms > 0 else 0
    return {
        "trials": n,
        "mean_ms":    round(mean_ms, 4),
        "median_ms":  round(float(np.median(arr)), 4),
        "std_ms":     round(std_ms, 4),
        "variance_ms2": round(float(np.var(arr, ddof=1)), 4),
        "ci_95_ms":   round(ci_95, 4),
        "p5_ms":      round(float(np.percentile(arr, 5)), 4),
        "p95_ms":     round(float(np.percentile(arr, 95)), 4),
        "p99_ms":     round(float(np.percentile(arr, 99)), 4),
        "min_ms":     round(float(arr.min()), 4),
        "max_ms":     round(float(arr.max()), 4),
        "mean_gbps":  round(mean_gbps, 4),
        "std_gbps":   round(std_gbps, 4),
        "cv":         round(std_ms / mean_ms, 4) if mean_ms > 0 else 0,  # Coefficient of Variation
    }


# ─── Core Measurement ─────────────────────────────────────────────────────────

def measure_pcie_transfer(size_bytes: int, n_trials: int, use_pinned: bool) -> list:
    """
    Measure RAM → VRAM transfer time for a given size and memory type.
    Returns list of raw times in ms.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    n_elements = size_bytes // 4  # float32 = 4 bytes

    # Warmup
    for _ in range(WARMUP_TRIALS):
        if use_pinned:
            cpu_t = torch.zeros(n_elements, dtype=dtype).pin_memory()
        else:
            cpu_t = torch.zeros(n_elements, dtype=dtype)
        gpu_t = cpu_t.to(device, non_blocking=False)
        torch.cuda.synchronize()
        del gpu_t
        torch.cuda.empty_cache()

    raw_times_ms = []
    for _ in range(n_trials):
        if use_pinned:
            cpu_t = torch.zeros(n_elements, dtype=dtype).pin_memory()
        else:
            cpu_t = torch.zeros(n_elements, dtype=dtype)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_t = cpu_t.to(device, non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        raw_times_ms.append((t1 - t0) * 1000.0)
        del gpu_t
        torch.cuda.empty_cache()
        time.sleep(0.005)  # Small settle between trials

    return raw_times_ms


# ─── Plot Generation ──────────────────────────────────────────────────────────

def generate_plots(all_results: list, exp_dir: Path):
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    modes = ["pinned", "pageable"]
    colors = {"pinned": "#2196F3", "pageable": "#F44336"}

    # 1. Bandwidth vs Block Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp02 — PCIe RAM→VRAM Bandwidth Distribution", fontsize=13)

    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue
        sizes = [r["block_size_mb"] for r in mode_results]
        means = [r["mean_gbps"] for r in mode_results]
        stds = [r["std_gbps"] for r in mode_results]
        ax1.errorbar(sizes, means, yerr=stds, fmt="-o", label=f"{mode.capitalize()}",
                     capsize=5, color=colors[mode])

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel("Block Size (MB)")
    ax1.set_ylabel("Throughput (GB/s)")
    ax1.set_title("Throughput vs Block Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Coefficient of Variation (predictability)
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue
        sizes = [r["block_size_mb"] for r in mode_results]
        cvs = [r["cv"] for r in mode_results]
        ax2.plot(sizes, cvs, "-s", label=f"{mode.capitalize()}", color=colors[mode])

    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel("Block Size (MB)")
    ax2.set_ylabel("Coefficient of Variation (σ/μ)")
    ax2.set_title("Predictability vs Block Size\n(Lower = More Deterministic)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "bandwidth_and_cv.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Latency Histograms per block size (showing both modes)
    block_sizes = sorted(set(r["block_size_mb"] for r in all_results))
    n_cols = 2
    n_rows = math.ceil(len(block_sizes) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    fig.suptitle("Exp02 — Latency Histograms (PCIe RAM→VRAM)", fontsize=13)
    axes = axes.flatten()

    for i, bs in enumerate(block_sizes):
        ax = axes[i]
        for mode in modes:
            result = next((r for r in all_results if r["mode"] == mode and r["block_size_mb"] == bs), None)
            if result and "raw_times_ms" in result:
                ax.hist(result["raw_times_ms"], bins=12, alpha=0.6,
                        label=f"{mode} (P95={result['p95_ms']:.1f}ms)",
                        color=colors[mode], edgecolor="black")
        ax.set_title(f"{bs} MB")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_dir / "histograms.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plots saved to {plots_dir}")


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment(block_sizes_mb: list, n_trials: int):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / f"exp02_{timestamp_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAMR Phase 1 — Experiment 2: PCIe Bandwidth Distribution (RAM→VRAM)")
    print("=" * 70)

    if not HAS_TORCH:
        print("[ERROR] PyTorch with CUDA is required for this experiment.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    hw_before = get_thermal_snapshot()

    all_results = []

    for mode, use_pinned in [("pinned", True), ("pageable", False)]:
        print(f"\n--- MODE: {mode.upper()} MEMORY ---")
        print(f"{'Block Size':>12} | {'Mean (ms)':>10} | {'Std (ms)':>9} | {'P95 (ms)':>10} | {'CV':>7} | {'GB/s':>8}")
        print("-" * 72)

        for size_mb in block_sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            print(f"{size_mb:>10} MB | ", end="", flush=True)

            raw_times = measure_pcie_transfer(size_bytes, n_trials, use_pinned)
            stats = full_stats(raw_times, size_bytes)
            stats["block_size_mb"] = size_mb
            stats["mode"] = mode
            stats["raw_times_ms"] = [round(t, 4) for t in raw_times]
            all_results.append(stats)

            print(f"{stats['mean_ms']:>10.3f} | {stats['std_ms']:>9.3f} | "
                  f"{stats['p95_ms']:>10.3f} | {stats['cv']:>7.4f} | "
                  f"{stats['mean_gbps']:>8.3f}")

    hw_after = get_thermal_snapshot()

    print("\nGenerating plots...")
    generate_plots(all_results, exp_dir)

    # raw.json
    with open(exp_dir / "raw.json", "w") as f:
        json.dump({"experiment_id": "exp02", "timestamp": timestamp_str,
                   "results": all_results}, f, indent=2)

    # hardware_snapshot.json
    with open(exp_dir / "hardware_snapshot.json", "w") as f:
        json.dump({
            "before": hw_before, "after": hw_after,
            "gpu": gpu_name,
            "system": {"os": sys.platform, "python": sys.version,
                       "hostname": os.uname().nodename}
        }, f, indent=2)

    # summary.json — also includes P95 comparison (SSD vs PCIe)
    summary = {
        "experiment_id": "exp02",
        "objective": "Measure PCIe RAM→VRAM transfer time distribution",
        "hypothesis": ["H1", "H4"],
        "git_commit": get_git_commit(),
        "question": "Is PCIe more predictable than SSD?",
        "results_summary": [
            {"mode": r["mode"], "block_mb": r["block_size_mb"],
             "mean_gbps": r["mean_gbps"], "std_gbps": r["std_gbps"],
             "p95_ms": r["p95_ms"], "cv": r["cv"]}
            for r in all_results
        ]
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # notebook.md
    with open(exp_dir / "notebook.md", "w") as f:
        f.write(f"# Experiment 02 — PCIe Bandwidth Distribution\n\n")
        f.write(f"**Date:** {timestamp_str}\n")
        f.write(f"**Objective:** Measure PCIe RAM→VRAM transfer time as a stochastic distribution\n")
        f.write(f"**Hypothesis:** H1, H4\n")
        f.write(f"**Git Commit:** {summary['git_commit']}\n")
        f.write(f"**GPU:** {gpu_name}\n\n")
        f.write(f"**Key Question:** Is PCIe more predictable (lower CV) than the SSD?\n\n")
        f.write(f"**SSD Reference (from exp01b):** 32MB block: Mean=25.3ms, P95=51.1ms, CV≈0.5\n\n")
        f.write(f"## Observations\n(Fill in manually after reviewing plots)\n\n")
        f.write(f"## Conclusions\n(Fill in manually)\n\n")
        f.write(f"## Next Action\n(Fill in manually)\n")

    print(f"\nAll outputs saved to: {exp_dir}/")
    print("\n--- SSD (exp01b) vs PCIe (exp02) COMPARISON at 32MB ---")
    ssd_p95 = 51.15
    ssd_cv = 25.30 / 51.15
    print(f"  SSD  (exp01b): Mean=25.30ms, P95={ssd_p95}ms")
    for r in all_results:
        if r["block_size_mb"] == 32:
            print(f"  PCIe ({r['mode']:>8}): Mean={r['mean_ms']:.2f}ms, P95={r['p95_ms']:.2f}ms, CV={r['cv']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="GAMR Phase 1: PCIe Bandwidth Distribution (RAM→VRAM)")
    parser.add_argument("--block-sizes", default=",".join(str(s) for s in DEFAULT_BLOCK_SIZES_MB))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()
    block_sizes = [int(s.strip()) for s in args.block_sizes.split(",")]
    run_experiment(block_sizes, args.trials)


if __name__ == "__main__":
    main()
