"""
experiments/exp02b_pcie_4mb_rerun.py
======================================
Phase 1, Experiment 2b — Investigating the 4MB Pinned PCIe Anomaly

VERSION: 1.0.0

PURPOSE
───────
Exp02 showed an anomalous CV=0.77 for 4MB pinned memory transfers.
One trial took ~3.25ms while the rest took ~0.6ms.

This experiment runs 500 trials of 4MB pinned memory transfer to determine:
  - Is the spike reproducible or a one-off?
  - What is the true P95 and P99?
  - Is the distribution bimodal (two distinct latency modes)?
  - Is this a CUDA runtime event, OS jitter, or genuine hardware behavior?

RESULT INTERPRETATION
─────────────────────
  If P99 ≈ P95 (tight tail):  → Single outlier, treat 4MB as normal
  If P99 >> P95 (fat tail):   → Genuine CUDA/HW jitter, use P99 in model
  If bimodal histogram:       → Two operating modes (e.g. CUDA context switch)
"""

import os
import sys
import json
import time
import math
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42


def run_experiment(block_size_mb: int = 4, n_trials: int = 500):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / f"exp02b_{timestamp_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAMR Phase 1 — Experiment 2b: 4MB Pinned PCIe Anomaly Investigation")
    print(f"Trials: {n_trials}  |  Block Size: {block_size_mb}MB  |  Memory: PINNED")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        sys.exit(1)

    size_bytes = block_size_mb * 1024 * 1024
    n_elements = size_bytes // 4  # float32

    # Warmup
    for _ in range(10):
        cpu_t = torch.zeros(n_elements, dtype=torch.float32).pin_memory()
        gpu_t = cpu_t.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        del gpu_t
        torch.cuda.empty_cache()

    raw_times_ms = []
    spike_threshold_ms = 1.5  # Anything > 1.5ms is a "spike" for 4MB

    print(f"\nRunning {n_trials} trials...")
    for i in range(n_trials):
        cpu_t = torch.zeros(n_elements, dtype=torch.float32).pin_memory()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_t = cpu_t.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        raw_times_ms.append(elapsed_ms)
        del gpu_t
        torch.cuda.empty_cache()
        time.sleep(0.002)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_trials} completed")

    arr = np.array(raw_times_ms)
    n_spikes = int(np.sum(arr > spike_threshold_ms))

    print(f"\n{'='*60}")
    print(f"RESULTS — 4MB Pinned PCIe ({n_trials} trials)")
    print(f"{'='*60}")
    print(f"  Mean:   {np.mean(arr):.4f} ms")
    print(f"  Median: {np.median(arr):.4f} ms")
    print(f"  Std:    {np.std(arr, ddof=1):.4f} ms")
    print(f"  P95:    {np.percentile(arr, 95):.4f} ms")
    print(f"  P99:    {np.percentile(arr, 99):.4f} ms")
    print(f"  Max:    {np.max(arr):.4f} ms")
    print(f"  CV:     {np.std(arr, ddof=1)/np.mean(arr):.4f}")
    print(f"  Spikes (>{spike_threshold_ms}ms): {n_spikes}/{n_trials} ({100*n_spikes/n_trials:.1f}%)")

    if n_spikes > 0:
        spike_vals = arr[arr > spike_threshold_ms]
        print(f"  Spike values: {sorted(spike_vals)}")

    # Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Exp02b — 4MB Pinned PCIe ({n_trials} trials)\nAnomaly Investigation", fontsize=13)

    # Full histogram
    ax1.hist(arr, bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(np.mean(arr), color="red", linestyle="--", label=f"Mean={np.mean(arr):.2f}ms")
    ax1.axvline(np.percentile(arr, 95), color="orange", linestyle="--", label=f"P95={np.percentile(arr,95):.2f}ms")
    ax1.axvline(np.percentile(arr, 99), color="purple", linestyle="--", label=f"P99={np.percentile(arr,99):.2f}ms")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Full Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zoom: normal range only (exclude top 1%)
    p99_val = np.percentile(arr, 99)
    normal = arr[arr <= p99_val]
    ax2.hist(normal, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax2.axvline(np.mean(normal), color="red", linestyle="--", label=f"Mean={np.mean(normal):.2f}ms")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Zoomed (excluding top 1%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / "histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save raw
    result = {
        "experiment_id": "exp02b",
        "block_size_mb": block_size_mb,
        "memory_type": "pinned",
        "n_trials": n_trials,
        "spike_threshold_ms": spike_threshold_ms,
        "stats": {
            "mean_ms": round(float(np.mean(arr)), 4),
            "median_ms": round(float(np.median(arr)), 4),
            "std_ms": round(float(np.std(arr, ddof=1)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "max_ms": round(float(arr.max()), 4),
            "cv": round(float(np.std(arr, ddof=1) / np.mean(arr)), 4),
            "n_spikes": n_spikes,
            "spike_pct": round(100.0 * n_spikes / n_trials, 2),
        },
        "raw_times_ms": [round(t, 4) for t in raw_times_ms]
    }

    with open(exp_dir / "raw.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {exp_dir}/")

    verdict = ""
    if n_spikes < 5:
        verdict = "Isolated outlier. Safe to treat 4MB pinned as normal; use median for model."
    elif np.percentile(arr, 99) > 3 * np.median(arr):
        verdict = "Fat-tailed distribution. Use P99 in model, not mean."
    else:
        verdict = "Moderate jitter. Use P95 in model."

    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    run_experiment(n_trials=args.trials)
