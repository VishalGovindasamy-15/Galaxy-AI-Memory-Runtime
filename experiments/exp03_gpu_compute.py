"""
experiments/exp03_gpu_compute.py
=================================
Phase 1, Experiment 3 — GPU Execution Time Distribution

VERSION: 1.0.0
HYPOTHESIS: H1, H4

PURPOSE
───────
Measure GPU execution time for the operations that dominate transformer inference.
This provides T_compute(B) — the final missing term in:

    P(stall | B) = P(T_SSD(B) + T_PCIe(B) > T_compute(B))

After this experiment we can compute P(stall) for the first time.

OPERATIONS BENCHMARKED
──────────────────────
1. GEMM          — Linear layers (dominant compute component)
2. LayerNorm     — Low arithmetic intensity; memory-bound
3. Softmax       — Common non-GEMM kernel in attention
4. Attention     — Full scaled dot-product attention block
5. FFN Block     — Two GEMMs + activation (realistic transformer sub-block)

MEASUREMENT PHILOSOPHY
──────────────────────
We use CUDA Events (not perf_counter) for timing.
CUDA Events measure GPU-side time exclusively, eliminating CPU-side jitter.

For each block size B:
  - Infer a weight matrix shape corresponding to B MB of float16 weights
  - Benchmark GEMM for that shape
  - Report: mean, P95, P99, achieved TFLOPS, arithmetic intensity

THE KEY OUTPUT
──────────────
A table cross-referencing T_compute vs T_transfer(SSD+PCIe) from exp01+exp02.
This is the first empirical evaluation of H1.
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
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BLOCK_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_TRIALS = 50   # More trials: GPU kernels are fast, statistics improve
WARMUP_TRIALS = 10
SEED = 42

DTYPE = torch.float16   # Realistic inference dtype
BATCH_SIZE = 1          # Single-token inference (streaming decode)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_thermal_snapshot():
    snap = {"timestamp": datetime.now().isoformat()}
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            snap["cpu_temp_c"] = float(f.read().strip()) / 1000.0
    except Exception:
        snap["cpu_temp_c"] = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem",
             "--format=csv,noheader"]).decode().strip().split(",")
        snap["gpu_temp_c"] = float(out[0].strip())
        snap["gpu_clock_sm_mhz"] = out[1].strip()
        snap["gpu_clock_mem_mhz"] = out[2].strip()
    except Exception:
        snap["gpu_temp_c"] = None
    return snap


def cuda_timed(fn, warmup: int, trials: int) -> list:
    """Run fn() with CUDA Event timing. Returns list of ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def full_stats(raw_times_ms: list, flops: float = 0) -> dict:
    arr = np.array(raw_times_ms)
    n = len(arr)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    tflops = (flops / (mean_ms / 1000.0)) / 1e12 if mean_ms > 0 and flops > 0 else None
    return {
        "trials": n,
        "mean_ms":    round(mean_ms, 5),
        "median_ms":  round(float(np.median(arr)), 5),
        "std_ms":     round(std_ms, 5),
        "p5_ms":      round(float(np.percentile(arr, 5)), 5),
        "p95_ms":     round(float(np.percentile(arr, 95)), 5),
        "p99_ms":     round(float(np.percentile(arr, 99)), 5),
        "min_ms":     round(float(arr.min()), 5),
        "max_ms":     round(float(arr.max()), 5),
        "cv":         round(std_ms / mean_ms, 5) if mean_ms > 0 else 0.0,
        "tflops_achieved": round(tflops, 3) if tflops else None,
        "raw_times_ms": [round(float(t), 5) for t in raw_times_ms],
    }


def block_size_to_gemm_shape(block_size_mb: int, dtype=torch.float16):
    """
    Given a weight block of `block_size_mb` MB in `dtype`,
    return a plausible [H_in, H_out] weight matrix shape.

    For float16: 2 bytes/element
    block_size_mb MB = block_size_mb * 1024**2 / 2 elements
    We choose H_in = H_out = sqrt(n_elements) for a square-ish matrix,
    then round to nearest power-of-64 for GPU efficiency.
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    n_elements = (block_size_mb * 1024 * 1024) // bytes_per_element
    side = int(math.isqrt(n_elements))
    # Round to nearest multiple of 64 for alignment
    side = max(64, (side // 64) * 64)
    # Recalculate actual H_out given H_in=side, staying within budget
    h_out = n_elements // side
    h_out = max(64, (h_out // 64) * 64)
    return side, h_out


# ─── Operation Benchmarks ─────────────────────────────────────────────────────

def bench_gemm(h_in: int, h_out: int, trials: int) -> dict:
    """Benchmark: y = x @ W  (single-token Linear layer)"""
    W = torch.randn(h_in, h_out, dtype=DTYPE, device="cuda")
    x = torch.randn(BATCH_SIZE, h_in, dtype=DTYPE, device="cuda")
    flops = 2 * BATCH_SIZE * h_in * h_out
    times = cuda_timed(lambda: torch.mm(x, W), WARMUP_TRIALS, trials)
    result = full_stats(times, flops)
    result["h_in"] = h_in
    result["h_out"] = h_out
    result["flops"] = flops
    return result


def bench_layernorm(h_in: int, trials: int) -> dict:
    """Benchmark: LayerNorm over hidden dim h_in"""
    ln = nn.LayerNorm(h_in, dtype=DTYPE, device="cuda")
    x = torch.randn(BATCH_SIZE, h_in, dtype=DTYPE, device="cuda")
    flops = 5 * BATCH_SIZE * h_in  # approx: mean, var, normalize, scale, shift
    times = cuda_timed(lambda: ln(x), WARMUP_TRIALS, trials)
    result = full_stats(times, flops)
    result["h_in"] = h_in
    return result


def bench_softmax(seq_len: int, trials: int) -> dict:
    """Benchmark: Softmax over attention scores of length seq_len"""
    x = torch.randn(BATCH_SIZE, 1, seq_len, dtype=DTYPE, device="cuda")
    times = cuda_timed(lambda: F.softmax(x, dim=-1), WARMUP_TRIALS, trials)
    result = full_stats(times)
    result["seq_len"] = seq_len
    return result


def bench_attention(h_in: int, n_heads: int, seq_len: int, trials: int) -> dict:
    """Benchmark: Scaled dot-product attention (single layer)"""
    head_dim = h_in // n_heads
    # QKV projections (pre-computed)
    q = torch.randn(BATCH_SIZE, n_heads, 1, head_dim, dtype=DTYPE, device="cuda")
    k = torch.randn(BATCH_SIZE, n_heads, seq_len, head_dim, dtype=DTYPE, device="cuda")
    v = torch.randn(BATCH_SIZE, n_heads, seq_len, head_dim, dtype=DTYPE, device="cuda")
    # FLOPS for attention: 2 * n_heads * seq_len * head_dim (QK^T) + same for AV
    flops = 4 * BATCH_SIZE * n_heads * seq_len * head_dim
    times = cuda_timed(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False),
        WARMUP_TRIALS, trials
    )
    result = full_stats(times, flops)
    result["h_in"] = h_in
    result["n_heads"] = n_heads
    result["seq_len"] = seq_len
    return result


def bench_ffn_block(h_in: int, h_ffn: int, trials: int) -> dict:
    """
    Benchmark: Full FFN sub-block = Linear(h_in→h_ffn) + GELU + Linear(h_ffn→h_in)
    This is the dominant compute block in transformer inference.
    """
    W1 = torch.randn(h_in, h_ffn, dtype=DTYPE, device="cuda")
    W2 = torch.randn(h_ffn, h_in, dtype=DTYPE, device="cuda")
    x = torch.randn(BATCH_SIZE, h_in, dtype=DTYPE, device="cuda")
    flops = 2 * (2 * BATCH_SIZE * h_in * h_ffn)  # Two GEMMs

    def ffn():
        h = torch.mm(x, W1)
        h = F.gelu(h)
        return torch.mm(h, W2)

    times = cuda_timed(ffn, WARMUP_TRIALS, trials)
    result = full_stats(times, flops)
    result["h_in"] = h_in
    result["h_ffn"] = h_ffn
    result["flops"] = flops
    return result


# ─── H1 Cross-Reference Table ─────────────────────────────────────────────────

# From exp01b and exp02 at corresponding block sizes (measured values)
EXP01_MEAN_MS = {1: 1.026, 2: 1.562, 4: 3.453, 8: 6.653,
                 16: 17.304, 32: 21.642, 64: 45.163, 128: 82.919}
EXP01_P95_MS  = {1: 2.1, 2: 3.1, 4: 5.5, 8: 12.0,
                 16: 35.0, 32: 51.15, 64: 89.0, 128: 155.0}  # from raw histogram
EXP02_MEAN_MS_PINNED = {1: 0.270, 2: 0.394, 4: 0.762, 8: 1.078,
                        16: 2.072, 32: 3.308, 64: 6.121, 128: 12.034}

def compute_h1_table(gemm_results: dict) -> list:
    """Compute the P(stall) proxy table comparing T_transfer vs T_compute."""
    rows = []
    for bs_mb, gemm in gemm_results.items():
        t_ssd_mean = EXP01_MEAN_MS.get(bs_mb, 0)
        t_ssd_p95  = EXP01_P95_MS.get(bs_mb, 0)
        t_pcie     = EXP02_MEAN_MS_PINNED.get(bs_mb, 0)
        t_transfer_mean = t_ssd_mean + t_pcie
        t_transfer_p95  = t_ssd_p95 + t_pcie
        t_compute       = gemm["mean_ms"]

        margin_mean = t_compute - t_transfer_mean
        margin_p95  = t_compute - t_transfer_p95

        rows.append({
            "block_mb": bs_mb,
            "T_compute_ms": round(t_compute, 3),
            "T_transfer_mean_ms": round(t_transfer_mean, 3),
            "T_transfer_p95_ms":  round(t_transfer_p95, 3),
            "margin_mean_ms":  round(margin_mean, 3),
            "margin_p95_ms":   round(margin_p95, 3),
            "H1_mean": "COMPUTE>TRANSFER" if margin_mean > 0 else "STALL",
            "H1_p95":  "COMPUTE>TRANSFER" if margin_p95 > 0 else "STALL",
        })
    return rows


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment(block_sizes_mb: list, n_trials: int):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / f"exp03_{timestamp_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAMR Phase 1 — Experiment 3: GPU Execution Time Distribution")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA required.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    tflops_fp16_peak = device_props.multi_processor_count * 128 * 2 * device_props.clock_rate * 1e3 / 1e12

    print(f"GPU:  {device_name}")
    print(f"VRAM: {device_props.total_memory / 1e9:.1f} GB")
    print(f"Dtype: {DTYPE} | Batch size: {BATCH_SIZE}")
    print()

    hw_before = get_thermal_snapshot()
    all_gemm  = {}
    all_ffn   = {}
    other_ops = {}

    # --- GEMM ---
    print(f"{'Block':>8} | {'Shape':>14} | {'Mean(ms)':>10} | {'P95(ms)':>9} | {'CV':>8} | {'TFLOPS':>8}")
    print("-" * 75)
    for bs_mb in block_sizes_mb:
        h_in, h_out = block_size_to_gemm_shape(bs_mb)
        result = bench_gemm(h_in, h_out, n_trials)
        all_gemm[bs_mb] = result
        tflops_str = f"{result['tflops_achieved']:.3f}" if result["tflops_achieved"] else "N/A"
        print(f"{bs_mb:>6}MB | {h_in:>6}×{h_out:<6} | "
              f"{result['mean_ms']:>10.4f} | {result['p95_ms']:>9.4f} | "
              f"{result['cv']:>8.4f} | {tflops_str:>8}")

    print("\n--- LayerNorm & Softmax (at 32MB-equivalent hidden dim) ---")
    h_ref, _ = block_size_to_gemm_shape(32)
    SEQ_LEN = 512  # Representative KV cache length

    ln_result  = bench_layernorm(h_ref, n_trials)
    sm_result  = bench_softmax(SEQ_LEN, n_trials)
    attn_result = bench_attention(h_ref, n_heads=16, seq_len=SEQ_LEN, trials=n_trials)
    ffn_result  = bench_ffn_block(h_ref, h_ref * 4, n_trials)

    other_ops = {
        "layernorm": {**ln_result, "shape": f"h={h_ref}"},
        "softmax":   {**sm_result, "shape": f"seq_len={SEQ_LEN}"},
        "attention": {**attn_result, "shape": f"h={h_ref} heads=16 seq={SEQ_LEN}"},
        "ffn_block": {**ffn_result, "shape": f"h={h_ref} h_ffn={h_ref*4}"},
    }

    for name, r in other_ops.items():
        print(f"  {name:>12}: mean={r['mean_ms']:.4f}ms  P95={r['p95_ms']:.4f}ms  CV={r['cv']:.4f}")

    # --- FFN per block size ---
    print("\n--- FFN Block per block size ---")
    for bs_mb in block_sizes_mb:
        h_in, h_out = block_size_to_gemm_shape(bs_mb)
        r = bench_ffn_block(h_in, h_out, n_trials)
        all_ffn[bs_mb] = r
        print(f"  {bs_mb}MB  FFN[{h_in}→{h_out}→{h_in}]: mean={r['mean_ms']:.4f}ms P95={r['p95_ms']:.4f}ms")

    hw_after = get_thermal_snapshot()

    # --- H1 Cross-Reference Table ---
    h1_table = compute_h1_table(all_gemm)
    print("\n" + "=" * 80)
    print("H1 EMPIRICAL EVALUATION (GEMM compute vs SSD+PCIe transfer)")
    print("=" * 80)
    print(f"{'Block':>8} | {'T_compute':>10} | {'T_xfer(mean)':>13} | {'T_xfer(P95)':>12} | {'M(mean)':>9} | {'M(P95)':>9} | {'Verdict (P95)':>16}")
    print("-" * 90)
    for row in h1_table:
        verdict = "✓ COMPUTE>TRANSFER" if row["H1_p95"] == "COMPUTE>TRANSFER" else "✗ STALL"
        print(f"{row['block_mb']:>6}MB | {row['T_compute_ms']:>10.3f} | "
              f"{row['T_transfer_mean_ms']:>13.3f} | {row['T_transfer_p95_ms']:>12.3f} | "
              f"{row['margin_mean_ms']:>9.3f} | {row['margin_p95_ms']:>9.3f} | {verdict:>16}")

    # --- Save plots ---
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Compute vs Transfer comparison
    blocks = [r["block_mb"] for r in h1_table]
    t_compute = [r["T_compute_ms"] for r in h1_table]
    t_xfer_mean = [r["T_transfer_mean_ms"] for r in h1_table]
    t_xfer_p95  = [r["T_transfer_p95_ms"] for r in h1_table]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(blocks, t_compute,    "g-o",  label="GPU Compute (GEMM)", linewidth=2)
    ax.plot(blocks, t_xfer_mean,  "b--s", label="SSD+PCIe Transfer (Mean)", linewidth=2)
    ax.plot(blocks, t_xfer_p95,   "r--^", label="SSD+PCIe Transfer (P95)", linewidth=2)
    ax.fill_between(blocks, t_compute, t_xfer_p95,
                    where=[c > x for c, x in zip(t_compute, t_xfer_p95)],
                    alpha=0.15, color="green", label="Safe margin (P95)")
    ax.fill_between(blocks, t_compute, t_xfer_p95,
                    where=[c <= x for c, x in zip(t_compute, t_xfer_p95)],
                    alpha=0.15, color="red", label="Stall zone (P95)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(blocks)
    ax.set_xticklabels([f"{b}MB" for b in blocks])
    ax.set_xlabel("Block Size (MB)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("H1 Empirical Evaluation: GPU Compute vs SSD+PCIe Transfer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "h1_compute_vs_transfer.png", dpi=150, bbox_inches="tight")
    plt.close()

    # TFLOPS achieved
    tflops_vals = [all_gemm[b]["tflops_achieved"] or 0 for b in block_sizes_mb]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f"{b}MB" for b in block_sizes_mb], tflops_vals, color="steelblue", edgecolor="black")
    ax.axhline(tflops_fp16_peak * 0.7, linestyle="--", color="red", label=f"70% peak ({0.7*tflops_fp16_peak:.1f} TFLOPS)")
    ax.set_xlabel("Block Size (MB)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title("GPU TFLOPS Achieved per Block Size (GEMM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "tflops_achieved.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Save outputs ---
    raw_data = {
        "experiment_id": "exp03",
        "timestamp": timestamp_str,
        "objective": "Measure GPU execution time distribution for transformer operations",
        "hypothesis": ["H1", "H4"],
        "git_commit": get_git_commit(),
        "gpu": device_name,
        "dtype": str(DTYPE),
        "batch_size": BATCH_SIZE,
        "gemm_results": {str(k): v for k, v in all_gemm.items()},
        "ffn_results": {str(k): v for k, v in all_ffn.items()},
        "other_ops": other_ops,
        "h1_table": h1_table,
    }
    with open(exp_dir / "raw.json", "w") as f:
        json.dump(raw_data, f, indent=2)

    with open(exp_dir / "hardware_snapshot.json", "w") as f:
        json.dump({"before": hw_before, "after": hw_after}, f, indent=2)

    with open(exp_dir / "notebook.md", "w") as f:
        f.write(f"# Experiment 03 — GPU Execution Time\n\n")
        f.write(f"**Date:** {timestamp_str}\n")
        f.write(f"**GPU:** {device_name}\n")
        f.write(f"**Hypothesis:** H1, H4\n")
        f.write(f"**Git Commit:** {get_git_commit()}\n\n")
        f.write(f"## H1 Empirical Evaluation\n\n")
        f.write(f"| Block | T_compute | T_xfer(mean) | T_xfer(P95) | M(P95) | Verdict |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for row in h1_table:
            verdict = "✓ OK" if row["H1_p95"] == "COMPUTE>TRANSFER" else "✗ STALL"
            f.write(f"| {row['block_mb']}MB | {row['T_compute_ms']}ms | "
                    f"{row['T_transfer_mean_ms']}ms | {row['T_transfer_p95_ms']}ms | "
                    f"{row['margin_p95_ms']}ms | {verdict} |\n")
        f.write(f"\n## Observations\n(Fill in manually)\n\n")
        f.write(f"## Conclusions\n(Fill in manually)\n\n")
        f.write(f"## Next Action\n(Fill in manually)\n")

    print(f"\nAll outputs saved to: {exp_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GAMR Phase 1: GPU Execution Time Distribution")
    parser.add_argument("--block-sizes", default=",".join(str(s) for s in DEFAULT_BLOCK_SIZES_MB))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()
    block_sizes = [int(s.strip()) for s in args.block_sizes.split(",")]
    run_experiment(block_sizes, args.trials)


if __name__ == "__main__":
    main()
