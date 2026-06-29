"""
experiments/exp04_pipeline_overlap.py
======================================
Phase 1, Experiment 4 — Sustained Pipeline Overlap (W_overlap measurement)

VERSION: 1.0.0
HYPOTHESIS: H1

PURPOSE
───────
This is the ONLY experiment that can properly evaluate H1.

Exp03 showed that isolated single-token GEMM is ~100× faster than SSD transfer.
But that does NOT evaluate H1 correctly, because H1 is about:

    P(T_transfer > W_overlap)

where W_overlap is the total useful GPU work available while blocks transfer.

This experiment directly measures what happens when we run a sustained pipeline:
    - A reader thread loads blocks from SSD into a shared queue
    - A GPU worker thread consumes blocks and runs compute
    - We vary the prefetch depth D = {1, 2, 4, 8, 16}

For each (block_size, prefetch_depth) pair we measure:
    - GPU idle % (time spent waiting for data)
    - Stall count (how many times GPU had to wait)
    - Effective throughput (blocks processed / wall time)
    - Pipeline efficiency (GPU_busy / total_time)

DESIGN NOTES
────────────
The "compute" work per block is a GEMM of the corresponding shape (same as exp03).
This is intentionally a lower bound on W_overlap — real inference includes many
more operations per weight block. If the pipeline already shows benefit at this
lower bound, then real inference will only be better.
"""

import os
import sys
import json
import time
import math
import threading
import queue
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42
DTYPE = torch.float16

# How many blocks to process in total per configuration
TOTAL_BLOCKS = 40
WARMUP_BLOCKS = 5  # first N blocks excluded from statistics


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
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader"]).decode().strip()
        snap["gpu_temp_c"] = float(out.strip())
    except Exception:
        snap["gpu_temp_c"] = None
    try:
        # NVMe temperature
        import glob
        for hwmon in glob.glob("/sys/class/hwmon/hwmon*/temp1_input"):
            with open(hwmon) as f:
                val = float(f.read().strip()) / 1000.0
                if 30.0 < val < 90.0:  # plausible SSD range
                    snap["nvme_temp_c"] = val
                    break
    except Exception:
        snap["nvme_temp_c"] = None
    return snap


def block_size_to_gemm_shape(block_size_mb: int):
    """Same as exp03: derive square-ish GEMM shape from block size."""
    n_elements = (block_size_mb * 1024 * 1024) // 2  # float16
    side = int(math.isqrt(n_elements))
    side = max(64, (side // 64) * 64)
    h_out = n_elements // side
    h_out = max(64, (h_out // 64) * 64)
    return side, h_out


def create_ssd_test_file(work_dir: Path, file_size_bytes: int) -> Path:
    """Create a random temp file for SSD reads."""
    filepath = work_dir / ".exp04_bench_file"
    if filepath.exists() and filepath.stat().st_size >= file_size_bytes:
        return filepath  # reuse
    print(f"  Creating {file_size_bytes / 1e6:.0f} MB temp file...")
    chunk = os.urandom(min(64 * 1024 * 1024, file_size_bytes))
    with open(filepath, "wb") as f:
        written = 0
        while written < file_size_bytes:
            to_write = min(len(chunk), file_size_bytes - written)
            f.write(chunk[:to_write])
            written += to_write
        f.flush()
        os.fsync(f.fileno())
    return filepath


# ─── SSD Reader Thread ────────────────────────────────────────────────────────

def ssd_reader_thread(filepath: Path, block_size_bytes: int, n_blocks: int,
                      prefetch_queue: queue.Queue, rng: np.random.Generator,
                      timing_log: list):
    """
    Read blocks from SSD and put them into the prefetch queue.
    The queue has maxsize = prefetch_depth, so this thread blocks
    when the queue is full (backpressure from GPU consumer).
    """
    file_size = filepath.stat().st_size
    max_offset = (file_size // block_size_bytes) * block_size_bytes - block_size_bytes

    # Open with O_DIRECT if possible
    flags = os.O_RDONLY
    try:
        flags |= os.O_DIRECT
        fd = os.open(str(filepath), flags)
        # Test read
        os.lseek(fd, 0, os.SEEK_SET)
        os.read(fd, block_size_bytes)
    except OSError:
        os.close(fd)
        flags = os.O_RDONLY
        fd = os.open(str(filepath), flags)

    try:
        for i in range(n_blocks):
            offset = int(rng.integers(0, max_offset // block_size_bytes)) * block_size_bytes
            os.lseek(fd, offset, os.SEEK_SET)

            t0 = time.perf_counter()
            data = os.read(fd, block_size_bytes)
            t1 = time.perf_counter()

            timing_log.append({
                "block_idx": i,
                "read_ms": (t1 - t0) * 1000.0,
                "size_bytes": len(data),
            })

            # Put data into the prefetch queue (blocks if full)
            prefetch_queue.put((i, data))
    finally:
        os.close(fd)

    # Sentinel to signal completion
    prefetch_queue.put(None)


# ─── GPU Consumer Thread ──────────────────────────────────────────────────────

def gpu_consumer_thread(prefetch_queue: queue.Queue, h_in: int, h_out: int,
                        block_size_bytes: int, timing_log: list):
    """
    Consume blocks from the queue, transfer to GPU, run GEMM.
    Records wait time (stall) and compute time separately.
    """
    device = torch.device("cuda")
    W = torch.randn(h_in, h_out, dtype=DTYPE, device=device)

    while True:
        # Wait for data
        t_wait_start = time.perf_counter()
        item = prefetch_queue.get()
        t_wait_end = time.perf_counter()

        if item is None:
            break

        block_idx, data = item

        # Transfer: CPU → GPU (simulate via tensor creation from raw bytes)
        t_xfer_start = time.perf_counter()
        n_elements = block_size_bytes // 2  # float16
        cpu_tensor = torch.frombuffer(bytearray(data), dtype=DTYPE).reshape(1, -1)
        # We only use the first h_in elements for the GEMM input
        x = cpu_tensor[:, :h_in].to(device)
        torch.cuda.synchronize()
        t_xfer_end = time.perf_counter()

        # Compute: GEMM
        t_compute_start = time.perf_counter()
        # Run multiple GEMMs to simulate a more realistic per-block compute load
        # In a real transformer, one weight block participates in at least one linear layer
        result = torch.mm(x, W)
        torch.cuda.synchronize()
        t_compute_end = time.perf_counter()

        wait_ms = (t_wait_end - t_wait_start) * 1000.0
        xfer_ms = (t_xfer_end - t_xfer_start) * 1000.0
        compute_ms = (t_compute_end - t_compute_start) * 1000.0

        timing_log.append({
            "block_idx": block_idx,
            "wait_ms": wait_ms,
            "xfer_ms": xfer_ms,
            "compute_ms": compute_ms,
            "stalled": wait_ms > 0.5,  # >0.5ms wait = meaningful stall
        })

        del result, x
        torch.cuda.empty_cache()


# ─── Run One Configuration ────────────────────────────────────────────────────

def run_one_config(filepath: Path, block_size_mb: int, prefetch_depth: int,
                   n_blocks: int) -> dict:
    block_size_bytes = block_size_mb * 1024 * 1024
    h_in, h_out = block_size_to_gemm_shape(block_size_mb)

    rng = np.random.default_rng(SEED)

    prefetch_q = queue.Queue(maxsize=prefetch_depth)
    reader_log = []
    consumer_log = []

    t_start = time.perf_counter()

    reader = threading.Thread(
        target=ssd_reader_thread,
        args=(filepath, block_size_bytes, n_blocks, prefetch_q, rng, reader_log))
    consumer = threading.Thread(
        target=gpu_consumer_thread,
        args=(prefetch_q, h_in, h_out, block_size_bytes, consumer_log))

    reader.start()
    consumer.start()
    reader.join()
    consumer.join()

    t_end = time.perf_counter()
    wall_time_ms = (t_end - t_start) * 1000.0

    # Skip warmup blocks from statistics
    consumer_stats = consumer_log[WARMUP_BLOCKS:]
    if not consumer_stats:
        return {"error": "No data after warmup"}

    wait_times = [e["wait_ms"] for e in consumer_stats]
    compute_times = [e["compute_ms"] for e in consumer_stats]
    xfer_times = [e["xfer_ms"] for e in consumer_stats]
    stall_count = sum(1 for e in consumer_stats if e["stalled"])

    total_compute = sum(compute_times)
    total_wait = sum(wait_times)
    total_xfer = sum(xfer_times)
    total_active = total_compute + total_xfer
    gpu_busy_pct = (total_compute / (total_compute + total_wait)) * 100 if (total_compute + total_wait) > 0 else 0
    gpu_idle_pct = 100 - gpu_busy_pct

    read_times = [e["read_ms"] for e in reader_log[WARMUP_BLOCKS:]]

    return {
        "block_size_mb": block_size_mb,
        "prefetch_depth": prefetch_depth,
        "n_blocks": n_blocks,
        "warmup_blocks": WARMUP_BLOCKS,
        "effective_blocks": len(consumer_stats),
        "h_in": h_in,
        "h_out": h_out,
        "wall_time_ms": round(wall_time_ms, 2),
        "gpu_busy_pct": round(gpu_busy_pct, 3),
        "gpu_idle_pct": round(gpu_idle_pct, 3),
        "stall_count": stall_count,
        "stall_pct": round(100 * stall_count / len(consumer_stats), 2),
        "blocks_per_sec": round(len(consumer_stats) / (wall_time_ms / 1000), 2),
        "compute_ms": {"mean": round(np.mean(compute_times), 4),
                       "p95": round(float(np.percentile(compute_times, 95)), 4)},
        "wait_ms":    {"mean": round(np.mean(wait_times), 4),
                       "p95": round(float(np.percentile(wait_times, 95)), 4)},
        "xfer_ms":    {"mean": round(np.mean(xfer_times), 4),
                       "p95": round(float(np.percentile(xfer_times, 95)), 4)},
        "ssd_read_ms": {"mean": round(np.mean(read_times), 4),
                        "p95": round(float(np.percentile(read_times, 95)), 4)},
        "raw_wait_ms": [round(w, 4) for w in wait_times],
        "raw_compute_ms": [round(c, 4) for c in compute_times],
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def generate_plots(all_results: list, exp_dir: Path):
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Group by block size
    block_sizes = sorted(set(r["block_size_mb"] for r in all_results))
    depths = sorted(set(r["prefetch_depth"] for r in all_results))

    # 1. GPU Idle % vs Prefetch Depth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp04 — Pipeline Overlap: GPU Utilization vs Prefetch Depth", fontsize=13)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(block_sizes)))
    for i, bs in enumerate(block_sizes):
        bs_results = [r for r in all_results if r["block_size_mb"] == bs]
        ds = [r["prefetch_depth"] for r in bs_results]
        idle = [r["gpu_idle_pct"] for r in bs_results]
        ax1.plot(ds, idle, "-o", label=f"{bs}MB", color=colors[i])

    ax1.set_xlabel("Prefetch Depth (D)")
    ax1.set_ylabel("GPU Idle %")
    ax1.set_title("GPU Idle vs Prefetch Depth\n(Lower = Better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Stall count vs Prefetch Depth
    for i, bs in enumerate(block_sizes):
        bs_results = [r for r in all_results if r["block_size_mb"] == bs]
        ds = [r["prefetch_depth"] for r in bs_results]
        stalls = [r["stall_pct"] for r in bs_results]
        ax2.plot(ds, stalls, "-s", label=f"{bs}MB", color=colors[i])

    ax2.set_xlabel("Prefetch Depth (D)")
    ax2.set_ylabel("Stall % (blocks that waited >0.5ms)")
    ax2.set_title("Stall Frequency vs Prefetch Depth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "pipeline_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Wait time histograms for one block size across depths
    ref_bs = 32 if 32 in block_sizes else block_sizes[len(block_sizes)//2]
    ref_results = [r for r in all_results if r["block_size_mb"] == ref_bs]
    n_depths = len(ref_results)
    if n_depths > 0:
        fig, axes = plt.subplots(1, min(n_depths, 5), figsize=(4 * min(n_depths, 5), 4))
        if n_depths == 1:
            axes = [axes]
        fig.suptitle(f"Wait Time Distribution ({ref_bs}MB blocks)", fontsize=13)

        for i, r in enumerate(ref_results[:5]):
            ax = axes[i]
            ax.hist(r["raw_wait_ms"], bins=20, color="coral", edgecolor="black", alpha=0.8)
            ax.set_title(f"D={r['prefetch_depth']}\nIdle={r['gpu_idle_pct']:.1f}%")
            ax.set_xlabel("Wait (ms)")
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(plots_dir / f"wait_histograms_{ref_bs}MB.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Plots saved to {plots_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_experiment(block_sizes_mb: list, depths: list, n_blocks: int):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / f"exp04_{timestamp_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAMR Phase 1 — Experiment 4: Sustained Pipeline Overlap (W_overlap)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA required.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Blocks per config: {n_blocks} (warmup: {WARMUP_BLOCKS})")
    print(f"Block sizes: {block_sizes_mb}")
    print(f"Prefetch depths: {depths}")

    hw_before = get_thermal_snapshot()

    # Create SSD test file (large enough for all block sizes)
    max_bs = max(block_sizes_mb) * 1024 * 1024
    file_size = max(1024 * 1024 * 1024, max_bs * 20)
    work_dir = Path(__file__).parent.parent
    filepath = create_ssd_test_file(work_dir, file_size)

    all_results = []

    print(f"\n{'Block':>8} | {'Depth':>6} | {'GPU Idle%':>10} | {'Stalls':>8} | {'Blk/s':>8} | {'Wait(mean)':>12} | {'Compute(mean)':>14}")
    print("-" * 85)

    for bs_mb in block_sizes_mb:
        for depth in depths:
            result = run_one_config(filepath, bs_mb, depth, n_blocks)
            all_results.append(result)

            if "error" not in result:
                print(f"{bs_mb:>6}MB | {depth:>6} | {result['gpu_idle_pct']:>9.2f}% | "
                      f"{result['stall_count']:>8} | {result['blocks_per_sec']:>8.1f} | "
                      f"{result['wait_ms']['mean']:>10.3f}ms | {result['compute_ms']['mean']:>12.4f}ms")
            else:
                print(f"{bs_mb:>6}MB | {depth:>6} | ERROR: {result['error']}")

    hw_after = get_thermal_snapshot()

    print("\nGenerating plots...")
    generate_plots(all_results, exp_dir)

    # Save outputs
    with open(exp_dir / "raw.json", "w") as f:
        json.dump({"experiment_id": "exp04", "timestamp": timestamp_str,
                   "results": all_results}, f, indent=2)

    with open(exp_dir / "hardware_snapshot.json", "w") as f:
        json.dump({"before": hw_before, "after": hw_after, "gpu": gpu_name}, f, indent=2)

    # Notebook
    with open(exp_dir / "notebook.md", "w") as f:
        f.write(f"# Experiment 04 — Sustained Pipeline Overlap\n\n")
        f.write(f"**Date:** {timestamp_str}\n")
        f.write(f"**GPU:** {gpu_name}\n")
        f.write(f"**Hypothesis:** H1\n")
        f.write(f"**Git Commit:** {get_git_commit()}\n\n")
        f.write(f"**Key Question:** Does increasing prefetch depth D reduce GPU idle %?\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Block | Depth | GPU Idle% | Stalls | Blk/s |\n")
        f.write(f"|---|---|---|---|---|\n")
        for r in all_results:
            if "error" not in r:
                f.write(f"| {r['block_size_mb']}MB | {r['prefetch_depth']} | "
                        f"{r['gpu_idle_pct']:.2f}% | {r['stall_count']} | "
                        f"{r['blocks_per_sec']:.1f} |\n")
        f.write(f"\n## H1 Evaluation\n(Fill in after reviewing)\n\n")
        f.write(f"## Observations\n(Fill in manually)\n\n")
        f.write(f"## Conclusions\n(Fill in manually)\n\n")
        f.write(f"## Next Action\n(Fill in manually)\n")

    print(f"\nAll outputs saved to: {exp_dir}/")

    # Cleanup temp file
    try:
        filepath.unlink()
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GAMR Phase 1: Sustained Pipeline Overlap")
    parser.add_argument("--block-sizes", default="8,16,32,64")
    parser.add_argument("--depths", default="1,2,4,8,16")
    parser.add_argument("--blocks", type=int, default=TOTAL_BLOCKS)
    args = parser.parse_args()
    block_sizes = [int(s.strip()) for s in args.block_sizes.split(",")]
    depths = [int(d.strip()) for d in args.depths.split(",")]
    run_experiment(block_sizes, depths, args.blocks)


if __name__ == "__main__":
    main()
