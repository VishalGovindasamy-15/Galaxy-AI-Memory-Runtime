"""
experiments/exp01b_repeatability.py
===================================
Phase 1, Experiment 1b — Repeatability and Latency Tails

PURPOSE
───────
Run 100 iterations of a single block size (32MB) to get a true distribution
of latency, looking for rare spikes (long tails) which ruin the streaming pipeline.
"""

import os
import json
import time
import tempfile
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42

def run_repeatability(block_size_mb: int, n_trials: int, use_direct: bool):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / f"exp01b_{timestamp_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GAMR Phase 1 — Experiment 1b: SSD Repeatability (Long Tails)")
    print("=" * 70)
    
    block_size_bytes = block_size_mb * 1024 * 1024
    file_size_bytes = max(1024 * 1024 * 1024, block_size_bytes * 10)
    
    work_dir = Path(__file__).parent.parent
    tmpfile = tempfile.NamedTemporaryFile(dir=str(work_dir), suffix=".gamr_bench", delete=False)
    
    try:
        print(f"Creating {file_size_bytes / 1e6:.0f} MB temp file...")
        chunk = os.urandom(min(64 * 1024 * 1024, file_size_bytes))
        written = 0
        while written < file_size_bytes:
            to_write = min(len(chunk), file_size_bytes - written)
            tmpfile.write(chunk[:to_write])
            written += to_write
        tmpfile.flush()
        os.fsync(tmpfile.fileno())
        tmpfile.close()

        flags = os.O_RDONLY
        if use_direct and hasattr(os, "O_DIRECT"):
            flags |= os.O_DIRECT
            print("[INFO] Using O_DIRECT")
        
        fd = os.open(tmpfile.name, flags)
        raw_times_ms = []
        rng = np.random.default_rng(SEED)
        max_blocks = file_size_bytes // block_size_bytes
        
        print(f"\nRunning {n_trials} trials of {block_size_mb} MB random reads...")
        try:
            # 5 warmups
            for i in range(5):
                offset = int(rng.integers(0, max_blocks)) * block_size_bytes
                os.lseek(fd, offset, os.SEEK_SET)
                try:
                    os.read(fd, block_size_bytes)
                except OSError:
                    # Fallback if O_DIRECT alignment issue
                    flags = os.O_RDONLY
                    os.close(fd)
                    fd = os.open(tmpfile.name, flags)
                    use_direct = False
                    print("[WARN] O_DIRECT failed, falling back to cached read")
                    os.lseek(fd, offset, os.SEEK_SET)
                    os.read(fd, block_size_bytes)

            for i in range(n_trials):
                offset = int(rng.integers(0, max_blocks)) * block_size_bytes
                os.lseek(fd, offset, os.SEEK_SET)
                
                if not use_direct:
                    os.system("sync && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")
                    
                time.sleep(0.01)
                
                t_start = time.perf_counter()
                os.read(fd, block_size_bytes)
                t_end = time.perf_counter()
                
                elapsed_ms = (t_end - t_start) * 1000.0
                raw_times_ms.append(elapsed_ms)
                
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{n_trials}")
                    
        finally:
            os.close(fd)
            
        data = {
            "block_size_mb": block_size_mb,
            "n_trials": n_trials,
            "use_direct": use_direct,
            "raw_times_ms": raw_times_ms
        }
        
        with open(exp_dir / "raw_1b.json", "w") as f:
            json.dump(data, f, indent=2)
            
        # Also plot the histogram directly
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(raw_times_ms, bins=50, color='coral', edgecolor='black')
        plt.title(f"Latency Distribution: {block_size_mb}MB Random Read ({n_trials} trials)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        mean_ms = np.mean(raw_times_ms)
        p95 = np.percentile(raw_times_ms, 95)
        p99 = np.percentile(raw_times_ms, 99)
        plt.axvline(mean_ms, color='red', linestyle='dashed', label=f"Mean: {mean_ms:.2f}ms")
        plt.axvline(p95, color='orange', linestyle='dashed', label=f"P95: {p95:.2f}ms")
        plt.axvline(p99, color='purple', linestyle='dashed', label=f"P99: {p99:.2f}ms")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(exp_dir / "histogram.png", dpi=150)
        
        print(f"\nMean latency: {mean_ms:.2f} ms")
        print(f"P95 latency:  {p95:.2f} ms")
        print(f"P99 latency:  {p99:.2f} ms")
        print(f"Max latency:  {np.max(raw_times_ms):.2f} ms")
        print(f"Saved to {exp_dir}")

    finally:
        try:
            os.unlink(tmpfile.name)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    
    use_direct = hasattr(os, "O_DIRECT")
    run_repeatability(args.block_size, args.trials, use_direct)
