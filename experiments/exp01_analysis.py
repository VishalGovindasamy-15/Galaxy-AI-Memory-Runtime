import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python exp01_analysis.py <path_to_raw_json>")
        return

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    modes = set(r["mode"] for r in results)
    
    out_dir = json_path.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    # 1. Bandwidth vs Block Size
    plt.figure(figsize=(10, 6))
    for mode in modes:
        mode_results = [r for r in results if r["mode"] == mode]
        sizes = [r["block_size_mb"] for r in mode_results]
        means = [r["mean_gbps"] for r in mode_results]
        # 95% CI for GB/s error bars
        errors = [r["std_gbps"] for r in mode_results]
        plt.errorbar(sizes, means, yerr=errors, fmt='-o', label=f"{mode.capitalize()} Read", capsize=5)

    plt.xscale('log', base=2)
    plt.xticks(sizes, sizes)
    plt.xlabel('Block Size (MB)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('Throughput vs Block Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bandwidth_vs_size.png", dpi=150)
    plt.close()

    # 2. Coefficient of Variation (CV) vs Block Size
    plt.figure(figsize=(10, 6))
    for mode in modes:
        mode_results = [r for r in results if r["mode"] == mode]
        sizes = [r["block_size_mb"] for r in mode_results]
        # CV = std / mean
        cvs = [r["std_ms"] / r["mean_ms"] for r in mode_results]
        plt.plot(sizes, cvs, '-s', label=f"{mode.capitalize()} Read")

    plt.xscale('log', base=2)
    plt.xticks(sizes, sizes)
    plt.xlabel('Block Size (MB)')
    plt.ylabel('Coefficient of Variation (σ / μ)')
    plt.title('Latency Variation vs Block Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cv_vs_size.png", dpi=150)
    plt.close()

    # 3. Latency Histograms
    for mode in modes:
        mode_results = [r for r in results if r["mode"] == mode]
        n_sizes = len(mode_results)
        fig, axes = plt.subplots(int(np.ceil(n_sizes/2)), 2, figsize=(12, 3 * np.ceil(n_sizes/2)))
        fig.suptitle(f'Latency Histograms ({mode.capitalize()} Read)', fontsize=14)
        axes = axes.flatten()
        
        for i, r in enumerate(mode_results):
            ax = axes[i]
            raw_times = r["raw_times_ms"]
            ax.hist(raw_times, bins=10, color='skyblue', edgecolor='black')
            ax.set_title(f"{r['block_size_mb']} MB")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Count")
            
            # Add mean and 95th percentile lines
            ax.axvline(r["mean_ms"], color='red', linestyle='dashed', linewidth=1, label=f'Mean: {r["mean_ms"]:.2f}')
            ax.axvline(r["p95_ms"], color='orange', linestyle='dotted', linewidth=1, label=f'P95: {r["p95_ms"]:.2f}')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(out_dir / f"histograms_{mode}.png", dpi=150)
        plt.close()

    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
