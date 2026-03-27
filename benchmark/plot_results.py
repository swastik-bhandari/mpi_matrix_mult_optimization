import csv
import sys
import os
from collections import defaultdict

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
csv_path = os.path.join(results_dir, 'raw_timings.csv')

if not os.path.exists(csv_path):
    print(f"No results file found at {csv_path}")
    print("Run benchmark/run_benchmarks.sh first")
    sys.exit(1)

data = defaultdict(lambda: defaultdict(dict))

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        algo = row['algorithm']
        n = int(row['n'])
        p = int(row['procs'])
        t = float(row['time_s'])
        data[algo][n][p] = t

print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

all_n = sorted({n for algo in data for n in data[algo]})
all_p = sorted({p for algo in data for n in data[algo] for p in data[algo][n]})

for n in all_n:
    print(f"\nMatrix size: {n}x{n}")
    print(f"{'Procs':<8} {'Naive (s)':<14} {'Cannon (s)':<14} {'Speedup (vs Naive p=1)':<24} {'Efficiency'}")
    print("-" * 75)

    baseline_naive = data.get('naive', {}).get(n, {}).get(1, None)

    for p in all_p:
        naive_t = data.get('naive', {}).get(n, {}).get(p, None)
        cannon_t = data.get('cannon', {}).get(n, {}).get(p, None)

        naive_str = f"{naive_t:.4f}" if naive_t else "  -   "
        cannon_str = f"{cannon_t:.4f}" if cannon_t else "  -   "

        if cannon_t and baseline_naive:
            speedup = baseline_naive / cannon_t
            efficiency = speedup / p
            speedup_str = f"{speedup:.2f}x"
            eff_str = f"{efficiency:.2%}"
        elif naive_t and baseline_naive:
            speedup = baseline_naive / naive_t
            efficiency = speedup / p
            speedup_str = f"{speedup:.2f}x (naive)"
            eff_str = f"{efficiency:.2%}"
        else:
            speedup_str = "  -"
            eff_str = "  -"

        print(f"{p:<8} {naive_str:<14} {cannon_str:<14} {speedup_str:<24} {eff_str}")

print("\n")
print("=" * 60)
print("COMMUNICATION ANALYSIS")
print("=" * 60)
print("""
Alpha-Beta Communication Model:
  T_comm = alpha + beta * message_size

Naive (row-wise scatter/gather):
  - Scatter A:    alpha + beta * (n^2 / p)      per process
  - Broadcast B:  alpha*log(p) + beta * n^2      ALL processes get full B
  - Gather C:     alpha + beta * (n^2 / p)       per process
  ------------------------------------------------------------------
  Total comm per process ~ O(n^2)                independent of p!
  B broadcast is the killer: every rank receives n^2 doubles.

Cannon's Algorithm:
  - Initial skew:  2 * (alpha + beta * (n/sqrt(p))^2)
  - Per step:      2 * (alpha + beta * (n/sqrt(p))^2)
  - Total steps:   sqrt(p)
  ------------------------------------------------------------------
  Total comm ~ O(n^2 / sqrt(p))
  Each process only ever holds and exchanges (n/sqrt(p))^2 blocks.

Reduction in bytes transferred (per process):
  Naive:   beta * n^2         (dominant: full B broadcast)
  Cannon:  beta * n^2/sqrt(p) * sqrt(p) = beta * n^2 ... wait—
           but no ALL-to-ALL: only nearest-neighbor shifts.
           Peak memory/bandwidth per process: O(n^2 / p)

The real win: Cannon uses only point-to-point shifts on a torus.
No collective broadcast. Each process moves O(n^2 / p) data total.
""")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MPI Matrix Multiplication: Naive vs Cannon', fontsize=14, fontweight='bold')

    colors = {'naive': '#e74c3c', 'cannon': '#2ecc71'}

    ax = axes[0]
    ax.set_title("Time vs Processes (n=512)")
    for algo in ['naive', 'cannon']:
        if 512 in data.get(algo, {}):
            pts = sorted(data[algo][512].items())
            xs, ys = zip(*pts)
            ax.plot(xs, ys, 'o-', label=algo.capitalize(), color=colors[algo], linewidth=2, markersize=7)
    ax.set_xlabel("Processes")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Speedup vs Processes (n=512)")
    for algo in ['naive', 'cannon']:
        if 512 in data.get(algo, {}) and 1 in data.get(algo, {}).get(512, {}):
            base = data[algo][512][1]
            pts = sorted(data[algo][512].items())
            xs = [p for p, _ in pts]
            ys = [base / t for _, t in pts]
            ax.plot(xs, ys, 'o-', label=algo.capitalize(), color=colors[algo], linewidth=2, markersize=7)
    ideal_p = sorted(all_p)
    ax.plot(ideal_p, ideal_p, 'k--', label='Ideal', alpha=0.5)
    ax.set_xlabel("Processes")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.set_title("Time vs Matrix Size (p=4)")
    for algo in ['naive', 'cannon']:
        sizes_with_data = [n for n in all_n if 4 in data.get(algo, {}).get(n, {})]
        if sizes_with_data:
            ys = [data[algo][n][4] for n in sizes_with_data]
            ax.plot(sizes_with_data, ys, 'o-', label=algo.capitalize(), color=colors[algo], linewidth=2, markersize=7)
    ax.set_xlabel("Matrix Size n")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'benchmark_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")

except ImportError:
    print("matplotlib not installed — skipping plot generation")
    print("Install with: pip install matplotlib")
