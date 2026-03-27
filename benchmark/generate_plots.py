import csv
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
csv_path = os.path.join(results_dir, 'raw_timings.csv')

data = defaultdict(lambda: defaultdict(dict))

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        algo = row['algorithm']
        n = int(row['n'])
        p = int(row['procs'])
        t = float(row['time_s'])
        data[algo][n][p] = t

all_n = sorted({n for algo in data for n in data[algo]})
all_p = sorted({p for algo in data for n in data[algo] for p in data[algo][n]})

COLORS = {
    'naive':  '#e74c3c',
    'cannon': '#2980b9',
}
MARKERS = {'naive': 'o', 'cannon': 's'}

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

fig.suptitle('MPI Matrix Multiplication: Naive vs Cannon\'s Algorithm',
             fontsize=15, fontweight='bold', y=0.98, color='#2c3e50')

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50', pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color='#555')
    ax.set_ylabel(ylabel, fontsize=9, color='#555')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_facecolor('#fdfdfd')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ddd')

for ax, n in zip([ax1, ax2, ax3], [256, 512, 1024]):
    style_ax(ax, f'Time vs Processes  (n={n})', 'Processes', 'Time (s)')
    for algo in ['naive', 'cannon']:
        if n in data.get(algo, {}):
            pts = sorted(data[algo][n].items())
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=MARKERS[algo], color=COLORS[algo],
                    linewidth=2.2, markersize=8, label=algo.capitalize(), zorder=3)
    ax.legend(fontsize=8)
    ax.set_xticks(all_p)

for ax, n in zip([ax4, ax5, ax6], [256, 512, 1024]):
    style_ax(ax, f'Speedup & Efficiency  (n={n})', 'Processes', 'Speedup')
    ax2r = ax.twinx()
    ax2r.set_ylabel('Efficiency', fontsize=8, color='#888')

    for algo in ['naive', 'cannon']:
        if n in data.get(algo, {}) and 1 in data[algo][n]:
            base = data[algo][n][1]
            pts = sorted(data[algo][n].items())
            xs = [p for p, _ in pts]
            speedups = [base / t for _, t in pts]
            efficiencies = [s / p for s, p in zip(speedups, xs)]
            ax.plot(xs, speedups, marker=MARKERS[algo], color=COLORS[algo],
                    linewidth=2.2, markersize=8, label=f'{algo.capitalize()} speedup', zorder=3)
            ax2r.plot(xs, efficiencies, marker=MARKERS[algo], color=COLORS[algo],
                      linewidth=1.2, markersize=5, linestyle=':', alpha=0.7,
                      label=f'{algo.capitalize()} efficiency')

    ax.plot(all_p, all_p, 'k--', alpha=0.35, linewidth=1.5, label='Ideal')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xticks(all_p)
    ax2r.set_ylim(0, 1.1)
    ax2r.tick_params(labelsize=7)

out_path = os.path.join(results_dir, 'benchmark_plots.png')
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
