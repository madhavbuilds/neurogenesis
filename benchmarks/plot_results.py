"""Turn benchmarks/results.md into assets/benchmark.png (small multiples, one per dataset)."""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurogenesis.viz import BG, DIM, FG  # noqa: E402

FIXED, NG = "#4a90e2", "#c08a1f"  # validated: dark-mode band, CVD ΔE ≥ 26
HERE = os.path.dirname(__file__)
text = open(os.path.join(HERE, "results.md")).read()

summary, sweep = {}, {}
for line in text.splitlines():
    cells = [c.strip() for c in line.strip("|").split("|")]
    m = re.match(r"\*\*([\d.]+)\*\*", cells[1]) if len(cells) > 2 else None
    if m and "(" in cells[2]:
        lo, hi = map(float, re.search(r"\(([\d.]+)–([\d.]+)\)", cells[2]).groups())
        summary[cells[0]] = (float(m.group(1)), float(re.search(r"\*\*([\d.]+)\*\*", cells[2]).group(1)), lo, hi)
    elif len(cells) == 8 and cells[0] in summary:
        sweep[cells[0]] = np.array([float(c) for c in cells[1:]])
widths = np.array([2, 4, 8, 16, 32, 64, 128])

fig, axes = plt.subplots(1, len(summary), figsize=(15, 3.9), dpi=110, facecolor=BG)
for i, (ax, (name, (score, neu, lo, hi))) in enumerate(zip(axes, summary.items())):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(DIM)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(axis="y", color=DIM, alpha=0.18, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.minorticks_off()
    ax.plot(widths, sweep[name], color=FIXED, lw=2, marker="o", ms=5, label="fixed-width MLP (you pick the size)", zorder=2)
    ax.errorbar([neu], [score], xerr=[[neu - lo], [hi - neu]], fmt="D", ms=9, color=NG, mec=BG, mew=2,
                ecolor=NG, elinewidth=1.5, capsize=3, label="Neurogenesis (picks its own size)", zorder=3)
    ax.set_title(name, color=FG, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("hidden neurons", color=DIM, fontsize=8.5)
    if i == 0:
        ax.set_ylabel("test score (accuracy / R²)", color=DIM, fontsize=8.5)
        ax.annotate("Neurogenesis", (neu, score), xytext=(8, -16), textcoords="offset points", color=FG, fontsize=8)
        ax.annotate("fixed width", (8, sweep[name][2]), xytext=(6, -12), textcoords="offset points", color=FG, fontsize=8)
    lo_y = min(sweep[name].min(), score)
    ax.set_ylim(lo_y - 0.03 * (1 - lo_y + 0.05), 1.005)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=FG, bbox_to_anchor=(0.99, 1.0))
fig.text(0.01, 0.95, "One run vs. a whole width sweep", color=FG, fontsize=14, weight="bold")
fig.text(0.01, 0.885, "Mean over 3 seeds. Diamond = Neurogenesis at the width it chose (bar = min–max over seeds).",
         color=DIM, fontsize=9)
fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.15, wspace=0.28)
out = os.path.join(HERE, "..", "assets", "benchmark.png")
fig.savefig(out, facecolor=BG)
print("saved", os.path.abspath(out))
