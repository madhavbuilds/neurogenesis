"""Turn benchmarks/real_results.json into assets/real_benchmark.png (a dot plot, one row per dataset)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from real_data import DATASETS, DEV, SWEEP  # noqa: E402

from neurogenesis.viz import BG, DIM, FG  # noqa: E402

# Validated with the dataviz palette checker against BG (dark mode, all pairs): CVD ΔE ≥ 9.4, contrast ≥ 3:1.
# Marker shapes double-encode the series, so identity never depends on color alone.
SERIES = [
    ("neurogenesis", "Neurogenesis, 1 layer", "#3987e5", "o"),
    ("neurogenesis-2L", "Neurogenesis, 2 layers", "#d95926", "D"),
    ("hgb", "boosted trees (reference)", "#199e70", "s"),
]

with open(os.path.join(HERE, "real_results.json")) as fh:
    results = json.load(fh)
by = {}
for r in results:
    by.setdefault((r["dataset"], r["method"]), {})
    key = r["seed"] if r["method"] != "fixed" else (r["seed"], tuple(r["widths"]))
    by[(r["dataset"], r["method"])][key] = r

rows = []
for name in DATASETS:
    ng = by.get((name, "neurogenesis"), {})
    if not ng:
        continue
    fixed = by[(name, "fixed")]
    gaps = {m: [] for m, *_ in SERIES}
    for s in sorted(ng):
        cand = [fixed[(s, tuple([w] * d))] for w, d in SWEEP if (s, tuple([w] * d)) in fixed]
        tuned = max(cand, key=lambda r: (r["val"]["score"], -r["params"]))["test"]["score"]
        for m, *_ in SERIES:
            if s in by.get((name, m), {}):
                gaps[m].append(100 * (by[(name, m)][s]["test"]["score"] - tuned))
    rows.append((name, {m: np.mean(v) for m, v in gaps.items()}))
rows.sort(key=lambda r: r[1]["neurogenesis"])
held = [g["neurogenesis"] for n, g in rows if n not in DEV]

fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2.4), dpi=150, facecolor=BG)
ax.set_facecolor(BG)
y = np.arange(len(rows))
for i in y:
    ax.axhline(i, color=DIM, alpha=0.12, lw=0.6, zorder=0)
ax.axvline(0, color=FG, alpha=0.55, lw=1.2, zorder=1)
for j, (m, label, color, marker) in enumerate(SERIES):
    xs = [g[m] for _, g in rows]
    ax.scatter(xs, y + (j - 1) * 0.18, s=46, c=color, marker=marker, edgecolors=BG, linewidths=1.6, zorder=3, label=label)
ax.set_yticks(y, [f"{n} †" if n in DEV else n for n, _ in rows], color=FG, fontsize=9)
ax.tick_params(axis="x", colors=DIM, labelsize=8.5)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(DIM)
lo, hi = ax.get_xlim()
lim = max(abs(lo), abs(hi))
ax.set_xlim(-lim, lim)
ax.text(lim * 0.02, -0.9, "better than the tuned MLP →", color=DIM, fontsize=8.5, ha="left", va="center")
ax.text(-lim * 0.02, -0.9, "← worse", color=DIM, fontsize=8.5, ha="right", va="center")
ax.set_xlabel("test score minus the tuned fixed MLP's (points of accuracy or R² × 100)", color=DIM, fontsize=9)
ax.set_ylim(-1.3, len(rows) - 0.5)
leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=FG,
                handletextpad=0.3, columnspacing=1.6)
fig.text(0.02, 0.985, f"Neurogenesis vs. a tuned MLP on {len(rows)} real tabular datasets", color=FG, fontsize=13,
         weight="bold", va="top")
fig.text(0.02, 0.945,
         f"0 = the best of 12 fixed MLPs (width 8–256 × depth 1–2), chosen on validation. Mean of 5 seeds. "
         f"Held-out median gap: {np.median(held):+.2f} pts.  † = used to tune defaults.",
         color=DIM, fontsize=8.5, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, "..", "assets", "real_benchmark.png")
fig.savefig(out, facecolor=BG)
print(f"saved {os.path.abspath(out)}")
