"""
Does a network that sizes itself beat ordinary networks whose size you have to guess?

For each dataset and seed:
  * Neurogenesis starts from 1 hidden neuron and stops on its own.
  * Fixed-width MLPs (same optimiser, same tanh, same depth, same validation split, early stopping
    and best-checkpoint restore) get a budget of up to 30k steps each.

Run:  python benchmarks/benchmark.py            (writes benchmarks/results.md)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurogenesis import Neurogenesis  # noqa: E402
from neurogenesis import datasets as D  # noqa: E402

SEEDS = [0, 1, 2]
WIDTHS = [2, 4, 8, 16, 32, 64, 128]
TASKS = {**D.ALL, "wavy (regression)": D.wavy}


def run(name, make):
    ng, fixed = [], {w: [] for w in WIDTHS}
    for seed in SEEDS:
        X, y = make(seed=seed)
        Xtr, Xte, ytr, yte = D.split(X, y, seed=seed)
        t = time.time()
        m = Neurogenesis(random_state=seed).fit(Xtr, ytr)
        ng.append((m.score(Xte, yte), m.n_neurons_, m.n_params_, m.n_steps_, time.time() - t))
        for w in WIDTHS:
            t = time.time()
            f = Neurogenesis(init_width=w, static=True, random_state=seed).fit(Xtr, ytr)
            fixed[w].append((f.score(Xte, yte), w, f.n_params_, f.n_steps_, time.time() - t))
        print(f"  {name} seed={seed}: NG={ng[-1][0]:.3f} @ {ng[-1][1]} neurons", flush=True)
    return np.array(ng), {w: np.array(v) for w, v in fixed.items()}


def main():
    lines = [
        "| dataset | Neurogenesis score | neurons (self-chosen) | steps | best fixed MLP | smallest fixed MLP within 1 pt of Neurogenesis |",
        "|---|---|---|---|---|---|",
    ]
    detail = ["", "### Fixed-width sweep (mean test score over seeds; up to 30k steps each, early-stopped on validation)", "",
              "| dataset | " + " | ".join(f"w={w}" for w in WIDTHS) + " |", "|---|" + "---|" * len(WIDTHS)]
    for name, make in TASKS.items():
        print(name, flush=True)
        ng, fixed = run(name, make)
        acc, neu = ng[:, 0].mean(), ng[:, 1].mean()
        means = {w: v[:, 0].mean() for w, v in fixed.items()}
        best_w = max(means, key=means.get)
        match = [w for w in WIDTHS if means[w] >= acc - 0.01]
        match_s = f"w={match[0]} ({means[match[0]]:.3f})" if match else "none"
        lines.append(
            f"| {name} | **{acc:.3f}** ± {ng[:, 0].std():.3f} | **{neu:.1f}** ({ng[:, 1].min():.0f}–{ng[:, 1].max():.0f}) | "
            f"{ng[:, 3].mean():.0f} | {means[best_w]:.3f} (w={best_w}) | {match_s} |"
        )
        detail.append(f"| {name} | " + " | ".join(f"{means[w]:.3f}" for w in WIDTHS) + " |")
    out = "\n".join(
        ["# Benchmark results", "", f"Seeds: {SEEDS}. Score = test accuracy (classification) or R² (regression). "
         "One hidden layer, tanh, Adam, identical training code (incl. early stopping) for all models.", ""] + lines + detail
    )
    print("\n" + out)
    with open(os.path.join(os.path.dirname(__file__), "results.md"), "w") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    main()
