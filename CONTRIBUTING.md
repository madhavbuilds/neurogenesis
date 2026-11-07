# Contributing to Neurogenesis

Thanks for helping. This project values **honest results over impressive ones**.

## Setup

```bash
git clone https://github.com/madhavbuilds/neurogenesis && cd neurogenesis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"
pytest
```

## Good first contributions

- **Head-to-head with other growth methods.** GradMax, Cascade-Correlation, Self-Expanding NNs, Adaptive Width NNs, in the same
  harness as `benchmarks/real_data.py`. Roadmap item 6.
- **More real data.** The other datasets of the Grinsztajn benchmark, then one frozen-code run on TabArena. Roadmap item 2.
- **Calibrated hiring.** Hire only when the audition beats a permutation null. Roadmap item 5.
- **Docs and examples.** Notebooks, a clearer explanation, translations.

## Running the benchmarks

```bash
pip install -e ".[bench]"
python benchmarks/benchmark.py        # toy data, ~5 min
python benchmarks/real_data.py        # 16 real datasets x 5 seeds, ~30 min on 8 cores (downloads from OpenML once)
python benchmarks/plot_real.py        # assets/real_benchmark.png
```

Don't tune defaults on the held-out datasets. The four development datasets are listed in `real_data.py` (`DEV`);
tune on their *validation* scores only.

## Rules of thumb

1. **Keep `core.py` dependency-free** (NumPy only) and readable. Clarity beats cleverness.
2. **Every behaviour change needs a test.** The guarantees in `docs/HOW_IT_WORKS.md` §7 must keep passing.
3. **Claims need evidence.** If a PR changes results, include the benchmark command, the seeds and the numbers, including the ones that got worse.
4. **Use seeds.** Every experiment should be reproducible with `random_state`.
5. Credit prior work. If an idea comes from a paper, cite it in the docstring or docs.

## Reporting results

When you share benchmark results, please include the dataset, split, number of seeds, mean ± std (or a CI), the baseline settings, and wall-clock time.
