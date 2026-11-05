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

- **Real-data benchmarks.** Run Neurogenesis on public tabular datasets (OpenML) against fixed-width MLPs. See roadmap item 2.
- **Validation-based stopping.** Roadmap item 1.
- **The multi-layer bottleneck.** Roadmap item 3, with a written diagnosis in `docs/RESEARCH_ROADMAP.md`.
- **Docs and examples.** Notebooks, a clearer explanation, translations.

## Rules of thumb

1. **Keep `core.py` dependency-free** (NumPy only) and readable. Clarity beats cleverness.
2. **Every behaviour change needs a test.** The guarantees in `docs/HOW_IT_WORKS.md` §7 must keep passing.
3. **Claims need evidence.** If a PR changes results, include the benchmark command, the seeds and the numbers, including the ones that got worse.
4. **Use seeds.** Every experiment should be reproducible with `random_state`.
5. Credit prior work. If an idea comes from a paper, cite it in the docstring or docs.

## Reporting results

When you share benchmark results, please include the dataset, split, number of seeds, mean ± std (or a CI), the baseline settings, and wall-clock time.
