# Changelog

## 0.2.0 (2026-11-07)

Real data, validation-based decisions, and a working multi-layer mode.

- **Validation-based life cycle.** 10% of the training data (stratified) is held out, or pass `fit(..., eval_set=(X_val, y_val))`.
  Growth is judged, early stopping is decided, and the best checkpoint is chosen on validation loss. The best checkpoint is restored at the end.
- **Projection-based audition.** Candidates are scored on what they add *beyond* the existing neurons (orthogonal matching pursuit),
  replacing the 0.95-correlation heuristic.
- **Multi-layer fix.** Layer choice uses a scale-invariant score, so deep networks no longer collapse to 1-neuron bottlenecks
  (`[35, 1]` → `[21, 8]` on spiral, 2 layers; 3-layer test accuracy 92.2% → 97.7%).
- **Geometric growth** (`grow_frac=0.25`) and a gentler maturity rule (`min_gain=0.01`), tuned on 4 development datasets only.
- **Real-data benchmark** on 16 datasets from Grinsztajn et al. (2022) against a 12-model tuned MLP sweep, a from-scratch control,
  two ablations and gradient-boosted trees (`benchmarks/real_data.py`).
- `save` / `load` (NumPy `.npz`, no pickle), scikit-learn `get_params` / `set_params` / `clone` / `cross_val_score` support,
  per-layer `init_width`, parameter validation, analytic FLOP counter (`flops_`), `audition="random"` ablation.
- New events: `stopped` (early stopping) and `restore`. 21 tests.

Behaviour change: default results differ from 0.1.0. Set `validation_fraction=0, patience=0, restore_best=False, grow_frac=0, min_gain=0.05`
to approximate 0.1.0 (the audition scoring also changed).

## 0.1.0 (2026-11-05)

First public release.

- Self-growing MLP with an audition-based birth, death with bias compensation, regression-based fusion, and automatic maturity.
- Adam with per-weight step counters that survives every shape change.
- Input validation and clear errors (not fitted, NaN/inf, feature-count mismatch, `predict_proba` on regression).
- GIF visualizer (`neurogenesis.viz`) and an audition explainer animation.
- Benchmark against fixed-width MLPs on 5 toy datasets (3 seeds), with a chart.
- Docs: how it works, research roadmap. 9 tests and CI.

Known limitations: multi-layer growth can leave a bottleneck; growth decisions use training data; toy datasets only.
