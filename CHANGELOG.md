# Changelog

## 0.1.0 (2026-11-05)

First public release.

- Self-growing MLP with an audition-based birth, death with bias compensation, regression-based fusion, and automatic maturity.
- Adam with per-weight step counters that survives every shape change.
- Input validation and clear errors (not fitted, NaN/inf, feature-count mismatch, `predict_proba` on regression).
- GIF visualizer (`neurogenesis.viz`) and an audition explainer animation.
- Benchmark against fixed-width MLPs on 5 toy datasets (3 seeds), with a chart.
- Docs: how it works, research roadmap. 9 tests and CI.

Known limitations: multi-layer growth can leave a bottleneck; growth decisions use training data; toy datasets only.
