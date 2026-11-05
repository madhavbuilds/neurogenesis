# Research roadmap

This page answers one question: **what would it take for Neurogenesis to be a real research contribution, and not just a nice demo?**
Items are ranked by scientific importance. Each one is open for contributors. Open an issue before starting a big one.

## Where we honestly stand

Every part of the mechanism has a published precedent (see the README's *Is this new?* table).
The current evidence is toy 2-D/1-D data with 3 seeds. So right now this is a **clean, readable implementation of known ideas**.
It becomes a contribution only through (a) a rigorous real-data study and (b) a well-supported improvement.

## Ranked roadmap

| # | Work item | Why it matters | Gate |
|---|---|---|---|
| **1** | **Validation-based stopping + best-checkpoint restore + growth rollback** | Growth decisions currently use training data, so no evaluation is methodologically clean without this | — |
| **2** | **Real tabular benchmark on a development set** (≈10 datasets from Grinsztajn et al. 2022), testing H1–H3 below | Decides whether there's anything to claim | If it's not on par with a validation-tuned fixed MLP, or has no compute advantage → publish as a negative/empirical study, or stop |
| **3** | **Scale-invariant growth placement** (fixes the multi-layer bottleneck) | A real, diagnosed defect | Must remove `[10, 10, 1]`-style failures across datasets |
| **4** | **Better audition math:** score candidates after projecting out what existing neurons already explain; Gauss-Newton weighting for cross-entropy | Replaces the 0.95-correlation heuristic with a principled one | Ablation must show a gain |
| **5** | **Statistically calibrated hiring** (permutation test) | The most plausible *distinctive* contribution | A full literature search must come back clean first |
| **6** | Shared-codebase comparison with GradMax, SENN, NORTH, Adaptive Width NNs, Cascade-Correlation, plus one confirmatory TabArena run with frozen code | Required for any venue | — |
| **7** | Release engineering: save/load, scikit-learn estimator compliance, lockfile | Credibility | — |
| **8** | PyTorch backend, CNN/transformer widths | Reach | Only if 2 and 6 are positive |

## Hypotheses (register before running)

| | Hypothesis | Prior |
|---|---|---|
| H1 | Default Neurogenesis is no worse than a validation-tuned fixed MLP (within a fixed margin on normalized log loss), with much less tuning compute | plausible |
| H2 | Growing beats **the same final architecture trained from scratch** for the same number of steps | uncertain: this is the key control |
| H3 | The width it picks tracks the validation-best width across datasets (Spearman ρ > 0) | plausible |
| H4 | Each of pruning, fusion and the audition contributes (ablations) | uncertain |
| H5 | At equal compute, it's at least as good as SENN, Adaptive Width NNs or GradMax | unlikely without items 3–5 |

## Benchmark protocol

- **Data:** develop on a subset of the Grinsztajn et al. (NeurIPS 2022) benchmark. Confirm **once**, with frozen code, on TabArena (NeurIPS 2025).
- **Preprocessing (identical for every method):** quantile-transform numeric features, one-hot encode low-cardinality categoricals.
- **Splits:** the benchmark's outer folds, a 20% inner validation split, and ≥ 10 seeds for the growth methods.
- **Comparators:**
  - fixed MLPs: width {8…512} × depth {1, 2, 3}, validation-selected; the test-selected best is reported only as a labelled oracle;
  - the from-scratch control (H2);
  - train-large-then-prune;
  - the growth methods above;
  - RealMLP and GBDT defaults (XGBoost, LightGBM, CatBoost) as a reference ceiling.
- **Ablations:** no pruning; no fusion; random candidates only; SVD candidates only (≈ GradMax); Cascor's score; global vs per-weight Adam counters.
- **Metrics:** log loss (primary), accuracy, ROC-AUC and balanced accuracy for classification; RMSE (primary) and R² for regression.
  Also parameter count, analytic FLOPs including the audition, wall-clock time, and width stability across seeds.
- **Statistics:** bootstrap 95% confidence intervals per dataset. Across datasets: Wilcoxon signed-rank with Holm correction, Friedman/Nemenyi critical-difference diagrams,
  and IQM with stratified-bootstrap CIs plus probability of improvement (the `rliable` approach). Report effect sizes.

## Diagnosis: the multi-layer bottleneck

One run, 3 hidden layers, spiral data. The audition always found valid winners in layer 3, but upstream layers kept winning the
layer-selection comparison, ending at widths `[10, 10, 1]`. Over the growth rounds, the upstream error-signal norm
rose (0.007 → 0.125) as downstream weights grew, while layer 3's fell (0.022 → 0.007).

**Likely cause:** the score is a first-order gradient norm. It isn't invariant to the scale of downstream weights, and it can't see that a
1-neuron layer caps capacity.

**Fixes, in order:**
1. Score candidates by expected loss decrease under a Gauss-Newton or natural-gradient metric, which is scale-invariant (this is the insight behind Self-Expanding NNs).
2. **Lookahead oracle:** tentatively add the best candidate in each layer, train ~100 steps, and keep the layer with the best validation loss. Use it to validate fix 1.
3. **Safety rule:** width ≥ output dimension, and force growth in a layer whose activations have hit their effective-rank limit.

## Audition math: proposed improvements

1. **Project out existing neurons.** The output layer re-fits after a hire, so only the *new* part of a candidate's signal matters.
   For squared error, the gain after re-fitting is ‖(P⊥h)ᵀR‖² / (2‖P⊥h‖²), where P⊥ removes everything the existing neurons (plus a bias) can express.
   Choosing several neurons greedily this way is orthogonal matching pursuit. (TINY solves a related problem optimally.)
2. **Cross-entropy:** weight by the softmax curvature (iteratively reweighted least squares), turning (1) into a weighted least-squares gain.
3. **Refine the top candidates** with a few gradient-ascent steps on their gain (as Cascade-Correlation does). This matters for high-dimensional inputs.
4. **Calibrated hiring:** shuffle the error rows relative to the inputs, recompute the best score in the pool, and hire only if the real best beats the
   95th percentile of the shuffled bests. Compute it on held-out residuals.
   *Close precedent:* the Lee–White–Granger (1993) neglected-nonlinearity test correlates random hidden units with residuals.

## Possible venues

- **TMLR:** a rigorous empirical study ("does growing an MLP beat tuning its width on tabular data?"). Negative results are acceptable if done carefully.
- **AutoML Conference**, and efficiency or tabular-learning workshops at NeurIPS/ICML: if items 3–5 work.
- **JOSS / JMLR open-source track:** for the software, after item 7. Check their current requirements first.

## Key references

- Ash (1989), *Dynamic Node Creation in Backpropagation Networks*, Connection Science
- Fahlman & Lebiere (1989/90), *The Cascade-Correlation Learning Architecture*, NeurIPS
- Lee, White & Granger (1993), *Testing for neglected nonlinearity in time series models*, J. Econometrics
- Chen, Goodfellow & Shlens (2016), *Net2Net*, ICLR · Wei et al. (2016), *Network Morphism*, ICML
- Wu et al. (2020), *Firefly Neural Architecture Descent*, NeurIPS
- Evci et al. (2022), *GradMax: Growing Neural Networks using Gradient Information*, ICLR, arXiv:2201.05125
- Maile et al. (2022), *When, where, and how to add new neurons to ANNs*, AutoML, arXiv:2202.08539
- Mitchell et al. (2023), *Self-Expanding Neural Networks*, arXiv:2307.04526
- Verbockhaven et al. (2024), *Growing Tiny Networks: Spotting Expressivity Bottlenecks and Fixing Them Optimally*, TMLR, arXiv:2405.19816
- Errica et al. (2025/26), *Adaptive Width Neural Networks*, ICLR 2026, arXiv:2501.15889
- Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS, arXiv:2207.08815
- Holzmüller, Grinsztajn & Steinwart (2024), *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data*, NeurIPS
- Erickson et al. (2025), *TabArena: A Living Benchmark for Machine Learning on Tabular Data*, NeurIPS D&B, arXiv:2506.16791
- Agarwal et al. (2021), *Deep RL at the Edge of the Statistical Precipice*, NeurIPS, arXiv:2108.13264
- Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR
