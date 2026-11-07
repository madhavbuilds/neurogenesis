"""
Real-data benchmark: does a network that sizes itself match one whose size you tune?

16 datasets from the tabular benchmark of Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022),
8 classification and 8 regression, downloaded from OpenML.

Protocol (identical for every method):
  * per seed: random test split (30%, at most 10,000 rows); of the rest, at most 10,000 rows
    are used, 80% for training and 20% for validation
  * features: quantile-transformed to a normal distribution (fit on the training rows)
  * every MLP uses the same code, optimizer, tanh activation, early stopping on the validation
    split and best-checkpoint restore. Only the architecture differs.

Methods:
  neurogenesis       starts from 1 neuron, 1 hidden layer, grows and stops on its own
  neurogenesis-2L    the same with 2 hidden layers
  fixed w=.. d=..    a sweep of 12 fixed MLPs: width {8..256} x depth {1, 2}
                     "tuned" = the one with the best *validation* score (what you'd pick in practice)
                     "oracle" = the one with the best *test* score (an unfair upper bound, labelled as such)
  scratch            neurogenesis's final architecture, trained from scratch as a fixed MLP (control)
  ng-random          ablation: hires random candidates instead of auditioning them
  ng-no-prune        ablation: no death, no fusion
  hgb                scikit-learn HistGradientBoosting, default settings (reference: trees usually win here)

Run:  pip install -e ".[bench]"
      python benchmarks/real_data.py              # ~1 hour on 8 cores; writes real_results.{json,md}
      python benchmarks/real_data.py --summarize  # rebuild the markdown from the JSON
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one process per core instead of oversubscribed BLAS threads

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from neurogenesis import Neurogenesis  # noqa: E402

CACHE = os.path.join(HERE, ".cache")
CLASSIFICATION = {
    "credit": 44089, "electricity": 44120, "pol (clf)": 44122, "house_16H": 44123,
    "MagicTelescope": 44125, "bank-marketing": 44126, "eye_movements": 44130, "heloc": 45026,
}
REGRESSION = {
    "cpu_act": 44132, "pol (reg)": 44133, "elevators": 44134, "wine_quality": 44136,
    "Ailerons": 44137, "houses": 44138, "Bike_Sharing": 44142, "superconduct": 44148,
}
DATASETS = {**CLASSIFICATION, **REGRESSION}
# Defaults (grow_frac, min_gain) were tuned on the *validation* scores of these 4 datasets only.
# They are marked † and left out of the head-to-head statistics, which use the other 12.
DEV = ["credit", "electricity", "wine_quality", "superconduct"]
SEEDS = [0, 1, 2, 3, 4]
SWEEP = [(w, d) for d in (1, 2) for w in (8, 16, 32, 64, 128, 256)]


# ---------------------------------------------------------------------------------- data
def download(did: int) -> str:
    os.makedirs(CACHE, exist_ok=True)
    meta_path = os.path.join(CACHE, f"{did}.json")
    pq_path = os.path.join(CACHE, f"{did}.pq")
    if not os.path.exists(meta_path):
        with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{did}") as r:
            meta = json.load(r)["data_set_description"]
        with open(meta_path, "w") as fh:
            json.dump({"name": meta["name"], "target": meta["default_target_attribute"], "parquet": meta["parquet_url"]}, fh)
    if not os.path.exists(pq_path):
        with open(meta_path) as fh:
            url = json.load(fh)["parquet"]
        try:
            urllib.request.urlretrieve(url, pq_path + ".part")
        except OSError:  # parquet mirror down: fall back to the ARFF file via scikit-learn
            from sklearn.datasets import fetch_openml

            print(f"  parquet unavailable for {did}, fetching ARFF", flush=True)
            fetch_openml(data_id=did, as_frame=True, parser="liac-arff", data_home=CACHE).frame.to_parquet(pq_path + ".part")
        os.replace(pq_path + ".part", pq_path)
    return pq_path


def load(name: str):
    import pandas as pd

    did = DATASETS[name]
    df = pd.read_parquet(os.path.join(CACHE, f"{did}.pq"))
    with open(os.path.join(CACHE, f"{did}.json")) as fh:
        target = json.load(fh)["target"]
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    if name in CLASSIFICATION:
        return X, df[target].astype(str).to_numpy(), "classification"
    return X, df[target].to_numpy(dtype=float), "regression"


def prepare(name: str, seed: int):
    from sklearn.preprocessing import QuantileTransformer

    X, y, task = load(name)
    idx = np.random.default_rng(seed).permutation(len(X))
    n_test = min(int(0.3 * len(X)), 10_000)
    test, rest = idx[:n_test], idx[n_test:]
    rest = rest[:10_000]
    n_val = int(0.2 * len(rest))
    val, train = rest[:n_val], rest[n_val:]
    qt = QuantileTransformer(n_quantiles=min(1000, len(train)), output_distribution="normal", random_state=seed)
    qt.fit(X[train])
    return (qt.transform(X[train]), y[train], qt.transform(X[val]), y[val], qt.transform(X[test]), y[test], task)


# ------------------------------------------------------------------------------- methods
def metrics(model, X, y, task) -> dict:
    if task == "classification":
        P = model.predict_proba(X)
        classes = list(model.classes_)
        known = np.isin(y, classes)
        idx = np.array([classes.index(v) for v in y[known]])
        return {
            "score": float((np.asarray(model.classes_)[P.argmax(1)] == y).mean()),
            "loss": float(-np.log(np.clip(P[known][np.arange(known.sum()), idx], 1e-12, 1)).mean()),
        }
    p = np.asarray(model.predict(X), dtype=float).ravel()
    ss = ((y - y.mean()) ** 2).sum()
    return {"score": float(1 - ((y - p) ** 2).sum() / ss), "loss": float(np.sqrt(((y - p) ** 2).mean()))}


def run_job(job: dict) -> dict:
    name, seed, method, widths = job["dataset"], job["seed"], job["method"], job.get("widths")
    Xtr, ytr, Xva, yva, Xte, yte, task = prepare(name, seed)
    t0 = time.time()
    ng = dict(random_state=seed, task=task)
    if method == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        model = cls(random_state=seed).fit(Xtr, ytr)
    else:
        if method == "neurogenesis":
            model = Neurogenesis(**ng)
        elif method == "neurogenesis-2L":
            model = Neurogenesis(hidden_layers=2, **ng)
        elif method == "ng-random":
            model = Neurogenesis(audition="random", **ng)
        elif method == "ng-no-prune":
            model = Neurogenesis(prune_utility=0.0, fuse_corr=1.0, **ng)
        elif method in ("fixed", "scratch", "scratch-2L"):
            model = Neurogenesis(hidden_layers=len(widths), init_width=list(widths), static=True, **ng)
        else:
            raise ValueError(method)
        model.fit(Xtr, ytr, eval_set=(Xva, yva))
    seconds = time.time() - t0
    out = {**job, "task": task, "seconds": seconds, "val": metrics(model, Xva, yva, task), "test": metrics(model, Xte, yte, task)}
    if method != "hgb":
        out.update(widths=model.widths_, params=model.n_params_, flops=model.flops_, steps=model.n_steps_)
    return out


def run_all(datasets, seeds, workers, out_path):
    for name in datasets:
        download(DATASETS[name])
    results = []
    if os.path.exists(out_path):
        with open(out_path) as fh:
            results = json.load(fh)
    done = {(r["dataset"], r["seed"], r["method"], tuple(r["widths"]) if r["method"] == "fixed" else None) for r in results}

    def key(j):
        return (j["dataset"], j["seed"], j["method"], tuple(j["widths"]) if j["method"] == "fixed" else None)

    phase1 = []
    for name in datasets:
        for seed in seeds:
            for m in ("neurogenesis", "neurogenesis-2L", "ng-random", "ng-no-prune", "hgb"):
                phase1.append({"dataset": name, "seed": seed, "method": m})
            for w, d in SWEEP:
                phase1.append({"dataset": name, "seed": seed, "method": "fixed", "widths": [w] * d})
    _execute([j for j in phase1 if key(j) not in done], workers, results, out_path)

    # Control: neurogenesis's final architecture trained from scratch.
    phase2 = []
    for r in list(results):
        if r["dataset"] in datasets and r["seed"] in seeds and r["method"] in ("neurogenesis", "neurogenesis-2L"):
            m = "scratch" if r["method"] == "neurogenesis" else "scratch-2L"
            phase2.append({"dataset": r["dataset"], "seed": r["seed"], "method": m, "widths": r["widths"]})
    _execute([j for j in phase2 if key(j) not in done], workers, results, out_path)
    return results


def _execute(jobs, workers, results, out_path):
    if not jobs:
        return
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs  ({time.time() - t0:.0f}s)", flush=True)
                with open(out_path, "w") as fh:
                    json.dump(results, fh)


# ------------------------------------------------------------------------------- summary
def summarize(results, datasets) -> str:
    from scipy.stats import spearmanr, wilcoxon

    by = {}
    for r in results:
        by.setdefault((r["dataset"], r["method"]), []).append(r)

    def seeds_of(name, method, widths=None):
        rs = by.get((name, method), [])
        if widths is not None:
            rs = [r for r in rs if r["widths"] == list(widths)]
        return {r["seed"]: r for r in rs}

    tuned_mean = {}
    rows, diffs, scratch_d, rand_d, nop_d, nop_n, rand_n, ng2_d = [], [], [], [], [], [], [], []
    ng_w, tuned_w1, flop_ratio, time_ratio, tuned_vs_hgb, ng_vs_hgb, ng_vs_oracle = [], [], [], [], [], [], []
    for name in datasets:
        ng = seeds_of(name, "neurogenesis")
        seeds = sorted(ng)
        if not seeds:
            continue
        sweep = {(w, d): seeds_of(name, "fixed", [w] * d) for w, d in SWEEP}
        tuned, oracle, tuned_cfg, tuned_1l, sweep_flops, sweep_time = [], [], [], [], [], []
        for s in seeds:
            cand = {c: v[s] for c, v in sweep.items() if s in v}
            best = max(cand, key=lambda c: (cand[c]["val"]["score"], -cand[c]["params"]))
            tuned.append(cand[best]["test"]["score"])
            tuned_cfg.append(best)
            oracle.append(max(v["test"]["score"] for v in cand.values()))
            one = {c: v for c, v in cand.items() if c[1] == 1}
            tuned_1l.append(max(one, key=lambda c: (one[c]["val"]["score"], -one[c]["params"]))[0])
            sweep_flops.append(sum(v["flops"] for v in cand.values()))
            sweep_time.append(sum(v["seconds"] for v in cand.values()))
        s_ng = np.array([ng[s]["test"]["score"] for s in seeds])
        n_ng = np.array([ng[s]["widths"][0] for s in seeds])
        hgb = np.array([r["test"]["score"] for r in seeds_of(name, "hgb").values()])
        ng2 = seeds_of(name, "neurogenesis-2L")
        s_ng2 = np.array([ng2[s]["test"]["score"] for s in seeds if s in ng2])
        w_ng2 = [ng2[s]["widths"] for s in seeds if s in ng2]
        scratch = seeds_of(name, "scratch")
        rnd, nop = seeds_of(name, "ng-random"), seeds_of(name, "ng-no-prune")

        tuned, oracle = np.array(tuned), np.array(oracle)
        tuned_mean[name] = tuned.mean()
        task = ng[seeds[0]]["task"]
        common = max(set(tuned_cfg), key=tuned_cfg.count)
        flops_r = np.mean(sweep_flops) / np.mean([ng[s]["flops"] for s in seeds])
        time_r = np.mean(sweep_time) / np.mean([ng[s]["seconds"] for s in seeds])
        rows.append(
            f"| {name}{' †' if name in DEV else ''} | {'acc' if task == 'classification' else 'R²'} | **{s_ng.mean():.3f}** ± {s_ng.std():.3f} | {n_ng.mean():.0f} ({n_ng.min()}–{n_ng.max()}) "
            f"| {s_ng2.mean():.3f} | {_fmt_widths(w_ng2)} | {tuned.mean():.3f} ± {tuned.std():.3f} | w={common[0]} d={common[1]} "
            f"| {oracle.mean():.3f} | {hgb.mean():.3f} | {time_r:.0f}× |"
        )
        if name in DEV:
            continue
        diffs.append(s_ng.mean() - tuned.mean())
        ng_vs_oracle.append(s_ng.mean() - oracle.mean())
        ng2_d.append(s_ng2.mean() - tuned.mean())
        tuned_vs_hgb.append(tuned.mean() - hgb.mean())
        ng_vs_hgb.append(s_ng.mean() - hgb.mean())
        scratch_d.append(np.mean([ng[s]["test"]["score"] - scratch[s]["test"]["score"] for s in seeds if s in scratch]))
        rand_d.append(np.mean([ng[s]["test"]["score"] - rnd[s]["test"]["score"] for s in seeds if s in rnd]))
        rand_n.append(np.mean([rnd[s]["widths"][0] for s in seeds if s in rnd]))
        nop_d.append(np.mean([ng[s]["test"]["score"] - nop[s]["test"]["score"] for s in seeds if s in nop]))
        nop_n.append(np.mean([nop[s]["widths"][0] for s in seeds if s in nop]))
        ng_w.append(n_ng.mean())
        tuned_w1.append(np.mean(tuned_1l))
        flop_ratio.append(flops_r)
        time_ratio.append(time_r)

    def wil(x):
        x = np.asarray(x)
        if np.allclose(x, 0):
            return 1.0
        return float(wilcoxon(x).pvalue)

    def wtl(x, tol=0.002):
        x = np.asarray(x)
        return f"{(x > tol).sum()} / {(np.abs(x) <= tol).sum()} / {(x < -tol).sum()}"

    held = [d for d in datasets if d not in DEV and seeds_of(d, "neurogenesis")]

    def mean_test(name, method, widths=None):
        return np.mean([r["test"]["score"] for r in seeds_of(name, method, widths).values()])

    ng_mean = {d: mean_test(d, "neurogenesis") for d in held}
    untuned_rows = [f"| Neurogenesis, 1 layer | {np.mean(list(ng_mean.values())):.4f} | – | – | – |",
                    f"| Neurogenesis, 2 layers | {np.mean([mean_test(d, 'neurogenesis-2L') for d in held]):.4f} | – | – | – |",
                    f"| tuned fixed MLP (best of 12 on validation; 12 training runs) | {np.mean([tuned_mean[d] for d in held]):.4f} | – | – | – |"]
    for w, d in SWEEP:
        fx = {n: mean_test(n, "fixed", [w] * d) for n in held}
        x = np.array([ng_mean[n] - fx[n] for n in held])
        untuned_rows.append(f"| fixed w={w} d={d} | {np.mean(list(fx.values())):.4f} | {100 * np.median(x):+.2f} | {wtl(x)} | {wil(x):.3f} |")
    rho = spearmanr(ng_w, tuned_w1)
    n = len(diffs)
    lines = [
        "# Real-data benchmark results",
        "",
        f"{len(rows)} datasets from the Grinsztajn et al. (2022) tabular benchmark · seeds {SEEDS} · "
        "score = test accuracy (classification) or R² (regression), mean ± std over seeds.",
        "Every MLP shares the same code, optimizer, early stopping and checkpoint restore. "
        "Generated by `python benchmarks/real_data.py`.",
        "",
        "| dataset | metric | Neurogenesis (1 layer) | neurons it chose | Neurogenesis (2 layers) | widths it chose (2L) "
        "| tuned fixed MLP (val-selected from 12) | config usually picked | oracle fixed MLP (test-selected) | HistGradientBoosting | sweep wall-clock ÷ NG wall-clock |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "† = development dataset: Neurogenesis defaults were tuned on its validation scores. Excluded from everything below.",
        "",
        f"## Head-to-head on the {n} held-out datasets",
        "",
        "Differences in score points (×100). Win / tie / loss uses a ±0.2 pt tie band. p = Wilcoxon signed-rank test across datasets.",
        "",
        "| comparison | mean diff | median diff | win / tie / loss | p-value |",
        "|---|---|---|---|---|",
        f"| Neurogenesis 1L − tuned fixed MLP | {100 * np.mean(diffs):+.2f} | {100 * np.median(diffs):+.2f} | {wtl(diffs)} | {wil(diffs):.3f} |",
        f"| Neurogenesis 2L − tuned fixed MLP | {100 * np.mean(ng2_d):+.2f} | {100 * np.median(ng2_d):+.2f} | {wtl(ng2_d)} | {wil(ng2_d):.3f} |",
        f"| Neurogenesis 1L − oracle fixed MLP | {100 * np.mean(ng_vs_oracle):+.2f} | {100 * np.median(ng_vs_oracle):+.2f} | {wtl(ng_vs_oracle)} | {wil(ng_vs_oracle):.3f} |",
        f"| Neurogenesis 1L − same architecture from scratch | {100 * np.mean(scratch_d):+.2f} | {100 * np.median(scratch_d):+.2f} | {wtl(scratch_d)} | {wil(scratch_d):.3f} |",
        f"| Neurogenesis 1L − random-candidate ablation | {100 * np.mean(rand_d):+.2f} | {100 * np.median(rand_d):+.2f} | {wtl(rand_d)} | {wil(rand_d):.3f} |",
        f"| Neurogenesis 1L − no-prune/fuse ablation | {100 * np.mean(nop_d):+.2f} | {100 * np.median(nop_d):+.2f} | {wtl(nop_d)} | {wil(nop_d):.3f} |",
        f"| Neurogenesis 1L − HistGradientBoosting | {100 * np.mean(ng_vs_hgb):+.2f} | {100 * np.median(ng_vs_hgb):+.2f} | {wtl(ng_vs_hgb)} | {wil(ng_vs_hgb):.3f} |",
        f"| tuned fixed MLP − HistGradientBoosting | {100 * np.mean(tuned_vs_hgb):+.2f} | {100 * np.median(tuned_vs_hgb):+.2f} | {wtl(tuned_vs_hgb)} | {wil(tuned_vs_hgb):.3f} |",
        "",
        "## Versus a single fixed size with no tuning",
        "",
        "The question a skeptic should ask: would one sensible default size (with the same early stopping) do just as well?",
        "",
        "| model | mean score over held-out datasets | Neurogenesis 1L − this (median pts) | win / tie / loss | p-value |",
        "|---|---|---|---|---|",
        *untuned_rows,
        "",
        "## Size and compute",
        "",
        f"- Wall-clock of the 12-model sweep ÷ one Neurogenesis run: median **{np.median(time_ratio):.1f}×** "
        f"(range {min(time_ratio):.1f}–{max(time_ratio):.1f}×). In analytic FLOPs (incl. the audition) the ratio is larger, median {np.median(flop_ratio):.0f}×, "
        "because the sweep's big models dominate FLOPs but NumPy overheads dominate small-model wall-clock.",
        f"- Neurons chosen (1 layer): median {np.median(ng_w):.0f}, range {min(ng_w):.0f}–{max(ng_w):.0f}. "
        f"Random-candidate ablation: median {np.median(rand_n):.0f}. No-prune/fuse ablation: median {np.median(nop_n):.0f}.",
        f"- Does the width it picks track the validation-best 1-layer width? Spearman ρ = {rho.statistic:.2f} (p = {rho.pvalue:.3f}, n = {n}).",
    ]
    return "\n".join(lines) + "\n"


def _fmt_widths(ws):
    if not ws:
        return "–"
    a = np.array(ws)
    return "[" + ", ".join(f"{v:.0f}" for v in a.mean(0)) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "real_results.json"))
    ap.add_argument("--summarize", action="store_true", help="only rebuild the markdown from the JSON")
    a = ap.parse_args()
    if a.summarize:
        with open(a.out) as fh:
            results = json.load(fh)
    else:
        results = run_all(a.datasets, a.seeds, a.workers, a.out)
    md = summarize(results, a.datasets)
    print(md)
    with open(os.path.splitext(a.out)[0] + ".md", "w") as fh:
        fh.write(md)


if __name__ == "__main__":
    main()
