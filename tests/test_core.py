import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurogenesis import Neurogenesis  # noqa: E402
from neurogenesis import datasets as D  # noqa: E402


def _trained(**kw):
    X, y = D.spiral(n=600)
    m = Neurogenesis(random_state=0, max_steps=3000, **kw).fit(X, y)
    return m, X, y


def test_learns_and_grows_from_one_neuron():
    X, y = D.circles()
    m = Neurogenesis(random_state=0).fit(X, y)
    assert m.score(X, y) > 0.97
    assert any(e.kind == "birth" for e in m.events_)
    assert 2 <= m.n_neurons_ <= 12  # circles need ~3 tanh units; it should not bloat


def test_birth_preserves_function_exactly():
    m, X, y = _trained()
    rng = np.random.default_rng(1)
    Xs = m._prep(X)
    before = m.predict_proba(X)
    acts, out = m._forward(Xs)
    _, dout = m._loss(out, np.eye(len(m.classes_))[np.searchsorted(m.classes_, y)] / 1)
    _, _, deltas = m._backward(acts, dout)
    w0 = m.n_neurons_
    assert m._grow(99999, acts, deltas, rng)
    assert m.n_neurons_ > w0
    np.testing.assert_allclose(m.predict_proba(X), before, atol=1e-12)


def test_birth_candidates_have_positive_gradient_pressure():
    m, X, y = _trained()
    Xs = m._prep(X)
    acts, out = m._forward(Xs)
    _, dout = m._loss(out, np.eye(len(m.classes_))[np.searchsorted(m.classes_, y)])
    _, _, deltas = m._backward(acts, dout)
    W, b, s = m._audition(acts[0], deltas[1], acts[1], 4, np.random.default_rng(0))
    assert W.shape == (2, len(s)) and len(s) > 0 and (s > 0).all()
    assert np.all(np.diff(s) <= 1e-12)  # winners come out best-first


def test_death_of_constant_neuron_is_exact():
    # A saturated neuron outputs a constant; moving that constant into the bias loses nothing.
    m, X, _ = _trained()
    m._W[0].append(np.zeros((2, 1)), axis=1)
    m._b[0].append(np.array([-50.0]), axis=0)  # tanh(-50) == -1 everywhere
    m._W[1].append(np.full((1, m._W[1].val.shape[1]), 0.7), axis=0)
    m._ids[0], m._born[0] = np.r_[m._ids[0], 999], np.r_[m._born[0], 0]
    before = m.predict_proba(X)
    acts, _ = m._forward(m._prep(X))
    m._remove(0, np.array([m.widths_[0] - 1]), acts[1], compensate=True)
    np.testing.assert_allclose(m.predict_proba(X), before, atol=1e-12)


def test_death_bias_compensation_beats_plain_removal():
    m, X, _ = _trained()
    acts, _ = m._forward(m._prep(X))
    H = acts[1]
    weakest = np.array([(H.std(0) * np.linalg.norm(m._W[1].val, axis=1)).argmin()])
    before = m.predict_proba(X)
    errs = []
    for compensate in (True, False):
        state = [(P.val.copy()) for P in m._W + m._b], [i.copy() for i in m._ids], [b.copy() for b in m._born]
        m._remove(0, weakest, H, compensate=compensate)
        errs.append(np.abs(m.predict_proba(X) - before).mean())
        vals, m._ids, m._born = state
        for P, v in zip(m._W + m._b, vals):
            P.val = v
            P.m, P.v, P.t = np.zeros_like(v), np.zeros_like(v), np.zeros(v.shape, dtype=np.int64)
    assert errs[0] <= errs[1]


def test_fusion_of_duplicate_neuron_is_lossless():
    m, X, _ = _trained()
    # clone neuron 0 into a new, old-enough neuron with split fan-out
    W0, W1 = m._W[0], m._W[1]
    W0.append(W0.val[:, :1].copy(), axis=1)
    m._b[0].append(m._b[0].val[:1].copy(), axis=0)
    half = W1.val[:1] * 0.5
    W1.val[0] *= 0.5
    W1.append(half, axis=0)
    m._ids[0] = np.r_[m._ids[0], 999]
    m._born[0] = np.r_[m._born[0], -10_000]
    m._born[0][:] = -10_000
    before = m.predict_proba(X)
    w = m.n_neurons_
    m.events_ = []
    acts, _ = m._forward(m._prep(X))
    m.prune_utility = 0.0
    m._prune_and_fuse(0, acts)
    assert m.n_neurons_ == w - 1
    assert any(e.kind == "fusion" for e in m.events_)
    np.testing.assert_allclose(m.predict_proba(X), before, atol=1e-6)


def test_optimizer_state_tracks_structure():
    m, *_ = _trained()
    for P in m._W + m._b:
        assert P.val.shape == P.m.shape == P.v.shape == P.t.shape
    for h in range(m.hidden_layers):
        assert len(m._ids[h]) == len(m._born[h]) == m.widths_[h]


def test_regression_and_deep():
    X, y = D.wavy()
    m = Neurogenesis(random_state=0).fit(X, y)
    assert m.task_ == "regression" and m.score(X, y) > 0.9
    X, y = D.moons()
    m = Neurogenesis(hidden_layers=2, random_state=0, max_steps=8000).fit(X, y)
    assert m.score(X, y) > 0.95 and len(m.widths_) == 2


def test_static_mode_never_changes_shape():
    X, y = D.moons()
    m = Neurogenesis(init_width=8, static=True, max_steps=1000, random_state=0).fit(X, y)
    assert m.widths_ == [8] and not m.events_


def test_input_validation_and_clear_errors():
    import pytest

    X, y = D.moons()
    with pytest.raises(RuntimeError, match="not fitted"):
        Neurogenesis().predict(X)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        Neurogenesis().fit(Xn, y)
    m = Neurogenesis(max_steps=200, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        m.predict(np.zeros((3, 5)))
    Xr, yr = D.wavy()
    r = Neurogenesis(max_steps=200, random_state=0).fit(Xr, yr)
    with pytest.raises(AttributeError):
        r.predict_proba(Xr)


def test_validation_split_and_best_checkpoint_restore():
    X, y = D.moons()
    Xv, yv = D.moons(n=300, seed=7)
    m = Neurogenesis(random_state=0, max_steps=4000).fit(X, y, eval_set=(Xv, yv))
    assert m.has_validation_ and len(m.history_["val_loss"]) == len(m.history_["loss"])
    # the returned weights are exactly the best validation checkpoint
    _, out = m._forward(m._prep(Xv))
    val_loss = m._loss(out, m._encode(yv))[0]
    assert abs(val_loss - m.best_val_loss_) < 1e-12
    assert val_loss <= min(m.history_["val_loss"]) + 1e-12


def test_holdout_is_stratified_and_disjoint():
    X, y = D.spiral(n=600)
    m = Neurogenesis(validation_fraction=0.2)
    m.task_, m.classes_ = "classification", np.unique(y)
    tr, va = m._holdout(y, np.random.default_rng(0))
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(y)
    assert np.allclose(np.bincount(y[va]) / len(va), np.bincount(y) / len(y), atol=0.01)


def test_early_stopping_halts_static_model():
    X, y = D.moons(n=200)
    m = Neurogenesis(init_width=64, static=True, patience=5, max_steps=30_000, random_state=0).fit(X, y)
    assert m.n_steps_ < 30_000 and any(e.kind == "stopped" for e in m.events_)


def test_deep_growth_has_no_one_neuron_bottleneck():
    # v0.1 grew [10, 10, 1]-style networks here; scale-invariant layer choice should not.
    X, y = D.spiral()
    m = Neurogenesis(hidden_layers=3, random_state=0).fit(X, y)
    assert min(m.widths_) >= 3, m.widths_
    assert m.score(X, y) > 0.95


def test_geometric_growth_adds_more_per_spurt():
    X, y = D.spiral(n=600)
    small = Neurogenesis(grow_frac=0.0, random_state=0, max_steps=6000).fit(X, y)
    big = Neurogenesis(grow_frac=1.0, random_state=0, max_steps=6000).fit(X, y)

    def biggest_spurt(m):
        steps = [e.step for e in m.events_ if e.kind == "birth"]
        return max(steps.count(s) for s in set(steps))

    assert biggest_spurt(small) == 2 and biggest_spurt(big) > 2


def test_random_audition_ablation_runs():
    X, y = D.moons()
    m = Neurogenesis(audition="random", random_state=0, max_steps=3000).fit(X, y)
    assert any(e.kind == "birth" for e in m.events_)


def test_save_load_roundtrip(tmp_path):
    X, y = D.spiral(n=300)
    labels = np.array(["red", "green", "blue"])[y]
    m = Neurogenesis(random_state=0, max_steps=2000).fit(X, labels)
    m.save(tmp_path / "clf.npz")
    m2 = Neurogenesis.load(tmp_path / "clf.npz")
    assert (m2.predict(X) == m.predict(X)).all() and m2.widths_ == m.widths_
    np.testing.assert_allclose(m2.predict_proba(X), m.predict_proba(X))

    Xr, yr = D.wavy(n=300)
    r = Neurogenesis(random_state=0, max_steps=2000).fit(Xr, yr)
    r.save(tmp_path / "reg.npz")
    np.testing.assert_allclose(Neurogenesis.load(tmp_path / "reg.npz").predict(Xr), r.predict(Xr))


def test_init_width_per_layer_and_param_validation():
    import pytest

    X, y = D.moons(n=200)
    m = Neurogenesis(hidden_layers=2, init_width=[6, 3], static=True, max_steps=200).fit(X, y)
    assert m.widths_ == [6, 3]
    for bad in (dict(hidden_layers=0), dict(hidden_layers=2, init_width=[2]), dict(validation_fraction=1.0), dict(audition="x"),
                dict(task="ranking"), dict(learning_rate=0), dict(grow_frac=-1), dict(init_width=500)):
        with pytest.raises(ValueError):
            Neurogenesis(**bad).fit(X, y)


def test_scikit_learn_compatibility():
    import pytest

    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    m = Neurogenesis(max_steps=1500, random_state=0)
    c = clone(m)
    assert c.get_params() == m.get_params() and c is not m
    assert m.set_params(grow_k=3).grow_k == 3
    X, y = D.moons(n=300)
    scores = cross_val_score(Neurogenesis(max_steps=1500, random_state=0), X, y, cv=3)
    assert len(scores) == 3 and scores.mean() > 0.9
    assert "max_steps=1500" in repr(m) and sklearn is not None


def test_numpy_typed_params_repr_and_save(tmp_path):
    X, y = D.moons(n=200)
    m = Neurogenesis(hidden_layers=2, init_width=np.array([4, 3]), random_state=np.int64(0), static=True, max_steps=200)
    assert "init_width" in repr(m)
    m.fit(X, y).save(tmp_path / "m.npz")
    assert Neurogenesis.load(tmp_path / "m.npz").widths_ == [4, 3]


def test_tiny_class_keeps_a_training_row():
    y = np.r_[np.zeros(60, int), np.ones(2, int)]
    m = Neurogenesis(validation_fraction=0.9)
    m.task_, m.classes_ = "classification", np.unique(y)
    tr, _ = m._holdout(y, np.random.default_rng(0))
    assert set(y[tr]) == {0, 1}
