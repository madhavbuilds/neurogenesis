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


def test_death_compensates_with_bias():
    m, X, _ = _trained()
    Xs = m._prep(X)
    acts, _ = m._forward(Xs)
    H = acts[1]
    util = H.std(0) * np.linalg.norm(m._W[1].val, axis=1)
    weakest = np.array([util.argmin()])
    before = m.predict_proba(X)
    m._remove(0, weakest, H, compensate=True)
    after = m.predict_proba(X)
    assert np.abs(after - before).mean() < 0.05


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
