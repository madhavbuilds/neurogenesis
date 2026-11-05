"""Tiny, dependency-free toy datasets."""
import numpy as np


def spiral(n=1200, arms=3, turns=1.5, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    per = n // arms
    X, y = [], []
    for k in range(arms):
        t = np.sqrt(rng.uniform(0, 1, per)) * turns * 2 * np.pi
        r = t / (turns * 2 * np.pi) * 3
        a = t + 2 * np.pi * k / arms
        X.append(np.c_[r * np.cos(a), r * np.sin(a)] + rng.normal(0, noise * r[:, None] / 3, (per, 2)))
        y.append(np.full(per, k))
    return np.vstack(X), np.concatenate(y)


def moons(n=1000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    h = n // 2
    t1, t2 = rng.uniform(0, np.pi, h), rng.uniform(0, np.pi, n - h)
    X = np.r_[np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]]
    return X + rng.normal(0, noise, X.shape), np.r_[np.zeros(h, int), np.ones(n - h, int)]


def circles(n=1000, noise=0.08, factor=0.5, seed=0):
    rng = np.random.default_rng(seed)
    h = n // 2
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.r_[np.ones(h), np.full(n - h, factor)]
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    return X + rng.normal(0, noise, X.shape), np.r_[np.zeros(h, int), np.ones(n - h, int)]


def checkerboard(n=2000, tiles=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((np.floor((X[:, 0] + 1) * tiles / 2) + np.floor((X[:, 1] + 1) * tiles / 2)) % 2).astype(int)
    return X, y


def wavy(n=1000, noise=0.1, seed=0):
    """1-D regression: a chirp, a signal whose frequency rises across the input range."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    return x[:, None], np.sin(x * (1.5 + 0.5 * x)) + 0.3 * x + rng.normal(0, noise, n)


def split(X, y, test=0.25, seed=0):
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(X))
    k = int(len(X) * (1 - test))
    return X[idx[:k]], X[idx[k:]], y[idx[:k]], y[idx[k:]]


ALL = {"spiral": spiral, "moons": moons, "circles": circles, "checkerboard": checkerboard}
