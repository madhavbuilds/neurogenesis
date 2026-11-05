"""
Neurogenesis: a neural network that grows its own brain.

You never choose the number of neurons. The network starts with a single
neuron per hidden layer and runs a life cycle while it trains:

  BIRTH    when learning plateaus, a pool of candidate neurons "auditions".
           Each candidate is scored by the exact gradient norm its outgoing
           weights would receive, which is the initial rate at which it would
           reduce the loss. The best, mutually-diverse candidates are born with
           zero outgoing weights, so a birth never changes the network's output.
  DEATH    neurons whose contribution (activation spread x outgoing weight
           norm) withers are removed. Their mean output moves into the next
           layer's bias so the function barely changes.
  FUSION   two neurons that fire in lock-step (|corr| > threshold) are merged
           into one by linear regression, which keeps the output nearly the same.
  MATURITY when repeated growth spurts stop paying off, growth stops. The
           network fine-tunes and halts on its own.

Pure NumPy. No architecture search, no width hyperparameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Neurogenesis", "Event"]


@dataclass
class Event:
    step: int
    kind: str  # "birth" | "death" | "fusion" | "mature" | "converged"
    layer: int
    neuron: int
    info: str = ""

    def __str__(self) -> str:
        icon = {"birth": "+", "death": "x", "fusion": "&", "mature": "*", "converged": "#"}[self.kind]
        where = f" L{self.layer} n{self.neuron}" if self.neuron >= 0 else ""
        return f"[{self.step:>6}] {icon} {self.kind:<9}{where} {self.info}"


class _Param:
    """A parameter tensor with its own Adam state. The state can be sliced and grown with the tensor."""

    __slots__ = ("val", "m", "v", "t")

    def __init__(self, val: np.ndarray):
        self.val = val
        self.m = np.zeros_like(val)
        self.v = np.zeros_like(val)
        self.t = np.zeros(val.shape, dtype=np.int64)

    def select(self, keep: np.ndarray, axis: int) -> None:
        for name in self.__slots__:
            setattr(self, name, np.take(getattr(self, name), keep, axis=axis))

    def append(self, new: np.ndarray, axis: int) -> None:
        self.val = np.concatenate([self.val, new], axis=axis)
        self.m = np.concatenate([self.m, np.zeros_like(new)], axis=axis)
        self.v = np.concatenate([self.v, np.zeros_like(new)], axis=axis)
        self.t = np.concatenate([self.t, np.zeros(new.shape, dtype=np.int64)], axis=axis)

    def adam(self, g: np.ndarray, lr: float, wd: float = 0.0, b1=0.9, b2=0.999, eps=1e-8) -> None:
        # Per-element step counters: newborn weights get proper bias correction, with no
        # oversized first step.
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mhat = self.m / (1 - b1 ** self.t)
        vhat = self.v / (1 - b2 ** self.t)
        self.val = self.val - lr * (mhat / (np.sqrt(vhat) + eps) + wd * self.val)


class Neurogenesis:
    """Self-growing, self-pruning multilayer perceptron (scikit-learn style API).

    Parameters
    ----------
    hidden_layers : number of hidden layers (depth stays fixed; width is learned).
    init_width : neurons each hidden layer is born with.
    max_width : hard cap per hidden layer.
    learning_rate, batch_size, max_steps : usual optimisation knobs.
    check_every : steps between "health checks" (plateau test, pruning, fusion).
    plateau_window, plateau_tol : learning has plateaued when the probe loss improved by less
        than ``plateau_tol`` (relative) over the last ``plateau_window`` checks.
    grow_k : neurons born per growth spurt.
    candidates : size of the audition pool per spurt.
    min_gain : a growth spurt that improves loss by less than this (relative) is a "strike".
    maturity_patience : strikes in a row before the network stops growing.
    prune_utility : neurons whose utility falls below this fraction of the layer mean die.
    fuse_corr : |correlation| above which two neurons are fused.
    min_age : steps a newborn is protected from death and fusion.
    weight_decay : decoupled weight decay. It slowly starves useless neurons.
    task : "auto", "classification" or "regression".
    static : disable the life cycle (a plain fixed-width MLP trained for ``max_steps``). Use it for baselines.
    """

    def __init__(
        self,
        hidden_layers: int = 1,
        init_width: int = 1,
        max_width: int = 128,
        learning_rate: float = 0.01,
        batch_size: int = 128,
        max_steps: int = 30_000,
        check_every: int = 100,
        plateau_window: int = 4,
        plateau_tol: float = 0.03,
        grow_k: int = 2,
        candidates: int = 512,
        min_gain: float = 0.05,
        maturity_patience: int = 3,
        prune_utility: float = 0.05,
        fuse_corr: float = 0.995,
        min_age: int = 600,
        weight_decay: float = 1e-4,
        task: str = "auto",
        static: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.hidden_layers = hidden_layers
        self.init_width = init_width
        self.max_width = max_width
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.check_every = check_every
        self.plateau_window = plateau_window
        self.plateau_tol = plateau_tol
        self.grow_k = grow_k
        self.candidates = candidates
        self.min_gain = min_gain
        self.maturity_patience = maturity_patience
        self.prune_utility = prune_utility
        self.fuse_corr = fuse_corr
        self.min_age = min_age
        self.weight_decay = weight_decay
        self.task = task
        self.static = static
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ public API
    @property
    def widths_(self) -> list[int]:
        return [self._W[h].val.shape[1] for h in range(self.hidden_layers)]

    @property
    def n_neurons_(self) -> int:
        return int(sum(self.widths_))

    @property
    def n_params_(self) -> int:
        return int(sum(W.val.size + b.val.size for W, b in zip(self._W, self._b)))

    def fit(self, X, y, callback=None) -> "Neurogenesis":
        """Train and grow. ``callback(model, step, probe_loss)`` is called at every health check."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(X) < 2:
            raise ValueError(f"X must be 2-D with one row per label; got X{X.shape}, y{y.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        self.task_ = self._infer_task(y)
        if self.task_ == "regression" and not np.isfinite(y.astype(float)).all():
            raise ValueError("y contains NaN or infinite values")

        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self._x_mu) / self._x_sd
        if self.task_ == "classification":
            self.classes_, yi = np.unique(y, return_inverse=True)
            Y = np.eye(len(self.classes_))[yi]
        else:
            Y = y.reshape(len(y), -1).astype(float)
            self._y_mu, self._y_sd = Y.mean(0), Y.std(0) + 1e-8
            Y = (Y - self._y_mu) / self._y_sd

        n, d_in = Xs.shape
        dims = [d_in] + [self.init_width] * self.hidden_layers + [Y.shape[1]]
        self._W = [_Param(rng.normal(0, 1 / np.sqrt(dims[i]), (dims[i], dims[i + 1]))) for i in range(len(dims) - 1)]
        self._b = [_Param(np.zeros(dims[i + 1])) for i in range(len(dims) - 1)]
        self._next_id = 0
        self._ids = [self._new_ids(w) for w in dims[1:-1]]
        self._born = [np.zeros(w, dtype=np.int64) for w in dims[1:-1]]

        probe = rng.choice(n, size=min(n, 2048), replace=False)
        Xp, Yp = Xs[probe], Y[probe]

        self.events_: list[Event] = []
        self.history_ = {"step": [], "loss": [], "neurons": [], "widths": []}
        self.mature_ = False
        strikes, growth_base, since_growth = 0, None, 0
        bs = min(self.batch_size, n)

        for step in range(1, self.max_steps + 1):
            idx = rng.integers(0, n, bs)
            acts, out = self._forward(Xs[idx])
            _, dout = self._loss(out, Y[idx])
            gW, gb, _ = self._backward(acts, dout)
            for i in range(len(self._W)):
                self._W[i].adam(gW[i], self.learning_rate, self.weight_decay)
                self._b[i].adam(gb[i], self.learning_rate)

            if step % self.check_every:
                continue

            # ---------------------------------------------------------- health check
            acts, out = self._forward(Xp)
            loss, dout = self._loss(out, Yp)
            self._record(step, loss)
            since_growth += 1
            if callback is not None:
                callback(self, step, loss)
            if self.static:
                continue
            self._prune_and_fuse(step, acts)

            L = self.history_["loss"]
            w = self.plateau_window
            plateau = since_growth > w and (L[-1 - w] - L[-1]) / max(L[-1 - w], 1e-12) < self.plateau_tol
            if not plateau:
                continue

            if self.mature_:
                self._log(Event(step, "converged", -1, -1, f"loss={loss:.4f} neurons={self.n_neurons_}"))
                break

            perfect = self.task_ == "classification" and self._accuracy(out, Yp) >= 0.999 and loss < 0.05
            if growth_base is not None:
                gain = (growth_base - loss) / max(growth_base, 1e-12)
                strikes = strikes + 1 if (gain < self.min_gain or perfect) else 0
            if strikes >= self.maturity_patience or perfect or self.n_neurons_ >= self.max_width * self.hidden_layers:
                self.mature_ = True
                self._log(Event(step, "mature", -1, -1, f"stopped growing at {self.n_neurons_} neurons, widths={self.widths_}"))
                since_growth = 0
                continue

            acts, out = self._forward(Xp)  # structure may have changed during pruning
            _, dout = self._loss(out, Yp)
            _, _, deltas = self._backward(acts, dout)
            if self._grow(step, acts, deltas, rng):
                growth_base, since_growth = loss, 0
            else:
                self.mature_ = True
                self._log(Event(step, "mature", -1, -1, "no useful candidates left"))

        self.n_steps_ = step
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._check_fitted().task_ != "classification":
            raise AttributeError("predict_proba is only available for classification")
        _, out = self._forward(self._prep(X))
        return _softmax(out)

    def predict(self, X) -> np.ndarray:
        _, out = self._forward(self._prep(X))
        if self.task_ == "classification":
            return self.classes_[out.argmax(1)]
        pred = out * self._y_sd + self._y_mu
        return pred[:, 0] if pred.shape[1] == 1 else pred

    def score(self, X, y) -> float:
        """Accuracy for classification, R^2 for regression."""
        p = self.predict(X)
        y = np.asarray(y)
        if self.task_ == "classification":
            return float((p == y).mean())
        y = y.reshape(p.shape).astype(float)
        return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

    def life_story(self) -> str:
        counts = {k: sum(e.kind == k for e in self.events_) for k in ("birth", "death", "fusion")}
        head = (
            f"Neurogenesis: {self.n_neurons_} neurons {self.widths_}, {self.n_params_} params, "
            f"{self.n_steps_} steps | births={counts['birth']} deaths={counts['death']} fusions={counts['fusion']}"
        )
        return "\n".join([head] + [str(e) for e in self.events_])

    def snapshot(self) -> dict:
        """Lightweight copy of the current network (used for animation)."""
        return {
            "W": [W.val.copy() for W in self._W],
            "b": [b.val.copy() for b in self._b],
            "born": [b.copy() for b in self._born],
            "x_mu": self._x_mu,
            "x_sd": self._x_sd,
        }

    # ------------------------------------------------------------------ internals
    def _infer_task(self, y: np.ndarray) -> str:
        if self.task != "auto":
            return self.task
        if y.dtype.kind in "biuOUS":
            return "classification"
        if y.ndim == 1 and np.all(np.mod(y, 1) == 0) and len(np.unique(y)) <= 32:
            # Ambiguous: integer-valued floats. Pass task="regression" for counts or ratings.
            return "classification"
        return "regression"

    def _check_fitted(self) -> "Neurogenesis":
        if not hasattr(self, "_W"):
            raise RuntimeError("This Neurogenesis model is not fitted yet. Call fit(X, y) first.")
        return self

    def _prep(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] != len(self._x_mu):
            raise ValueError(f"X has {X.shape[1]} features, but the model was fitted with {len(self._x_mu)}")
        return (X - self._x_mu) / self._x_sd

    def _new_ids(self, k: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + k)
        self._next_id += k
        return ids

    def _forward(self, X):
        acts = [X]
        for i in range(len(self._W) - 1):
            acts.append(np.tanh(acts[-1] @ self._W[i].val + self._b[i].val))
        return acts, acts[-1] @ self._W[-1].val + self._b[-1].val

    def _loss(self, out, Y):
        n = len(Y)
        if self.task_ == "classification":
            p = _softmax(out)
            return float(-np.log(p[Y > 0] + 1e-12).mean()), (p - Y) / n
        r = out - Y
        return float(0.5 * (r * r).sum(1).mean()), r / n

    def _backward(self, acts, dout):
        L = len(self._W)
        gW, gb, deltas = [None] * L, [None] * L, [None] * L
        delta = dout
        for i in reversed(range(L)):
            deltas[i] = delta
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self._W[i].val.T) * (1 - acts[i] ** 2)
        return gW, gb, deltas

    @staticmethod
    def _accuracy(out, Y) -> float:
        return float((out.argmax(1) == Y.argmax(1)).mean())

    def _record(self, step, loss):
        h = self.history_
        h["step"].append(step)
        h["loss"].append(loss)
        h["neurons"].append(self.n_neurons_)
        h["widths"].append(list(self.widths_))

    def _log(self, e: Event):
        self.events_.append(e)
        if self.verbose:
            print(e)

    # ---------------------------------------------------------------- BIRTH
    def _grow(self, step, acts, deltas, rng) -> bool:
        # Grow in the layer with the most unexplained gradient signal.
        best = None
        for h in range(self.hidden_layers):
            room = self.max_width - self.widths_[h]
            if room <= 0:
                continue
            Win, bin_, scores = self._audition(acts[h], deltas[h + 1], acts[h + 1], min(self.grow_k, room), rng)
            if len(scores) and (best is None or scores.max() > best[3].max()):
                best = (h, Win, bin_, scores)
        if best is None:
            return False
        h, Win, bin_, scores = best
        k = Win.shape[1]
        d_out = self._W[h + 1].val.shape[1]
        self._W[h].append(Win, axis=1)
        self._b[h].append(bin_, axis=0)
        self._W[h + 1].append(np.zeros((k, d_out)), axis=0)  # zero fan-out: function preserved exactly
        new = self._new_ids(k)
        self._ids[h] = np.concatenate([self._ids[h], new])
        self._born[h] = np.concatenate([self._born[h], np.full(k, step)])
        for nid, s in zip(new, scores):
            self._log(Event(step, "birth", h, int(nid), f"pressure={s:.4f} -> layer width {self.widths_[h]}"))
        return True

    def _audition(self, A, D, H_existing, k, rng):
        """Score a pool of candidate neurons by the gradient their fan-out would receive.

        A: inputs to the layer (n, d). D: dLoss/d(pre-activation) of the next layer (n, d_out).
        Returns the weights, biases and scores of up to k winners, which are high-scoring and
        not redundant with each other or with existing neurons.
        """
        Wc, bc, scores, Hc, Hn = self._candidate_pool(A, D, rng)
        return self._select(Wc, bc, scores, Hc, Hn, H_existing, k)

    def _candidate_pool(self, A, D, rng):
        """Build and score the whole audition pool. Returns (W, b, scores, raw acts, standardized acts)."""
        n, d = A.shape
        C = self.candidates
        Wc = rng.normal(size=(d, C))
        # Seed the pool with the linear-optimal directions: top singular vectors of A^T D.
        U, S, _ = np.linalg.svd((A - A.mean(0)).T @ D, full_matrices=False)
        r = min(int((S > 1e-9 * max(S.max(), 1e-30)).sum()), C // 4)
        Wc[:, :r] = U[:, :r]
        proj_sd = (A @ Wc).std(0) + 1e-8
        sharp = rng.uniform(0.5, 4.0, C)  # mixture of soft and sharp candidates
        sharp[:r] = 1.0
        Wc = Wc * (sharp / proj_sd)
        anchors = A[rng.integers(0, n, C)]  # each hyperplane passes through a real data point
        bc = -np.einsum("cd,dc->c", anchors, Wc)
        bc[:r] = -(A.mean(0) @ Wc[:, :r])

        Hc = np.tanh(A @ Wc + bc)
        Hn = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-8)
        scores = np.linalg.norm(Hn.T @ D, axis=1)  # ||dL/d(fan-out)|| for a unit-variance neuron
        return Wc, bc, scores, Hc, Hn

    @staticmethod
    def _select(Wc, bc, scores, Hc, Hn, H_existing, k):
        """Greedy pick of the best k candidates, skipping any that duplicate existing neurons or each other."""
        n = len(Hn)
        En =(H_existing - H_existing.mean(0)) / (H_existing.std(0) + 1e-8)
        chosen: list[int] = []
        for c in np.argsort(-scores):
            if len(chosen) == k or scores[c] < 1e-8:
                break
            hc = Hn[:, c]
            if Hc[:, c].std() < 1e-3:
                continue
            if En.shape[1] and np.abs(En.T @ hc).max() / n > 0.95:
                continue
            if chosen and np.abs(Hn[:, chosen].T @ hc).max() / n > 0.95:
                continue
            chosen.append(int(c))
        return Wc[:, chosen], bc[chosen], scores[chosen]

    # ------------------------------------------------------- DEATH and FUSION
    def _prune_and_fuse(self, step, acts):
        for h in range(self.hidden_layers):
            H = acts[h + 1]
            old = (step - self._born[h]) >= self.min_age

            # FUSION: lock-step twins merge, using linear regression h_b = c*h_a + d.
            if self.widths_[h] > 1 and old.sum() > 1:
                sd = H.std(0)
                live = np.where(old & (sd > 1e-6))[0]
                if len(live) > 1:
                    Hc = H[:, live] - H[:, live].mean(0)
                    corr = (Hc.T @ Hc) / (np.outer(sd[live], sd[live]) * len(H))
                    np.fill_diagonal(corr, 0)
                    i, j = np.unravel_index(np.abs(corr).argmax(), corr.shape)
                    if abs(corr[i, j]) > self.fuse_corr:
                        a, b = live[i], live[j]
                        c = (Hc[:, j] @ Hc[:, i]) / (Hc[:, i] @ Hc[:, i])
                        dd = H[:, b].mean() - c * H[:, a].mean()
                        Wn = self._W[h + 1].val
                        Wn[a] += c * Wn[b]
                        self._b[h + 1].val += dd * Wn[b]
                        self._log(Event(step, "fusion", h, int(self._ids[h][b]), f"-> n{self._ids[h][a]} (corr={corr[i, j]:+.3f})"))
                        self._remove(h, np.array([b]), H, compensate=False)
                        H = np.delete(H, b, axis=1)
                        old = np.delete(old, b)

            # DEATH: utility = activation spread x fan-out strength, relative to layer mean.
            if self.widths_[h] > 1:
                util = H.std(0) * np.linalg.norm(self._W[h + 1].val, axis=1)
                rel = util / (util.mean() + 1e-12)
                dead = np.where(old & (rel < self.prune_utility))[0]
                if len(dead) >= self.widths_[h]:
                    dead = dead[np.argsort(rel[dead])][: self.widths_[h] - 1]
                if len(dead):
                    for d_ in dead:
                        self._log(Event(step, "death", h, int(self._ids[h][d_]), f"utility={rel[d_]:.3f}"))
                    self._remove(h, dead, H, compensate=True)

    def _remove(self, h, idx, H, compensate):
        if compensate:  # move the neuron's mean output into the next layer's bias
            self._b[h + 1].val += H[:, idx].mean(0) @ self._W[h + 1].val[idx]
        keep = np.setdiff1d(np.arange(self.widths_[h]), idx)
        self._W[h].select(keep, axis=1)
        self._b[h].select(keep, axis=0)
        self._W[h + 1].select(keep, axis=0)
        self._ids[h] = self._ids[h][keep]
        self._born[h] = self._born[h][keep]


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
