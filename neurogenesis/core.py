"""
Neurogenesis: a neural network that grows its own brain.

You never choose the number of neurons. The network starts with a single
neuron per hidden layer and runs a life cycle while it trains:

  BIRTH    when learning plateaus, a pool of candidate neurons "auditions".
           Each candidate is scored by how much of the remaining error signal
           it can explain that the existing neurons cannot (its gradient
           pressure after projecting out the current layer). The best,
           mutually-novel candidates are born with zero outgoing weights, so a
           birth never changes the network's output.
  DEATH    neurons whose contribution (activation spread x outgoing weight
           norm) withers are removed. Their mean output moves into the next
           layer's bias so the function barely changes.
  FUSION   two neurons that fire in lock-step (|corr| > threshold) are merged
           into one by linear regression, which keeps the output nearly the same.
  MATURITY when growth spurts stop improving the *validation* loss, growth stops.
           The network fine-tunes, early-stops, and restores its best
           validation checkpoint.

Pure NumPy. No architecture search, no width hyperparameter.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass

import numpy as np

__all__ = ["Neurogenesis", "Event"]

_ICONS = {"birth": "+", "death": "x", "fusion": "&", "mature": "*", "converged": "#", "stopped": "!", "restore": "<"}
_MIN_NOVELTY = 0.1  # a candidate must keep >= 10% of its signal after projecting out existing neurons


@dataclass
class Event:
    step: int
    kind: str  # "birth" | "death" | "fusion" | "mature" | "converged" | "stopped" | "restore"
    layer: int
    neuron: int
    info: str = ""

    def __str__(self) -> str:
        where = f" L{self.layer} n{self.neuron}" if self.neuron >= 0 else ""
        return f"[{self.step:>6}] {_ICONS[self.kind]} {self.kind:<9}{where} {self.info}"


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
    init_width : neurons each hidden layer is born with (an int, or one int per layer).
    max_width : hard cap per hidden layer.
    learning_rate, batch_size, max_steps : usual optimisation knobs.
    check_every : steps between "health checks" (plateau test, pruning, fusion, validation).
    plateau_window, plateau_tol : learning has plateaued when the training probe loss improved by
        less than ``plateau_tol`` (relative) over the last ``plateau_window`` checks.
    grow_k : minimum neurons born per growth spurt.
    grow_frac : a spurt adds at least this fraction of the layer's current width (geometric growth).
        0.5 grows faster and suits larger real-world data; 0 is the v0.1 behaviour.
    candidates : size of the audition pool per spurt.
    min_gain : a growth spurt that improves the validation loss by less than this (relative) is a "strike".
    maturity_patience : strikes in a row before the network stops growing.
    prune_utility : neurons whose utility falls below this fraction of the layer mean die (0 disables death).
    fuse_corr : |correlation| above which two neurons are fused (>= 1 disables fusion).
    min_age : steps a newborn is protected from death and fusion.
    weight_decay : decoupled weight decay. It slowly starves useless neurons.
    validation_fraction : share of the training data held out to judge growth, early-stop and pick the
        best checkpoint (stratified for classification). Ignored when ``eval_set`` is passed to ``fit``.
        0 disables validation: decisions then use training data, as in v0.1.
    patience : stop after this many health checks without a validation improvement (0 disables).
    restore_best : at the end of training, restore the checkpoint with the best validation loss.
    audition : "gradient" (score candidates) or "random" (hire random valid candidates; an ablation).
    task : "auto", "classification" or "regression".
    static : disable the life cycle (a plain fixed-width MLP with the same training and early stopping).
        Use it for baselines.
    """

    def __init__(
        self,
        hidden_layers: int = 1,
        init_width=1,
        max_width: int = 128,
        learning_rate: float = 0.01,
        batch_size: int = 128,
        max_steps: int = 30_000,
        check_every: int = 100,
        plateau_window: int = 4,
        plateau_tol: float = 0.03,
        grow_k: int = 2,
        grow_frac: float = 0.25,
        candidates: int = 512,
        min_gain: float = 0.01,
        maturity_patience: int = 3,
        prune_utility: float = 0.05,
        fuse_corr: float = 0.995,
        min_age: int = 600,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.1,
        patience: int = 20,
        restore_best: bool = True,
        audition: str = "gradient",
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
        self.grow_frac = grow_frac
        self.candidates = candidates
        self.min_gain = min_gain
        self.maturity_patience = maturity_patience
        self.prune_utility = prune_utility
        self.fuse_corr = fuse_corr
        self.min_age = min_age
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.restore_best = restore_best
        self.audition = audition
        self.task = task
        self.static = static
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ scikit-learn protocol
    @classmethod
    def _param_names(cls) -> list[str]:
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self, **params) -> "Neurogenesis":
        names = self._param_names()
        for k, v in params.items():
            if k not in names:
                raise ValueError(f"Invalid parameter {k!r} for Neurogenesis. Valid parameters: {names}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        defaults = {k: p.default for k, p in inspect.signature(type(self).__init__).parameters.items() if k != "self"}
        changed = [f"{k}={v!r}" for k, v in self.get_params().items() if repr(v) != repr(defaults[k])]
        return f"Neurogenesis({', '.join(changed)})"

    def __sklearn_tags__(self):  # only called by scikit-learn, so the import is safe here
        from sklearn.utils import Tags, TargetTags

        return Tags(estimator_type=None, target_tags=TargetTags(required=True))

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

    def fit(self, X, y, callback=None, eval_set=None) -> "Neurogenesis":
        """Train and grow.

        ``callback(model, step, probe_loss)`` is called at every health check.
        ``eval_set=(X_val, y_val)`` supplies your own validation data; otherwise
        ``validation_fraction`` of ``X`` is held out.
        """
        self._check_params()
        rng = np.random.default_rng(self.random_state)
        X, y = self._check_xy(X, y)
        self.task_ = self._infer_task(y)
        if self.task_ == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) < 2:
                raise ValueError("classification needs at least 2 classes in y")
        self.n_features_in_ = X.shape[1]

        if eval_set is not None:
            Xv, yv = self._check_xy(*eval_set)
            if Xv.shape[1] != X.shape[1]:
                raise ValueError(f"eval_set has {Xv.shape[1]} features, X has {X.shape[1]}")
        elif self.validation_fraction > 0 and len(X) >= 50:
            tr, va = self._holdout(y, rng)
            X, y, Xv, yv = X[tr], y[tr], X[va], y[va]
        else:
            Xv = yv = None
        self.has_validation_ = Xv is not None

        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-8
        if self.task_ == "regression":
            Yr = y.reshape(len(y), -1).astype(float)
            self._y_mu, self._y_sd = Yr.mean(0), Yr.std(0) + 1e-8
        Xs, Y = self._scale(X), self._encode(y)

        n, d_in = Xs.shape
        dims = [d_in] + self._init_widths() + [Y.shape[1]]
        self._W = [_Param(rng.normal(0, 1 / np.sqrt(dims[i]), (dims[i], dims[i + 1]))) for i in range(len(dims) - 1)]
        self._b = [_Param(np.zeros(dims[i + 1])) for i in range(len(dims) - 1)]
        self._next_id = 0
        self._ids = [self._new_ids(w) for w in dims[1:-1]]
        self._born = [np.zeros(w, dtype=np.int64) for w in dims[1:-1]]

        probe = rng.choice(n, size=min(n, 2048), replace=False)
        Xp, Yp = Xs[probe], Y[probe]
        if self.has_validation_:
            Xvs, Yv = self._scale(Xv), self._encode(yv)
            if len(Xvs) > 8192:
                sub = rng.choice(len(Xvs), 8192, replace=False)
                Xvs, Yv = Xvs[sub], Yv[sub]

        self.events_: list[Event] = []
        self.history_ = {"step": [], "loss": [], "val_loss": [], "neurons": [], "widths": []}
        self.mature_ = False
        self.flops_ = 0.0
        self._best = None
        best_val, bad_checks = np.inf, 0
        strikes, growth_base, since_growth, best_since_growth = 0, None, 0, np.inf
        bs = min(self.batch_size, n)

        for step in range(1, self.max_steps + 1):
            idx = rng.integers(0, n, bs)
            acts, out = self._forward(Xs[idx])
            _, dout = self._loss(out, Y[idx])
            gW, gb, _ = self._backward(acts, dout)
            for i in range(len(self._W)):
                self._W[i].adam(gW[i], self.learning_rate, self.weight_decay)
                self._b[i].adam(gb[i], self.learning_rate)
            self.flops_ += 6.0 * bs * self.n_params_

            if step % self.check_every:
                continue

            # ---------------------------------------------------------- health check
            acts, out = self._forward(Xp)
            loss, dout = self._loss(out, Yp)
            vloss = self._loss(self._forward(Xvs)[1], Yv)[0] if self.has_validation_ else loss
            self.flops_ += 2.0 * (len(Xp) + (len(Xvs) if self.has_validation_ else 0)) * self.n_params_
            self._record(step, loss, vloss)
            if vloss < best_val * (1 - 1e-4):
                best_val, bad_checks = vloss, 0
                if self.restore_best:
                    self._best = self._checkpoint(step, vloss)
            else:
                bad_checks += 1
            best_since_growth = min(best_since_growth, vloss)
            since_growth += 1
            if callback is not None:
                callback(self, step, loss)
            if self.patience and bad_checks >= self.patience:
                self._log(Event(step, "stopped", -1, -1, f"no validation improvement for {bad_checks * self.check_every} steps"))
                break
            if self.static:
                continue
            self._prune_and_fuse(step, acts)

            L = self.history_["loss"]
            w = self.plateau_window
            plateau = since_growth > w and (L[-1 - w] - L[-1]) / max(L[-1 - w], 1e-12) < self.plateau_tol
            if not plateau:
                continue

            if self.mature_:
                self._log(Event(step, "converged", -1, -1, f"loss={loss:.4f} val_loss={vloss:.4f} neurons={self.n_neurons_}"))
                break

            perfect = self.task_ == "classification" and self._accuracy(out, Yp) >= 0.999 and loss < 0.05
            if growth_base is not None:
                gain = (growth_base - best_since_growth) / max(growth_base, 1e-12)
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
                growth_base, since_growth, best_since_growth = vloss, 0, vloss
            else:
                self.mature_ = True
                self._log(Event(step, "mature", -1, -1, "no useful candidates left"))

        self.n_steps_ = step
        self.best_val_loss_ = float(best_val)
        if self._best is not None:
            self._restore(self._best)
        self._best = None
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

    def save(self, path) -> None:
        """Save a fitted model to a ``.npz`` file (no pickle). Load it with ``Neurogenesis.load``."""
        self._check_fitted()
        arrays = {f"W{i}": W.val for i, W in enumerate(self._W)}
        arrays.update({f"b{i}": b.val for i, b in enumerate(self._b)})
        arrays.update(x_mu=self._x_mu, x_sd=self._x_sd)
        if self.task_ == "classification":
            classes = np.asarray(self.classes_)
            if classes.dtype.kind == "O":
                raise ValueError("labels of dtype object can't be saved without pickle; convert y to str or int")
            arrays["classes"] = classes
        else:
            arrays.update(y_mu=self._y_mu, y_sd=self._y_sd)
        meta = {"params": self.get_params(), "task": self.task_, "n_steps": self.n_steps_, "format": 1}
        meta_json = json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))  # NumPy scalars/arrays
        np.savez(path, meta=np.array(meta_json), **arrays)

    @classmethod
    def load(cls, path) -> "Neurogenesis":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            m = cls(**meta["params"])
            n_layers = m.hidden_layers + 1
            m._W = [_Param(f[f"W{i}"]) for i in range(n_layers)]
            m._b = [_Param(f[f"b{i}"]) for i in range(n_layers)]
            m._x_mu, m._x_sd = f["x_mu"], f["x_sd"]
            m.task_ = meta["task"]
            if m.task_ == "classification":
                m.classes_ = f["classes"]
            else:
                m._y_mu, m._y_sd = f["y_mu"], f["y_sd"]
        m.n_features_in_ = m._W[0].val.shape[0]
        m._next_id = 0
        m._ids = [m._new_ids(w) for w in m.widths_]
        m._born = [np.zeros(w, dtype=np.int64) for w in m.widths_]
        m.n_steps_, m.events_, m.history_, m.mature_ = meta["n_steps"], [], {}, True
        return m

    # ------------------------------------------------------------------ internals
    def _check_params(self) -> None:
        def bad(msg):
            raise ValueError(f"Invalid Neurogenesis parameter: {msg}")

        if not (isinstance(self.hidden_layers, (int, np.integer)) and self.hidden_layers >= 1):
            bad(f"hidden_layers must be an int >= 1, got {self.hidden_layers!r}")
        widths = self._init_widths()
        if len(widths) != self.hidden_layers or min(widths) < 1:
            bad(f"init_width must be an int >= 1 or one per hidden layer, got {self.init_width!r}")
        if not self.static and max(widths) > self.max_width:
            bad(f"init_width {self.init_width!r} exceeds max_width {self.max_width}")
        for name in ("learning_rate",):
            if not getattr(self, name) > 0:
                bad(f"{name} must be > 0")
        for name in ("batch_size", "max_steps", "check_every", "plateau_window", "grow_k", "candidates", "maturity_patience"):
            v = getattr(self, name)
            if not (isinstance(v, (int, np.integer)) and v >= 1):
                bad(f"{name} must be an int >= 1, got {v!r}")
        if not self.grow_frac >= 0:
            bad(f"grow_frac must be >= 0, got {self.grow_frac!r}")
        if not 0 <= self.validation_fraction < 1:
            bad(f"validation_fraction must be in [0, 1), got {self.validation_fraction!r}")
        if not (isinstance(self.patience, (int, np.integer)) and self.patience >= 0):
            bad(f"patience must be an int >= 0, got {self.patience!r}")
        if self.audition not in ("gradient", "random"):
            bad(f"audition must be 'gradient' or 'random', got {self.audition!r}")
        if self.task not in ("auto", "classification", "regression"):
            bad(f"task must be 'auto', 'classification' or 'regression', got {self.task!r}")

    def _init_widths(self) -> list[int]:
        if np.ndim(self.init_width) == 0:
            return [int(self.init_width)] * self.hidden_layers
        return [int(w) for w in self.init_width]

    @staticmethod
    def _check_xy(X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(X) < 2:
            raise ValueError(f"X must be 2-D with one row per label; got X{X.shape}, y{y.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        return X, y

    def _infer_task(self, y: np.ndarray) -> str:
        if self.task != "auto":
            task = self.task
        elif y.dtype.kind in "biuOUS":
            task = "classification"
        elif y.ndim == 1 and np.all(np.mod(y, 1) == 0) and len(np.unique(y)) <= 32:
            # Ambiguous: integer-valued floats. Pass task="regression" for counts or ratings.
            task = "classification"
        else:
            task = "regression"
        if task == "regression" and not np.isfinite(y.astype(float)).all():
            raise ValueError("y contains NaN or infinite values")
        return task

    def _holdout(self, y, rng):
        """Stratified (classification) or random hold-out split. Returns (train_idx, val_idx)."""
        n, f = len(y), self.validation_fraction
        if self.task_ == "classification":
            val = []
            for c in self.classes_:
                idx = rng.permutation(np.flatnonzero(y == c))
                val.append(idx[: min(int(round(f * len(idx))), len(idx) - 1)])  # every class keeps a training row
            val = np.concatenate(val)
        else:
            val = rng.permutation(n)[: max(1, int(round(f * n)))]
        mask = np.zeros(n, bool)
        mask[val] = True
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    def _scale(self, X) -> np.ndarray:
        return (X - self._x_mu) / self._x_sd

    def _encode(self, y) -> np.ndarray:
        if self.task_ == "classification":
            idx = np.clip(np.searchsorted(self.classes_, y), 0, len(self.classes_) - 1)
            if not np.all(self.classes_[idx] == y):
                raise ValueError("y contains labels that are not in the training data")
            return np.eye(len(self.classes_))[idx]
        return (y.reshape(len(y), -1).astype(float) - self._y_mu) / self._y_sd

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
        return self._scale(X)

    def _new_ids(self, k: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + k)
        self._next_id += k
        return ids

    def _checkpoint(self, step, vloss) -> dict:
        return {
            "step": step,
            "val_loss": vloss,
            "W": [W.val.copy() for W in self._W],
            "b": [b.val.copy() for b in self._b],
            "ids": [i.copy() for i in self._ids],
            "born": [b.copy() for b in self._born],
        }

    def _restore(self, ck) -> None:
        current = [W.val.shape for W in self._W]
        same = current == [W.shape for W in ck["W"]] and all(np.array_equal(a.val, b) for a, b in zip(self._W, ck["W"]))
        if same:
            return
        self._W = [_Param(W) for W in ck["W"]]
        self._b = [_Param(b) for b in ck["b"]]
        self._ids, self._born = ck["ids"], ck["born"]
        self._log(Event(self.n_steps_, "restore", -1, -1,
                        f"best validation checkpoint from step {ck['step']}: val_loss={ck['val_loss']:.4f}, widths={self.widths_}"))

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

    def _record(self, step, loss, vloss):
        h = self.history_
        h["step"].append(step)
        h["loss"].append(loss)
        h["val_loss"].append(vloss)
        h["neurons"].append(self.n_neurons_)
        h["widths"].append(list(self.widths_))

    def _log(self, e: Event):
        self.events_.append(e)
        if self.verbose:
            print(e)

    # ---------------------------------------------------------------- BIRTH
    def _grow(self, step, acts, deltas, rng) -> bool:
        # Grow in the layer whose best candidate explains the largest *fraction* of that layer's
        # error signal. The fraction is scale-invariant, so a layer can't win just because the
        # weights downstream of it are large.
        best = None
        for h in range(self.hidden_layers):
            room = self.max_width - self.widths_[h]
            if room <= 0:
                continue
            k = min(max(self.grow_k, int(round(self.grow_frac * self.widths_[h]))), room)
            Win, bin_, scores = self._audition(acts[h], deltas[h + 1], acts[h + 1], k, rng)
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
        """Score a pool of candidate neurons and hire up to k of them.

        A: inputs to the layer (n, d). D: dLoss/d(pre-activation) of the next layer (n, d_out).
        H_existing: current activations of the layer (n, w).
        Returns the weights, biases and scores of the winners, best first.
        """
        Wc, bc, _, Hc, _ = self._candidate_pool(A, D, rng)
        return self._select(Wc, bc, Hc, D, H_existing, k, rng)

    def _candidate_pool(self, A, D, rng):
        """Build and score the whole audition pool. Returns (W, b, scores, raw acts, standardized acts).

        The scores here are the plain gradient pressure of each candidate on its own:
        ``||h^T D|| / (||h|| ||D||)``, the fraction of the error signal a unit-variance candidate
        lines up with. ``_select`` refines them against the existing neurons.
        """
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
        scores = np.linalg.norm(Hn.T @ D, axis=1) / (np.sqrt(n) * np.linalg.norm(D) + 1e-30)
        self.flops_ = getattr(self, "flops_", 0.0) + 2.0 * n * C * (2 * d + D.shape[1])
        return Wc, bc, scores, Hc, Hn

    def _select(self, Wc, bc, Hc, D, H_existing, k, rng):
        """Greedy, projection-based pick of k candidates (orthogonal matching pursuit).

        Each candidate is first projected onto the part of activation space the existing neurons
        (plus a bias) can't reach. Its score is the fraction of the error signal that this *new*
        part explains: ``||r^T D|| / (||r|| ||D||)``. After each hire, the remaining candidates are
        projected again, so two winners never duplicate each other. Candidates that keep less than
        ``_MIN_NOVELTY`` of their signal after projection are skipped as redundant.
        """
        n = len(Hc)
        B = np.c_[np.ones(n), H_existing]
        U, s, _ = np.linalg.svd(B, full_matrices=False)
        Q = U[:, s > 1e-8 * s.max()]
        Hc0 = Hc - Hc.mean(0)
        base = np.linalg.norm(Hc0, axis=0) + 1e-12
        R = Hc - Q @ (Q.T @ Hc)
        Dn = np.linalg.norm(D) + 1e-30
        ok = Hc.std(0) > 1e-3
        self.flops_ = getattr(self, "flops_", 0.0) + 4.0 * n * Q.shape[1] * Hc.shape[1]

        chosen, chosen_scores = [], []
        for _ in range(k):
            rn = np.linalg.norm(R, axis=0)
            valid = ok & (rn / base > _MIN_NOVELTY)
            valid[chosen] = False
            if not valid.any():
                break
            score = np.linalg.norm(R.T @ D, axis=1) / (rn * Dn + 1e-30)
            if self.audition == "random":
                c = int(rng.choice(np.flatnonzero(valid)))
            else:
                score_v = np.where(valid, score, -np.inf)
                c = int(np.argmax(score_v))
                if score[c] < 1e-8:
                    break
            chosen.append(c)
            chosen_scores.append(float(score[c]))
            q = R[:, c] / rn[c]
            R = R - np.outer(q, q @ R)
            self.flops_ += 6.0 * n * Hc.shape[1]
        order = np.argsort(chosen_scores)[::-1]
        chosen = np.array(chosen, dtype=int)[order]
        return Wc[:, chosen], bc[chosen], np.array(chosen_scores)[order]

    # ------------------------------------------------------- DEATH and FUSION
    def _prune_and_fuse(self, step, acts):
        for h in range(self.hidden_layers):
            H = acts[h + 1]
            old = (step - self._born[h]) >= self.min_age
            self.flops_ = getattr(self, "flops_", 0.0) + 2.0 * len(H) * H.shape[1] ** 2

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
            if self.widths_[h] > 1 and self.prune_utility > 0:
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
