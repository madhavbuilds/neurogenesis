"""Watch a network grow. Records snapshots during ``fit`` and renders them to a GIF."""
from __future__ import annotations

import numpy as np

BG, FG, DIM = "#0b0f19", "#e6edf3", "#5b6b82"
CLASS_COLORS = np.array([[0.30, 0.65, 1.00], [1.00, 0.42, 0.42], [1.00, 0.82, 0.40], [0.40, 0.90, 0.60], [0.75, 0.55, 1.00]])
NEWBORN, ADULT = "#ffd166", "#06d6a0"


class Recorder:
    """Pass ``recorder`` as ``callback`` to ``Neurogenesis.fit``."""

    def __init__(self, every: int = 1):
        self.every, self.frames, self._n = every, [], 0

    def __call__(self, model, step, loss):
        self._n += 1
        if self._n % self.every:
            return
        snap = model.snapshot()
        snap.update(
            step=step,
            hist={k: list(v) for k, v in model.history_.items()},
            events=list(model.events_),
            task=model.task_,
            mature=model.mature_,
        )
        self.frames.append(snap)


def _forward(snap, Xs):
    a = Xs
    for W, b in zip(snap["W"][:-1], snap["b"][:-1]):
        a = np.tanh(a @ W + b)
    out = a @ snap["W"][-1] + snap["b"][-1]
    if snap["task"] == "classification":
        e = np.exp(out - out.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return out


def _draw_network(ax, snap):
    from matplotlib.collections import LineCollection

    Ws = snap["W"]
    sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
    xs = np.linspace(0, 1, len(sizes))
    pos = []
    for x, n in zip(xs, sizes):
        span = min(1.0, 0.06 * n)
        pos.append(np.c_[np.full(n, x), np.linspace(-span, span, n) if n > 1 else [0.0]])
    segs, cols = [], []
    for i, W in enumerate(Ws):
        m = np.abs(W).max() + 1e-12
        for a in range(W.shape[0]):
            for b in range(W.shape[1]):
                w = W[a, b]
                alpha = min(1.0, abs(w) / m) ** 0.8
                if alpha < 0.04:
                    continue
                segs.append([pos[i][a], pos[i + 1][b]])
                cols.append((0.35, 0.65, 1.0, 0.85 * alpha) if w > 0 else (1.0, 0.42, 0.42, 0.85 * alpha))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8))
    ax.scatter(*pos[0].T, s=60, c=DIM, zorder=3)
    ax.scatter(*pos[-1].T, s=60, c=FG, zorder=3)
    for h, born in enumerate(snap["born"]):
        age = snap["step"] - born
        young = age < 800
        p = pos[h + 1]
        if young.any():
            ax.scatter(*p[young].T, s=260, c=NEWBORN, alpha=0.25, zorder=2, linewidths=0)
        ax.scatter(*p.T, s=np.where(young, 70, 45), c=np.where(young, NEWBORN, ADULT), zorder=3, linewidths=0)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")


def _draw_task(ax, snap, X, y, grid):
    if X.shape[1] == 1:
        xs = np.linspace(X.min(), X.max(), 400)[:, None]
        out = _forward(snap, (xs - snap["x_mu"]) / snap["x_sd"])
        ax.scatter(X[:, 0], y, s=6, c=DIM, alpha=0.6, linewidths=0)
        mu, sd = snap.get("y_mu", 0), snap.get("y_sd", 1)
        ax.plot(xs[:, 0], out[:, 0] * sd + mu, c=NEWBORN, lw=2.5)
        ax.set_xticks([]), ax.set_yticks([])
        return
    xx, yy, G = grid
    P = _forward(snap, (G - snap["x_mu"]) / snap["x_sd"])
    img = (P @ CLASS_COLORS[: P.shape[1]]).reshape(len(yy), len(xx), 3)
    img = 0.12 + 0.55 * img * (P.max(1).reshape(len(yy), len(xx), 1) ** 2)
    ax.imshow(img, extent=(xx[0], xx[-1], yy[0], yy[-1]), origin="lower", aspect="auto", interpolation="bilinear")
    ax.scatter(X[:, 0], X[:, 1], s=7, c=CLASS_COLORS[y % len(CLASS_COLORS)], edgecolors=BG, linewidths=0.3)
    ax.set_xticks([]), ax.set_yticks([])


def render_gif(frames, X, y, path, fps=14, hold=2.5, dpi=100, title="Neurogenesis"):
    """Render recorded frames to ``path`` (.gif). Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from PIL import Image

    X, y = np.asarray(X, float), np.asarray(y)
    grid = None
    if X.shape[1] == 2:
        pad = 0.08 * (X.max(0) - X.min(0))
        xx = np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], 160)
        yy = np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], 160)
        grid = (xx, yy, np.c_[np.repeat(xx[None], len(yy), 0).ravel(), np.repeat(yy, len(xx))])
        _, y = np.unique(y, return_inverse=True)

    last = frames[-1]
    max_neurons = max(max(f["hist"]["neurons"]) for f in frames)
    max_step = last["step"]
    images = []
    for f in frames:
        fig = plt.figure(figsize=(12, 4.6), dpi=dpi, facecolor=BG)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1.05], height_ratios=[1, 1], wspace=0.08, hspace=0.35,
                              left=0.02, right=0.96, top=0.84, bottom=0.08)
        a_net, a_task = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
        a_n, a_l = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
        for a in (a_net, a_task, a_n, a_l):
            a.set_facecolor(BG)
            for s in a.spines.values():
                s.set_color(DIM)
            a.tick_params(colors=DIM, labelsize=8)

        _draw_network(a_net, f)
        _draw_task(a_task, f, X, y, grid)

        h = f["hist"]
        a_n.step(h["step"], h["neurons"], where="post", c=ADULT, lw=2)
        a_n.set_xlim(0, max_step), a_n.set_ylim(0, max_neurons * 1.15)
        a_n.set_title("neurons (self-chosen)", color=FG, fontsize=9, loc="left")
        a_l.plot(h["step"], h["loss"], c=NEWBORN, lw=1.8)
        a_l.set_yscale("log"), a_l.set_xlim(0, max_step)
        a_l.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for a in (a_n, a_l):
            a.yaxis.tick_right()
        a_l.set_title("loss", color=FG, fontsize=9, loc="left")
        for e in f["events"]:
            if e.kind == "mature":
                for a in (a_n, a_l):
                    a.axvline(e.step, c=DIM, ls="--", lw=1)

        ev = f["events"]
        births = sum(e.kind == "birth" for e in ev)
        deaths = sum(e.kind == "death" for e in ev)
        fusions = sum(e.kind == "fusion" for e in ev)
        n = sum(W.shape[1] for W in f["W"][:-1])
        state = "MATURE · fine-tuning" if f["mature"] else "GROWING"
        fig.text(0.02, 0.93, title, color=FG, fontsize=17, weight="bold")
        fig.text(0.02, 0.875, "starts with 1 neuron · grows where the gradient asks · prunes what it doesn't need",
                 color=DIM, fontsize=9.5)
        fig.text(0.98, 0.93, f"step {f['step']:,}   ·   {n} neurons   ·   {state}", color=FG, fontsize=11, ha="right")
        fig.text(0.98, 0.875, f"born {births}   died {deaths}   fused {fusions}", color=DIM, fontsize=9.5, ha="right")

        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    pal = images[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    dur = [int(1000 / fps)] * len(q)
    dur[-1] = int(hold * 1000)
    q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True)
    return path


def grow_and_render(model, X, y, path, every=1, **kw):
    """Convenience: fit ``model`` while recording, then render the GIF."""
    rec = Recorder(every)
    model.fit(X, y, callback=rec)
    if model.task_ == "regression":
        for fr in rec.frames:
            fr["y_mu"], fr["y_sd"] = model._y_mu[0], model._y_sd[0]
    return render_gif(rec.frames, X, y, path, **kw)
