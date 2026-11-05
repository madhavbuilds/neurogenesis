"""
Animated explainer: one "audition", step by step, on real data.

    python examples/explain_audition.py      # -> assets/audition.gif

Each candidate neuron is a line, where its tanh fires 50/50. The animation shows
  1. the network stuck at a plateau, with the points it still gets wrong glowing,
  2. hundreds of candidate lines proposed through the data,
  3. every candidate scored by how fast it would cut the error,
  4. the winners hired with zero outgoing weights, so the output is unchanged,
  5. the network after training on with its new neurons.
Everything shown is computed by the library itself. Nothing is mocked.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from neurogenesis import Neurogenesis  # noqa: E402
from neurogenesis import datasets as D  # noqa: E402
from neurogenesis.viz import ADULT, BG, CLASS_COLORS, DIM, FG, NEWBORN  # noqa: E402

SEED, GROWTH_ROUND = 0, 1
X, y = D.moons()
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "audition.gif")

# 1. Find when the chosen growth round happens, then stop training just before it.
full = Neurogenesis(random_state=SEED).fit(X, y)
growth_steps = sorted({e.step for e in full.events_ if e.kind == "birth"})
stop = growth_steps[min(GROWTH_ROUND, len(growth_steps)) - 1] - 1
model = Neurogenesis(random_state=SEED, max_steps=stop).fit(X, y)

Xs = model._prep(X)
Y = np.eye(len(model.classes_))[np.searchsorted(model.classes_, y)]


def state():
    acts, out = model._forward(Xs)
    _, dout = model._loss(out, Y)
    _, _, deltas = model._backward(acts, dout)
    return acts, out, deltas


acts, out, deltas = state()
err = np.linalg.norm(deltas[-1] * len(X), axis=1)  # per-point error |p - y|

# 2. The full audition pool, then the real growth (same seed, so the same pool).
Wc, bc, scores, Hc, Hn = model._candidate_pool(acts[0], deltas[1], np.random.default_rng(42))
n_before = model.n_neurons_
proba_before = model.predict_proba(X)
model._grow(stop, acts, deltas, np.random.default_rng(42))
assert np.allclose(model.predict_proba(X), proba_before), "birth must not change the output"
W_new = model._W[0].val[:, n_before:]
winners = [int(np.argmin(np.abs(Wc - w[:, None]).sum(0))) for w in W_new.T]


# 3. Keep training the grown network (the same Adam loop that fit() uses).
def train(steps, rng=np.random.default_rng(1)):
    for _ in range(steps):
        idx = rng.integers(0, len(X), 128)
        a, o = model._forward(Xs[idx])
        _, d = model._loss(o, Y[idx])
        gW, gb, _ = model._backward(a, d)
        for i in range(len(model._W)):
            model._W[i].adam(gW[i], model.learning_rate, model.weight_decay)
            model._b[i].adam(gb[i], model.learning_rate)


# ------------------------------------------------------------------ drawing helpers
pad = 0.25
x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
G = np.c_[gx.ravel(), gy.ravel()]


def regions(ax):
    P = model.predict_proba(G)
    img = (P @ CLASS_COLORS[: P.shape[1]]).reshape(200, 200, 3)
    img = 0.10 + 0.40 * img * (P.max(1).reshape(200, 200, 1) ** 2)
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="lower", aspect="auto", interpolation="bilinear")


def line(w, b):
    """Return the endpoints of the line where a neuron is 50/50, in original coordinates."""
    a = w / model._x_sd
    c = b - (w * model._x_mu / model._x_sd).sum()
    if abs(a[1]) > abs(a[0]):
        xs = np.array([x0, x1])
        return xs, -(a[0] * xs + c) / a[1]
    ys = np.array([y0, y1])
    return -(a[1] * ys + c) / a[0], ys


def points(ax, glow=None):
    ax.scatter(X[:, 0], X[:, 1], s=10, c=CLASS_COLORS[y], edgecolors=BG, linewidths=0.3, zorder=3)
    if glow is not None:
        bad = glow > 0.5
        ax.scatter(X[bad, 0], X[bad, 1], s=90, facecolors="none", edgecolors=NEWBORN, linewidths=1.2, zorder=4)


def existing(ax, upto=None, color=FG, lw=1.4, alpha=0.9):
    W, b = model._W[0].val, model._b[0].val
    for j in range(W.shape[1] if upto is None else upto):
        ax.plot(*line(W[:, j], b[j]), c=color, lw=lw, alpha=alpha, zorder=2)


_vir = plt.get_cmap("viridis")


def cmap(v):  # purple -> teal; yellow is reserved for "hired"
    return _vir(0.05 + 0.65 * v)
s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
order = np.random.default_rng(0).permutation(len(scores))
SHOW = 220  # candidate lines drawn in total (of the full pool), so the frame stays readable
show_idx = np.r_[winners, np.argsort(-scores)[: SHOW // 2], order[: SHOW // 2]]
show_idx = np.array(list(dict.fromkeys(show_idx.tolist())))


def frame(step_no, title, subtitle, draw, bars=None):
    fig = plt.figure(figsize=(12, 5.4), dpi=90, facecolor=BG)
    ax = fig.add_axes([0.02, 0.04, 0.56, 0.78])
    ax.set_facecolor(BG)
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(DIM)
    draw(ax)
    fig.text(0.02, 0.92, f"Step {step_no}", color=NEWBORN, fontsize=13, weight="bold")
    fig.text(0.10, 0.92, title, color=FG, fontsize=15, weight="bold")
    fig.text(0.02, 0.86, subtitle, color=DIM, fontsize=10.5)
    side = fig.add_axes([0.62, 0.10, 0.35, 0.68])
    side.set_facecolor(BG)
    for s in side.spines.values():
        s.set_color(DIM)
    side.tick_params(colors=DIM, labelsize=8)
    if bars is not None:
        bars(side)
    else:
        side.axis("off")
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return im


def score_bars(highlight_winners, upto=None):
    def draw(ax):
        top = np.argsort(-scores)[:40]
        if upto is not None:
            top = top[:upto]
        cols = [NEWBORN if (highlight_winners and t in winners) else cmap(0.15 + 0.8 * s_norm[t]) for t in top]
        ax.barh(np.arange(len(top))[::-1], scores[top], color=cols, height=0.8)
        ax.set_yticks([])
        ax.set_xlim(0, scores.max() * 1.1)
        ax.set_ylim(-1, 40)
        ax.set_title("top 40 candidates · score = how fast it would cut the error", color=FG, fontsize=9, loc="left")
        if highlight_winners:
            ax.set_title("top 40 candidates · yellow = hired", color=NEWBORN, fontsize=9, loc="left")
    return draw


frames, holds = [], []


def add(im, hold=0.12):
    frames.append(im)
    holds.append(hold)


neurons_now = n_before
acc0 = (out.argmax(1) == Y.argmax(1)).mean()
add(frame(1, f"The network is stuck at {neurons_now} neuron{'s' if neurons_now > 1 else ''} ({acc0:.1%} training accuracy)",
          "Loss has plateaued. Rings mark points it still gets wrong. White lines are its current neurons.",
          lambda ax: (regions(ax), existing(ax, upto=n_before), points(ax, err))), 3.0)

k_steps = 14
for i in range(1, k_steps + 1):
    sub = show_idx[: int(len(show_idx) * i / k_steps)]

    def draw(ax, sub=sub):
        regions(ax)
        for c in sub:
            ax.plot(*line(Wc[:, c], bc[c]), c=cmap(0.15 + 0.8 * s_norm[c]), lw=0.8, alpha=0.35, zorder=1)
        existing(ax, upto=n_before)
        points(ax, err)

    add(frame(2, f"{len(scores)} candidate neurons audition",
              "Each line is one candidate. Most pass through a real data point. (Showing a sample.)", draw),
        0.12 if i < k_steps else 1.5)

for i in range(1, 9):
    def draw(ax, i=i):
        regions(ax)
        for c in show_idx:
            ax.plot(*line(Wc[:, c], bc[c]), c=cmap(0.15 + 0.8 * s_norm[c]), lw=0.5 + 2.5 * s_norm[c] ** 3,
                    alpha=0.15 + 0.75 * s_norm[c] ** 2, zorder=1)
        existing(ax, upto=n_before)
        points(ax, err)
    add(frame(3, "Each candidate is scored",
              "Score = the gradient its outgoing weights would get = how fast it would reduce the error. Bright = high.",
              draw, score_bars(False, upto=5 * i)), 0.15 if i < 8 else 2.5)


def hired(ax):
    regions(ax)
    for c in show_idx:
        ax.plot(*line(Wc[:, c], bc[c]), c=DIM, lw=0.5, alpha=0.12, zorder=1)
    existing(ax, upto=n_before)
    for c in winners:
        ax.plot(*line(Wc[:, c], bc[c]), c=NEWBORN, lw=3.2, zorder=5)
    points(ax, err)


add(frame(4, f"The best {len(winners)} are hired, with outgoing weights set to zero",
          "Redundant look-alikes are skipped. Zero outgoing weights mean the output is unchanged (checked to 1e-12).",
          hired, score_bars(True)), 3.2)

for t in range(1, 9):
    train(150)
    acts, out, deltas = state()
    err_t = np.linalg.norm(deltas[-1] * len(X), axis=1)
    acc = (out.argmax(1) == Y.argmax(1)).mean()

    def draw(ax, err_t=err_t):
        regions(ax)
        existing(ax, upto=n_before)
        W, b = model._W[0].val, model._b[0].val
        for j in range(n_before, W.shape[1]):
            ax.plot(*line(W[:, j], b[j]), c=NEWBORN, lw=2.5, zorder=5)
        points(ax, err_t)

    add(frame(5, "The new neurons learn",
              f"Training continues ({t * 150} more steps). Training accuracy now {acc:.1%}. Then: repeat until growth stops helping.",
              draw), 0.25 if t < 8 else 3.5)

pal = frames[-1].quantize(colors=160, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
q[0].save(OUT, save_all=True, append_images=q[1:], duration=[int(h * 1000) for h in holds], loop=0, optimize=True)
print(f"saved {os.path.abspath(OUT)} ({len(frames)} frames, winners={winners}, neurons {n_before}->{model.n_neurons_})")
