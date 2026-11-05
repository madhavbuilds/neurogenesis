"""
Watch a neural network grow its own brain.

    python examples/grow.py                # spiral -> assets/spiral.gif
    python examples/grow.py checkerboard
    python examples/grow.py wavy           # 1-D regression
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neurogenesis import Neurogenesis  # noqa: E402
from neurogenesis import datasets as D  # noqa: E402
from neurogenesis.viz import grow_and_render  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "spiral"
make = {**D.ALL, "wavy": D.wavy}[name]
X, y = make()
model = Neurogenesis(random_state=0, verbose=True)
out = os.path.join(os.path.dirname(__file__), "..", "assets", f"{name}.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)
grow_and_render(model, X, y, out, every=2, title=f"Neurogenesis · {name}")
print()
print(model.life_story().splitlines()[0])
print(f"train score: {model.score(X, y):.3f}")
print(f"saved {os.path.abspath(out)}")
