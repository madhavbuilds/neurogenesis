# Benchmark results

Seeds: [0, 1, 2]. Score = test accuracy (classification) or R² (regression). One hidden layer, tanh, Adam, identical training code for all models.

| dataset | Neurogenesis score | neurons (self-chosen) | steps | best fixed MLP | smallest fixed MLP within 1 pt of Neurogenesis |
|---|---|---|---|---|---|
| spiral | **0.981** ± 0.009 | **22.0** (19–27) | 26933 | 0.988 (w=64) | w=16 (0.978) |
| moons | **0.992** ± 0.003 | **10.7** (5–14) | 9000 | 0.991 (w=4) | w=4 (0.991) |
| circles | **0.997** ± 0.004 | **3.7** (3–5) | 9033 | 0.997 (w=4) | w=4 (0.997) |
| checkerboard | **0.972** ± 0.005 | **22.7** (20–25) | 17933 | 0.979 (w=32) | w=16 (0.974) |
| wavy (regression) | **0.990** ± 0.001 | **8.7** (6–11) | 7400 | 0.990 (w=32) | w=4 (0.987) |

### Fixed-width sweep (mean test score over seeds, 30k steps each)

| dataset | w=2 | w=4 | w=8 | w=16 | w=32 | w=64 | w=128 |
|---|---|---|---|---|---|---|---|
| spiral | 0.522 | 0.621 | 0.839 | 0.978 | 0.987 | 0.988 | 0.984 |
| moons | 0.896 | 0.991 | 0.989 | 0.989 | 0.988 | 0.985 | 0.983 |
| circles | 0.780 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| checkerboard | 0.582 | 0.722 | 0.869 | 0.974 | 0.979 | 0.977 | 0.979 |
| wavy (regression) | 0.900 | 0.987 | 0.990 | 0.988 | 0.990 | 0.989 | 0.988 |
