# Benchmark results

Seeds: [0, 1, 2]. Score = test accuracy (classification) or R² (regression). One hidden layer, tanh, Adam, identical training code (incl. early stopping) for all models.

| dataset | Neurogenesis score | neurons (self-chosen) | steps | best fixed MLP | smallest fixed MLP within 1 pt of Neurogenesis |
|---|---|---|---|---|---|
| spiral | **0.982** ± 0.003 | **34.3** (18–44) | 21500 | 0.989 (w=64) | w=16 (0.976) |
| moons | **0.992** ± 0.006 | **6.3** (3–10) | 8100 | 0.996 (w=64) | w=4 (0.992) |
| circles | **0.999** ± 0.002 | **3.3** (3–4) | 9533 | 0.997 (w=4) | w=4 (0.997) |
| checkerboard | **0.970** ± 0.004 | **42.0** (35–48) | 17933 | 0.976 (w=64) | w=16 (0.967) |
| wavy (regression) | **0.990** ± 0.001 | **10.3** (10–11) | 5533 | 0.989 (w=32) | w=4 (0.983) |

### Fixed-width sweep (mean test score over seeds; up to 30k steps each, early-stopped on validation)

| dataset | w=2 | w=4 | w=8 | w=16 | w=32 | w=64 | w=128 |
|---|---|---|---|---|---|---|---|
| spiral | 0.489 | 0.639 | 0.817 | 0.976 | 0.982 | 0.989 | 0.984 |
| moons | 0.893 | 0.992 | 0.988 | 0.992 | 0.993 | 0.996 | 0.995 |
| circles | 0.761 | 0.997 | 0.997 | 0.996 | 0.996 | 0.996 | 0.996 |
| checkerboard | 0.583 | 0.694 | 0.830 | 0.967 | 0.963 | 0.976 | 0.975 |
| wavy (regression) | 0.812 | 0.983 | 0.989 | 0.989 | 0.989 | 0.986 | 0.986 |
