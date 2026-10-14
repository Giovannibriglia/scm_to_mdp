"""Summary statistics over seeds: means with percentile-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values, level: float = 0.95, n_boot: int = 10_000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean and a percentile-bootstrap ``level`` interval of the mean of ``values``.

    A run with an infinite value (e.g. a row the optimal policy uses but the data never visit) makes
    the mean and the interval infinite. With fewer than two values the interval is the value itself.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values")
    if np.isnan(x).any():
        raise ValueError("NaN in values")
    if np.isinf(x).any():
        inf = float(np.inf) if np.any(x == np.inf) else float(-np.inf)
        return inf, inf, inf
    mean = float(np.mean(x))
    if x.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return mean, float(lo), float(hi)
