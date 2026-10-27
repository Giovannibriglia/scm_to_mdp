"""How the figures summarise runs over seeds (S9): the interquartile mean (IQM) with a 95% percentile
bootstrap confidence interval.

Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep Reinforcement Learning at the Edge of the
Statistical Precipice", NeurIPS 2021 (arXiv:2108.13264): the IQM (the mean of the middle 50% of the
runs) is robust to outlying seeds and less noisy than the median; the uncertainty about it is given by
a bootstrap confidence interval over the runs (stratified by task when tasks are pooled: iqm_ci with
`strata`). With 10 seeds the IQM averages the middle runs (scipy.stats.trim_mean, 25% cut each side).

Rates (a share of runs, e.g. how often a promise held) are plain means without a band: an IQM of
0/1 values would drop exactly the runs that matter.
"""

import numpy as np
from scipy.stats import trim_mean

N_BOOT = 2000
LEVEL = 0.95


def iqm(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(trim_mean(x, 0.25)) if len(x) else float("nan")


def iqm_ci(x, strata=None, n_boot: int = N_BOOT, level: float = LEVEL, seed: int = 0) -> tuple:
    """(IQM, lower, upper): the IQM of x and a percentile bootstrap interval. strata: one label per
    run (e.g. the domain); runs are resampled within each stratum (stratified bootstrap)."""
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x)
    x = x[keep]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    centre = iqm(x)
    if len(x) == 1:
        return centre, centre, centre
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    else:
        labels = np.asarray(strata)[keep]
        groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
        idx = np.concatenate([g[rng.integers(0, len(g), size=(n_boot, len(g)))] for g in groups], axis=1)
    boots = trim_mean(x[idx], 0.25, axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return centre, float(lo), float(hi)
