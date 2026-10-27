"""How the figures summarise runs over seeds (S9): the interquartile mean (IQM) with a 95% percentile
bootstrap confidence interval; the transitions needed (censored) by the median with its 95% interval.

Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep Reinforcement Learning at the Edge of the
Statistical Precipice", NeurIPS 2021 (arXiv:2108.13264): the IQM (the mean of the middle 50% of the
runs) is robust to outlying seeds and less noisy than the median; the uncertainty about it is given by
a bootstrap confidence interval over the runs (stratified by task when tasks are pooled: iqm_ci with
`strata`). With 10 seeds the IQM averages the middle runs (scipy.stats.trim_mean, 25% cut each side).

Rates (a share of runs, e.g. how often a promise held) are plain means without a band: an IQM of
0/1 values would drop exactly the runs that matter.

The transitions needed for a near-optimal policy are censored: a seed that never gets there has
t* = inf, and the IQM is inf as soon as more than a quarter of the seeds fail. They are summarised by
the median with its distribution-free 95% confidence interval from order statistics (median_ci): the
k-th smallest and k-th largest values, with k the largest such that
P(X_(k) <= median <= X_(n+1-k)) = 1 - 2 P(Bin(n, 1/2) <= k - 1) >= 95%. It only ranks the seeds, so
inf needs no special handling; with 10 seeds it is [2nd, 9th] (coverage 97.9%).
"""

import numpy as np
from scipy.stats import binom, trim_mean

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


def median_ci(x, level: float = LEVEL) -> tuple:
    """(median, lower, upper): the median of x (the lower one for an even count, so a value of x) and
    its order-statistic confidence interval (see the module docstring); inf allowed, nan dropped.
    With 5 runs or fewer no such interval reaches 95%: it is then [min, max] (its coverage is lower)."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    median = float(np.quantile(x, 0.5, method="inverted_cdf"))
    k = 1  # the 1-based rank of the lower end
    while k + 1 <= (n + 1) // 2 and 1 - 2 * binom.cdf(k, n, 0.5) >= level:
        k += 1
    return median, float(x[k - 1]), float(x[n - k])
