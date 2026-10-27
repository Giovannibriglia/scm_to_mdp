"""Conditional independence tests for discrete data (steps S2 and the baselines' structure learning).

The G^2 statistic of x independent of y given discrete strata z, compared with its chi-square
distribution, is badly anti-conservative when the strata are sparse: with x, y independent by
construction it rejects at alpha = 0.01 in 1% of the runs at 200 rows per stratum but in 100% at 20 rows
per stratum (measured 2026-10-27). Spurious dependences then add parents (PCMCI+, FOCUS) or merge
action groups (DiFaRL). Two remedies:

    g2_permutation_pvalue   the p-value of the G^2 statistic under permutations of y within each
                            stratum (exact at any sample size; conditional permutation test). It stops
                            early by Besag & Clifford's sequential rule (Biometrika 78(2), 1991): after
                            `enough` permutations at least as extreme as the data, p = enough / done, so
                            a clearly independent pair costs a few dozen permutations, a dependent one
                            at most n_perm (smallest p-value 1 / (n_perm + 1))
    enough_rows             whether the chi-square p-value can be trusted: the usual rule of thumb
                            that the expected count of a cell should be at least about 5 (Cochran,
                            Biometrics 10(4), 1954), applied to the mean count over the cells of the
                            tables actually used (the values seen in each stratum), with 10 rather than
                            5 (ROWS_PER_CELL, measured). The total number
                            of rows per degree of freedom is not enough: 200 strata of 20 rows pass
                            "5 rows per dof" with 3 x 3 tables, and the test still rejects always
"""

import numpy as np

ROWS_PER_CELL = 10  # Cochran's rule of thumb is 5; across many strata we measured 8% false rejections
# at alpha = 1% with 5.6 rows per cell (3 x 3 tables), and 1% at 22


def stratified_g2(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> tuple[float, int, int]:
    """(G^2, degrees of freedom, cells) of the stratified contingency tables of x and y (integer
    codes); the degrees of freedom and the cells count only the values seen in each stratum."""
    _, z = np.unique(strata, return_inverse=True)
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    cx, cy, nz = int(x.max()) + 1, int(y.max()) + 1, int(z.max()) + 1
    table = np.bincount((z * cx + x) * cy + y, minlength=nz * cx * cy).reshape(nz, cx, cy).astype(float)
    return _g2(table), _dof(table), _cells(table)


def _g2(table: np.ndarray) -> float:
    total = table.sum((1, 2), keepdims=True)
    expected = table.sum(2, keepdims=True) * table.sum(1, keepdims=True) / np.maximum(total, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(2 * np.where(table > 0, table * np.log(table / expected), 0.0).sum())


def _dof(table: np.ndarray) -> int:
    rows_used = (table.sum(2) > 0).sum(1)
    cols_used = (table.sum(1) > 0).sum(1)
    return int(((rows_used - 1) * (cols_used - 1)).clip(min=0).sum())


def _cells(table: np.ndarray) -> int:
    """The cells of the tables used: (values of x seen) x (values of y seen), summed over the strata."""
    return int(((table.sum(2) > 0).sum(1) * (table.sum(1) > 0).sum(1)).sum())


def enough_rows(n_rows: int, cells: int, rows_per_cell: float = ROWS_PER_CELL) -> bool:
    """Whether the chi-square p-value of a test is trusted (see the module docstring)."""
    return cells > 0 and n_rows >= rows_per_cell * cells


def g2_permutation_pvalue(
    x: np.ndarray, y: np.ndarray, strata: np.ndarray, n_perm: int = 1999, enough: int = 10, seed: int = 0
) -> float:
    """p-value of x independent of y given the strata, from permutations of y within each stratum (see
    the module docstring). 1.0 when no stratum has two different values of both x and y."""
    _, z = np.unique(strata, return_inverse=True)
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    cx, cy, nz = int(x.max()) + 1, int(y.max()) + 1, int(z.max()) + 1

    def statistic(y_codes):
        table = np.bincount((z * cx + x) * cy + y_codes, minlength=nz * cx * cy).reshape(nz, cx, cy)
        return _g2(table.astype(float))

    observed = statistic(y)
    if observed <= 1e-12:
        return 1.0
    rng = np.random.default_rng(seed)
    by_stratum = np.argsort(z, kind="stable")  # rows grouped by stratum
    extreme = 0
    for done in range(1, n_perm + 1):
        order = by_stratum[np.lexsort((rng.random(len(z)), z[by_stratum]))]  # shuffled within strata
        shuffled = np.empty_like(y)
        shuffled[by_stratum] = y[order]
        extreme += statistic(shuffled) >= observed - 1e-9
        if extreme >= enough:  # sequential stop: clearly not significant
            return extreme / done
    return (extreme + 1) / (n_perm + 1)


def g2_pvalue(x: np.ndarray, y: np.ndarray, strata: np.ndarray, n_perm: int = 1999, seed: int = 0) -> float:
    """p-value of x independent of y given the strata: the chi-square p-value of G^2 where it is reliable
    (enough_rows), the permutation p-value elsewhere (sparse strata). 1.0 without degrees of freedom."""
    from scipy.stats import chi2

    g, dof, cells = stratified_g2(x, y, strata)
    if dof == 0:
        return 1.0
    if enough_rows(len(x), cells):
        return float(chi2.sf(g, dof))
    return g2_permutation_pvalue(x, y, strata, n_perm=n_perm, seed=seed)
