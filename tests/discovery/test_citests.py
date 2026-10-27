"""The conditional independence tests (citests.py) and their use in PCMCI+: calibrated with sparse
strata, where the chi-square p-value of G^2 finds dependences that are not there."""

import numpy as np

from scm_to_mdp.citests import enough_rows, g2_permutation_pvalue, g2_pvalue, stratified_g2
from scm_to_mdp.discovery.pcmci_plus import FastGsquared


def sparse(rng, dependent: bool, rows_per_stratum: int = 20, n: int = 4000):
    z = rng.integers(0, n // rows_per_stratum, n)
    x = rng.integers(0, 3, n)
    y = np.where(rng.random(n) < 0.3, x, rng.integers(0, 3, n)) if dependent else rng.integers(0, 3, n)
    return x, y, z


def test_sparse_strata_are_calibrated_and_keep_their_power():
    rng = np.random.default_rng(0)
    false = sum(g2_pvalue(*sparse(rng, False)) < 0.01 for _ in range(60))
    found = sum(g2_pvalue(*sparse(rng, True)) < 0.01 for _ in range(20))
    assert false <= 4  # nominal 1% of 60 runs (the chi-square p-value: all 60)
    assert found == 20


def test_pcmci_plus_uses_the_permutation_p_value_on_sparse_tables():
    rng = np.random.default_rng(1)
    test = FastGsquared(significance="analytic")

    def p_value(x, y, z):
        array, xyz = np.vstack([x, y, z]).astype(float), np.array([0, 1, 2])
        value = test.get_dependence_measure(array, xyz)
        return test.get_analytic_significance(value=value, T=len(x), dim=3, xyz=xyz)

    assert sum(p_value(*sparse(rng, False)) < 0.01 for _ in range(30)) <= 3
    assert all(p_value(*sparse(rng, True)) < 0.01 for _ in range(10))


def test_dense_tables_use_the_chi_square_p_value():
    rng = np.random.default_rng(2)
    x, y, z = sparse(rng, True, rows_per_stratum=2000)
    g, dof, cells = stratified_g2(x, y, z)
    assert enough_rows(len(x), cells) and dof == 2 * 4
    from scipy.stats import chi2

    assert g2_pvalue(x, y, z) == chi2.sf(g, dof)


def test_permutation_p_value_has_its_floor_and_is_one_without_dependence():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, 400)
    assert g2_permutation_pvalue(x, x, np.zeros(400), n_perm=99) == 1 / 100  # the smallest possible
    assert g2_pvalue(x, np.zeros(400, dtype=int), np.zeros(400)) == 1.0  # y constant: no dof
