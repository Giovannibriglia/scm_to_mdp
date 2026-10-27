"""The conditional independence tests (citests.py) and their use in PCMCI+: calibrated with sparse
strata, where the chi-square p-value of G^2 finds dependences that are not there."""

import importlib.util
import sys
import types

import numpy as np
import pytest

from scm_to_mdp.citests import enough_rows, g2_permutation_pvalue, g2_pvalue, stratified_g2
from scm_to_mdp.data import read_dataset
from scm_to_mdp.discovery import pcmci_plus
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


def test_the_fast_shuffle_gives_the_reference_p_value():
    """Both shuffles sort the rows by (stratum, the same noise): the same permutations, the same p-value."""
    rng = np.random.default_rng(4)
    for dependent in (True, False):
        x, y, z = sparse(rng, dependent, n=1000)
        fast = g2_permutation_pvalue(x, y, z, n_perm=199, seed=5)
        assert fast == g2_permutation_pvalue(x, y, z, n_perm=199, seed=5, shuffle="lexsort")
    with pytest.raises(ValueError, match="shuffle"):
        g2_permutation_pvalue(x, y, z, shuffle="bogus")


@pytest.fixture
def without_numba(monkeypatch):
    """Tigramite's CMIsymb imports numba only for @jit(forceobj=True), i.e. object mode, plain Python:
    when numba is not installed a pass-through jit stands in (the test is the same)."""
    if importlib.util.find_spec("numba") is None:
        stub = types.ModuleType("numba")
        stub.jit = lambda *args, **kwargs: args[0] if args and callable(args[0]) else (lambda f: f)
        monkeypatch.setitem(sys.modules, "numba", stub)
        monkeypatch.delitem(sys.modules, "tigramite.independence_tests.cmisymb", raising=False)


def test_our_permutation_test_is_tigramites_local_permutation_test(without_numba):
    """Cross-check with Tigramite's CMIsymb (CMI = G^2 / 2n, x permuted within the strata of z): the same
    p-values up to Monte Carlo error (999 permutations: standard error <= 0.016 each) on sparse tables."""
    from tigramite.independence_tests.cmisymb import CMIsymb

    rng = np.random.default_rng(6)
    for dependent in (True, False, False):
        x, y, z = sparse(rng, dependent, rows_per_stratum=8, n=400)
        ours = g2_permutation_pvalue(x, y, z, n_perm=999, enough=1000)
        test = CMIsymb(significance="shuffle_test", sig_samples=999, seed=1)
        array, xyz = np.vstack([x, y, z]), np.array([0, 1, 2])
        theirs = test.get_shuffle_significance(array, xyz, test.get_dependence_measure(array, xyz))
        assert abs(ours - theirs) < 0.08, (dependent, ours, theirs)


def test_pcmci_plus_runs_with_every_ci_test(env, without_numba):
    """400 rows (CMIsymb's permutations are quadratic in the rows: ~20 s here). 199 permutations: the
    smallest permutation p-value, 1/200, is below alpha = 0.01 (with 49 it is 0.02: nothing is found)."""
    data = read_dataset(env.sample(20, 20, "repair_broken"), env.action_vars, env.cards, env.reward)
    graphs = {name: pcmci_plus.fit(data, 0.01, ci_test_name=name, n_perm=199) for name in pcmci_plus.CI_TESTS}
    assert graphs["ours_fast"] == graphs["ours"]  # the same p-values
    assert all(g.directed for g in graphs.values())
    with pytest.raises(ValueError, match="ci_test"):
        pcmci_plus.ci_test("bogus")
