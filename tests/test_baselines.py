import numpy as np

from scm_to_mdp.baselines import difarl, flat_mdp, flat_rmdp
from scm_to_mdp.evaluation import evaluate, optimal_values, random_starts
from scm_to_mdp.solving import robust_vi

GAMMA = 0.9


def test_flat_mdp_rows_are_distributions(dataset):
    mdp = flat_mdp.learn(dataset)
    assert np.allclose(mdp.L.sum(-1), 1.0)


def test_flat_rmdp_is_valid(truth, dataset):
    mdp = flat_rmdp.learn(dataset, delta=0.05)
    assert np.all(mdp.L <= mdp.U)
    assert evaluate(
        truth,
        dataset,
        robust_vi.solve(mdp, GAMMA),
        GAMMA,
        optimal_values(truth, GAMMA),
        random_starts(truth, 20, 0),
    )["safe"]


def test_difarl_groups_partition_the_actions(dataset):
    groups = difarl.groups(dataset, 0.005)
    assert sorted(i for g in groups for i in g) == list(range(len(dataset.action_vars)))


def test_difarl_fqi_is_finite(dataset):
    solution = difarl.fqi(dataset, difarl.groups(dataset, 0.005), GAMMA)
    assert np.isfinite(solution.V).all() and solution.V.max() < 3 / (1 - GAMMA)


def test_g2_detects_dependence():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 3000)
    strata = np.zeros(3000, dtype=np.int64)
    assert difarl.g2_pvalue(x, x, strata) < 1e-6
    assert difarl.g2_pvalue(x, rng.integers(0, 2, 3000), strata) > 0.01
