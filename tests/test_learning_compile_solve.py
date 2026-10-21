import numpy as np

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import IntervalMDP
from scm_to_mdp.discovery import oracle
from scm_to_mdp.evaluation import evaluate, optimal_values, random_starts
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.solving import robust_vi, value_iteration, vertex_robust_vi

GAMMA = 0.9


def learned(env, dataset):
    bounds = nbn_intervals.learn(dataset, oracle.fit(dataset, env.true_parents(dataset)), delta=0.05)
    return bounds, product_bounds.compile(bounds)


def test_intervals_contain_the_true_kernel(env, dataset):
    _, mdp = learned(env, dataset)
    P = env.P[:, env.env_actions(dataset)]  # true kernel over the learner's actions [S, K, S']
    assert np.all(mdp.L <= P + 1e-12) and np.all(P <= mdp.U + 1e-12)


def test_robust_vi_without_uncertainty_is_vi(env, dataset):
    P = env.P[:, env.env_actions(dataset)]
    exact = value_iteration.solve(P, dataset.reward, GAMMA, tol=1e-10)
    robust = robust_vi.solve(IntervalMDP(P, P, dataset.reward), GAMMA, tol=1e-10)
    assert np.allclose(exact.V, robust.V, atol=1e-6)


def test_promises_are_ordered_and_valid(env, dataset):
    bounds, mdp = learned(env, dataset)
    ia = robust_vi.solve(mdp, GAMMA)
    exact = vertex_robust_vi.solve(bounds, GAMMA)
    assert np.all(ia.V <= exact.V + 1e-6)  # the relaxation is more pessimistic
    for solution in (ia, exact):
        assert evaluate(env, dataset, solution, GAMMA, optimal_values(env, GAMMA), random_starts(env, 20, 0))[
            "safe"
        ]
