import numpy as np

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import IntervalMDP
from scm_to_mdp.discovery import oracle
from scm_to_mdp.evaluation import evaluate, optimal_values, random_starts
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given
from scm_to_mdp.solving import robust_vi, value_iteration, vertex_robust_vi

GAMMA = 0.9


def learned(env, dataset):
    graph = given.select(oracle.fit(dataset, env.true_parents()), dataset)
    bounds = nbn_intervals.learn(dataset, graph, delta=0.05)
    return bounds, product_bounds.compile(bounds)


def test_intervals_contain_the_true_kernel(env, truth, dataset):
    _, mdp = learned(env, dataset)
    P = truth.P[:, truth.actions_of(dataset)]  # true kernel over the learner's actions [S, K, S']
    assert np.all(mdp.L <= P + 1e-12) and np.all(P <= mdp.U + 1e-12)


def test_robust_vi_without_uncertainty_is_vi(truth, dataset):
    P = truth.P[:, truth.actions_of(dataset)]
    exact = value_iteration.solve(P, dataset.reward, GAMMA, tol=1e-10)
    robust = robust_vi.solve(IntervalMDP(P, P, dataset.reward), GAMMA, tol=1e-10)
    assert np.allclose(exact.V, robust.V, atol=1e-6)


def test_promises_are_ordered_and_valid(env, truth, dataset):
    bounds, mdp = learned(env, dataset)
    ia = robust_vi.solve(mdp, GAMMA)
    exact = vertex_robust_vi.solve(bounds, GAMMA)
    assert np.all(ia.V <= exact.V + 1e-6)  # the relaxation is more pessimistic
    for solution in (ia, exact):
        assert evaluate(
            truth, dataset, solution, GAMMA, optimal_values(truth, GAMMA), random_starts(truth, 20, 0)
        )["safe"]


def test_value_iteration_stops_within_tol_of_the_fixed_point(truth):
    """tol bounds the distance to the fixed point, also for a discount close to 1."""
    for gamma in (0.9, 0.99):
        solution = value_iteration.solve(truth.P, truth.reward, gamma, tol=1e-6)
        exact = value_iteration.policy_value(truth.P, truth.reward, solution.pi, gamma)
        assert np.abs(solution.V - exact).max() < 1e-6, gamma


def test_robust_promise_is_a_lower_bound_even_with_a_loose_tolerance():
    """One state, reward -1 forever: value iteration from 0 falls towards -1/(1-gamma), so a stopped
    iterate is ABOVE the true value. The returned promise must still be below it, at any tolerance."""
    from scm_to_mdp.core import Reward

    P = np.ones((1, 1, 1))
    reward = Reward.immediate(np.array([[-1.0]]))
    for gamma in (0.9, 0.99):
        for tol in (1e-1, 1e-3, 1e-7):
            solution = robust_vi.solve(IntervalMDP(P, P, reward), gamma, tol=tol)
            true_value = -1 / (1 - gamma)
            assert solution.V[0] <= true_value + 1e-12, (gamma, tol)
            assert solution.V[0] >= true_value - tol - 1e-9, (gamma, tol)  # and still within tol
