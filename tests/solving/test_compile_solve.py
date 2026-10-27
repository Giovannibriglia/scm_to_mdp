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


def _robust_vi_reference(L, U, reward, gamma, n_iter=400):
    """The textbook form: per (s, a), sort the next states by value and fill the room greedily."""
    V = np.zeros(L.shape[0])
    for _ in range(n_iter):
        W = reward.backup(V, gamma)
        Q = np.empty(L.shape[:2])
        for s in range(L.shape[0]):
            for k in range(L.shape[1]):
                p, free = L[s, k].copy(), 1.0 - L[s, k].sum()
                for j in np.argsort(W):
                    add = min(U[s, k, j] - L[s, k, j], free)
                    p[j] += add
                    free -= add
                Q[s, k] = reward.R[s, k] + p @ W
        V = Q.max(1)
    return V


def test_robust_vi_matches_the_textbook_form():
    from scm_to_mdp.core import Reward

    rng = np.random.default_rng(1)
    S, K = 12, 3
    P = rng.dirichlet(np.ones(S) * 0.3, size=(S, K))
    L, U = np.clip(P - 0.05, 0, 1), np.clip(P + 0.05, 0, 1)
    reward = Reward(rng.random((S, K)), rng.random(S), rng.random(S) < 0.2)
    fast = robust_vi.solve(IntervalMDP(L, U, reward), GAMMA, tol=1e-10)
    assert np.allclose(fast.V, _robust_vi_reference(L, U, reward, GAMMA), atol=1e-6)


def _optimistic_reference(mdp, gamma, tol=1e-7):
    """The optimistic value iteration as it was written before the shared loop (copies at every sweep)."""
    from scm_to_mdp.solving.stopping import stop_at

    L, U = mdp.L, mdp.U
    V = np.zeros(L.shape[0])
    while True:
        W = mdp.reward.backup(V, gamma)
        order = np.argsort(-W)
        lower, room, w = L[..., order], (U - L)[..., order], W[order]
        remaining = 1.0 - lower.sum(-1, keepdims=True)
        given_before = np.cumsum(room, -1) - room
        extra = np.clip(remaining - given_before, 0.0, room)
        V_new = (mdp.reward.R + ((lower + extra) * w).sum(-1)).max(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            return V_new + gamma / (1 - gamma) * step
        V = V_new


def _random_imdp(S, K, width, seed):
    from scm_to_mdp.core import Reward

    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 0.2, size=(S, K))
    L, U = np.clip(P - width, 0, 1), np.clip(P + width, 0, 1)
    unseen = rng.random((S, K)) < 0.2  # rows never seen: the whole simplex
    L[unseen], U[unseen] = 0.0, 1.0
    return IntervalMDP(L, U, Reward(rng.normal(size=(S, K)), rng.normal(size=S), rng.random(S) < 0.1))


def test_cooperative_nature_is_the_optimistic_mirror(env, dataset):
    import pytest

    from scm_to_mdp.evaluation import optimistic_values

    _, learned_mdp = learned(env, dataset)
    for mdp, gamma in [
        (_random_imdp(30, 3, 0.05, 0), 0.9),
        (_random_imdp(60, 4, 0.2, 1), 0.95),
        (learned_mdp, GAMMA),
    ]:
        up = robust_vi.solve(mdp, gamma, nature="cooperative")
        assert np.allclose(up.V, _optimistic_reference(mdp, gamma), atol=1e-9)  # the old loop, same values
        assert np.all(up.V >= robust_vi.solve(mdp, gamma).V - 1e-9)  # optimistic >= pessimistic
        fixed = robust_vi.solve(mdp, gamma, tol=1e-13, nature="cooperative").V  # (corrected upward)
        loose = optimistic_values(mdp, gamma)  # the certificate's tol 1e-5: still an upper bound ...
        assert np.all(loose >= fixed - 1e-11) and np.max(loose - fixed) <= 1e-5  # ... within tol
    with pytest.raises(ValueError, match="nature"):
        robust_vi.solve(learned_mdp, GAMMA, nature="neutral")


def test_a_cooperative_solver_carries_no_guarantee():
    from scm_to_mdp.tool import Method

    assert Method("flat_rmdp", "robust_vi").guarantee
    assert not Method("flat_rmdp", "robust_vi", solver_options={"nature": "cooperative"}).guarantee


def _evaluate_on_every_action(mdp, policy, gamma, nature, n_iter=600):
    """The policy evaluation before on_policy: nature solved at every (s, a), then weighted."""
    inner = robust_vi.Nature(mdp, nature)
    V = np.zeros(mdp.L.shape[0])
    for _ in range(n_iter):
        V = (policy * (mdp.reward.R + inner.expectation(mdp.reward.backup(V, gamma)))).sum(1)
    return V


def test_evaluation_on_the_policy_actions_is_exact():
    """on_policy keeps only the actions the policy plays: the same values (deterministic or stochastic
    policy, either nature), from any start, and the same one-step backup."""
    rng = np.random.default_rng(7)
    mdp = _random_imdp(40, 5, 0.1, 3)
    S, K = mdp.L.shape[:2]
    deterministic = np.eye(K)[rng.integers(0, K, S)]
    stochastic = rng.dirichlet(np.ones(K), S) * (rng.random((S, K)) < 0.5)
    stochastic[np.arange(S), rng.integers(0, K, S)] += 0.1  # every state plays at least one action
    stochastic /= stochastic.sum(1, keepdims=True)
    for policy in (deterministic, stochastic):
        restricted, probs = robust_vi.on_policy(mdp, policy)
        assert restricted.L.shape[1] == int((policy > 0).sum(1).max()) and np.allclose(probs.sum(1), 1)
        for nature in robust_vi.NATURES:
            reference = _evaluate_on_every_action(mdp, policy, GAMMA, nature)
            for V0 in (None, rng.normal(size=S) * 10):
                V = robust_vi.evaluate(mdp, policy, GAMMA, tol=1e-11, nature=nature, V0=V0)
                assert np.allclose(V, reference, atol=1e-8), (nature, V0 is None)
            W = rng.normal(size=S)
            full = (policy * robust_vi.Nature(mdp, nature).expectation(W)).sum(1)
            assert np.allclose((probs * robust_vi.Nature(restricted, nature).expectation(W)).sum(1), full)
    assert robust_vi.on_policy(mdp, deterministic)[0].L.shape == (S, 1, S)


def test_a_warm_start_gives_the_same_robust_solution():
    mdp = _random_imdp(40, 4, 0.1, 5)
    for nature in robust_vi.NATURES:
        cold = robust_vi.solve(mdp, GAMMA, tol=1e-10, nature=nature)
        warm = robust_vi.solve(mdp, GAMMA, tol=1e-10, nature=nature, V0=cold.V + 0.01)
        assert np.allclose(cold.V, warm.V, atol=1e-8) and warm.iterations < cold.iterations


def test_nature_reuses_its_fill_only_while_the_order_holds():
    """A Nature that keeps its fill gives what a fresh one gives: for W in the same order as before
    (reused), in a new order (refilled) and with ties; robust VI still matches the textbook loop."""
    rng = np.random.default_rng(11)
    mdp = _random_imdp(50, 4, 0.1, 8)
    W1 = rng.normal(size=50)
    for W2 in (2 * W1 + 1, rng.normal(size=50), np.round(W1)):  # same order, new order, ties
        for nature in robust_vi.NATURES:
            kept = robust_vi.Nature(mdp, nature)
            kept.expectation(W1)
            fresh = robust_vi.Nature(mdp, nature).expectation(W2)
            assert np.allclose(kept.expectation(W2), fresh, atol=1e-12)
    kept = robust_vi.Nature(mdp)
    kept.expectation(W1)
    kept.expectation(2 * W1 + 1)
    assert kept.reused == 1
    fast = robust_vi.solve(mdp, GAMMA, tol=1e-10).V
    assert np.allclose(fast, _robust_vi_reference(mdp.L, mdp.U, mdp.reward, GAMMA), atol=1e-6)
