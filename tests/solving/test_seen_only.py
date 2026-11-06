""""Seen pairs only" (solver_options seen_only=True): the max over actions only over the pairs (s, a) whose
every factor's parent row has data, a state with none at the value floor (solving/stopping.greedy)."""

import numpy as np
import pytest

from scm_to_mdp.core import IntervalMDP, Reward
from scm_to_mdp.solving import robust_vi, value_iteration
from scm_to_mdp.solving.stopping import greedy, value_floor
from scm_to_mdp.tool import Method, evaluate, learn, solve

GAMMA = 0.9
TRUE = {"discovery": "oracle", "selection": "given"}


def test_greedy_keeps_to_the_allowed_actions_and_floors_the_empty_states():
    Q = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 0.0]])
    allowed = np.array([[True, False], [True, True], [False, False]])
    V, pi = greedy(Q, allowed, floor=-7.0)
    assert V.tolist() == [1.0, 3.0, -7.0] and pi[:2].tolist() == [0, 1]
    assert greedy(Q)[0].tolist() == [5.0, 3.0, 4.0]  # without a mask: the plain max


def test_the_floor_is_below_every_value():
    reward = Reward(np.array([[1.0, -2.0], [0.5, 0.0]]), np.array([0.0, 0.5]), np.array([False, False]))
    assert value_floor(reward, GAMMA) == pytest.approx(-2.0 / (1 - GAMMA))
    ends = Reward(np.array([[1.0, 2.0]]), np.array([0.5]), np.array([True]))  # an episode may end: 0
    assert value_floor(ends, GAMMA) == 0.0


def _random_set(rng, S=6, K=3, unseen=0.4):
    """A true kernel P and an interval set around it; unseen pairs get [0, 1]."""
    P = rng.dirichlet(np.ones(S), size=(S, K))
    seen = rng.random((S, K)) > unseen
    seen[0] = False  # a state with no seen action
    L, U = np.clip(P - 0.1, 0, 1), np.clip(P + 0.1, 0, 1)
    L[~seen], U[~seen] = 0.0, 1.0
    reward = Reward(rng.normal(size=(S, K)), np.zeros(S), np.zeros(S, dtype=bool))
    return P, IntervalMDP(L, U, reward, seen=seen)


@pytest.mark.parametrize("seed", range(5))
def test_the_seen_only_promise_is_a_lower_bound_and_below_the_unrestricted_one(seed):
    P, mdp = _random_set(np.random.default_rng(seed))
    sol = robust_vi.solve(mdp, GAMMA, allowed=mdp.seen)
    free = robust_vi.solve(mdp, GAMMA)
    true = value_iteration.policy_value(P, mdp.reward, sol.pi, GAMMA)
    assert np.all(sol.V <= true + 1e-9)  # sound (P is in the set)
    assert np.all(sol.V <= free.V + 1e-9)  # fewer actions: a lower value
    has = mdp.seen.any(1)
    assert np.all(mdp.seen[has, sol.pi[has]])  # only seen pairs where there is one
    assert sol.V[0] == pytest.approx(value_floor(mdp.reward, GAMMA), abs=1e-6)


def test_plain_vi_with_allowed_keeps_to_them():
    P, mdp = _random_set(np.random.default_rng(7))
    sol = value_iteration.solve(P, mdp.reward, GAMMA, allowed=mdp.seen)
    has = mdp.seen.any(1)
    assert np.all(mdp.seen[has, sol.pi[has]])


SEEN_ARMS = {
    "ia": Method("causal", "box_robust_vi", solver_options={"seen_only": True}, **TRUE),
    "exact": Method("causal", "exact_robust_vi", solver_options={"seen_only": True}, **TRUE),
    "plug-in": Method("causal", "vi", solver_options={"seen_only": True}, **TRUE),
    "flat": Method("flat_rmdp", "robust_vi", solver_options={"seen_only": True}),
    "flat plug-in": Method("flat_mdp", "vi", solver_options={"seen_only": True}),
}


@pytest.mark.parametrize("name", list(SEEN_ARMS))
def test_seen_only_arms_end_to_end(name, truth, dataset):
    result = solve(learn(dataset, SEEN_ARMS[name], true_parents=truth.true_parents), dataset, GAMMA)
    assert result.ok, result.error
    seen = result.model.seen
    has = seen.any(1)
    assert np.all(seen[has, result.solution.pi[has]])
    assert 0 <= result.info["states_no_seen"] <= 1 and 0 <= result.info["seen_pairs"] <= 1
    assert result.info["seen_pairs"] == pytest.approx(seen.mean())
    metrics = evaluate(result, dataset, truth, GAMMA)
    if SEEN_ARMS[name].guarantee:
        assert metrics["safe"], (name, metrics["promise"], metrics["true"])


def test_the_factored_seen_pairs_need_every_factors_row(truth, dataset):
    m = Method("causal", "box_robust_vi", **TRUE)
    bounds = learn(dataset, m, true_parents=truth.true_parents).model
    unseen_row = np.isclose(bounds.hi, 1.0).all(-1) & np.isclose(bounds.lo, 0.0).all(-1)  # [S, K, n]
    assert np.array_equal(bounds.seen, ~unseen_row.any(-1))  # a factor at [0, 1] <=> its row has no data


def test_seen_only_is_off_by_default_and_checked():
    m = Method("causal", "box_robust_vi", **TRUE)
    assert "seen_only" not in m.solver_options
    with pytest.raises(ValueError):
        Method("factored_spibb", "spibb", solver_options={"seen_only": True}, **TRUE)
    with pytest.raises(ValueError):
        Method("causal", "worst_mdp_vi", solver_options={"seen_only": True}, **TRUE)
    with pytest.raises(ValueError):
        Method("causal", "box_robust_vi", solver_options={"allowed": None}, **TRUE)


def test_seen_only_changes_nothing_when_every_pair_is_seen(truth, dataset):
    plain = Method("causal", "box_robust_vi", **TRUE)
    seen = Method("causal", "box_robust_vi", solver_options={"seen_only": True}, **TRUE)
    a = solve(learn(dataset, plain, true_parents=truth.true_parents), dataset, GAMMA)
    b_learned = learn(dataset, seen, true_parents=truth.true_parents)
    if not b_learned.model.seen.all():
        pytest.skip("the shared data leave some pairs unseen")
    b = solve(b_learned, dataset, GAMMA)
    assert np.allclose(a.solution.V, b.solution.V) and np.array_equal(a.solution.pi, b.solution.pi)
