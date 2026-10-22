"""SPIBB (flat and with structure learning) and the discrete FOCUS (baselines/spibb.py, focus.py)."""

import numpy as np
import pytest

from scm_to_mdp.baselines import focus, spibb
from scm_to_mdp.core import Solution
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.evaluation import structure_distances
from scm_to_mdp.solving import value_iteration
from scm_to_mdp.tool import Method, evaluate, learn, solve, standard_methods
from scm_to_mdp.tool.methods import Result

GAMMA = 0.9


def as_result(probs):
    """A stochastic policy as a solved result (its promise: 0)."""
    return Result(Method("flat_mdp", "vi"), solution=Solution(np.zeros(len(probs)), probs.argmax(1), probs))


@pytest.fixture(scope="module")
def problem():
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    expert = truth.learner_spec.read(env.sample(400, 20, env.behaviour("expert", GAMMA, epsilon=0.3)))
    uniform = truth.learner_spec.read(env.sample(400, 20, env.behaviour("uniform", GAMMA)))
    return truth, expert, uniform


# ---- SPIBB ----------------------------------------------------------------------------------------
def test_spibb_keeps_the_behaviour_on_bootstrapped_pairs(problem):
    _, expert, _ = problem
    model = spibb.flat_model(expert, m=30)
    sol = spibb.solve(model, GAMMA)
    boot, pi_b = model.bootstrapped, model.behaviour
    assert np.allclose(sol.probs.sum(1), 1)
    assert np.allclose(sol.probs[boot], pi_b[boot])  # Pi_b-SPIBB: same probability where uncertain
    free = ~boot.all(1)
    extra = (sol.probs - np.where(boot, pi_b, 0)).sum(1)  # the rest goes to one non-bootstrapped action
    assert np.allclose(extra[free], 1 - np.where(boot, pi_b, 0).sum(1)[free])


def test_spibb_with_no_trusted_pair_returns_the_behaviour(problem):
    _, expert, _ = problem
    sol = spibb.solve(spibb.flat_model(expert, m=10**9), GAMMA)
    assert np.allclose(sol.probs, spibb.estimated_behaviour(expert))


def test_structure_learning_spibb_improves_on_the_logger(problem):
    """On this data (seed 0, 400 rollouts); SPIBB's improvement holds with high probability, not
    always: with 100 rollouts it fails on 1 seed out of 6."""
    truth, expert, _ = problem
    logger = spibb.estimated_behaviour(expert)
    base = evaluate(as_result(logger), expert, truth, GAMMA)
    sl = solve(learn(expert, Method("sl_spibb", "spibb", learning_options={"d": 3})), expert, GAMMA)
    flat = solve(learn(expert, Method("spibb", "spibb")), expert, GAMMA)
    v_sl, v_flat = (evaluate(r, expert, truth, GAMMA)["true"] for r in (sl, flat))
    assert v_sl >= base["true"] - 1e-6 and v_sl > v_flat  # structure: fewer bootstrapped pairs


def test_a_stochastic_policy_is_evaluated_exactly(problem):
    truth, expert, _ = problem
    probs = spibb.estimated_behaviour(expert)
    out = evaluate(as_result(probs), expert, truth, GAMMA)
    full = np.zeros((truth.n_states, truth.P.shape[1]))
    full[:, truth.actions_of(expert)] = probs
    by_hand = truth.start @ value_iteration.policy_value(truth.P, truth.reward, full, GAMMA)
    assert np.isclose(out["true"], by_hand)


# ---- FOCUS ----------------------------------------------------------------------------------------
def test_focus_threshold_rule():
    # a clear gap after the small p-values: the threshold is the last of them
    assert focus.threshold(np.array([0.0, 0.001, 0.002, 0.6, 0.7, 0.8, 0.9])) == 0.002


def test_focus_finds_the_ring_with_uniform_data(problem):
    truth, _, uniform = problem
    graph = focus.discover(uniform)
    d = structure_distances(graph, truth.parents_in(uniform), uniform)
    assert d["missing"] == 0
    result = solve(learn(uniform, Method("focus", "vi")), uniform, GAMMA)
    assert result.ok and result.graph is not None
    assert evaluate(result, uniform, truth, GAMMA)["optimal_share"] > 0.95


def test_the_penalty_lowers_the_value(problem):
    _, expert, _ = problem
    graph = focus.discover(expert)
    plain, penalised = (focus.penalised_mdp(expert, graph, lam=lam) for lam in (0.0, 1.0))
    V0 = value_iteration.solve(plain.L, plain.reward, GAMMA).V
    V1 = value_iteration.solve(penalised.L, penalised.reward, GAMMA).V
    assert (V1 <= V0 + 1e-9).all() and (V1 < V0).any()


def test_the_new_baselines_are_standard_methods():
    methods = standard_methods(state_parents=2)
    assert methods["SL-SPIBB"].learning_options == {"d": 2}
    for name in ("flat SPIBB", "SL-SPIBB", "FOCUS"):
        assert methods[name].role == "baseline" and not methods[name].guarantee
    with pytest.raises(ValueError, match="unknown options"):
        Method("sl_spibb", "spibb", learning_options={"depth": 2})
