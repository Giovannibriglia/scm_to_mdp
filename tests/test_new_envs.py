"""Coffee, Taxi, Chemistry and the confounded Sepsis: valid models, exact true parents, and the
evaluation with a hidden variable."""

import numpy as np
import pytest

from scm_to_mdp.envs import Chemistry, Coffee, Sepsis, SysAdmin, Taxi
from scm_to_mdp.envs.taxi import step_square

ENVS = {
    "coffee": lambda: Coffee("coffee"),
    "coffee+snack": lambda: Coffee("coffee+snack"),
    "taxi": lambda: Taxi("deterministic"),
    "taxi noisy": lambda: Taxi("noisy"),
    "chemistry 0.0": lambda: Chemistry(0.0, seed=0),
    "chemistry 0.7": lambda: Chemistry(0.7, seed=1),
    "sysadmin ring": lambda: SysAdmin(4, 0, topology="ring"),
    "sysadmin bidirectional": lambda: SysAdmin(4, 0, topology="bidirectional"),
    "sysadmin star": lambda: SysAdmin(4, 0, topology="star"),
}


@pytest.mark.parametrize("name", list(ENVS))
def test_model_is_an_mdp(name):
    truth = ENVS[name]().ground_truth()  # validates P, the start and the reward
    assert truth.n_states == truth.spec.n_states


def sufficient(table: np.ndarray, dist: np.ndarray, columns: list[int]) -> bool:
    """Do equal values of these input columns always give the same next distribution?"""
    keys = {}
    for row, key in enumerate(map(tuple, table[:, columns])):
        if key in keys:
            if not np.allclose(dist[row], dist[keys[key]], atol=1e-12):
                return False
        else:
            keys[key] = row
    return True


@pytest.mark.parametrize("name", list(ENVS))
def test_true_parents_are_exact(name):
    """Over every (state, action): the true parents of x' determine its next distribution
    (sufficient), and none of them can be left out (minimal)."""
    env = ENVS[name]()
    S, A = len(env.B), env.n_actions
    states, actions = np.repeat(env.B, A, axis=0), np.tile(np.arange(A), S)
    table = np.hstack([states, env.action_columns(actions)])  # inputs: state_vars + action_vars
    inputs = env.state_vars + env.action_vars
    dist = env.next_distributions(states, actions)
    for v, parents in env.true_parents().items():
        cols = [inputs.index(p) for p in parents]
        assert sufficient(table, dist[:, env.state_vars.index(v)], cols), (name, v, "not sufficient")
        for drop in cols:
            rest = [c for c in cols if c != drop]
            assert not sufficient(table, dist[:, env.state_vars.index(v)], rest), (
                name,
                v,
                inputs[drop],
                "not needed",
            )


def test_taxi_walls_and_rewards():
    assert step_square(0, "east") == 1 and step_square(1, "east") == 1  # wall between columns 1 and 2
    assert step_square(0, "north") == 0 and step_square(20, "east") == 20  # border; Y's wall
    env = Taxi("deterministic")
    truth = env.ground_truth()
    V = truth.optimal_values(0.99)
    assert truth.start @ V > 0  # a delivery is worth it from the start


def test_chemistry_density_controls_the_graph():
    edges = [np.mean([Chemistry(d, seed=s).n_edges for s in range(10)]) for d in (0.0, 0.3, 0.9)]
    assert edges[0] == 0 and edges[0] < edges[1] < edges[2]


def test_confounded_sepsis_hides_diabetes_and_evaluates_in_the_full_model():
    from scm_to_mdp.tool import Method, evaluate, learn, solve

    env = Sepsis("custom", seed=0, vitals=("hr", "sysbp"), diabetes=True, hide_diabetes=True)
    truth = env.ground_truth()
    assert truth.hidden == ["diabetic"] and truth.learner_spec.n_states * 2 == truth.n_states
    df = env.hide(env.sample(300, 20, "diabetes_aware"))
    assert "diabetic" not in df.columns
    truth.check_data(df)
    data = truth.learner_spec.read(df)
    result = solve(learn(data, Method("flat_mdp", "vi")), data, 0.9)
    out = evaluate(result, data, truth, 0.9)
    assert out["true"] <= truth.start @ truth.optimal_values(0.9) + 1e-9  # V* sees diabetes: an upper bound
    with pytest.raises(ValueError, match="hide_diabetes"):
        Sepsis("small", hide_diabetes=True)


def test_hidden_variable_policy_is_the_same_in_both_halves():
    """A policy over the observed states acts the same whatever the hidden value."""
    env = Sepsis("custom", seed=0, vitals=("hr", "sysbp"), diabetes=True, hide_diabetes=True)
    truth = env.ground_truth()
    obs = truth.observed_index()
    half = truth.n_states // 2
    assert np.array_equal(obs[:half], obs[half:])  # diabetic is the first (most significant) variable
