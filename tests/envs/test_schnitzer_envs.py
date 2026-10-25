"""The environments after Schnitzer, Abate, Parker (AAAI 2026): stock trading, Frozen Lake, aircraft
collision avoidance, drone delivery. A valid model, true parents that match the model, a positive
optimal value, data with the declared columns, and the three behaviour policies."""

import numpy as np
import pytest

from scm_to_mdp.core import NEXT_MARK
from scm_to_mdp.envs import Aircraft, Drone, FrozenLake, StockTrading

CASES = {
    "stock 2x2": lambda: StockTrading(2, 2, 0),
    "stock 2x2 news": lambda: StockTrading(2, 2, 0, news=0.3),
    "lake 4x4": lambda: FrozenLake("4x4", 1, 0),
    "lake 4x4 two agents wind": lambda: FrozenLake("4x4", 2, 0, wind=0.3),
    "lake 4x4 not slippery": lambda: FrozenLake("4x4", 1, 0, slippery=False),
    "lake 4x4 two agents joint axes": lambda: FrozenLake("4x4", 2, 0, wind=0.3, axes="joint"),
    "lake 4x4 joint axes": lambda: FrozenLake("4x4", 1, 0, axes="joint"),
    "aircraft 6x4": lambda: Aircraft(6, 4, 0),
    "aircraft 6x4 turbulence": lambda: Aircraft(6, 4, 0, turbulence=0.3),
    "drone 3": lambda: Drone(3, 1, 0),
    "drone 3 two gust": lambda: Drone(3, 2, 0, gust=0.3),
}


@pytest.fixture(scope="module", params=list(CASES))
def env(request):
    return CASES[request.param]()


def test_model_is_valid_and_worth_solving(env):
    truth = env.ground_truth()
    assert np.allclose(truth.P.sum(-1), 1.0) and (truth.P >= 0).all()
    assert truth.start @ truth.optimal_values(0.95) > 0


def test_true_parents_match_the_model(env):
    """Changing an input that is not a parent of x_i' never changes x_i''s next distribution."""
    parents = env.true_parents()
    columns = env.action_columns(np.arange(env.n_actions))
    states = np.repeat(env.B, env.n_actions, axis=0)
    actions = np.tile(np.arange(env.n_actions), len(env.B))
    dist = env.next_distributions(states, actions)  # [S * A, n, C]
    inputs = np.concatenate([states, columns[actions]], axis=1)
    names = env.state_vars + env.action_vars
    for i, v in enumerate(env.state_vars):
        keep = [names.index(p) for p in parents[v] if not p.endswith(NEXT_MARK)]  # the slice-t parents
        _, group = np.unique(inputs[:, keep], axis=0, return_inverse=True)
        for g in np.unique(group):
            rows = dist[group == g, i]
            assert np.allclose(rows, rows[0]), (v, parents[v])


def test_data_and_behaviour_policies(env):
    df = env.hide(env.sample(20, 15, "random"))
    observed = [v for v in env.state_vars if v not in env.hidden_vars]
    assert list(df.columns) == observed + env.action_vars + [v + NEXT_MARK for v in observed]
    names = ["uniform", "expert"] + (["confounded"] if env.confounder is not None else [])
    for name in names:
        table = env.behaviour(name, 0.95)
        assert np.allclose(table.sum(1), 1.0)


def test_same_step_edges_match_the_coupling(env):
    """A pair from coupled_next has a joint law that is not the product of its marginals (the
    same-step edge y' -> x' of the true graph is needed); no pair, no same-step parent."""
    states = np.repeat(env.B, env.n_actions, axis=0)
    actions = np.tile(np.arange(env.n_actions), len(env.B))
    pairs = env.coupled_next(states, actions)
    same_step = {v: [p for p in ps if p.endswith(NEXT_MARK)] for v, ps in env.true_parents().items()}
    assert {(env.state_vars.index(p.removesuffix(NEXT_MARK)), env.state_vars.index(v))
            for v, ps in same_step.items() for p in ps} == set(pairs)  # fmt: skip
    dist = env.next_distributions(states, actions)
    for (i, j), joint in pairs.items():
        assert np.allclose(joint.sum(2), dist[:, i, : joint.shape[1]])  # the marginals agree
        assert np.allclose(joint.sum(1), dist[:, j, : joint.shape[2]])
        product = dist[:, i, : joint.shape[1], None] * dist[:, j, None, : joint.shape[2]]
        assert not np.allclose(joint, product)


def test_independent_axes_stay_independent_with_the_hidden_wind():
    """Axes "independent": the wind pushes agent 0's row only, so once the hidden wind is averaged out
    the observed kernel of agent 0's cell is still the product of its x0' and y0' laws (a model with
    independent axes is exact). A wind on both axes (axes "joint") gives a kernel that is not."""
    for axes, product in (("independent", True), ("joint", False)):
        env = FrozenLake("4x4", 1, 0, wind=0.3, axes=axes)
        K = env.ground_truth().observed_kernel()  # [S_obs, A, S_obs], states (x0, y0) in C order
        cell = K.reshape(K.shape[0], K.shape[1], 4, 4)  # [s, a, x0', y0']
        outer = cell.sum(3)[..., :, None] * cell.sum(2)[..., None, :]
        assert np.allclose(cell, outer) == product, axes
    env = FrozenLake("4x4", 1, 0, wind=0.3)
    calm, windy = env.B[env.B[:, 2] == 0], env.B[env.B[:, 2] == 1]  # the same cells, wind off / on
    actions = np.arange(env.n_actions).repeat(len(calm))
    off = env.next_distributions(np.tile(calm, (env.n_actions, 1)), actions)
    on = env.next_distributions(np.tile(windy, (env.n_actions, 1)), actions)
    assert np.allclose(off[:, 0], on[:, 0]) and not np.allclose(off[:, 1], on[:, 1])  # x0' vs y0'


def test_two_agent_lake_ends_when_both_have_finished():
    """No done variable: an agent on a hole or the goal stays there, and the episode ends (and pays
    the number of agents on the goal) only when both agents have finished."""
    env = FrozenLake("4x4", 2, 0)
    B, reward = env.B, env.ground_truth().reward
    cell = {name: (B[:, 2 * i + 1] * 4 + B[:, 2 * i]) for i, name in enumerate(("a", "b"))}
    goal, hole = 15, 5  # G at (3, 3); a hole at (1, 1)
    both = (cell["a"] == goal) & (cell["b"] == hole)
    one = (cell["a"] == goal) & (cell["b"] == 0)
    assert reward.terminal[both].all() and reward.r_next[both].tolist() == [1.0]
    assert not reward.terminal[one].any() and reward.r_next[one].tolist() == [0.0]
    s = int(np.flatnonzero(one)[0])
    assert np.isclose(env.P[s][:, cell["a"] == goal].sum(1), 1.0).all()  # agent a stays on G
