import numpy as np
import pytest

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import encode
from scm_to_mdp.data import read_dataset
from scm_to_mdp.discovery import oracle
from scm_to_mdp.envs import Sepsis
from scm_to_mdp.envs.sepsis import state_index
from scm_to_mdp.evaluation import evaluate, optimal_values, random_starts
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given
from scm_to_mdp.solving import robust_vi, value_iteration

GAMMA = 0.9


@pytest.fixture(scope="module")
def full():
    return Sepsis("full", seed=0)


@pytest.mark.parametrize("size,states", [("small", 72), ("medium", 144), ("full", 1440)])
def test_sizes(size, states):
    env = Sepsis(size, seed=0)
    assert len(env.B) == states
    assert np.allclose(env.P.sum(-1), 1.0)
    assert np.isclose(env.start.sum(), 1.0) and env.start[env.true_reward.terminal].sum() == 0


def test_custom_size():
    env = Sepsis("custom", vitals=("hr", "sysbp", "o2", "glucose"), diabetes=False)
    assert len(env.B) == 720 and "diabetic" not in env.state_vars
    with pytest.raises(ValueError):
        Sepsis("huge")


def test_glucose_fix(full):
    """Non-diabetic, glucose high (3), no treatment: up by one -> 4 w.p. 0.1 (the original code: -> 1)."""
    s = state_index(full, diabetic=0, hr=0, sysbp=0, o2=0, glucose=3, abx_on=0, vaso_on=0, vent_on=0)
    dist = full.next_distributions(full.B[[s]], np.array([0]))[0, full.state_vars.index("glucose")]
    assert np.allclose(dist, [0, 0, 0.1, 0.8, 0.1])
    bug = Sepsis("full", seed=0, glucose_bug=True)
    dist = bug.next_distributions(bug.B[[s]], np.array([0]))[0, bug.state_vars.index("glucose")]
    assert np.allclose(dist, [0, 0.1, 0.1, 0.8, 0])


def test_true_parents_are_exact(full):
    """Changing a non-parent never changes a variable's next distribution; every parent matters."""
    rng = np.random.default_rng(0)
    data = read_dataset(full.sample(50, 5, "random"), full.action_vars, full.cards, full.reward)
    parents = full.true_parents()
    states = full.B[rng.integers(len(full.B), size=300)]
    actions = rng.integers(0, 8, 300)
    base = full.next_distributions(states, actions)
    matters = {v: set() for v in full.state_vars}
    for name in data.inputs:
        for value in range(full.cards[name]):
            s, a = states.copy(), actions.copy()
            if name in full.state_vars:
                s[:, full.state_vars.index(name)] = value
            else:  # an action column: set that treatment
                cols = full.action_columns(a)
                cols[:, full.action_vars.index(name)] = value
                a = cols[:, 0] * 4 + cols[:, 1] * 2 + cols[:, 2]
            changed = np.abs(full.next_distributions(s, a) - base).max(axis=(0, 2)) > 1e-12
            for i, v in enumerate(full.state_vars):
                if changed[i]:
                    matters[v].add(name)
    for v in full.state_vars:
        assert matters[v] == set(parents[v]), v


def test_episodes_stop_at_terminal_states(full):
    df = full.sample(200, 20, "random")
    next_states = df[[v + "'" for v in full.state_vars]].values
    ends = full.true_reward.terminal[encode(next_states, full.state_cards)]
    last = df.groupby(level="episode").tail(1).index
    is_last = df.index.isin(last)
    assert not (ends & ~is_last).any()  # nothing is recorded after a terminal state


def test_policy_value_matches_simulation():
    env = Sepsis("small", seed=1)
    pi = np.zeros(len(env.B), dtype=np.int64)  # never treat
    V = value_iteration.policy_value(env.P, env.true_reward, pi, GAMMA)
    s = env.initial_states(20000)
    total, discount, alive = np.zeros(len(s)), 1.0, np.ones(len(s), dtype=bool)
    for _ in range(200):
        s = env.step(s, pi[encode(s, env.state_cards)])
        r = env.true_reward.r_next[encode(s, env.state_cards)]
        total += discount * r * alive
        alive &= ~env.is_terminal(s)
        discount *= GAMMA
    assert abs(total.mean() - env.start @ V) < 0.02


def test_true_graph_gives_a_valid_promise():
    env = Sepsis("small", seed=0)
    data = read_dataset(env.sample(2000, 20, "random"), env.action_vars, env.cards, env.reward)
    bounds = nbn_intervals.learn(data, given.select(oracle.fit(data, env.true_parents()), data), 0.05)
    mdp = product_bounds.compile(bounds)
    truth = env.ground_truth()
    P = truth.P[:, truth.actions_of(data)]
    assert np.all(mdp.L <= P + 1e-12) and np.all(P <= mdp.U + 1e-12)
    assert evaluate(
        truth,
        data,
        robust_vi.solve(mdp, GAMMA),
        GAMMA,
        optimal_values(truth, GAMMA),
        random_starts(truth, 20, 0),
    )["safe"]
