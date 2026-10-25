"""The alternative rewards of every environment (Env.alternative_rewards, reward transfer, S9): each is a
valid reward on the same dynamics, never reads a hidden variable, leaves the start state alive, has
something to gain, and is a different task from the original."""

import numpy as np
import pytest

from scm_to_mdp.envs import Aircraft, Chemistry, Drone, FrozenLake, Sepsis, StockTrading, SysAdmin

CASES = {
    "sysadmin": lambda: SysAdmin(3, 0, storm=0.3),
    "sepsis small": lambda: Sepsis("small", 0),
    "sepsis full hidden": lambda: Sepsis("full", 0, hide_diabetes=True),
    "chemistry": lambda: Chemistry(0.4, 0, n_objects=3, catalyst=0.3),
    "stock 2x2 news": lambda: StockTrading(2, 2, 0, news=0.3),
    "lake 4x4": lambda: FrozenLake("4x4", 1, 0),
    "lake 4x4 two agents wind": lambda: FrozenLake("4x4", 2, 0, wind=0.3),
    "aircraft 6x4 turbulence": lambda: Aircraft(6, 4, 0, turbulence=0.3),
    "drone 3": lambda: Drone(3, 1, 0),
    "drone 5": lambda: Drone(5, 1, 0),
    "drone 3 two gust": lambda: Drone(3, 2, 0, gust=0.3),
}
GAMMA = 0.95


@pytest.fixture(scope="module", params=list(CASES))
def env(request):
    return CASES[request.param]()


def test_every_task_is_valid_and_different(env):
    tasks = env.alternative_rewards()
    assert len(tasks) >= 3
    truth = env.ground_truth()
    V0 = truth.optimal_values(GAMMA)
    alive = truth.start > 0
    for name, _ in tasks.items():
        task_truth, observed = env.task(name)
        assert task_truth.P is truth.P, name  # the same dynamics
        r = task_truth.reward
        assert r.R.shape == truth.reward.R.shape and np.isfinite(r.R).all() and np.isfinite(r.r_next).all()
        assert not r.terminal[alive].any(), name  # the episode does not end before it starts
        V = task_truth.optimal_values(GAMMA)
        V_random = task_truth.random_policy_values(GAMMA)
        assert truth.start @ V > truth.start @ V_random + 1e-6, name  # something to gain over random
        assert not np.allclose(V, V0), name  # another task
        data = task_truth.learner_spec.read(env.hide(env.sample(5, 5, "random")))
        assert np.isfinite(observed(data.B, data.actions).R).all(), name  # the learner's reward


def test_rewards_ignore_hidden_variables(env):
    if not env.hidden_vars:
        pytest.skip("no hidden variable")
    truth = env.ground_truth()
    hidden = [env.state_vars.index(v) for v in env.hidden_vars]
    for name, reward in env.alternative_rewards().items():
        B = truth.B
        flipped = B.copy()
        flipped[:, hidden] = 1 - flipped[:, hidden]
        a, b = reward(B, truth.actions), reward(flipped, truth.actions)
        assert np.array_equal(a.R, b.R) and np.array_equal(a.r_next, b.r_next), name
        assert np.array_equal(a.terminal, b.terminal), name
