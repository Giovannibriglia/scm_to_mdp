"""S4 from a KWIK structure (scm_learning/kwik_intervals.py): the hulls contain the true factor
probabilities; HALT is a lower bound on every value, also with terminal rewards."""

import sys
from pathlib import Path

import numpy as np
import pytest

from scm_to_mdp.core import Reward
from scm_to_mdp.discovery import kwik_structure as ks
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.scm_learning import kwik_intervals as ki
from scm_to_mdp.solving import value_iteration

sys.path.insert(0, str(Path(__file__).parent))
from test_widen import true_factors  # noqa: E402


def test_hulls_and_kwik_bounds_contain_the_truth():
    """Guaranteed at 1 - delta (here: every entry): the fixed hull (both learners) and Strehl's accepted
    hull. Nominal only (the papers' m, Met-Rmax may eliminate the true meteorologist): Met-Rmax's accepted
    hull and the KWIK arms; checked as a regression floor."""
    env = SysAdmin(4, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(1000, 20, "random"))
    T = true_factors(truth, data)

    def covered(b):
        inside = (b.lo[..., :2] <= T[..., :2] + 1e-9) & (T[..., :2] <= b.hi[..., :2] + 1e-9)
        return inside.all(-1).mean()

    strehl, met = ks.strehl_sl(data, k=3, m=30, eps1=0.2), ks.met_rmax(data, D=3, m=50, eps=0.2)
    for b in (ki.learn(data, strehl, 0.05, "fixed"), ki.learn(data, met, 0.05, "fixed"),
              ki.learn(data, strehl, 0.05, "accepted")):  # fmt: skip
        assert covered(b) == 1.0
    for b in (ki.learn(data, met, 0.05, "accepted"), ki.kwik_bounds(data, strehl), ki.kwik_bounds(data, met)):
        assert covered(b) >= 0.9


def test_halt_value_is_a_lower_bound_with_terminal_rewards():
    # Sepsis-like: no step reward, death (terminal) -1, discharge (terminal) +1, other states 0
    R = np.zeros((3, 2))
    r_next = np.array([0.0, -1.0, 1.0])
    terminal = np.array([False, True, True])
    assert ki.halt_value(Reward(R, r_next, terminal), 0.9) == -1.0
    # a reward that goes on: -2 per step forever is worse than dying now
    assert ki.halt_value(Reward(R - 2, r_next, terminal), 0.9) == pytest.approx(min(-2 / 0.1, -3 + -2 / 0.1))


def test_point_model_halt_pairs_are_pessimistic():
    env = SysAdmin(4, 0)
    data = env.ground_truth().learner_spec.read(env.sample(100, 20, "random"))
    s = ks.strehl_sl(data, k=3, m=30, eps1=0.2)
    point = ki.kwik_point(data, s)
    mdp = ki.exact_with_halt(point, 0.9)
    halt = point.info["_halt"]
    assert (
        halt.any()
        and np.all(mdp.L[halt] == 0)
        and np.all(mdp.reward.R[halt] == ki.halt_value(data.reward, 0.9))
    )
    sol = value_iteration.solve(mdp.L, mdp.reward, 0.9)
    assert np.all(sol.V >= ki.halt_value(data.reward, 0.9) - 1e-9)
