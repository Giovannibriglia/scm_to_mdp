"""S1 assumption checks from data alone: confounding gate, Markov, stationarity, episode split."""
from __future__ import annotations

import numpy as np

from scm_to_mdp.assumptions import (
    confounding_gate,
    episode_split,
    markov_test,
    profile_assumptions,
    stationarity_test,
)
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger


def _pair(env, second="specialist", episodes=200, seed=0, **kw):
    first = make_logger("specialist", env, region=(0, 1), eps=0.1)
    other = make_logger(second, env, **kw)
    return collect(env, [first, other], [episodes, episodes], 30, seed=seed)


def test_gate_does_not_reject_unconfounded_loggers_and_rejects_a_confounded_one():
    env = SysAdminRing(n=4, load_effect=0.4)
    clean = _pair(env, "specialist", region=(2, 3), eps=0.1)
    dirty = _pair(env, "confounded")
    assert min(confounding_gate(env, clean).values()) > 0.05 / 4
    assert min(confounding_gate(env, dirty).values()) <= 0.001          # the floor of 999 permutations


def test_markov_and_stationarity_hold_on_the_ring():
    env = SysAdminRing(n=4)
    data = _pair(env, "uniform", episodes=300, seed=1)
    assert min(markov_test(env, data).values()) > 0.05 / 4
    assert min(stationarity_test(env, data).values()) > 0.05 / 4


def test_markov_test_detects_a_hidden_persistent_variable():
    """A history-dependent logger is fine; a *persistent* hidden load is not first-order Markov in S."""
    env = SysAdminRing(n=4, load_effect=0.4, load_prob=0.5)
    data = _pair(env, "uniform", episodes=300, seed=2)
    # Make the load persistent within an episode by resampling each episode with one load value.
    rng = np.random.default_rng(3)
    from scm_to_mdp.data import Transitions

    parts = []
    for r, lg in enumerate([make_logger("uniform", env), make_logger("uniform", env)]):
        S, A, S2, W, E, T = [], [], [], [], [], []
        for ep in range(300):
            w = env.sample_load(rng)
            s = env.initial_state.copy()
            for t in range(30):
                a = lg.act(s, rng, w)
                s2 = env.step(s, a, rng, w)
                S.append(s), A.append(a), S2.append(s2), W.append(w), E.append(ep), T.append(t)
                s = s2
        parts.append(Transitions(np.array(S), np.array(A), np.array(S2), np.full(len(A), r),
                                 np.array(E), np.array(T), np.array(W)))
    persistent = Transitions.concat(parts)
    assert min(markov_test(env, persistent).values()) <= 0.001          # the floor of 999 permutations
    assert min(markov_test(env, data).values()) > 0.05 / 4


def test_profile_and_episode_split():
    env = SysAdminRing(n=4)
    data = _pair(env, "uniform", episodes=40, seed=4)
    rep = profile_assumptions(env, data)
    assert rep.episodes == 80 and rep.episodes_per_logger == {0: 40, 1: 40}
    assert not rep.rejected("gate") and 0 < rep.joint_coverage <= 1
    a, b = episode_split(data, 0.5, seed=0)
    assert len(a) + len(b) == len(data)
    for r in (0, 1):
        ea = set(a.episode[a.regime == r].tolist())
        eb = set(b.episode[b.regime == r].tolist())
        assert not ea & eb and len(ea) == len(eb) == 20
