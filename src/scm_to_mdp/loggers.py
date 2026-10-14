"""Behaviour policies (loggers) for the SysAdmin ring (pipeline step S1).

In setting 1 the loggers read only the observed state, so they cannot change the kernel: they decide
which rows are seen (coverage) and which candidate parent sets can be told apart (aliasing). The
families below span the cases discussed for E1:

==================  ==========================================================  =====================
family              behaviour                                                   stresses
==================  ==========================================================  =====================
``uniform``         every action with probability 1 / (N + 1)                   best-coverage reference
``specialist``      repairs a failed machine of its own region                  joint vs factor coverage
``aliasing``        repairs m_i exactly when m_{i+offset} is down               candidate graphs aliased
``softmax_q``       softmax over the optimal Q at temperature tau               data in good states only
``lazy``            mostly no-op                                                failures seen, repairs rare
``recent_failure``  repairs the machine that failed most recently (history)     non-Markov but unconfounded
``confounded``      repairs more when the hidden load is high                   breaks setting 1 (RQ2)
==================  ==========================================================  =====================

Every logger takes an exploration probability ``eps``: with probability ``eps`` it plays a uniform
random action. "Mixed sizes" is not a family: it is a dataset-level choice (episodes per logger).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.mdp import value_iteration


@dataclass
class Logger:
    """Base class. ``act`` returns an action in ``0 .. N`` (0 = no-op, i + 1 = repair machine i)."""

    env: SysAdminRing
    eps: float = 0.0
    name: str = "logger"
    reads_hidden: bool = field(default=False, init=False)
    history_dependent: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.eps <= 1.0:
            raise ValueError(f"eps must be a probability; got {self.eps}")

    def reset(self) -> None:
        """Called at the start of every episode."""

    def act(self, state: np.ndarray, rng: np.random.Generator, w: int = 0) -> int:
        if rng.random() < self.eps:
            return int(rng.integers(self.env.n_actions))
        return self._act(state, rng, w)

    def _act(self, state: np.ndarray, rng: np.random.Generator, w: int) -> int:
        raise NotImplementedError

    def observe(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        """Called after every transition (only history-dependent loggers use it)."""


def _repair_one(candidates: np.ndarray, rng: np.random.Generator) -> int:
    """Repair a uniformly chosen machine among ``candidates`` (indices); no-op if empty."""
    if candidates.size == 0:
        return 0
    return int(rng.choice(candidates)) + 1


@dataclass
class UniformRandom(Logger):
    name: str = "uniform"

    def _act(self, state, rng, w):
        return int(rng.integers(self.env.n_actions))


@dataclass
class RegionSpecialist(Logger):
    """Repairs a uniformly chosen failed machine of ``region``; no-op when its region is healthy."""

    region: tuple[int, ...] = (0, 1)
    name: str = "specialist"

    def _act(self, state, rng, w):
        region = np.asarray(self.region)
        return _repair_one(region[state[region] == 0], rng)


@dataclass
class AliasingRule(Logger):
    """Repairs ``m_i`` exactly when ``m_{i + offset}`` is down (the lowest such ``i``); else no-op.

    With ``offset = -1`` the repair flag ``rep_i`` moves with the left neighbour ``m_{i-1}``, the true
    parent of ``m_i'``: candidate parent sets with and without that neighbour explain the data almost
    equally well and disagree on rows the logger never plays. A second logger with another ``offset``
    breaks the tie.
    """

    offset: int = -1
    name: str = "aliasing"

    def _act(self, state, rng, w):
        n = self.env.n
        trigger = np.array([state[(i + self.offset) % n] == 0 for i in range(n)])
        hits = np.flatnonzero(trigger)
        return int(hits[0]) + 1 if hits.size else 0


@dataclass
class SoftmaxQ(Logger):
    """``pi(a | s)`` proportional to ``exp(Q*(s, a) / temperature)`` (as in Simão & Spaan, IJCAI 2019)."""

    temperature: float = 1.0
    name: str = "softmax_q"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        _, Q = value_iteration(self.env.kernel(), self.env.reward, self.env.gamma)
        z = (Q - Q.max(axis=1, keepdims=True)) / self.temperature
        self.probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    def _act(self, state, rng, w):
        return int(rng.choice(self.env.n_actions, p=self.probs[self.env.index(state)]))


@dataclass
class Lazy(Logger):
    """No-op with probability ``p_noop``; otherwise a uniformly chosen repair."""

    p_noop: float = 0.9
    name: str = "lazy"

    def _act(self, state, rng, w):
        if rng.random() < self.p_noop:
            return 0
        return int(rng.integers(1, self.env.n_actions))


@dataclass
class RecentFailure(Logger):
    """Repairs the machine that went down most recently (history-dependent, still unconfounded).

    It reads only observed past states, so the data still show the interventional kernel (no
    unobserved confounding: Bareinboim, *Causal AI* v0.27, Def. 8.2.1, p. 547). This relies on the
    hidden load being fresh at every step: with a persistent load, reading past states would open a
    path from the action to the next state through earlier loads.
    """

    name: str = "recent_failure"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.history_dependent = True
        self._failed_at = np.full(self.env.n, -1)
        self._t = 0

    def reset(self) -> None:
        self._failed_at[:] = -1
        self._t = 0

    def _act(self, state, rng, w):
        down = np.flatnonzero(state == 0)
        if down.size == 0:
            return 0
        latest = down[np.argmax(self._failed_at[down])]            # ties: lowest index
        return int(latest) + 1

    def observe(self, state, action, next_state):
        self._t += 1
        newly_down = (state == 1) & (next_state == 0)
        self._failed_at[newly_down] = self._t


@dataclass
class Confounded(Logger):
    """Reads the hidden load ``w``: under high load repairs a failed machine with probability
    ``p_high``, otherwise with probability ``p_low``. Confounds ``A -> S'`` when the load also changes
    the dynamics (``SysAdminRing.load_effect > 0``)."""

    p_high: float = 0.9
    p_low: float = 0.1
    name: str = "confounded"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reads_hidden = True

    def _act(self, state, rng, w):
        p = self.p_high if w == 1 else self.p_low
        if rng.random() >= p:
            return 0
        return _repair_one(np.flatnonzero(state == 0), rng)


FAMILIES = {
    "uniform": UniformRandom,
    "specialist": RegionSpecialist,
    "aliasing": AliasingRule,
    "softmax_q": SoftmaxQ,
    "lazy": Lazy,
    "recent_failure": RecentFailure,
    "confounded": Confounded,
}


def make_logger(family: str, env: SysAdminRing, **kwargs) -> Logger:
    """Build a logger by family name (see ``FAMILIES``)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown logger family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[family](env=env, **kwargs)
