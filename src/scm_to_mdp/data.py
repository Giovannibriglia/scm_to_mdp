"""Offline datasets from loggers (pipeline step S1): episodes of transitions with a regime label."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.loggers import Logger


@dataclass
class Transitions:
    """Columns of a dataset, one row per transition.

    ``state``/``next_state`` ``[T, N]`` bits, ``action`` ``[T]``, ``regime`` ``[T]`` (index of the
    logger that produced the row), ``episode`` and ``t`` ``[T]``. ``hidden_load`` ``[T]`` is the
    hidden variable: it is kept for evaluation and diagnostics only, never given to a learner.
    ``distractor`` ``[T]`` is an *observed* exogenous bit (i.i.d. per step, all zeros unless the data
    were collected with one): no machine depends on it, but a logger may read it (E1 ablation).
    """

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    regime: np.ndarray
    episode: np.ndarray
    t: np.ndarray
    hidden_load: np.ndarray
    distractor: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.distractor is None:
            self.distractor = np.zeros(len(self.action), dtype=np.int8)

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def columns(self, env: SysAdminRing, extra: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
        """Observed columns by name: ``m{i}``, ``rep{i}``, ``m{i}'``, ``regime`` (no hidden load), and
        ``x`` (the distractor bit) when ``"x"`` is in ``extra``."""
        out = {f"m{i}": self.state[:, i] for i in range(env.n)}
        flags = env.flags(self.action)
        out.update({f"rep{i}": flags[:, i] for i in range(env.n)})
        out.update({f"m{i}'": self.next_state[:, i] for i in range(env.n)})
        out["regime"] = self.regime
        if "x" in extra:
            out["x"] = self.distractor
        return out

    FIELDS: ClassVar[tuple[str, ...]] = ("state", "action", "next_state", "regime", "episode", "t",
                                         "hidden_load", "distractor")

    @staticmethod
    def concat(parts: list[Transitions]) -> Transitions:
        return Transitions(*(np.concatenate([getattr(p, f) for p in parts]) for f in Transitions.FIELDS))

    def take(self, mask: np.ndarray) -> Transitions:
        """The rows selected by a boolean mask (or an index array)."""
        return Transitions(*(getattr(self, f)[mask] for f in Transitions.FIELDS))

    def episode_ids(self) -> np.ndarray:
        """One integer id per (regime, episode) pair, consecutive from 0 (no packing overflow)."""
        _, ids = np.unique(np.stack([self.regime, self.episode], axis=1), axis=0, return_inverse=True)
        return ids.reshape(-1)


def rollout(env: SysAdminRing, logger: Logger, n_episodes: int, horizon: int,
            rng: np.random.Generator, regime: int = 0, start: np.ndarray | None = None,
            distractor_prob: float | None = None) -> Transitions:
    """``n_episodes`` episodes of ``horizon`` steps from ``start`` (default: every machine up).

    With ``distractor_prob`` an observed bit ``x_t ~ Bernoulli(distractor_prob)`` is drawn at every
    step (after the load, before the action) and shown to the logger (``logger.distractor``).
    """
    if n_episodes < 0 or horizon < 1:
        raise ValueError("n_episodes must be >= 0 and horizon >= 1")
    T = n_episodes * horizon
    S = np.empty((T, env.n), dtype=np.int8)
    S2 = np.empty((T, env.n), dtype=np.int8)
    A = np.empty(T, dtype=np.int64)
    W = np.empty(T, dtype=np.int8)
    X = np.zeros(T, dtype=np.int8)
    k = 0
    for _ in range(n_episodes):
        s = (env.initial_state if start is None else np.asarray(start, dtype=np.int8)).copy()
        logger.reset()
        for _ in range(horizon):
            w = env.sample_load(rng)
            if distractor_prob is not None:
                X[k] = int(rng.random() < distractor_prob)
                logger.distractor = int(X[k])
            a = logger.act(s, rng, w)
            s2 = env.step(s, a, rng, w)
            logger.observe(s, a, s2)
            S[k], A[k], S2[k], W[k] = s, a, s2, w
            s = s2
            k += 1
    episode = np.repeat(np.arange(n_episodes), horizon)
    t = np.tile(np.arange(horizon), n_episodes)
    return Transitions(S, A, S2, np.full(T, regime), episode, t, W, X)


def collect(env: SysAdminRing, loggers: list[Logger], episodes: list[int], horizon: int,
            seed: int, distractor_prob: float | None = None) -> Transitions:
    """One dataset per logger (regime = its position), pooled. ``episodes[k]`` episodes for logger k."""
    if len(loggers) != len(episodes):
        raise ValueError("one episode count per logger")
    rng = np.random.default_rng(seed)
    parts = [rollout(env, lg, n, horizon, rng, regime=k, distractor_prob=distractor_prob)
             for k, (lg, n) in enumerate(zip(loggers, episodes, strict=True))]
    return Transitions.concat(parts)
