"""SysAdmin ring (Guestrin et al., JAIR 2003): a small, factored, stochastic MDP with an exact model.

Pipeline step S0 (the unknown system). ``N`` machines on a one-way ring; machine ``i`` is up (1) or
down (0). Actions: ``0`` = no-op, ``i + 1`` = repair machine ``i``. The action enters machine ``i`` only
through the flag ``rep_i = [A = i + 1]`` (declared in S0), so the true 2-slice parents are

    Pa(m_i') = {m_i, m_{i-1}, rep_i}        (in-degree 3)

Transition of machine ``i`` (given its parents and the hidden load ``w``):

- repaired:                       up with probability ``p_repair - load_effect * w``;
- up, left neighbour up:          stays up with probability ``p_ok - load_effect * w``;
- up, left neighbour down:        stays up with probability ``p_bad - load_effect * w``;
- down, not repaired:             recovers with probability ``p_recover``.

A high load makes machines fail more often and repairs succeed less often.

The machines move independently given ``(s, a, w)``, so the joint kernel is the product of the
per-machine factors. The **hidden load** ``w`` (``W_t ~ Bernoulli(load_prob)``, fresh at every step,
never part of the observed state) matters only when ``load_effect > 0``. Then a logger that reads it
is confounded (setting 1 assumes no logger does): this is an MDP with unobserved confounders, whose
exogenous terms are drawn i.i.d. at every step (Bareinboim, *Causal AI* v0.27, Def. 9.4.4, p. 623),
and the observed kernel ``P(s' | s, a)`` differs from the interventional one (*ibid.*, Eq. 9.137).
The interventional kernel, which a planner that does not see ``w`` needs, averages ``w`` out:
``P(s' | s, do(a)) = sum_w P(w) P(s' | s, a, w)`` (*ibid.*, Ex. 7.9, Eqs. 7.44-7.45, p. 524), valid
because ``W_t`` is independent of ``S_t``.

Reward: the number of machines up, with machine 0 (the server) weighted ``server_weight``.
State index: C order over ``(m_0, ..., m_{N-1})`` (``m_{N-1}`` fastest), the row layout of NBN.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class SysAdminRing:
    n: int = 4
    p_ok: float = 0.95
    p_bad: float = 0.60
    p_repair: float = 0.95
    p_recover: float = 0.05
    server_weight: float = 2.0
    gamma: float = 0.95
    load_prob: float = 0.5
    load_effect: float = 0.0

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("a ring needs at least 2 machines")
        for name in ("p_ok", "p_bad", "p_repair", "p_recover", "load_prob"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a probability; got {v}")
        if not 0.0 <= self.load_effect <= min(self.p_ok, self.p_bad, self.p_repair):
            raise ValueError("load_effect must keep every probability in [0, 1]")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")

    # ------------------------------------------------------------------ spaces and names
    @property
    def n_states(self) -> int:
        return 2 ** self.n

    @property
    def n_actions(self) -> int:
        return self.n + 1

    @property
    def state_vars(self) -> list[str]:
        return [f"m{i}" for i in range(self.n)]

    @property
    def action_flags(self) -> list[str]:
        return [f"rep{i}" for i in range(self.n)]

    @cached_property
    def states(self) -> np.ndarray:
        """``[S, N]`` bits of every state, C order (row ``s`` is state index ``s``)."""
        grid = np.indices([2] * self.n).reshape(self.n, -1).T
        return grid.astype(np.int8)

    def index(self, bits: np.ndarray) -> np.ndarray:
        """State index of ``[..., N]`` bits."""
        return np.ravel_multi_index(np.moveaxis(np.asarray(bits), -1, 0), [2] * self.n)

    def flags(self, action: np.ndarray) -> np.ndarray:
        """``[..., N]`` repair flags of actions."""
        a = np.asarray(action)
        return (a[..., None] == np.arange(1, self.n + 1)).astype(np.int8)

    def true_parents(self) -> dict[str, tuple[str, ...]]:
        """The true 2-slice parents of every next-state variable."""
        return {f"m{i}'": (f"m{i}", f"m{(i - 1) % self.n}", f"rep{i}") for i in range(self.n)}

    # ------------------------------------------------------------------ dynamics
    def p_up(self, up: np.ndarray, left: np.ndarray, rep: np.ndarray,
             w: np.ndarray | float = 0.0) -> np.ndarray:
        """Probability that a machine is up next step, given its parents and the hidden load."""
        up, left, rep = (np.asarray(x) for x in (up, left, rep))
        drop = self.load_effect * np.asarray(w)
        stay = np.where(left == 1, self.p_ok, self.p_bad) - drop
        return np.where(rep == 1, self.p_repair - drop, np.where(up == 1, stay, self.p_recover))

    def factor_probs(self, w: float | None = None) -> np.ndarray:
        """``[S, A, N]``: P(m_i' = up | s, a), with the load averaged out when ``w`` is None."""
        s = self.states[:, None, :]                                   # [S, 1, N]
        rep = self.flags(np.arange(self.n_actions))[None, :, :]       # [1, A, N]
        left = np.roll(self.states, 1, axis=1)[:, None, :]            # m_{i-1}
        if w is None:
            return sum(prob * self.p_up(s, left, rep, wv) for wv, prob in self._load_values())
        return self.p_up(s, left, rep, w)

    def kernel(self, w: float | None = None) -> np.ndarray:
        """``[S, A, S']`` exact transition kernel (interventional when ``w`` is None).

        With the load averaged out the machines are no longer independent given ``(s, a)`` (they share
        ``w``), so the joint kernel is the average over ``w`` of the per-``w`` product kernels.
        """
        if w is None:
            return sum(prob * self.kernel(wv) for wv, prob in self._load_values())
        p = self.factor_probs(w)                                      # [S, A, N]
        nxt = self.states[None, None, :, :]                           # [1, 1, S', N]
        per = np.where(nxt == 1, p[:, :, None, :], 1.0 - p[:, :, None, :])
        return per.prod(-1)

    def _load_values(self) -> list[tuple[float, float]]:
        if self.load_effect == 0.0:
            return [(0.0, 1.0)]
        return [(0.0, 1.0 - self.load_prob), (1.0, self.load_prob)]

    @cached_property
    def reward(self) -> np.ndarray:
        """``[S]`` reward of a state: machines up, the server weighted."""
        weights = np.ones(self.n)
        weights[0] = self.server_weight
        return self.states @ weights

    @property
    def initial_state(self) -> np.ndarray:
        return np.ones(self.n, dtype=np.int8)

    def step(self, state: np.ndarray, action: int, rng: np.random.Generator, w: int = 0) -> np.ndarray:
        """Sample the next state (bits) given the state, the action and the hidden load."""
        rep = self.flags(action)
        p = self.p_up(state, np.roll(state, 1), rep, w)
        return (rng.random(self.n) < p).astype(np.int8)

    def sample_load(self, rng: np.random.Generator) -> int:
        return int(rng.random() < self.load_prob)
