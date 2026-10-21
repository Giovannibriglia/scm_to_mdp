"""The data classes passed between the pipeline steps.

    env.sample -> DataFrame -> read_dataset -> Dataset
    Dataset -> discovery -> Graph                         (S2-S3)
    Dataset + Graph -> scm_learning -> FactorBounds        (S4)
    FactorBounds -> compilation -> IntervalMDP             (S7)
    IntervalMDP (or FactorBounds) -> solving -> Solution   (S8)

Conventions used everywhere:
    Every variable is discrete with values 0..card-1.
    S = number of states (every combination of the state variables), K = number of action vectors
    seen in the data, n = number of state variables, C = the largest number of values of a variable.
    States are listed in C order (first variable = most significant digit), as NBN lists the rows of
    a conditional table; the index of a state is encode(values, cards).
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NEXT_MARK = "'"  # x' is the next-step value of x


# ---- states as vectors of values and as indices ---------------------------------------------------
def all_states(cards: list[int]) -> np.ndarray:
    """[prod(cards), n] every combination of values, C order."""
    return np.array(list(itertools.product(*[range(c) for c in cards])), dtype=np.int64).reshape(
        -1, len(cards)
    )


def encode(values: np.ndarray, cards: list[int]) -> np.ndarray:
    """Index of value vectors (last axis) in C order: a number in base (card_1, ..., card_n)."""
    weights = np.ones(len(cards), dtype=np.int64)
    for i in range(len(cards) - 2, -1, -1):
        weights[i] = weights[i + 1] * cards[i + 1]
    return values @ weights


# ---- reward ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reward:
    """Reward of a step s --a--> s': R[s, a] + r_next[s']. Arriving in a terminal state ends the
    episode (nothing follows). Known to the learner (a side input, not learned)."""

    R: np.ndarray  # [S, K] reward of taking action k in state s
    r_next: np.ndarray  # [S] reward of arriving in s'
    terminal: np.ndarray  # [S] bool, the episode ends on arrival

    def backup(self, V: np.ndarray, gamma: float) -> np.ndarray:
        """W[s'] = r_next[s'] + gamma * V[s'] (0 after a terminal state). Then Q = R + P @ W."""
        return self.r_next + gamma * np.where(self.terminal, 0.0, V)

    @staticmethod
    def immediate(R: np.ndarray) -> "Reward":
        """A reward that depends only on (s, a), with no terminal state."""
        S = R.shape[0]
        return Reward(R, np.zeros(S), np.zeros(S, dtype=bool))


# ---- what passes between the steps ---------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """Transitions as a table, plus what the learner is told about the columns."""

    df: pd.DataFrame  # columns: state vars, action vars, next-state vars (x')
    state_vars: list[str]
    action_vars: list[str]
    inputs: list[str]  # the slice-t columns: state vars and action vars, in table order
    cards: dict[str, int]  # number of values of every state and action variable
    B: np.ndarray  # [S, n] every state as a vector of values
    actions: np.ndarray  # [K, n_a] the action vectors seen in the data
    reward: Reward

    @property
    def next_vars(self) -> list[str]:
        return [v + NEXT_MARK for v in self.state_vars]

    @property
    def state_cards(self) -> list[int]:
        return [self.cards[v] for v in self.state_vars]

    @property
    def variables(self) -> dict:
        """NBN variable declaration of every column."""
        return {c: ("discrete", self.cards[c.removesuffix(NEXT_MARK)]) for c in self.df.columns}

    def tensors(self) -> dict:
        """Columns as torch tensors (for NBN). Called inside the step that needs them."""
        import torch

        return {c: torch.tensor(self.df[c].values) for c in self.df.columns}

    def state_index(self, values: np.ndarray) -> np.ndarray:
        return encode(values, self.state_cards)

    def action_index(self) -> np.ndarray:
        """[rows] index k of the action vector of every transition."""
        lookup = {tuple(a): k for k, a in enumerate(self.actions)}
        return np.array([lookup[tuple(a)] for a in self.df[self.action_vars].values])


@dataclass(frozen=True)
class Graph:
    """The chosen 2-slice graph: the slice-t parents of every next-state variable."""

    parents: dict[str, tuple[str, ...]]  # state var x -> parents of x' (names of inputs)
    report: dict = field(default_factory=dict)  # e.g. size of the equivalence class, ties


@dataclass(frozen=True)
class FactorBounds:
    """Per-factor intervals: lo[s, k, i, v] <= P(x_i' = v | s, a_k) <= hi[s, k, i, v].
    Values v >= card_i are padding (0)."""

    lo: np.ndarray  # [S, K, n, C]
    hi: np.ndarray  # [S, K, n, C]
    cards: list[int]  # number of values of every state variable
    B: np.ndarray  # [S, n]
    reward: Reward


@dataclass(frozen=True)
class IntervalMDP:
    """L <= P(s' | s, a) <= U entry-wise. An exact MDP is the case L == U."""

    L: np.ndarray  # [S, K, S]
    U: np.ndarray  # [S, K, S]
    reward: Reward


@dataclass(frozen=True)
class Solution:
    """Values and a deterministic policy (an action index k per state)."""

    V: np.ndarray  # [S] the value the method promises (for a robust method: a lower bound)
    pi: np.ndarray  # [S]
