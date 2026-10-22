"""From a table of transitions to a Dataset (step S1).

The learner is told (the DataSpec):
    - the state variables and the action columns;
    - that x' is the next-step value of x (the time order);
    - how many values every variable has;
    - the reward (a function of the states and action vectors).
Everything else (which action does what, the graph, the probabilities) is learned from the data.

The table has one row per transition: columns x (state at t), a (action columns at t) and x' (state
at t+1). Other columns are ignored. Index (episode, t) is needed by PCMCI+ and by sample sweeps.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scm_to_mdp.core import NEXT_MARK, Dataset, Reward, all_states

# reward(B [S, n], action vectors [K, n_a]) -> Reward
RewardFn = Callable[[np.ndarray, np.ndarray], Reward]


class DataError(ValueError):
    """The table does not match the DataSpec."""


@dataclass(frozen=True)
class DataSpec:
    """What the learner is told about the table. It is the only side input of learning."""

    state_vars: list[str]
    action_vars: list[str]
    cards: dict[str, int]  # number of values of every state and action variable (values 0..card-1)
    reward: RewardFn

    @property
    def next_vars(self) -> list[str]:
        return [v + NEXT_MARK for v in self.state_vars]

    @property
    def columns(self) -> list[str]:
        return self.state_vars + self.action_vars + self.next_vars

    @property
    def n_states(self) -> int:
        return int(np.prod([self.cards[v] for v in self.state_vars]))

    def check(self, df: pd.DataFrame) -> None:
        """Raise DataError if a column is missing or a value is outside 0..card-1."""
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise DataError(f"columns missing from the table: {missing}")
        for c in self.columns:
            values = df[c].to_numpy()
            card = self.cards[c.removesuffix(NEXT_MARK)]
            if not np.issubdtype(values.dtype, np.integer):
                raise DataError(f"column {c!r} must hold integers 0..{card - 1}, found {values.dtype}")
            if values.min() < 0 or values.max() >= card:
                raise DataError(f"column {c!r} has values outside 0..{card - 1}")

    def read(self, df: pd.DataFrame) -> Dataset:
        """Check the table and turn it into a Dataset (columns in the order of the spec)."""
        self.check(df)
        return read_dataset(df[self.columns], self.action_vars, self.cards, self.reward)


def read_dataset(
    df: pd.DataFrame,
    action_vars: list[str],
    cards: dict[str, int],
    reward: RewardFn,
) -> Dataset:
    """reward(B [S, n], actions [K, n_a]) -> Reward."""
    state_vars = [c.removesuffix(NEXT_MARK) for c in df.columns if c.endswith(NEXT_MARK)]
    inputs = [c for c in df.columns if not c.endswith(NEXT_MARK)]
    actions = np.array(sorted(set(df[action_vars].itertuples(index=False, name=None))), dtype=np.int64)
    B = all_states([cards[v] for v in state_vars])
    return Dataset(df, state_vars, list(action_vars), inputs, dict(cards), B, actions, reward(B, actions))


def slice_inputs(dataset: Dataset) -> np.ndarray:
    """X[s, k, j] = value of input j (in the order of dataset.inputs) at state s and action k."""
    S, K = len(dataset.B), len(dataset.actions)
    columns = {}
    for i, v in enumerate(dataset.state_vars):
        columns[v] = np.repeat(dataset.B[:, i, None], K, axis=1)  # [S, K]
    for j, a in enumerate(dataset.action_vars):
        columns[a] = np.repeat(dataset.actions[None, :, j], S, axis=0)  # [S, K]
    return np.stack([columns[c] for c in dataset.inputs], axis=-1)  # [S, K, n_inputs]


def first_rollouts(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The transitions of the first n rollouts (the n smallest episode ids) of a table indexed by
    (episode, t). Taking prefixes of one table makes the datasets of a sample sweep nested."""
    if "episode" not in (df.index.names or []):
        raise DataError("the table needs an index level 'episode' to take its first rollouts")
    episode = df.index.get_level_values("episode")
    keep = np.sort(episode.unique())[:n]
    if len(keep) < n:
        raise DataError(f"the table has {len(keep)} rollouts, fewer than {n}")
    return df[episode.isin(keep)]
