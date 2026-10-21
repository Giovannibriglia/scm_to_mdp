"""From a table of transitions to a Dataset.

The learner is told:
    - which columns are actions;
    - that x' is the next-step value of x (the time order);
    - how many values every variable has;
    - the reward (a function of the states and action vectors).
Everything else (which action does what, the graph, the probabilities) is learned from the data.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from scm_to_mdp.core import NEXT_MARK, Dataset, Reward, all_states


def read_dataset(
    df: pd.DataFrame,
    action_vars: list[str],
    cards: dict[str, int],
    reward: Callable[[np.ndarray, np.ndarray], Reward],
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
