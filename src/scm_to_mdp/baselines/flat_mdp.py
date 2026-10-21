"""Baseline without structure and without a guarantee: the maximum-likelihood flat MDP.

P_hat(s' | s, a) = count(s, a, s') / count(s, a); a row never seen is uniform. Solve it with
solving/value_iteration.py. Its "promise" is the value under P_hat, which can be too optimistic.
"""

import numpy as np

from scm_to_mdp.core import Dataset, IntervalMDP


def flat_counts(dataset: Dataset) -> np.ndarray:
    """counts[s, k, s']: how often action vector k in state s led to s'."""
    S, K = len(dataset.B), len(dataset.actions)
    s = dataset.state_index(dataset.df[dataset.state_vars].values)
    s_next = dataset.state_index(dataset.df[dataset.next_vars].values)
    counts = np.zeros((S, K, S))
    np.add.at(counts, (s, dataset.action_index(), s_next), 1)
    return counts


def learn(dataset: Dataset) -> IntervalMDP:
    """The plug-in MDP, as an interval MDP with L == U."""
    counts = flat_counts(dataset)
    n = counts.sum(-1, keepdims=True)
    P_hat = np.where(n > 0, counts / np.maximum(n, 1), 1.0 / counts.shape[-1])
    return IntervalMDP(P_hat, P_hat, dataset.reward)
