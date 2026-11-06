"""Baseline without structure, with a guarantee: the flat robust MDP learned from counts.

One Clopper-Pearson interval per entry P(s' | s, a) of the flat (s, a) x s' count table (NBN
interval_set_from_counts, budget delta split over all entries); a row never seen is [0, 1]. This is
the non-factored r-MDP baseline of Schnitzer, Abate, Parker (AAAI 2026, arXiv:2508.00707).
Solve it with solving/robust_vi.py.
"""

import torch
from nbn.uncertainty.builders import interval_set_from_counts

from scm_to_mdp.baselines.flat_mdp import flat_counts
from scm_to_mdp.core import Dataset, IntervalMDP
from scm_to_mdp.scm_learning.families import seen_info


def learn(dataset: Dataset, delta: float = 0.05) -> IntervalMDP:
    counts = flat_counts(dataset)  # [S, K, S']
    S, K, _ = counts.shape
    sets = interval_set_from_counts(
        torch.tensor(counts.reshape(S * K, S)), delta=delta, method="clopper_pearson", sampling="trajectory"
    )
    L, U = (b.double().numpy().reshape(S, K, S) for b in sets.bounds())
    unseen = counts.sum(-1) == 0
    L[unseen], U[unseen] = 0.0, 1.0
    return IntervalMDP(L, U, dataset.reward, info=seen_info(~unseen), seen=~unseen)
