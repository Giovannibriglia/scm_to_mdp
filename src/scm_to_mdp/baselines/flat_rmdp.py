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
from scm_to_mdp.scm_learning import data_only as rules
from scm_to_mdp.scm_learning.families import seen_info


def learn(dataset: Dataset, delta: float = 0.05, data_only: bool = False) -> IntervalMDP:
    """data_only: the intervals and the sinks of scm_learning/data_only.py (one factor: the joint next state;
    a row is the pair (s, a), its weight the product over the state's and the action's value codes)."""
    counts = flat_counts(dataset)  # [S, K, S']
    S, K, _ = counts.shape
    if data_only:
        rows_w = rules.code_weight(dataset.B)[:, None] * rules.code_weight(dataset.actions)[None, :]  # [S, K]
        L, U = rules.intervals(counts.reshape(S * K, S), rows_w.reshape(-1), rules.code_weight(dataset.B), delta)
        L, U = L.reshape(S, K, S), U.reshape(S, K, S)
        seen = counts.sum(-1) > 0
        masks, extra = rules.description(dataset, ~seen)
        return IntervalMDP(L, U, rules.model_reward(dataset), info=seen_info(seen) | extra, seen=seen,
                           data_only=masks)  # fmt: skip
    sets = interval_set_from_counts(
        torch.tensor(counts.reshape(S * K, S)), delta=delta, method="clopper_pearson", sampling="trajectory"
    )
    L, U = (b.double().numpy().reshape(S, K, S) for b in sets.bounds())
    unseen = counts.sum(-1) == 0
    L[unseen], U[unseen] = 0.0, 1.0
    return IntervalMDP(L, U, dataset.reward, info=seen_info(~unseen), seen=~unseen)
