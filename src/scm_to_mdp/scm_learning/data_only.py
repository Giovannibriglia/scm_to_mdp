"""Data-only S4 (learning option data_only=True): the model uses only the data, no value domains, no support,
no terminal states (and no done / terminated / truncated flags). The reward function is still known and enters
only value iteration.

What the learner may use: the roles of the columns (state, action, next state) and the data. The value domains
(Dataset.cards) are used only as storage for the index of a state; the intervals and the sinks do not depend on
them. The rules, for every factor x_i' (a flat model: one factor, the joint next state):

    a value never seen in the data (in any state or next-state column)   every state holding it is a sink
    a joint state never seen as a source (it may be terminal)            a sink
    a parent row never seen                                              the pair (s, a) sends all its mass
                                                                         to a sink ("doomed")
    a seen row, an observed value                                        Clopper-Pearson interval
    a seen row, a value not observed at that row                         [0, 1] (the residual mass of the
                                                                         simplex: 1 - sum of the observed
                                                                         lower bounds, Ashok et al. 2019)

A sink's value is fixed (solving/stopping.data_only_values): V_min = min(r_min, 0) / (1 - gamma) from the
smallest one-step reward r_min (0 covers an episode that ends), entered with the smallest arrival reward
when the state itself is unknown. A pessimistic value in place of an unknown one keeps the promise a lower
bound (the robust update is monotone): badings_claude, 2026-11-06 (general argument; closest, Ashok et al.,
CAV 2019, Eq. (1) and Lemma 1, where unestimated mass gets the worst value).

Confidence (agarwal_claude, 2026-11-06): the budget is split over a family fixed before the data, the factors,
the candidate parent sets (n_candidates), every possible parent row and value, with the weights
u(v) = 6 / (pi^2 (v + 1)^2) over the value codes v = 0, 1, 2, ... (they sum to 1, so no domain size is needed;
a row's weight is the product over its parents), and over the visit count n as delta / (n (n + 1)) (NBN's
anytime split for trajectory data). On the event that every interval of the family holds, the intervals of the
data-chosen rows and values hold too, and the unobserved mass is at most 1 - sum of the observed lower bounds.
The level holds for parent sets that contain the true parents (as for every S4 here).
"""

import math

import numpy as np
import torch
from nbn.uncertainty.binomial import anytime_delta, binomial_interval

from scm_to_mdp.core import Dataset, Reward


def value_weight(v: np.ndarray) -> np.ndarray:
    """u(v) = 6 / (pi^2 (v + 1)^2): a weight per value code, summing to 1 over v = 0, 1, 2, ..."""
    return 6.0 / (math.pi**2 * (np.asarray(v, dtype=float) + 1.0) ** 2)


def row_values(parents: tuple[str, ...], dataset: Dataset) -> np.ndarray:
    """[R, len(parents)] the value codes of every row of a table counted over parents (C order)."""
    cards = [dataset.cards[p] for p in parents]
    if not parents:
        return np.zeros((1, 0), dtype=np.int64)
    return np.stack(np.unravel_index(np.arange(int(np.prod(cards))), cards), axis=1)


def code_weight(codes: np.ndarray) -> np.ndarray:
    """[R] the weight of every tuple of value codes [R, P]: the product of u over its entries (1 if P = 0)."""
    codes = np.asarray(codes)
    return value_weight(codes).prod(1) if codes.shape[1] else np.ones(len(codes))


def intervals(counts: np.ndarray, row_w: np.ndarray, value_w: np.ndarray, delta: float):
    """(lo, hi) [R, C] for a count table counts [R, C]: Clopper-Pearson for the values observed at a row, at
    level delta * row_w[r] * value_w[v], anytime in the row's count; [0, 1] for the values not observed there and
    for unseen rows (module docstring). row_w [R], value_w [C]: code_weight of the rows' and values' codes."""
    level = delta * np.asarray(row_w)[:, None] * np.asarray(value_w)[None, :]  # [R, C]
    n = counts.sum(1, keepdims=True)
    k = torch.tensor(counts, dtype=torch.float64)
    trials = torch.tensor(np.broadcast_to(n, counts.shape).copy(), dtype=torch.float64)
    at = anytime_delta(torch.tensor(level, dtype=torch.float64), trials)
    lower, upper = binomial_interval(k, trials.clamp_min(1), at, method="clopper_pearson")
    lo, hi = lower.numpy().astype(float), upper.numpy().astype(float)
    observed = counts > 0
    lo, hi = np.where(observed, lo, 0.0), np.where(observed, hi, 1.0)  # unobserved at the row: free
    lo[n[:, 0] == 0], hi[n[:, 0] == 0] = 0.0, 1.0
    return lo, hi


def observed_values(dataset: Dataset) -> dict:
    """var -> [card] bool: the values of every state variable seen in the data (as state or next state)."""
    out = {}
    for v, v_next in zip(dataset.state_vars, dataset.next_vars, strict=True):
        seen = np.zeros(dataset.cards[v], dtype=bool)
        seen[np.unique(np.concatenate([dataset.df[v].values, dataset.df[v_next].values]).astype(int))] = True
        out[v] = seen
    return out


def sinks(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """([S] unknown, [S] never_source): states holding a value never seen, and joint states never seen as the
    source of a transition (possibly terminal). Both are sinks."""
    seen = observed_values(dataset)
    unknown = np.zeros(len(dataset.B), dtype=bool)
    for j, v in enumerate(dataset.state_vars):
        unknown |= ~seen[v][dataset.B[:, j]]
    source = np.zeros(len(dataset.B), dtype=bool)
    source[np.unique(dataset.state_index(dataset.df[dataset.state_vars].values))] = True
    return unknown, ~source & ~unknown


def model_reward(dataset: Dataset) -> Reward:
    """The known reward without terminal states (they are not given)."""
    r = dataset.reward
    return Reward(r.R, r.r_next, np.zeros_like(r.terminal, dtype=bool))


def description(dataset: Dataset, doomed: np.ndarray) -> dict:
    """The model's data-only masks (FactorBounds.data_only / IntervalMDP.data_only) and their row columns."""
    unknown, never_source = sinks(dataset)
    masks = {"unknown": unknown, "never_source": never_source, "doomed": doomed}
    info = {"sink_states": float((unknown | never_source).mean()),
            "doomed_pairs": float(doomed[~(unknown | never_source)].mean()) if (~(unknown | never_source)).any()
            else 1.0}  # fmt: skip
    return masks, info
