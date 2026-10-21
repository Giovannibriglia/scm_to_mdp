"""Exact BIC parent search for every next-state variable, under the time order (step S2).

For each m_i' we score every subset of the slice-t inputs (state and action columns) with at most
max_parents elements and keep the best BIC. The time order makes this a separate search per target
(Koller & Friedman, Probabilistic Graphical Models, 2009, Prop. 18.4, p. 810). With N samples BIC
picks at most O(log N) parents, so a bound costs little (de Campos & Ji, JMLR 12:663-689, 2011,
Cor. 3); max_parents=None searches every subset.
"""

import itertools
import math

import numpy as np
from nbn.structure import HypothesisSet

from scm_to_mdp.core import Dataset, Graph


def bic(counts: np.ndarray) -> float:
    """BIC of one family from its count table [parent rows, child values]."""
    n_row = counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lik = np.where(counts > 0, counts * np.log(counts / n_row), 0.0).sum()
    n_params = counts.shape[0] * (counts.shape[1] - 1)
    return log_lik - 0.5 * math.log(counts.sum()) * n_params


def fit(dataset: Dataset, max_parents: int | None = 3) -> Graph:
    inputs = dataset.inputs
    k_max = len(inputs) if max_parents is None else max_parents
    candidates = [c for k in range(k_max + 1) for c in itertools.combinations(inputs, k)]
    hs = HypothesisSet(dataset.variables, {v: candidates for v in dataset.next_vars})
    counts = hs.counts(dataset.tensors())  # NBN counts every (target, parent set) family
    parents = {}
    for v, v_next in zip(dataset.state_vars, dataset.next_vars, strict=True):
        scores = [bic(counts[(v_next, c)].numpy()) for c in candidates]
        parents[v] = candidates[int(np.argmax(scores))]
    return Graph(parents, report={"candidates": len(candidates)})
