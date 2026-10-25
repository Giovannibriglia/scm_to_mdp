"""Exact BIC parent search for every next-state variable, under the time order (step S2).

For each x_i' we score every subset of the candidate parents with at most max_parents elements: the
slice-t inputs (state and action columns) and the other next-state variables x_j' (same-step edges:
the search is not constrained beyond the time order). fit returns every candidate set with its BIC;
the selection rule (S3, selection/bic_best.py) keeps the best combination whose same-step edges form
a DAG. Without same-step parents the search is separate per target, since no choice of slice-t
parents can create a cycle (Koller & Friedman, Probabilistic Graphical Models, 2009, Sec. 18.4.2,
Prop. 18.4, p. 810); with them, bic_best finds the best acyclic combination exactly (dynamic
programming over the sets of next-state variables). With N samples BIC
picks at most O(log N) parents, so a bound costs little (de Campos & Ji, JMLR 12:663-689, 2011,
Cor. 3); max_parents=None searches every subset.

BIC = log-likelihood - (log N / 2) * Dim, with Dim the number of free parameters (Koller & Friedman, Thm
18.1, pp. 801-802): (number of parent rows) * (number of values - 1) for a table CPD. The penalty here
counts only the rows of a parent table that can occur (our extension: the CPD is defined on the parent rows
that can occur; the book does not treat impossible rows). Action columns are often one-hot (one action at a
time): of the 2^k combinations of k such columns only k + 1 are actions. Counting the impossible rows
charged BIC for parameters that no data can ever inform and made it drop true action parents (Taxi: the
taxi's square depends on 4 move columns, 400 rows of which 125 can occur). A row can occur if its action
part is the projection of an action of the learner's action set (the action vectors seen in the data,
Dataset.actions); every combination of the state parents counts.
"""

import itertools
import math

import numpy as np
from nbn.structure import HypothesisSet

from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset


def feasible_rows(parents: tuple[str, ...], cards: dict[str, int], action_vars, actions: np.ndarray) -> int:
    """Rows of the table of `parents` that can occur: every combination of the non-action parents
    (state variables at t or t+1), times the number of distinct values the action parents take over
    the action set [K, n_a]."""
    columns = [j for j, a in enumerate(action_vars) if a in parents]
    n = math.prod(cards[p.removesuffix(NEXT_MARK)] for p in parents if p not in action_vars)
    if columns:
        n *= len({tuple(a) for a in actions[:, columns]})
    return n


def bic(counts: np.ndarray, n_rows: int | None = None) -> float:
    """BIC of one family from its count table [parent rows, child values]. n_rows: the rows that can
    occur (default: every row of the table)."""
    n_row = counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lik = np.where(counts > 0, counts * np.log(counts / n_row), 0.0).sum()
    n_params = (counts.shape[0] if n_rows is None else n_rows) * (counts.shape[1] - 1)
    return log_lik - 0.5 * math.log(counts.sum()) * n_params


def fit(dataset: Dataset, max_parents: int | None = 3) -> Candidates:
    pairs = list(zip(dataset.state_vars, dataset.next_vars, strict=True))
    options = {v_next: dataset.inputs + [u for u in dataset.next_vars if u != v_next] for _, v_next in pairs}
    candidates = {
        v_next: [
            c
            for k in range(len(names) + 1 if max_parents is None else max_parents + 1)
            for c in itertools.combinations(names, k)
        ]
        for v_next, names in options.items()
    }
    tensors = dataset.tensors()
    counts = {  # one HypothesisSet per target: NBN refuses a union of candidate edges with a cycle,
        # which same-step candidates of two targets make (x' <- y' and y' <- x')
        v_next: HypothesisSet(dataset.variables, {v_next: sets}).counts(tensors)
        for v_next, sets in candidates.items()
    }
    every = {c for sets in candidates.values() for c in sets}
    rows = {c: feasible_rows(c, dataset.cards, dataset.action_vars, dataset.actions) for c in every}
    scores = {
        v: [(c, bic(counts[v_next][(v_next, c)].numpy(), rows[c])) for c in candidates[v_next]]
        for v, v_next in pairs
    }
    report = {"candidates": max(len(sets) for sets in candidates.values())}
    return Candidates("scored_sets", scores=scores, report=report)
