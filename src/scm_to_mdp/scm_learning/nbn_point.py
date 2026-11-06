"""Per-factor point estimates with NBN (step S4, no guarantee): the model of the "X -> MDP" methods.

For the chosen parents of every next-state variable, NBN counts the data (HypothesisSet.counts, the
counts behind the intervals of nbn_intervals.py) and every parent row gets the posterior mean under a
symmetric Dirichlet(alpha) prior,

    P(x_i' = v | parent row) = (count(v, row) + alpha) / (count(row) + alpha * card_i),

the estimate of NBN's SmoothedEmpiricalCategorical / CategoricalTableMechanism(alpha) (alpha = 1:
Laplace's rule; a row never seen is uniform; tests/test_nbn_point.py checks the numbers against
NBN's mechanism). The result is a FactorBounds with lo == hi: compiled (compilation/product_bounds.py)
it is an exact factored MDP, solved by value iteration. It is a certainty-equivalence model: the
value it promises is an estimate, not a bound.
"""

import numpy as np
from nbn.structure import HypothesisSet

from scm_to_mdp.core import Dataset, FactorBounds, Graph
from scm_to_mdp.data import slice_inputs
from scm_to_mdp.scm_learning.families import family, rows, same_step, seen_info, seen_pairs


def learn(dataset: Dataset, graph: Graph, alpha: float = 1.0) -> FactorBounds:
    """lo = hi [S, K, n, C]: the Dirichlet(alpha) posterior mean of P(x_i' = v | parents) at every
    state s and action k (a factor with same-step parents: FactorBounds.conditional, and lo = hi the
    hull over their values)."""
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, not {alpha}")
    pairs = list(zip(dataset.state_vars, dataset.next_vars, strict=True))
    families = {v: family(dataset, graph.parents[v]) for v, _ in pairs}
    hs = HypothesisSet(dataset.variables, {v_next: [families[v]] for v, v_next in pairs})
    counts = hs.counts(dataset.tensors())
    X = slice_inputs(dataset)  # [S, K, n_inputs]
    S, K = X.shape[:2]
    C = max(dataset.state_cards)
    lo, hi, conditional = np.zeros((S, K, len(pairs), C)), np.zeros((S, K, len(pairs), C)), {}
    for i, (v, v_next) in enumerate(pairs):
        parents, card = families[v], dataset.cards[v]
        table = counts[(v_next, parents)].double().numpy()  # [parent rows, card], rows in C order
        n = table.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            prob = np.where(n + alpha * card > 0, (table + alpha) / (n + alpha * card), 1.0 / card)
        row = rows(dataset, parents, X)  # [S, K, R]
        P_i = np.zeros((*row.shape, C))
        P_i[..., :card] = prob[row]
        lo[:, :, i], hi[:, :, i] = P_i.min(2), P_i.max(2)  # = the estimate without same-step parents
        if row.shape[2] > 1:
            conditional[i] = (same_step(dataset, parents), P_i, P_i.copy())
    pairs_seen = seen_pairs(dataset, {v: [p] for v, p in families.items()}, counts, X)
    info = {"alpha": alpha, **seen_info(pairs_seen)}
    return FactorBounds(
        lo, hi, dataset.state_cards, dataset.B, dataset.reward, info=info, conditional=conditional,
        seen=pairs_seen,
    )
