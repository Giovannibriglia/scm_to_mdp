"""Per-factor probability intervals with NBN (step S4).

For the chosen parents of every next-state variable, NBN counts the data and builds a Clopper-Pearson
interval for every value of every parent row. The confidence budget delta is split over all
intervals of all trajectories (union bound, sampling="trajectory"), so all intervals hold together
with probability >= 1 - delta. A parent row never seen in the data gets [0, 1]. The guarantee holds
when the parents contain the true parents; with a wrong graph the intervals bound a policy-dependent
average, not the mechanism.
"""

import numpy as np
from nbn.structure import HypothesisSet

from scm_to_mdp.core import Dataset, FactorBounds, Graph, encode
from scm_to_mdp.data import slice_inputs


def learn(dataset: Dataset, graph: Graph, delta: float = 0.05) -> FactorBounds:
    """lo, hi [S, K, n, C]: interval of P(x_i' = v | parents) at every state s and action k."""
    pairs = list(zip(dataset.state_vars, dataset.next_vars, strict=True))
    hs = HypothesisSet(dataset.variables, {v_next: [graph.parents[v]] for v, v_next in pairs})
    sets = hs.factor_sets(
        hs.counts(dataset.tensors()),
        builder="interval",
        delta=delta,
        sampling="trajectory",
        method="clopper_pearson",
    )
    X = slice_inputs(dataset)  # [S, K, n_inputs]
    column = {name: j for j, name in enumerate(dataset.inputs)}
    S, K = X.shape[:2]
    C = max(dataset.state_cards)
    lo, hi = np.zeros((S, K, len(pairs), C)), np.zeros((S, K, len(pairs), C))
    for i, (v, v_next) in enumerate(pairs):
        parents = graph.parents[v]
        lower, upper = sets[(v_next, parents)].bounds()  # [parent rows, card], rows in C order
        if parents:
            row = encode(X[..., [column[p] for p in parents]], [dataset.cards[p] for p in parents])  # [S, K]
        else:
            row = np.zeros((S, K), dtype=np.int64)
        card = dataset.cards[v]
        lo[:, :, i, :card] = lower.double().numpy()[row]
        hi[:, :, i, :card] = upper.double().numpy()[row]
    return FactorBounds(lo, hi, dataset.state_cards, dataset.B, dataset.reward)
