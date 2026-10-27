"""The full factored graph (step S2, no discovery): every next-state variable x_i' gets every slice-t
input as a parent, all state variables and all action variables, and no same-step parent.

    pa(x_i') = (x_1, ..., x_n, a_1, ..., a_m)   for every i

It keeps the factorisation of the next state over its variables (x_i' independent given the slice-t
inputs) and assumes nothing else, so it contains the true graph whenever the truth has no same-step
edges. It is the "no causal discovery" arm: recall 1 by construction, precision = true edges / all
edges. Each factor then has one row per (s, a) (as many rows as the flat model has), so S4 learns
n tables of S x K rows instead of one of S x K x S cells. Select it with the rule "given".
"""

from scm_to_mdp.core import Candidates, Dataset
from scm_to_mdp.discovery import oracle


def parents(dataset: Dataset) -> dict[str, tuple[str, ...]]:
    """Every slice-t input (state and action columns, in table order) for every state variable."""
    return {v: tuple(dataset.inputs) for v in dataset.state_vars}


def fit(dataset: Dataset) -> Candidates:
    return oracle.fit(dataset, parents(dataset))
