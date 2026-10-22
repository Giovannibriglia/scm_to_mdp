"""The true graph, given by the environment (no discovery). Used by the baseline of Schnitzer, Abate,
Parker (structure assumed known) and as the upper reference for the discovery arms. Select it with
the rule "given" (selection/given.py)."""

from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset


def fit(dataset: Dataset, true_parents: dict[str, tuple[str, ...]]) -> Candidates:
    edges = {(p, v + NEXT_MARK) for v in dataset.state_vars for p in true_parents[v]}
    return Candidates("graph_class", frozenset(edges))
