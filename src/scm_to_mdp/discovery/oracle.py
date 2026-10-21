"""The true graph, given by the environment (no discovery). Used by the baseline of Schnitzer, Abate,
Parker (structure assumed known) and as the upper reference for the discovery arms."""

from scm_to_mdp.core import Dataset, Graph


def fit(dataset: Dataset, true_parents: dict[str, tuple[str, ...]]) -> Graph:
    return Graph({v: tuple(true_parents[v]) for v in dataset.state_vars})
