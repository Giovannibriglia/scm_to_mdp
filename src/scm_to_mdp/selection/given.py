"""Selection rule "given" (step S3): the candidates must already be one fully directed graph (the
oracle's true graph); it is taken as it is."""

from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset, Graph


def select(candidates: Candidates, dataset: Dataset) -> Graph:
    if candidates.kind != "graph_class" or candidates.undirected:
        raise ValueError("selection 'given' needs one fully directed graph (e.g. from the oracle)")
    parents = {
        v: tuple(x for x in dataset.inputs if (x, v + NEXT_MARK) in candidates.directed)
        for v in dataset.state_vars
    }
    return Graph(parents, report=dict(candidates.report))
