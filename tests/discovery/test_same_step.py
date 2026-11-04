"""No same-step edges (Strehl, Diuk, Littman, AAAI 2007, Assumption 1): PCMCI+ and DYNOTEARS with
same_step=False return only slice-t -> next-state edges, and PCMCI+'s class is then a single DAG."""

from scm_to_mdp.core import NEXT_MARK
from scm_to_mdp.discovery import dynotears, pcmci_plus


def _same_step_edges(candidates):
    pairs = list(candidates.directed) + [tuple(e) for e in candidates.undirected]
    return [(a, b) for a, b in pairs if a.endswith(NEXT_MARK) and b.endswith(NEXT_MARK)]


def test_pcmci_link_assumptions_without_same_step_links(dataset):
    names = dataset.inputs
    states = [names.index(v) for v in dataset.state_vars]
    with_links, without = pcmci_plus.link_assumptions(dataset), pcmci_plus.link_assumptions(dataset, False)
    assert any((i, 0) in with_links[j] for j in states for i in states if i != j)
    assert not any((i, 0) in without[j] for j in states for i in states)
    for j in range(len(names)):  # every lagged link is still allowed
        assert all(without[j][(i, -1)] == "-?>" for i in range(len(names)))


def test_pcmci_without_same_step_is_one_dag(dataset):
    c = pcmci_plus.fit(dataset, same_step=False)
    assert not c.undirected and not _same_step_edges(c)


def test_dynotears_without_same_step_has_no_next_to_next_edge(dataset):
    c = dynotears.fit(dataset, same_step=False)
    assert not _same_step_edges(c) and c.report["intra_weights"] == 0
