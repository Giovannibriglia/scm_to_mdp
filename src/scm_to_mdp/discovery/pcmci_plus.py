"""PCMCI+ (Runge, UAI 2020, https://arxiv.org/abs/2003.03685) with Tigramite (step S2).

Each episode is one time series over the slice-t columns (m*, a*); the last row of an episode holds
the final state and missing actions. Settings: G^2 test, tau_max = 1. Link assumptions (the
pipeline's): every variable at t may depend on anything at t-1; within a step only state -> action
(the behaviour policy reads the state). The output is a class of graphs: one DAG is chosen with
cpdag_to_dag.choose_dag (every DAG enumerated, best BIC). Same-step links can be 'x-x'
(conflicting); then no orientation avoids every new collider, so only acyclicity is required.
"""

import numpy as np
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.pcmci import PCMCI

from scm_to_mdp.core import NEXT_MARK, Dataset, Graph
from scm_to_mdp.discovery.cpdag_to_dag import choose_dag

MISSING = -99.0


def episodes(dataset: Dataset) -> dict[int, np.ndarray]:
    """{episode: [T + 1, n_inputs]} time series; the table index must be (episode, t)."""
    df = dataset.df
    series = {}
    for e, rows in df.groupby(level="episode", sort=True):
        rows = rows.sort_index(level="t")
        last = np.concatenate(
            [rows[dataset.next_vars].values[-1], np.full(len(dataset.action_vars), MISSING)]
        )
        series[int(e)] = np.vstack([rows[dataset.inputs].values.astype(float), last])
    return series


def link_assumptions(dataset: Dataset) -> dict:
    names = dataset.inputs
    states = [names.index(v) for v in dataset.state_vars]
    actions = [names.index(a) for a in dataset.action_vars]
    links = {j: {(i, -1): "-?>" for i in range(len(names))} for j in range(len(names))}
    for j in states:
        for i in actions:
            links[j][(i, 0)] = "<?-"  # within a step: an action never causes a state
    for j in actions:
        for i in states:
            links[j][(i, 0)] = "-?>"  # within a step: the state may cause the action (the policy)
    return links


def mixed_graph(graph, dataset: Dataset):
    """Tigramite's graph as (directed, undirected) edges over the table columns. A lag-1 link x -> y
    becomes x -> y'; same-step links stay among slice-t columns ('o-o', 'x-x' = undirected)."""
    names = dataset.inputs
    directed, undirected = set(), set()
    for i, x in enumerate(names):
        for j, y in enumerate(names):
            if graph[i, j, 1] == "-->":
                directed.add((x, y + NEXT_MARK))
            if i < j:
                mark = graph[i, j, 0]
                if mark == "-->":
                    directed.add((x, y))
                elif mark == "<--":
                    directed.add((y, x))
                elif mark in ("o-o", "x-x"):
                    undirected.add(frozenset((x, y)))
    return directed, undirected


def fit(dataset: Dataset, alpha: float = 0.01) -> Graph:
    data = DataFrame(
        episodes(dataset), analysis_mode="multiple", var_names=dataset.inputs, missing_flag=MISSING
    )
    pcmci = PCMCI(dataframe=data, cond_ind_test=Gsquared(significance="analytic"), verbosity=0)
    result = pcmci.run_pcmciplus(
        tau_min=0, tau_max=1, pc_alpha=alpha, link_assumptions=link_assumptions(dataset)
    )
    directed, undirected = mixed_graph(result["graph"], dataset)
    return choose_dag(directed, undirected, dataset, check_vstructures=False)
