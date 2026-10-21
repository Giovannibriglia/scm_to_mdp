"""GES (Chickering, JMLR 2002) with pgmpy and the discrete BIC score (step S2).

Runs on all columns (m*, a*, m*') with no background knowledge, so it does not know the time order.
It returns a CPDAG; one DAG is chosen with cpdag_to_dag.choose_dag (every DAG of the class
enumerated, best BIC, no new v-structures). The time order is used only at the end: the parents of
m_i' are its slice-t neighbours in the chosen DAG.
"""

from pgmpy.estimators import GES

from scm_to_mdp.core import Dataset, Graph
from scm_to_mdp.discovery.cpdag_to_dag import choose_dag


def fit(dataset: Dataset) -> Graph:
    table = dataset.df.reset_index(drop=True).astype("category")
    cpdag = GES(table).estimate(scoring_method="bic-d")
    directed = {tuple(e) for e in cpdag.directed_edges}
    undirected = {frozenset(e) for e in cpdag.undirected_edges}
    return choose_dag(directed, undirected, dataset, check_vstructures=True)
