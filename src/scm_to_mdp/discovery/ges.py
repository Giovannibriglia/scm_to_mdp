"""GES (Chickering, JMLR 2002) with pgmpy and the discrete BIC score (step S2).

Runs on all columns (x*, a*, x*') with no background knowledge, so it does not know the time order.
It returns a CPDAG (Candidates with keep_vstructures=True: an orientation may not create a new
v-structure); one DAG is chosen in step S3 (selection/). The time order is used only at the end: the
parents of x_i' are its slice-t neighbours in the chosen DAG.
"""

from pgmpy.estimators import GES

from scm_to_mdp.core import Candidates, Dataset


def fit(dataset: Dataset) -> Candidates:
    table = dataset.df.reset_index(drop=True).astype("category")
    cpdag = GES(table).estimate(scoring_method="bic-d")
    directed = {tuple(e) for e in cpdag.directed_edges}
    undirected = {frozenset(e) for e in cpdag.undirected_edges}
    return Candidates("graph_class", frozenset(directed), frozenset(undirected), keep_vstructures=True)
