"""GES (Chickering, JMLR 2002) with pgmpy and the discrete BIC score (step S2).

Runs on all columns (x*, a*, x*') with no background knowledge, so it does not know the time order.
It returns a CPDAG (Candidates with keep_vstructures=True: an orientation may not create a new
v-structure); one DAG is chosen in step S3 (selection/). The time order is used only at the end: the
parents of x_i' are its slice-t neighbours in the chosen DAG.

pgmpy's search loops over Python sets of node names, and when two operators score the same the set
order picks one. The order of a set of strings changes from one Python process to the next (string
hashing is randomised, PYTHONHASHSEED), so the same data gave different graphs. The columns are
therefore passed as integers 0..n-1, whose hashes are the integers themselves: the set order, and the
graph, are the same in every process. The names are put back on the edges.
"""

from pgmpy.estimators import GES

from scm_to_mdp.core import Candidates, Dataset


def fit(dataset: Dataset) -> Candidates:
    table = dataset.df.reset_index(drop=True).astype("category")
    names = list(table.columns)
    table.columns = range(len(names))  # integer node names: a set order that is the same in every process
    cpdag = GES(table).estimate(scoring_method="bic-d")
    directed = {(names[a], names[b]) for a, b in cpdag.directed_edges}
    undirected = {frozenset((names[a], names[b])) for a, b in cpdag.undirected_edges}
    return Candidates("graph_class", frozenset(directed), frozenset(undirected), keep_vstructures=True)
