"""PCMCI+ (Runge, UAI 2020, https://arxiv.org/abs/2003.03685) with Tigramite (step S2).

Each episode is one time series over the slice-t columns (m*, a*); the last row of an episode holds
the final state and missing actions. Settings: G^2 test (on sparse tables, fewer than 10 rows per cell,
the p-value comes from permutations within the strata: the chi-square one is unreliable there,
citests.py), tau_max = 1. Link assumptions: only the time
order. Every variable at t may depend on anything at t-1; within a step, a state may cause the action
(the behaviour policy reads the state), an action never causes a state of the same step, and two state
variables may be linked either way (a same-step edge: in the 2-slice model, x' -> y'). The output is a
class of graphs (Candidates); one DAG is chosen in step S3 (selection/). Same-step links can be 'x-x'
(conflicting); then no orientation avoids every new collider, so only acyclicity is required
(keep_vstructures=False)."""

import numpy as np
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.pcmci import PCMCI

from scm_to_mdp.citests import enough_rows, g2_permutation_pvalue
from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset

MISSING = -99.0


class FastGsquared(Gsquared):
    """Tigramite's G^2 test, vectorised, with a permutation p-value on sparse tables
    (citests.enough_rows, citests.g2_permutation_pvalue): the same statistic and the same degrees of freedom
    (Gsquared.get_dependence_measure), but Tigramite loops in Python over every cell of the
    conditioning variables' joint table, empty cells included, whose number grows exponentially with
    the conditioning set; here only the observed strata are tabulated, with one bincount. A test
    with several X or Y variables goes to Tigramite's code."""

    def get_dependence_measure(self, array, xyz, data_type=None):
        if (xyz == 0).sum() != 1 or (xyz == 1).sum() != 1 or self.n_symbs is not None:
            return super().get_dependence_measure(array, xyz, data_type)
        x_levels, x = np.unique(array[xyz == 0][0], return_inverse=True)
        y_levels, y = np.unique(array[xyz == 1][0], return_inverse=True)
        z_rows = array[xyz == 2].T
        if z_rows.shape[1]:
            _, z = np.unique(z_rows, axis=0, return_inverse=True)
        else:
            z = np.zeros(len(x), dtype=np.int64)
        cx, cy, nz = len(x_levels), len(y_levels), int(z.max()) + 1
        table = np.bincount((z * cy + y) * cx + x, minlength=nz * cy * cx).reshape(nz, cy, cx).astype(float)
        rows, cols = table.sum(2, keepdims=True), table.sum(1, keepdims=True)
        total = table.sum((1, 2), keepdims=True)
        expected = rows * cols / np.maximum(total, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gsquare = 2 * np.where(table > 0, table * np.log(table / expected), 0.0).sum()
        zero_rows = (rows[..., 0] == 0).sum(1)  # [nz] all-zero rows of each stratum's table
        zero_cols = (cols[:, 0, :] == 0).sum(1)
        dof = int(((cy - 1 - zero_rows) * (cx - 1 - zero_cols)).sum())
        self._temp_dof = max(dof, 1)
        cells = int(((cy - zero_rows) * (cx - zero_cols)).sum())  # the cells of the tables used
        if not enough_rows(len(x), cells):  # sparse tables: the chi-square p-value is unreliable,
            self._p_value = g2_permutation_pvalue(x, y, z)  # use the permutation one (citests.py)
        return gsquare

    def get_analytic_significance(self, value, T, dim, xyz):
        """The chi-square p-value, or the permutation p-value of a test on sparse tables."""
        p_value = getattr(self, "_p_value", None)
        if p_value is None:
            return super().get_analytic_significance(value, T, dim, xyz)
        self._p_value = None
        del self._temp_dof
        return p_value


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
    for j in states:
        for i in states:
            if i != j:
                links[j][(i, 0)] = "o?o"  # within a step: two states, either direction
    return links


def mixed_graph(graph, dataset: Dataset):
    """Tigramite's graph as (directed, undirected) edges over the table columns. A lag-1 link x -> y
    becomes x -> y'; a same-step link between two state variables becomes one between x' and y' (the
    process is stationary: the link at t is the link at t + 1); other same-step links (state - action)
    stay among the slice-t columns ('o-o', 'x-x' = undirected)."""
    names = dataset.inputs
    states = set(dataset.state_vars)
    directed, undirected = set(), set()
    for i, x in enumerate(names):
        for j, y in enumerate(names):
            if graph[i, j, 1] == "-->":
                directed.add((x, y + NEXT_MARK))
            if i < j:
                a, b = (x + NEXT_MARK, y + NEXT_MARK) if x in states and y in states else (x, y)
                mark = graph[i, j, 0]
                if mark == "-->":
                    directed.add((a, b))
                elif mark == "<--":
                    directed.add((b, a))
                elif mark in ("o-o", "x-x"):
                    undirected.add(frozenset((a, b)))
    return directed, undirected


def fit(dataset: Dataset, alpha: float = 0.01) -> Candidates:
    data = DataFrame(
        episodes(dataset), analysis_mode="multiple", var_names=dataset.inputs, missing_flag=MISSING
    )
    pcmci = PCMCI(dataframe=data, cond_ind_test=FastGsquared(significance="analytic"), verbosity=0)
    result = pcmci.run_pcmciplus(
        tau_min=0, tau_max=1, pc_alpha=alpha, link_assumptions=link_assumptions(dataset)
    )
    directed, undirected = mixed_graph(result["graph"], dataset)
    return Candidates("graph_class", frozenset(directed), frozenset(undirected), keep_vstructures=False)
