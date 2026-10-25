"""DiFaRL: Ehrlichman, Tang, Dykstra, Makar, "Causal Discovery for Efficient Offline RL with Factored
Action Spaces", CLeaR 2026, PMLR 323:1424-1449 (code: https://github.com/cehr123/DiFaRL).

Two steps, model-free, no guarantee. The sub-actions are the action variables (categorical; a domain
with one action variable has one group, and DiFaRL is then tabular FQI):
1. groups: sub-actions a_i and a_j share a group when A_i is NOT independent of A_j given
   (S_t, S_{t+1}) (their Prop. 1). Tested here with a G^2 test at p = 0.005 (the threshold of their
   Sec. 5.1) instead of KCIT, as they allow for discrete data.
2. fqi: linearly decomposed fitted Q iteration, Q(s, a) = sum_k Q_k(s, a_{g_k}) (their Eq. 1), tabular,
   at most 30 iterations (their Sec. 6; they report the best iteration, chosen with the simulator,
   which offline data cannot do: we keep the last one). Its "promise" is max_a Q(s, a).
   With more than one group the least-squares fit can overshoot its targets, and FQI can diverge if
   run for long (seen on Sepsis: values grow ~x1.2 every 10 iterations); the cap keeps it bounded.
"""

import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.stats import chi2

from scm_to_mdp.core import Dataset, Solution, encode


def g2_pvalue(x: np.ndarray, y: np.ndarray, strata: np.ndarray, cx: int = 2, cy: int = 2) -> float:
    """G^2 test of x (values 0..cx-1) independent of y (0..cy-1) given the discrete strata (stratified
    cx x cy tables)."""
    ids, z = np.unique(strata, return_inverse=True)
    cells = cx * cy
    table = np.bincount(z * cells + x * cy + y, minlength=len(ids) * cells)
    table = table.reshape(len(ids), cx, cy).astype(float)
    total = table.sum((1, 2), keepdims=True)
    expected = table.sum(2, keepdims=True) * table.sum(1, keepdims=True) / np.maximum(total, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 2 * np.where(table > 0, table * np.log(table / expected), 0.0).sum()
    rows_used = (table.sum(2) > 0).sum(1)  # degrees of freedom: only values seen in each stratum
    cols_used = (table.sum(1) > 0).sum(1)
    dof = int(((rows_used - 1) * (cols_used - 1)).clip(min=0).sum())
    return 1.0 if dof == 0 else float(chi2.sf(g, dof))


def groups(dataset: Dataset, p_threshold: float = 0.005) -> list[list[int]]:
    """Groups of sub-actions (indices into dataset.action_vars), by union-find over dependent pairs."""
    df = dataset.df
    both = np.hstack([df[dataset.state_vars].values, df[dataset.next_vars].values])
    strata = encode(both, dataset.state_cards * 2)  # one stratum per (s, s')
    actions = dataset.action_vars
    root = list(range(len(actions)))

    def find(i):
        while root[i] != i:
            i = root[i]
        return i

    for i, j in itertools.combinations(range(len(actions)), 2):
        x, y = df[actions[i]].values, df[actions[j]].values
        if g2_pvalue(x, y, strata, dataset.cards[actions[i]], dataset.cards[actions[j]]) < p_threshold:
            root[find(i)] = find(j)  # dependent: same group
    found: dict[int, list[int]] = {}
    for i in range(len(actions)):
        found.setdefault(find(i), []).append(i)
    return list(found.values())


def fqi(
    dataset: Dataset, action_groups: list[list[int]], gamma: float, max_iter: int = 30, tol: float = 1e-6
) -> Solution:
    """Tabular decomposed FQI. One change from their Eq. 1: the max over next actions is taken only
    over the actions seen in that state in the data (all actions in a state never seen). Without it,
    an action never logged in a state gets a value assembled from the other actions' terms, which can
    exceed 1 / (1 - gamma) times the reward, and FQI diverges (offline extrapolation error)."""
    df = dataset.df
    s = dataset.state_index(df[dataset.state_vars].values)
    s_next = dataset.state_index(df[dataset.next_vars].values)
    A = df[dataset.action_vars].values
    k_data = dataset.action_index()
    S, K = len(dataset.B), len(dataset.actions)

    # one tabular block of parameters per group: theta_g[s, value of the group's sub-actions]
    group_cards = [[dataset.cards[dataset.action_vars[j]] for j in g] for g in action_groups]
    sizes = [int(np.prod(c)) for c in group_cards]
    offsets = np.concatenate([[0], np.cumsum([S * size for size in sizes])[:-1]])

    def features(states, action_matrix):  # [rows, groups] index of the active parameter of every group
        cols = [
            off + states * size + encode(action_matrix[:, g], cards)
            for g, cards, off, size in zip(action_groups, group_cards, offsets, sizes, strict=True)
        ]
        return np.stack(cols, 1)

    n_params = int(offsets[-1] + S * sizes[-1])
    index = features(s, A)
    X = csr_matrix(
        (np.ones(index.size), (np.repeat(np.arange(len(s)), len(action_groups)), index.ravel())),
        shape=(len(s), n_params),
    )
    every = features(np.repeat(np.arange(S), K), np.tile(dataset.actions, (S, 1)))  # [S * K, groups]
    reward = dataset.reward
    r = reward.R[s, k_data] + reward.r_next[s_next]
    goes_on = ~reward.terminal[s_next]  # nothing follows a terminal state
    seen = np.zeros((S, K), dtype=bool)
    seen[s, k_data] = True
    seen[~seen.any(1)] = True  # a state never seen: every action allowed

    def q_table(theta):
        return np.where(seen, theta[every].sum(1).reshape(S, K), -np.inf)

    theta = np.zeros(n_params)
    for _ in range(max_iter):
        y = r + gamma * goes_on * q_table(theta).max(1)[s_next]
        # small ridge term: parameters the data never pin down stay near 0
        theta_new = lsqr(X, y, damp=1e-3, atol=1e-10, btol=1e-10)[0]
        converged = np.abs(theta_new - theta).max() < tol
        theta = theta_new
        if converged:
            break
    Q = q_table(theta)
    return Solution(Q.max(1), Q.argmax(1))
