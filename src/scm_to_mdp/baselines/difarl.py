"""DiFaRL: Ehrlichman, Tang, Dykstra, Makar, "Causal Discovery for Efficient Offline RL with Factored
Action Spaces", CLeaR 2026, PMLR 323:1424-1449 (code: https://github.com/cehr123/DiFaRL).

Two steps, model-free, no guarantee. The sub-actions are the action variables (categorical; a domain
with one action variable has one group, and DiFaRL is then tabular FQI):
1. groups: sub-actions a_i and a_j share a group when A_i is NOT independent of A_j given
   (S_t, S_{t+1}) (their Prop. 1), with groups as connected components (their code). The test is the
   G^2 test at p = 0.005 (the threshold of their Sec. 5.1; the paper allows parametric tests where
   parametric assumptions hold, and our variables are discrete) instead of KCIT, with a permutation
   p-value on sparse tables (citests.g2_pvalue: the chi-square one finds dependences that are not
   there). A next-state variable that copies an action variable (in the data, a function of one
   action variable: Sepsis's treatment flags) is left out of the conditioning set: it is a child of
   that action alone, not a collider, and conditioning on it would make every test accept.
2. fqi: linearly decomposed fitted Q iteration, Q(s, a) = sum_k Q_k(s, a_{g_k}) (their Eq. 1, Sec.
   5.2.1: the linear variant; with a tabular state the attention variant of their Sec. 5.2.2 has
   the same function class), tabular, at most 30 iterations (their Sec. 6; they report the best
   iteration, chosen with the simulator, which offline data cannot do: we keep the last one). The
   target is their code's: a Polyak-averaged target table (tau = 0.2 every iteration) and a double-Q
   target, y = r + gamma * Q_target(s', argmax_a Q(s', a)), the max over every action vector of the
   data. Its "promise" is max_a Q(s, a).
"""

import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from scm_to_mdp.citests import g2_pvalue
from scm_to_mdp.core import NEXT_MARK, Dataset, Solution, encode


def action_copies(dataset: Dataset) -> list[str]:
    """Next-state variables that, in the data, are a function of a single action variable."""
    df = dataset.df
    return [
        v
        for v in dataset.next_vars
        if any((df.groupby(a)[v].nunique() <= 1).all() for a in dataset.action_vars)
    ]


def groups(dataset: Dataset, p_threshold: float = 0.005) -> list[list[int]]:
    """Groups of sub-actions (indices into dataset.action_vars), by union-find over dependent pairs."""
    df = dataset.df
    copies = set(action_copies(dataset))
    given = dataset.state_vars + [v for v in dataset.next_vars if v not in copies]  # (S_t, S_{t+1})
    strata = encode(df[given].values, [dataset.cards[v.removesuffix(NEXT_MARK)] for v in given])
    actions = dataset.action_vars
    root = list(range(len(actions)))

    def find(i):
        while root[i] != i:
            i = root[i]
        return i

    for i, j in itertools.combinations(range(len(actions)), 2):
        x, y = df[actions[i]].values, df[actions[j]].values
        if g2_pvalue(x, y, strata) < p_threshold:
            root[find(i)] = find(j)  # dependent: same group
    found: dict[int, list[int]] = {}
    for i in range(len(actions)):
        found.setdefault(find(i), []).append(i)
    return list(found.values())


def fqi(
    dataset: Dataset,
    action_groups: list[list[int]],
    gamma: float,
    max_iter: int = 30,
    tol: float = 1e-6,
    tau: float = 0.2,
    seen_only: bool = False,
) -> Solution:
    """Tabular decomposed FQI with their code's target: theta_target <- (1 - tau) theta_target + tau
    theta after every fit, and y = r + gamma * Q_target(s', argmax_a Q(s', a)) (double Q). seen_only:
    take the max (and the final policy) only over the actions seen in each state (all actions in a
    state never seen); not in their method, off by default."""
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
    allowed = np.ones((S, K), dtype=bool)  # the actions the max may take
    if seen_only:
        allowed[:] = False
        allowed[s, k_data] = True
        allowed[~allowed.any(1)] = True  # a state never seen: every action allowed

    def q_table(theta):
        return np.where(allowed, theta[every].sum(1).reshape(S, K), -np.inf)

    theta = theta_target = np.zeros(n_params)
    for _ in range(max_iter):
        greedy = q_table(theta).argmax(1)  # double Q: the online table picks, the target table values
        next_value = q_table(theta_target)[np.arange(S), greedy]
        y = r + gamma * goes_on * next_value[s_next]
        # small ridge term: parameters the data never pin down stay near 0
        theta_new = lsqr(X, y, damp=1e-3, atol=1e-10, btol=1e-10)[0]
        converged = np.abs(theta_new - theta).max() < tol
        theta = theta_new
        theta_target = (1 - tau) * theta_target + tau * theta
        if converged:
            break
    Q = q_table(theta)
    return Solution(Q.max(1), Q.argmax(1))
