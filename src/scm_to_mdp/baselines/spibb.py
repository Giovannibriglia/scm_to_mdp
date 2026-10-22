"""Safe policy improvement with baseline bootstrapping (SPIBB): flat and with structure learning.

    flat SPIBB   Pi_b-SPIBB of Laroche, Trichelair, Tachet des Combes, "Safe Policy Improvement with
                 Baseline Bootstrapping", ICML 2019 (https://arxiv.org/abs/1712.06924): the MLE flat
                 MDP; a pair (s, a) seen fewer than m times is "bootstrapped".
    SL-SPIBB     Structure Learning Pi_b-SPIBB of Simao & Spaan, "Structure Learning for Safe Policy
                 Improvement", IJCAI 2019 (https://doi.org/10.24963/ijcai.2019/479), Alg. 3, with the SL
                 subalgorithm of Strehl, Diuk, Littman (AAAI 2007) described there (Sec. 2.3, Eqs. 4-5):
                 for every state variable X_i and action a, the candidate parent sets are the subsets
                 of d state variables; at state s a candidate D_i is accepted when every pair union
                 D_i u D_j has at least m samples in s's configuration (Eq. 4) and the estimates
                 P(X_i' | s[D_i u D_j]) of all j differ by at most eps in L1 (Eq. 5); it returns
                 P(X_i' | s[D_i u D_j]) for the first j. (s, a) is bootstrapped when some factor
                 finds no candidate; otherwise P(s' | s, a) = prod_i P_i(s'_i).

Both then compute the Pi_b-SPIBB policy (Laroche et al., Alg. 1): policy iteration in the estimated
MDP, where in every state the policy keeps the behaviour policy's probability on bootstrapped
actions and puts the rest of the mass on the best non-bootstrapped action. Bootstrapped pairs have
no estimated transition (an absorbing row, as in Simao & Spaan, Alg. 3, line 9). The result is a
stochastic policy; its value in the estimated MDP is reported as the promise, which is an estimate
(the guarantee of SPIBB is an improvement over the behaviour policy w.h.p., not a value bound).

The behaviour policy is estimated from the data (counts n(s, a) / n(s); uniform in a state never
seen), as in Simao, Laroche, Tachet des Combes, "Safe Policy Improvement with an Estimated Baseline
Policy", AAMAS 2020 (https://arxiv.org/abs/1909.05236): the learner never sees the true logger,
which may read hidden variables.

Parameters (Simao & Spaan, Table 1, SysAdmin column): flat SPIBB m = 100; SL-SPIBB m = 10,
eps = 0.2. d is the maximum number of state-variable parents (per environment, as max_parents for
BIC). Our only change: actions are the action vectors seen in the data (K of them).
"""

import itertools
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.core import Dataset, Reward, Solution, encode
from scm_to_mdp.solving import value_iteration


@dataclass(frozen=True)
class SpibbModel:
    """The estimated MDP of SPIBB: transitions (zero rows for bootstrapped pairs), which pairs are
    bootstrapped, and the estimated behaviour policy."""

    P: np.ndarray  # [S, K, S]
    bootstrapped: np.ndarray  # [S, K] bool
    behaviour: np.ndarray  # [S, K] estimated pi_b
    reward: Reward


def _indices(dataset: Dataset):
    df = dataset.df
    s = dataset.state_index(df[dataset.state_vars].values)
    s_next = dataset.state_index(df[dataset.next_vars].values)
    return s, dataset.action_index(), s_next


def estimated_behaviour(dataset: Dataset) -> np.ndarray:
    """[S, K] n(s, a) / n(s); uniform in states the data never visit."""
    s, k, _ = _indices(dataset)
    S, K = len(dataset.B), len(dataset.actions)
    counts = np.zeros((S, K))
    np.add.at(counts, (s, k), 1)
    n = counts.sum(1, keepdims=True)
    return np.where(n > 0, counts / np.maximum(n, 1), 1.0 / K)


def flat_model(dataset: Dataset, m: int = 100) -> SpibbModel:
    """MLE flat MDP; pairs seen fewer than m times are bootstrapped."""
    s, k, s_next = _indices(dataset)
    S, K = len(dataset.B), len(dataset.actions)
    counts = np.zeros((S, K, S))
    np.add.at(counts, (s, k, s_next), 1)
    n = counts.sum(2)
    boot = n < m
    P = np.where(boot[:, :, None], 0.0, counts / np.maximum(n, 1)[:, :, None])
    return SpibbModel(P, boot, estimated_behaviour(dataset), dataset.reward)


def _l1_diameter(points: np.ndarray) -> np.ndarray:
    """points [J, S, c] -> [S] max over pairs j, k of ||p_j - p_k||_1, exactly, through the sign
    vectors: ||x||_1 = max over sigma in {-1, 1}^c of sigma . x."""
    c = points.shape[-1]
    best = np.zeros(points.shape[1])
    for signs in itertools.product((-1.0, 1.0), repeat=c - 1):  # sigma and -sigma give the same value
        proj = points @ np.array((1.0, *signs))
        best = np.maximum(best, proj.max(0) - proj.min(0))
    return best


def sl_factor(states, rows, values, cards, card, d, m, eps):
    """The SL subalgorithm for one (X_i, a): states [S, n] every state, rows [N, n] the states of the
    transitions with action a, values [N] their next value of X_i. Returns (P [S, card], known [S])."""
    n_vars, S = states.shape[1], len(states)
    cands = list(itertools.combinations(range(n_vars), min(d, n_vars)))

    def estimate(U):  # counts and conditional distribution of X_i' given s[U], at every state
        U = list(U)
        cu = [cards[u] for u in U]
        table = np.zeros((int(np.prod(cu)), card))
        np.add.at(table, (encode(rows[:, U], cu), values), 1)
        at = table[encode(states[:, U], cu)]
        n = at.sum(1)
        return n, at / np.maximum(n, 1)[:, None]

    if len(cands) == 1:  # d >= number of variables: one candidate, the whole state
        n, p = estimate(cands[0])
        return p, n >= m
    pair = {}
    for i, j in itertools.combinations(range(len(cands)), 2):
        pair[i, j] = estimate(sorted(set(cands[i]) | set(cands[j])))
    P, known = np.zeros((S, card)), np.zeros(S, dtype=bool)
    for i in range(len(cands)):
        others = [pair[min(i, j), max(i, j)] for j in range(len(cands)) if j != i]
        n = np.stack([o[0] for o in others])  # [C - 1, S]
        p = np.stack([o[1] for o in others])  # [C - 1, S, card]
        accept = (n >= m).all(0) & (_l1_diameter(p) <= eps) & ~known  # Eqs. 4 and 5
        P[accept], known[accept] = p[0][accept], True
        if known.all():
            break
    return P, known


def sl_model(dataset: Dataset, d: int = 3, m: int = 10, eps: float = 0.2) -> SpibbModel:
    """Simao & Spaan's Structure Learning Pi_b-SPIBB model (Alg. 3, lines 1-14)."""
    s, k, s_next = _indices(dataset)
    B, cards = dataset.B, dataset.state_cards
    S, K, n = len(B), len(dataset.actions), len(cards)
    factors = np.zeros((S, K, n, max(cards)))
    known = np.ones((S, K), dtype=bool)
    for a in range(K):
        mine = k == a
        rows, nxt = B[s[mine]], B[s_next[mine]]
        for i in range(n):
            p, ok = sl_factor(B, rows, nxt[:, i], cards, cards[i], d, m, eps)
            factors[:, a, i, : cards[i]] = p
            known[:, a] &= ok
    P = np.ones((S, K, S))
    for i in range(n):
        P *= factors[:, :, i, B[:, i]]  # the product of the factors, for every next state
    P[~known] = 0.0
    return SpibbModel(P, ~known, estimated_behaviour(dataset), dataset.reward)


def solve(model: SpibbModel, gamma: float, max_iter: int = 200) -> Solution:
    """Pi_b-SPIBB policy iteration in the estimated MDP (Laroche et al., Alg. 1)."""
    P, boot, pi_b, reward = model.P, model.bootstrapped, model.behaviour, model.reward
    fixed = np.where(boot, pi_b, 0.0)  # mass that stays on the bootstrapped actions
    free_mass = 1.0 - fixed.sum(1)
    pi = pi_b.copy()
    for _ in range(max_iter):
        V = value_iteration.policy_value(P, reward, pi, gamma)
        Q = reward.R + P @ reward.backup(V, gamma)
        best = np.where(boot, -np.inf, Q).argmax(1)
        new = fixed.copy()
        rows = np.flatnonzero(~boot.all(1))
        new[rows, best[rows]] += free_mass[rows]
        new[boot.all(1)] = pi_b[boot.all(1)]  # every action bootstrapped: the behaviour policy
        if np.allclose(new, pi):
            break
        pi = new
    V = value_iteration.policy_value(P, reward, pi, gamma)
    return Solution(V, pi.argmax(1), pi)
