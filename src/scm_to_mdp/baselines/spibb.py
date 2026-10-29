"""Safe policy improvement with baseline bootstrapping (SPIBB): flat and with structure learning.

    flat SPIBB   Pi_b-SPIBB of Laroche, Trichelair, Tachet des Combes, "Safe Policy Improvement with
                 Baseline Bootstrapping", ICML 2019 (https://arxiv.org/abs/1712.06924): the MLE flat
                 MDP; a pair (s, a) seen fewer than m times is "bootstrapped" (the paper, Sec. 3: "the
                 set of state-action pairs with counts smaller than N_wedge"; their code,
                 github.com/RomainLaroche/SPIBB, spibb.py, bootstraps counts <= N_wedge: one count
                 apart, we follow the paper). Unseen pairs get a zero row, as their modelTransitions
                 (zero_unseen=True).
    SL-SPIBB     Structure Learning Pi_b-SPIBB of Simao & Spaan, "Structure Learning for Safe Policy
                 Improvement", IJCAI 2019 (https://doi.org/10.24963/ijcai.2019/479), Alg. 3, with the SL
                 subalgorithm of Strehl, Diuk, Littman (AAAI 2007) described there (Sec. 2.3, Eqs. 4-5):
                 for every state variable X_i and action a, the candidate parent sets are the subsets
                 of d state variables; at state s a candidate D_i is accepted when every pair union
                 D_i u D_j has at least m samples in s's configuration (Eq. 4) and the estimates
                 P(X_i' | s[D_i u D_j]) of all j differ by at most eps in L1 (Eq. 5); it returns
                 P(X_i' | s[D_i u D_j]) for the first j. (s, a) is bootstrapped when some factor
                 finds no candidate; otherwise P(s' | s, a) = prod_i P_i(s'_i).
    Factored     Factored Pi_b-SPIBB of Simao & Spaan, "Safe Policy Improvement with Baseline
    SPIBB        Bootstrapping in Factored Environments", AAAI 2019, 33(01):4967-4974
                 (https://doi.org/10.1609/aaai.v33i01.33014967), Alg. 2: the dependency function D
                 (the parents of every state variable) is GIVEN ("the dependence between the factors
                 is known a priori"); here, the true graph. Each factor is estimated from its counts,
                 P(x_i' | j) = n(x_i', j) / n(j) with j = D(s, a, X_i) (Eq. 2), P(s' | s, a) = prod_i
                 (Eq. 3), and (s, a) is known when every factor's component has n(j) >= m_i (Eq. 4);
                 the other pairs are bootstrapped. One threshold m for every factor (their Taxi and
                 SysAdmin experiments: m_i = 20 and 10, the values of SL-SPIBB's m in our domains).

Both then compute the Pi_b-SPIBB policy: policy iteration in the estimated MDP (Laroche et al.,
Sec. 2.3, with the greedy projection of their Alg. 1; Simao & Spaan, Alg. 1), where in every state
the policy keeps the baseline's probability on bootstrapped actions and puts the rest of the mass on
the best non-bootstrapped action (ties: the action the current policy prefers, so that the
iteration cannot cycle between equal actions; Laroche et al.'s code takes the first index and stops when
Q stops changing: the same values). The number of iterations is reported
(Solution.iterations; max_iter reached: not converged). The result is a stochastic policy; its value
in the estimated MDP is reported as the promise, which is an estimate (the guarantee of SPIBB is an
improvement over the baseline w.h.p., not a value bound).

The estimated MDP: flat SPIBB keeps the maximum-likelihood transitions of every pair seen at least
once (Laroche et al., Thm 1 and Lemma 2: bootstrapped pairs are followed with the baseline's
probabilities, so their estimates are used; their code passes the MLE model), and only unseen pairs
have no estimate; SL-SPIBB gives every bootstrapped pair an absorbing (zero) row, as Simao & Spaan,
Alg. 3, line 9; Factored SPIBB keeps the product estimate of every pair whose components were all seen
(as flat SPIBB keeps its MLE), and a zero row where some component has no data.

The baseline policy (`behaviour`): "uniform", the uniformly random logger of our experiments (1 / K
over the action vectors of the data: both papers take the baseline as given), or "mle", estimated
from the data (n(s, a) / n(s); uniform in a state never seen), as in Simao, Laroche, Tachet des Combes,
"Safe Policy Improvement with an Estimated Baseline Policy", AAMAS 2020
(https://arxiv.org/abs/1909.05236), for a logger the learner does not know.

Parameters. The count threshold m (Laroche et al.'s N_wedge) and SL-SPIBB's eps are tuned per domain
in Simao & Spaan, Table 1: Taxi (flat m = 10; SL m = 20, eps = 0.01), SysAdmin (100; 10, 0.2), Stock
(10; 20, 0.3); the defaults here are SysAdmin's, and the experiments set the others per domain. d
is the maximum number of state-variable parents of a factor (SL-SPIBB's "maximum in-degree"; its
guarantee needs d at least the true in-degree): set per domain (experiments/domains.py). When d is
at least the number of state variables there is one candidate, the whole state, and SL-SPIBB is
flat SPIBB with threshold m and a product-of-marginals model (no structure is learned).

Our changes: actions are the action vectors seen in the data (K of them); counts are not capped at
m (a batch of data, not Strehl et al.'s online KWIK setting); the candidate is chosen per state (the
literal reading of Simao & Spaan, Sec. 2.3; their code is not public).
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


BEHAVIOURS = ("uniform", "mle")


def baseline_policy(dataset: Dataset, behaviour: str = "mle") -> np.ndarray:
    """[S, K] the baseline policy: "uniform" (1 / K) or "mle" (n(s, a) / n(s); uniform in states the
    data never visit)."""
    if behaviour not in BEHAVIOURS:
        raise ValueError(f"behaviour must be one of {BEHAVIOURS}, not {behaviour!r}")
    S, K = len(dataset.B), len(dataset.actions)
    if behaviour == "uniform":
        return np.full((S, K), 1.0 / K)
    s, k, _ = _indices(dataset)
    counts = np.zeros((S, K))
    np.add.at(counts, (s, k), 1)
    n = counts.sum(1, keepdims=True)
    return np.where(n > 0, counts / np.maximum(n, 1), 1.0 / K)


def flat_model(dataset: Dataset, m: int = 100, behaviour: str = "mle") -> SpibbModel:
    """MLE flat MDP (zero rows only for unseen pairs); pairs seen fewer than m times are bootstrapped."""
    s, k, s_next = _indices(dataset)
    S, K = len(dataset.B), len(dataset.actions)
    counts = np.zeros((S, K, S))
    np.add.at(counts, (s, k, s_next), 1)
    n = counts.sum(2)
    P = counts / np.maximum(n, 1)[:, :, None]  # unseen pairs: a zero row
    return SpibbModel(P, n < m, baseline_policy(dataset, behaviour), dataset.reward)


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


def sl_model(
    dataset: Dataset, d: int = 3, m: int = 10, eps: float = 0.2, behaviour: str = "mle"
) -> SpibbModel:
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
    return SpibbModel(P, ~known, baseline_policy(dataset, behaviour), dataset.reward)


def factored_model(
    dataset: Dataset, parents: dict[str, tuple[str, ...]], m: int = 10, behaviour: str = "mle"
) -> SpibbModel:
    """Simao & Spaan's Factored Pi_b-SPIBB model (AAAI 2019, Alg. 2, lines 1-2) on the given parents
    (state and action variables of slice t) of every state variable."""
    s, k, s_next = _indices(dataset)
    B, A = dataset.B, dataset.actions
    S, K = len(B), len(A)
    column = {v: (B[:, i][:, None], (S, K)) for i, v in enumerate(dataset.state_vars)}
    column |= {v: (A[:, j][None, :], (S, K)) for j, v in enumerate(dataset.action_vars)}
    P = np.ones((S, K, S))
    known = np.ones((S, K), dtype=bool)
    seen = np.ones((S, K), dtype=bool)
    for i, v in enumerate(dataset.state_vars):
        pa = tuple(parents[v])
        unknown = [x for x in pa if x not in column]
        if unknown:
            raise ValueError(f"parents of {v!r} must be slice-t state or action variables, not {unknown}")
        cards = [dataset.cards[x] for x in pa]
        values = (
            np.stack([np.broadcast_to(*column[x]) for x in pa], axis=-1) if pa else np.zeros((S, K, 0), int)
        )
        comp = encode(values, cards) if pa else np.zeros((S, K), dtype=np.int64)  # j = D(s, a, X_i)
        counts = np.zeros((int(np.prod(cards)) if pa else 1, dataset.cards[v]))
        np.add.at(counts, (comp[s, k], B[s_next, i]), 1)  # n(x_i', j)
        n = counts.sum(1)  # n(j)
        known &= n[comp] >= m  # Eq. 4
        seen &= n[comp] > 0
        P *= (counts / np.maximum(n, 1)[:, None])[comp][:, :, B[:, i]]  # Eqs. 2-3
    P[~seen] = 0.0
    return SpibbModel(P, ~known, baseline_policy(dataset, behaviour), dataset.reward)


def solve(model: SpibbModel, gamma: float, max_iter: int = 200) -> Solution:
    """Pi_b-SPIBB policy iteration in the estimated MDP (see the module docstring)."""
    P, boot, pi_b, reward = model.P, model.bootstrapped, model.behaviour, model.reward
    fixed = np.where(boot, pi_b, 0.0)  # mass that stays on the bootstrapped actions
    free_mass = 1.0 - fixed.sum(1)
    pi = pi_b.copy()
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        V = value_iteration.policy_value(P, reward, pi, gamma)
        Q = reward.R + P @ reward.backup(V, gamma)
        allowed = np.where(boot, -np.inf, Q)
        best = allowed.argmax(1)
        current = np.where(boot, -np.inf, pi).argmax(1)  # the action the current policy prefers
        keep = allowed[np.arange(len(best)), current] >= allowed[np.arange(len(best)), best] - 1e-12
        best = np.where(keep, current, best)  # ties: no switch
        new = fixed.copy()
        rows = np.flatnonzero(~boot.all(1))
        new[rows, best[rows]] += free_mass[rows]
        new[boot.all(1)] = pi_b[boot.all(1)]  # every action bootstrapped: the behaviour policy
        if np.allclose(new, pi):
            break
        pi = new
    V = value_iteration.policy_value(P, reward, pi, gamma)
    return Solution(V, pi.argmax(1), pi, iterations=iterations)
