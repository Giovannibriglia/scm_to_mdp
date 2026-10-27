"""FOCUS for discrete variables: causal structure from CI tests, then a MOPO-style penalised model.

Zhu, Chen, Tian, Zhang, Yu, "Offline Reinforcement Learning with Causal Structured World Models"
(https://arxiv.org/abs/2206.01474; Frontiers of Computer Science 19(4):194347, 2025). Their method:

  Stage 1, structure (Sec. 5.2, App. B.1): for every slice-t input X_t^i (state or action) and next
      variable X_{t+1}^j, a conditional independence test of X_t^i and X_{t+1}^j given all the OTHER
      slice-t inputs (their rule: condition on the other variables at t, never on those at t + 1);
      an edge when p <= p*, with one threshold for all tests chosen from the sorted p-values
      p_1 <= ... <= p_n (Eq. 13): p* = p_i* with i* = argmax_i (p_{i+1} / (i + 1) - p_i / i).
  Stage 2, policy (Sec. 5.3): the world model masked by that structure, inside MOPO (Yu et al.,
      NeurIPS 2020, https://arxiv.org/abs/2005.13239): rewards penalised by the model's uncertainty,
      r~(s, a) = r(s, a) - lambda * u(s, a), and a policy optimised in the model.

Our discrete version (not a reproduction; every choice fixed before any run):
  - the CI test is the G^2 test for discrete variables, stratified by the conditioning set (they
    use the kernel test KCI, for continuous variables), with a permutation p-value on sparse tables,
    where the chi-square one wrongly finds dependences (citests.g2_pvalue);
  - the masked world model is the factored maximum-likelihood model on the learned parents (a
    parent row never seen: uniform), i.e. the tabular counterpart of their masked network;
  - u(s, a), an ensemble-disagreement penalty, as in MOPO's "ens. pen." ablation (MOPO's default is
    the largest learned standard deviation, which has no tabular counterpart that penalises rows
    seen once): `ensemble` bootstrap resamples of the transitions (MOPO trains 7 networks and keeps
    5), each giving a factored model; u(s, a) = max over members of sum_j TV(member_j, model_j) at
    (s, a), with TV = 1 for a parent row the data never show (a sum of factor TVs bounds the TV of
    the joint next state); lambda is given in units of the range of the step reward
    (lambda * (r_max - r_min) * u);
  - the policy is optimal in the penalised MDP (value iteration), as MOPO's policy optimisation
    aims for. The promise is its value there: an estimate, not a certified bound;
  - lambda = 0.3, from a pilot before the grids (seed 0; SysAdmin N = 3, 6; Sepsis, 2 vitals;
    Chemistry, 5 objects; uniform and expert loggers; two budgets each): the mean share of V*_obs
    was 0.82 (lambda = 0.1), 0.81 (0.3), 0.78 (1); of the two best we kept the more pessimistic.
    The pilot read the true values, as MOPO's choice of lambda (from {0.5, 1, 2, 5}, the best per
    dataset) did; MOPO's theory would ask for lambda = gamma / (1 - gamma) with our u (9 at gamma
    0.9), which is far more pessimistic. FOCUS's paper gives no value.
Eq. 13 favours a jump at a low rank, so with few tests it can accept a large p-value (e.g. 0.4) and
add parents; this follows the paper.
"""

import numpy as np

from scm_to_mdp.citests import g2_pvalue
from scm_to_mdp.core import NEXT_MARK, Dataset, Graph, IntervalMDP, Reward, encode


def threshold(p_values: np.ndarray) -> float:
    """FOCUS's p* (App. B.1, Eq. 13) from all the p-values of stage 1."""
    p = np.sort(np.asarray(p_values, dtype=float))
    if len(p) < 2:
        return float(p[0]) if len(p) else 0.0
    i = np.arange(1, len(p))  # 1-based ranks of p_1 .. p_{n-1}
    score = p[1:] / (i + 1) - p[:-1] / i
    return float(p[int(np.argmax(score))])


def discover(dataset: Dataset) -> Graph:
    """Stage 1: the slice-t parents of every next-state variable."""
    df, inputs = dataset.df, dataset.inputs
    X = df[inputs].to_numpy()
    cards = [dataset.cards[c] for c in inputs]
    p = np.ones((len(inputs), len(dataset.state_vars)))
    for i in range(len(inputs)):
        others = [c for c in range(len(inputs)) if c != i]
        strata = (
            encode(X[:, others], [cards[c] for c in others]) if others else np.zeros(len(X), dtype=np.int64)
        )
        for j, v in enumerate(dataset.state_vars):
            y = df[v + NEXT_MARK].to_numpy()
            p[i, j] = g2_pvalue(X[:, i], y, strata)
    p_star = threshold(p.ravel())
    parents = {
        v: tuple(inputs[i] for i in range(len(inputs)) if p[i, j] <= p_star)
        for j, v in enumerate(dataset.state_vars)
    }
    return Graph(parents, report={"p_star": p_star})


def _factored(dataset: Dataset, parents: dict, rows: np.ndarray):
    """Factor tables P(x_j' | parents) at every (s, a) from the transitions `rows`, and which parent
    rows were seen: ([S, K, n, C], [S, K, n] bool)."""
    df, B, actions = dataset.df, dataset.B, dataset.actions
    S, K, cards = len(B), len(actions), dataset.state_cards
    assert dataset.inputs == dataset.state_vars + dataset.action_vars  # the column order of inputs_at
    inputs_at = np.concatenate([np.repeat(B[:, None, :], K, 1), np.repeat(actions[None, :, :], S, 0)], 2)
    F = np.zeros((S, K, len(cards), max(cards)))
    seen = np.zeros((S, K, len(cards)), dtype=bool)
    for j, v in enumerate(dataset.state_vars):
        cols = [dataset.inputs.index(x) for x in parents[v]]
        pc = [dataset.cards[x] for x in parents[v]]
        y = df[v + NEXT_MARK].to_numpy()[rows]
        code = (
            encode(df[list(parents[v])].to_numpy()[rows], pc) if cols else np.zeros(len(rows), dtype=np.int64)
        )
        table = np.zeros((int(np.prod(pc)) if cols else 1, cards[j]))
        np.add.at(table, (code, y), 1)
        at = (
            encode(inputs_at[:, :, cols].reshape(S * K, len(cols)), pc)
            if cols
            else np.zeros(S * K, dtype=np.int64)
        )
        counts = table[at].reshape(S, K, cards[j])
        n = counts.sum(2, keepdims=True)
        F[:, :, j, : cards[j]] = np.where(n > 0, counts / np.maximum(n, 1), 1.0 / cards[j])
        seen[:, :, j] = n[..., 0] > 0
    return F, seen


def penalised_mdp(
    dataset: Dataset, graph: Graph, lam: float = 0.3, ensemble: int = 5, seed: int = 0
) -> IntervalMDP:
    """Stage 2: the factored model on the learned parents, with MOPO's penalised reward (an exact
    MDP, L == U, solved by value iteration)."""
    B, n_rows = dataset.B, len(dataset.df)
    F, seen = _factored(dataset, graph.parents, np.arange(n_rows))
    rng = np.random.default_rng(seed)
    u = np.zeros(F.shape[:2])
    for _ in range(ensemble):
        Fe, _ = _factored(dataset, graph.parents, rng.integers(0, n_rows, n_rows))
        tv = 0.5 * np.abs(Fe - F).sum(3)  # [S, K, n]
        u = np.maximum(u, np.where(seen, tv, 1.0).sum(2))
    u = np.maximum(u, np.where(seen, 0.0, 1.0).sum(2))  # a row never seen is as uncertain as can be
    r = dataset.reward
    step = r.R[:, :, None] + r.r_next[None, None, :]
    scale = float(step.max() - step.min()) or 1.0
    P = np.ones((len(B), F.shape[1], len(B)))
    for j in range(B.shape[1]):
        P *= F[:, :, j, B[:, j]]
    penalised = Reward(r.R - lam * scale * u, r.r_next, r.terminal)
    # the penalty without the reward's range, so that methods.with_reward can re-plan with another reward
    return IntervalMDP(P, P, penalised, info={"penalty": lam * u})
