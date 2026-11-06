"""Probability sets from a KWIK structure (discovery/kwik_structure.py: Strehl et al. 2007, Met-Rmax), S4.

The structure is context-specific: for every factor x_i', action a_k and state s the learner either chose
parents (Strehl's accepted candidate; Met-Rmax's surviving meteorologists) or said "don't know". Two ways to
turn it into a factored interval MDP, and one into a point model:

    learn         our S4: Clopper-Pearson intervals from the counts of the candidate parent sets (NBN,
                  sampling="trajectory": valid for every visit count), and at every (s, a, i) the HULL (min
                  of the lower ends, max of the upper ends) over the candidates that may contain the true
                  parents. A superset of the true parents has the true conditional (Koller & Friedman,
                  Def. 3.1, the local independencies), so the hull covers the truth when one of its
                  candidates contains the true parents (k at least the true number of parents):
        hull="fixed"     every candidate of the learner (all subsets of k state variables), FIXED before the
                         data. Only the candidate that contains the true parents must hold, and it is fixed
                         (unknown but not chosen from the data), so the level needs no union over the
                         candidates: delta is split over the (factor, action) tables only. It does not use
                         the learner's decisions (the same for Strehl and Met-Rmax).
        hull="accepted"  the candidates the learner chose: Strehl's pair unions D u D_j of the accepted D (a
                         "don't know" row gets [0, 1]); Met-Rmax's surviving meteorologists. They are chosen
                         from the data, so delta is also split over every candidate table the choice could
                         pick (a union bound: Agarwal, Brantley, Jiang, Kakade, Sun, as in Lemma A.20).
                         Strehl: some union D u D_j contains the true parents whichever D was accepted (j the
                         true one, or any j when D is), so the level is 1 - delta. Met-Rmax: the hull covers
                         the truth only if a survivor contains the true parents; the learner may eliminate it
                         (with the paper's m its Thm 1 confidence is nominal), so the level is
                         1 - delta - P(that).
    kwik_bounds   the learners' own inference (the "KWIK" arms): the prediction +- its accuracy (Strehl's
                  2 eps1, proof of Thm 1; Met-Rmax's eps, Thm 1), [0, 1] where it says "don't know". Its
                  level is the papers' (with their m, or with a calibrated m).
    kwik_point    the point model for value iteration: the predictions; a "don't know" pair goes to an
                  absorbing HALT state with the worst reward, as MOReL's pessimistic MDP (Kidambi,
                  Rajeswaran, Netrapalli, Joachims, NeurIPS 2020, arXiv:2005.05951). Here HALT is a zero
                  transition row whose reward is a lower bound on any value (halt_value; compile with
                  exact_with_halt).
"""

import itertools

import numpy as np
import torch
from nbn.uncertainty import interval_set_from_counts

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import Dataset, FactorBounds, IntervalMDP, Reward, encode
from scm_to_mdp.discovery.kwik_structure import KwikStructure
from scm_to_mdp.scm_learning import data_only as rules

HULLS = ("fixed", "accepted")


def _per_action(dataset: Dataset):
    df = dataset.df
    states, nxt = df[dataset.state_vars].to_numpy(), df[dataset.next_vars].to_numpy()
    k = dataset.action_index()
    return {a: (states[k == a], nxt[k == a]) for a in range(len(dataset.actions))}


def _cp_at_states(B, rows, values, U, cards, card, delta, data_only=False):
    """Clopper-Pearson intervals [S, card] of P(x_i' = v | s[U], a) at every state, from ALL the samples of
    the action (rows, values), at level delta for the table (NBN splits it over its entries and counts), and
    [S] whether the state's row has no data. data_only: scm_learning/data_only.py's intervals (the level split
    over the rows' and values' codes, the values not observed at a row free)."""
    cu = [cards[u] for u in U]
    table = np.zeros((int(np.prod(cu)), card))
    np.add.at(table, (encode(rows[:, list(U)], cu), values), 1)
    at = encode(B[:, list(U)], cu)
    unseen = table.sum(1)[at] == 0
    if data_only:
        codes = np.stack(np.unravel_index(np.arange(len(table)), cu), axis=1) if U else np.zeros((1, 0), int)
        lower, upper = rules.intervals(table, rules.code_weight(codes), rules.value_weight(np.arange(card)), delta)
        return lower[at], upper[at], unseen
    sets = interval_set_from_counts(torch.as_tensor(table), delta, "clopper_pearson", "trajectory")
    return sets.lower.double().numpy()[at], sets.upper.double().numpy()[at], unseen


def learn(
    dataset: Dataset, structure: KwikStructure, delta: float = 0.05, hull: str = "accepted", data_only: bool = False
) -> FactorBounds:
    """lo, hi [S, K, n, C] (module docstring). data_only (scm_learning/data_only.py): the level of an action's
    tables is weighted by the action's value codes instead of 1 / K, a pair whose hull uses a row without data
    (or Strehl's "don't know") is sent to a sink, and the sinks of data_only.py."""
    if hull not in HULLS:
        raise ValueError(f"hull must be one of {HULLS}, not {hull!r}")
    B, cards = dataset.B, dataset.state_cards
    S, K, n, C = len(B), len(dataset.actions), len(cards), max(cards)
    lo, hi = np.zeros((S, K, n, C)), np.zeros((S, K, n, C))
    data = _per_action(dataset)
    tables = 0
    doomed = np.zeros((S, K), dtype=bool)
    # the share of delta of an action's tables: 1 / K, or (data-only) the weight of its value codes
    share = rules.code_weight(dataset.actions) if data_only else np.full(K, 1.0 / K)
    for (i, k), f in structure.factors.items():
        rows, nxt = data[k]
        card, values = cards[i], nxt[:, i]
        H = len(f.candidates)
        lo_i, hi_i = np.ones((S, card)), np.zeros((S, card))
        if hull == "fixed" or structure.method == "met_rmax":
            # fixed: every candidate, level per (factor, action) table; Met-Rmax: the survivors, level per
            # candidate table (any candidate could survive)
            use = range(H) if hull == "fixed" else f.survivors
            d = delta * share[k] / n if hull == "fixed" else delta * share[k] / (n * H)
            for h in use:
                lower, upper, unseen = _cp_at_states(B, rows, values, f.candidates[h], cards, card, d, data_only)
                lo_i, hi_i = np.minimum(lo_i, lower), np.maximum(hi_i, upper)
                doomed[:, k] |= unseen
            tables += len(use)
        else:  # Strehl, accepted: the pair unions of the accepted candidate; "don't know" -> [0, 1]
            pairs = list(itertools.combinations(range(H), 2)) or [(0, 0)]
            d = delta * share[k] / (n * len(pairs))
            lo_i[:], hi_i[:] = 0.0, 1.0
            doomed[:, k] |= ~f.known  # "don't know": nothing is known there
            for a in sorted(set(f.accepted[f.known].tolist())):
                states = np.flatnonzero(f.accepted == a)
                l_a, u_a = np.ones((len(states), card)), np.zeros((len(states), card))
                for j in [j for j in range(H) if j != a] or [a]:
                    U = tuple(sorted(set(f.candidates[a]) | set(f.candidates[j])))
                    lower, upper, unseen = _cp_at_states(B[states], rows, values, U, cards, card, d, data_only)
                    l_a, u_a = np.minimum(l_a, lower), np.maximum(u_a, upper)
                    doomed[states, k] |= unseen
                    tables += 1
                lo_i[states], hi_i[states] = l_a, u_a
        lo[:, k, i, :card], hi[:, k, i, :card] = lo_i, hi_i
    info = {"s4_hull": hull, "s4_kwik": structure.method, "s4_tables": tables}
    if not data_only:
        return FactorBounds(lo, hi, cards, B, dataset.reward, info=info)
    masks, extra = rules.description(dataset, doomed)
    return FactorBounds(lo, hi, cards, B, rules.model_reward(dataset), info=info | extra, data_only=masks)


def kwik_bounds(dataset: Dataset, structure: KwikStructure) -> FactorBounds:
    """The learner's predictions +- its accuracy, [0, 1] where it says "don't know"."""
    acc = structure.params["accuracy"]
    B, cards = dataset.B, dataset.state_cards
    S, K, n, C = len(B), len(dataset.actions), len(cards), max(cards)
    lo, hi = np.zeros((S, K, n, C)), np.zeros((S, K, n, C))
    for (i, k), f in structure.factors.items():
        card = cards[i]
        lo[:, k, i, :card] = np.where(f.known[:, None], np.clip(f.predict - acc, 0, 1), 0.0)
        hi[:, k, i, :card] = np.where(f.known[:, None], np.clip(f.predict + acc, 0, 1), 1.0)
    unknown = np.mean([1 - f.known.mean() for f in structure.factors.values()])
    return FactorBounds(
        lo, hi, cards, B, dataset.reward, info={"s4_kwik": structure.method, "s4_dont_know": unknown}
    )


def kwik_point(dataset: Dataset, structure: KwikStructure) -> FactorBounds:
    """The predictions as a point model (lo == hi). info["_halt"] [S, K]: the pairs where some factor says
    "don't know" (sent to HALT by exact_with_halt). Keys with "_" stay out of the results rows."""
    B, cards = dataset.B, dataset.state_cards
    S, K, n, C = len(B), len(dataset.actions), len(cards), max(cards)
    P = np.zeros((S, K, n, C))
    halt = np.zeros((S, K), dtype=bool)
    for (i, k), f in structure.factors.items():
        card = cards[i]
        p = f.predict / np.maximum(f.predict.sum(1, keepdims=True), 1e-12)  # the midpoint may not sum to 1
        P[:, k, i, :card] = np.where(f.known[:, None], p, 1.0 / card)  # HALT rows: any distribution
        halt[:, k] |= ~f.known
    info = {"_halt": halt, "s4_kwik": structure.method, "s4_dont_know": float(halt.mean())}
    return FactorBounds(P, P.copy(), cards, B, dataset.reward, info=info)


def halt_value(reward: Reward, gamma: float) -> float:
    """A lower bound on the value of any path: r1 = min R + min r_next over non-terminal states (a step that
    goes on), r2 = min R + min r_next over terminal states (a step that ends the episode); going on forever
    is worth at least r1 / (1 - gamma), ending after some steps at least r2 + min(0, r1) / (1 - gamma)."""
    terminal = reward.terminal
    out, r1 = [], 0.0
    if (~terminal).any():
        r1 = float(reward.R.min() + reward.r_next[~terminal].min())
        out.append(r1 / (1 - gamma))
    if terminal.any():
        r2 = float(reward.R.min() + reward.r_next[terminal].min())
        out.append(r2 + min(0.0, r1) / (1 - gamma))
    return min(out)


def exact_with_halt(bounds: FactorBounds, gamma: float) -> IntervalMDP:
    """product_bounds.exact, then every HALT pair (bounds.info["_halt"]) gets a zero row and the reward
    halt_value, so Q = a lower bound on any value (MOReL's pessimistic HALT, its penalty the worst value).
    The zero rows break the interval invariant sum L <= 1 <= sum U: this model is for value iteration."""
    mdp = product_bounds.exact(bounds)
    halt = bounds.info["_halt"]
    L, U = mdp.L.copy(), mdp.U.copy()
    L[halt], U[halt] = 0.0, 0.0
    R = mdp.reward.R.copy()
    R[halt] = halt_value(mdp.reward, gamma)
    reward = Reward(R, mdp.reward.r_next, mdp.reward.terminal)
    return IntervalMDP(L, U, reward, info={"halt_share": float(halt.mean())})
