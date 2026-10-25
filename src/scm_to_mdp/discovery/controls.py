"""Control graphs (step S2, for the evaluation only): wrong structures of a known kind, learned and solved
like the true graph, to tell "the right causal structure" apart from "a small model" (claims A and B).

Each is a parent set per next-state variable, given to the learner as the oracle's graph is (selection
"given"); none is a discovery method, and each needs the truth.

    logger_markov   the MAIN control: for every x_i', a smallest set S of slice-t inputs that makes x_i'
                    independent of the true parents given S UNDER THE LOGGER's distribution of (s, a):
                    I(x_i'; pa_i | S) <= tol, computed exactly from the true kernel and the logger's
                    occupancy (not estimated). It fits the logged data as well as the true parents, but
                    a (near-)deterministic logger makes actions and states stand in for each other
                    ("determinism induces additional independences": Peters, Janzing, Scholkopf,
                    Elements of Causal Inference, 2017, p. 206), so it may drop a true parent that
                    matters under a new policy. With a positive logger it is the true parent set
                    (known time order, no same-step edges: their p. 152).
    random_matched  a sanity control: for every x_i', as many action parents as the truth and state
                    parents drawn at random, among the sets with the number of free parameters
                    (card(x_i') - 1) * prod(cards of the parents) closest to the truth's (Koller &
                    Friedman, PGM, 2009, p. 802: model dimension), never the true set when another
                    exists. Usually misspecified: it mixes the bias of a missing parent with the
                    fragmentation of an extra one (their Thm 17.3, p. 774).
    plus_one        fragmentation only: the true parents and one more state variable, drawn at random
                    (a superset still contains the true mechanism; their pp. 784-785).
Same-step parents (x_j') of the truth are kept as they are by plus_one, and dropped by the other two.
"""

import itertools
from math import prod

import numpy as np

from scm_to_mdp.behaviour import data_occupancy, to_observed
from scm_to_mdp.core import NEXT_MARK, encode

CONTROLS = ("logger_markov", "random_matched", "plus_one")
TOL = 1e-3  # nats: I(x_i'; pa_i | S) at or below this counts as independence (logger_markov)


def _observed_inputs(truth) -> tuple[list[str], list[str], dict]:
    spec = truth.learner_spec
    return list(spec.state_vars), list(spec.action_vars), dict(spec.cards)


def _slice_t(parents: tuple[str, ...], cards: dict) -> list[str]:
    return [p for p in parents if not p.endswith(NEXT_MARK) and p in cards]


def random_matched(truth, seed: int) -> dict:
    states, actions, cards = _observed_inputs(truth)
    out = {}
    for i, v in enumerate(states):
        true = _slice_t(truth.true_parents[v], cards)
        k_a = [p for p in true if p in actions]
        k_s = [p for p in true if p in states]
        dim = prod(cards[p] for p in true)
        options = [set(c) for c in itertools.combinations(states, len(k_s)) if set(c) != set(k_s)]
        if not options:
            out[v] = tuple(true)
            continue
        gap = [abs(prod(cards[p] for p in c) * prod(cards[p] for p in k_a) - dim) for c in options]
        best = [c for c, g in zip(options, gap, strict=True) if g == min(gap)]
        pick = best[np.random.default_rng([seed, i]).integers(len(best))]
        out[v] = tuple(p for p in states + actions if p in pick or p in k_a)
    return out


def plus_one(truth, seed: int) -> dict:
    states, actions, cards = _observed_inputs(truth)
    out = {}
    for i, v in enumerate(states):
        parents = tuple(p for p in truth.true_parents[v] if p.removesuffix(NEXT_MARK) in cards)
        free = [p for p in states if p not in parents]
        if free:
            extra = free[np.random.default_rng([seed, i, 1]).integers(len(free))]
            names = states + actions + [s + NEXT_MARK for s in states]
            parents = tuple(p for p in names if p in parents or p == extra)
        out[v] = parents
    return out


def logger_markov(truth, logger: np.ndarray, horizon: int, tol: float = TOL) -> dict:
    """logger: [S_full, A] the behaviour policy table (Env.behaviour); horizon: the rollout length."""
    states, actions, cards = _observed_inputs(truth)
    nu = to_observed(truth, data_occupancy(truth, logger, horizon))  # [S_obs, A] the logged (s, a)
    P = truth.observed_kernel()  # [S_obs, A, S_obs]
    B = all_observed_states(truth)
    s_idx, a_idx = np.nonzero(nu > 0)
    w = nu[s_idx, a_idx]
    w = w / w.sum()
    values = {v: B[s_idx, j] for j, v in enumerate(states)}
    values |= {a: truth.actions[a_idx, j] for j, a in enumerate(actions)}
    inputs = states + actions
    out = {}
    for j, v in enumerate(states):
        true = _slice_t(truth.true_parents[v], cards)
        rows = P[s_idx, a_idx]  # [E, S_obs]
        q = np.stack([rows[:, B[:, j] == x].sum(1) for x in range(cards[v])], axis=1)  # [E, card] P(x_j'|s,a)
        chosen = None
        for size in range(len(true) + 1):
            scored = []
            for subset in itertools.combinations(inputs, size):
                cmi = _conditional_information(q, w, [values[p] for p in subset], [cards[p] for p in subset])
                if cmi <= tol:
                    overlap = len(set(subset) & set(true))
                    scored.append((cmi, -overlap, subset))
            if scored:
                chosen = min(scored)[2]
                break
        out[v] = tuple(p for p in inputs if p in (chosen if chosen is not None else true))
    return out


def all_observed_states(truth) -> np.ndarray:
    from scm_to_mdp.core import all_states

    spec = truth.learner_spec
    return all_states([spec.cards[v] for v in spec.state_vars])


def _conditional_information(q: np.ndarray, w: np.ndarray, columns: list, cards: list[int]) -> float:
    """I(x'; inputs | S) = E_w KL(q(.|s,a) || q(.|S)): q [E, C] the true next-value law at every logged
    (s, a) (it depends on the true parents only), w [E] their weights."""
    code = encode(np.stack(columns, axis=1), cards) if columns else np.zeros(len(w), dtype=np.int64)
    groups, inverse = np.unique(code, return_inverse=True)
    mass = np.bincount(inverse, weights=w, minlength=len(groups))
    pooled = np.stack(
        [np.bincount(inverse, weights=w * q[:, c], minlength=len(groups)) for c in range(q.shape[1])], 1
    )
    pooled = pooled / np.maximum(mass[:, None], 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(q > 0, q * (np.log(q) - np.log(pooled[inverse])), 0.0).sum(1)
    return float((w * kl).sum())
