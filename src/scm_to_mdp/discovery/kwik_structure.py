"""Structure learners that can abstain ("I don't know"; the KWIK model of Li, Littman, Walsh, ICML 2008),
for factored MDPs with an unknown dependency structure and a known bound on the number of parents (step S2).

    strehl_sl   the Structure-Learning algorithm of Strehl, Diuk, Littman, "Efficient Structure Learning
                in Factored-state MDPs", AAAI 2007 (pp. 645-650; https://cdn.aaai.org/AAAI/2007/AAAI07-102.pdf),
                the subroutine of SLF-Rmax: its basic version (one parent, pp. 647-648) extended to k
                parents as in the paper (p. 649: "the algorithm enumerates all possible elements [settings
                of k input positions] and keeps statistics ... for all possible pairs of elements")
    met_rmax    the Adaptive k-Meteorologists Algorithm of Diuk, Li, Leffler, "The Adaptive k-Meteorologists
                Problem and Its Application to Structure Learning and Feature Selection in Reinforcement
                Learning", ICML 2009 (pp. 249-256; doi:10.1145/1553374.1553406), Alg. 1, as the structure
                learner of Met-Rmax (Sec. 3.1: one meteorologist per subset of D candidate parents)

Both papers learn online, one long trajectory (the KWIK protocol: the inputs may be chosen in any way,
Strehl et al. Def. 3, Diuk et al. Sec. 2.1). Here the logged transitions are REPLAYED in their logged order
(episode, step), which is one such sequence of inputs, so the per-prediction guarantees carry over; the
exploration part of SLF-Rmax / Met-Rmax (optimism for "don't know") does not apply offline and is not used.

As in both papers, one learner per (next-state factor x_i', action): Strehl et al. p. 649 (Alg. 1, lines 1-3,
"for all factors i and actions a: initialize a new instantiation A_{i,a}") and Diuk et al. Sec. 3.1 ("for any
given factor f_i and action a"); its inputs are the state variables (the candidate parents), and it sees the
transitions taken with its action. The candidates are the subsets of exactly k state variables (Strehl: k
input positions; Diuk: the C(n, D) parent subsets).

Multi-valued factors (both papers state the binary case, Y = [0, 1] the probability of a 1): a value of x_i'
is a one-hot vector z, a prediction a vector p; distances are the largest difference over the values
(L_inf: for a binary factor exactly the papers' |p - p'|, because the two values differ by the same amount)
and the squared error of Met-Rmax is ||p - z||^2 (for a binary factor 2 (p_1 - z_1)^2, the same comparisons).
Ours (not in the papers): this vector extension.

Outputs (KwikStructure): for every factor i, action k and state s, the decision (the accepted candidate, the
surviving meteorologists, or "don't know") and the prediction; the S4 step turns them into intervals
(scm_learning/kwik_intervals.py) and the KWIK arms use the predictions themselves.
"""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from scm_to_mdp.core import Dataset, encode

DONT_KNOW = -1
TIE = 1e-12  # Met-Rmax, Alg. 1 line 19: Delta_ij within float error of 0 is a tie (removes j)


@dataclass
class KwikFactor:
    """The learner of one (factor i, action k). candidates: the subsets of state-variable indices;
    accepted [S]: per state the index of the candidate used (Strehl), or for Met-Rmax 0 where the final set
    R predicts (a "known" marker, not a candidate index); DONT_KNOW elsewhere; survivors: the candidates
    still in Met-Rmax's set R at the end (all of them for Strehl); predict [S, card]: the prediction where
    known (rows of DONT_KNOW states are 0); known [S] bool."""

    candidates: list[tuple[int, ...]]
    accepted: np.ndarray
    survivors: list[int]
    predict: np.ndarray
    known: np.ndarray


@dataclass
class KwikStructure:
    """The output of strehl_sl / met_rmax: factors[(i, k)] for every state variable i and action index k
    (dataset.actions), the method and its parameters."""

    method: str
    factors: dict[tuple[int, int], KwikFactor]
    params: dict = field(default_factory=dict)

    def parents(self, dataset: Dataset, view: str = "union") -> dict[str, tuple[str, ...]]:
        """One parent set per next-state variable (a DAG view, e.g. for Factored SPIBB; ours, not in the
        papers, whose structure is per input), plus the action columns (the per-action learners make every
        factor depend on the action). The candidates used: accepted somewhere (Strehl) or surviving
        (Met-Rmax), over the actions. view="union": every variable of one of them (a superset of the true
        parents when one used candidate contains them; with no evidence, e.g. no disagreement, every
        candidate survives and the union is every variable); "intersection": the variables in all of them
        (the true parents when every used candidate contains them; misses a parent if one does not)."""
        if view not in ("union", "intersection"):
            raise ValueError(f"view must be 'union' or 'intersection', not {view!r}")
        out = {}
        for i, v in enumerate(dataset.state_vars):
            used = []
            for k in range(len(dataset.actions)):
                f = self.factors[(i, k)]
                idx = set(f.accepted[f.known].tolist()) if self.method == "strehl" else set(f.survivors)
                used += [set(f.candidates[c]) for c in idx]
            chosen = (set().union(*used) if view == "union" else set.intersection(*used)) if used else set()
            out[v] = tuple(dataset.state_vars[j] for j in sorted(chosen)) + tuple(dataset.action_vars)
        return out


def _transitions(dataset: Dataset):
    """Per action k: (states [N_k, n] in logged order, next states [N_k, n]). The table is sorted by its
    (episode, t) index when it has one, so that "the first m" and the replay follow the logged order."""
    df = dataset.df
    if {"episode", "t"} <= set(df.index.names or []):
        df = df.sort_index(level=["episode", "t"])
    states = df[dataset.state_vars].to_numpy()
    nxt = df[dataset.next_vars].to_numpy()
    lookup = {tuple(a): j for j, a in enumerate(dataset.actions)}  # on the SORTED table (not dataset.df)
    k = np.array([lookup[tuple(a)] for a in df[dataset.action_vars].to_numpy()], dtype=np.int64)
    return {a: (states[k == a], nxt[k == a]) for a in range(len(dataset.actions))}


def _first_m(codes: np.ndarray, m: int) -> np.ndarray:
    """mask [N]: the sample is among the first m (in the given order) with its code."""
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    start = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    rank = np.arange(len(codes)) - np.repeat(start, np.diff(np.r_[start, len(codes)]))
    mask = np.zeros(len(codes), dtype=bool)
    mask[order] = rank < m
    return mask


def _first_m_table(rows, values, U, cards, card, m):
    """Counts of the first m samples of every setting of the variables U, and n = min(m, count)."""
    cu = [cards[u] for u in U]
    codes = encode(rows[:, list(U)], cu) if U else np.zeros(len(rows), dtype=np.int64)
    keep = _first_m(codes, m) if len(codes) else np.zeros(0, dtype=bool)
    table = np.zeros((int(np.prod(cu)) if U else 1, card))
    np.add.at(table, (codes[keep], values[keep]), 1)
    return table, cu


# ==========================================================================================
# Strehl, Diuk, Littman (AAAI 2007)
# ==========================================================================================
def strehl_events(dataset: Dataset, k: int) -> int:
    """The number of (factor, action, pair of candidates, setting of their union, value) first-m estimates.
    The proof of Strehl et al.'s Thm 1 (p. 648) needs only the pairs that contain the correct factor (about
    4n per learner, binary case); this counts every pair (the correct one is unknown when choosing m), so it
    is conservative. A binary factor counts once (its two values are one event), a factor with c > 2 values
    c times (one per value indicator, the L_inf distance)."""
    cards, n = dataset.state_cards, len(dataset.state_vars)
    cands = list(itertools.combinations(range(n), min(k, n)))
    pairs = list(itertools.combinations(cands, 2)) or [(cands[0], cands[0])]
    settings = sum(int(np.prod([cards[u] for u in sorted(set(a) | set(b))])) for a, b in pairs)
    return len(dataset.actions) * sum(settings * (1 if c == 2 else c) for c in cards)


def strehl_calibrated_m(dataset: Dataset, k: int, eps1: float, delta: float) -> int:
    """m such that every first-m estimate is within eps1 of its probability with probability >= 1 - delta:
    Hoeffding, P(|C/m - P| > eps1) <= 2 exp(-2 m eps1^2), with a union bound over strehl_events (the steps of
    the proof of Thm 1, p. 648; the explicit constant is ours: the paper states m proportional to
    ln(n / delta) / eps^2)."""
    return math.ceil(math.log(2 * strehl_events(dataset, k) / delta) / (2 * eps1**2))


def _strehl_factor(B, rows, values, cards, card, k, m, eps1):
    n = B.shape[1]
    cands = list(itertools.combinations(range(n), min(k, n)))
    S = len(B)
    accepted = np.full(S, DONT_KNOW)
    predict = np.zeros((S, card))
    if len(cands) == 1:  # k >= n: one candidate, the whole state (no pair to compare)
        table, cu = _first_m_table(rows, values, cands[0], cards, card, m)
        at = table[encode(B[:, list(cands[0])], cu)]
        ok = at.sum(1) >= m
        accepted[ok], predict[ok] = 0, at[ok] / m
        return KwikFactor(cands, accepted, [0], predict, ok)
    union = {}  # (a, b) with a < b: [S, card] first-m counts of the union's setting at every state
    for a, b in itertools.combinations(range(len(cands)), 2):
        U = tuple(sorted(set(cands[a]) | set(cands[b])))
        table, cu = _first_m_table(rows, values, U, cards, card, m)
        union[(a, b)] = table[encode(B[:, list(U)], cu)]
    for i in range(len(cands)):  # "the algorithm searches for a bit position i such that ..." (p. 647)
        others = [union[(min(i, j), max(i, j))] for j in range(len(cands)) if j != i]
        counts = np.stack(others)  # [C - 1, S, card]
        full = (counts.sum(-1) >= m).all(0)  # C(i, j, (x(i), x(j))) = m for all j != i
        est = counts / m
        spread = est.max(0) - est.min(0)  # [S, card]: the largest difference over pairs j, j'
        ok = full & (spread.max(-1) <= eps1 + 1e-12) & (accepted == DONT_KNOW)  # |C(..j) - C(..j')| <= m eps1
        accepted[ok] = i
        predict[ok] = est[0][ok]  # "for any j != i (chosen arbitrarily)": the first
    return KwikFactor(cands, accepted, list(range(len(cands))), predict, accepted != DONT_KNOW)


def strehl_sl(
    dataset: Dataset, k: int = 2, m: int | str = 20, eps1: float = 0.2, delta: float = 0.05
) -> KwikStructure:
    """Strehl et al.'s Structure-Learning algorithm (module docstring), one learner per (factor, action).
    m: the experience threshold ("C(i, j, z) is the minimum of m and the number of timesteps ...", p. 647;
    every estimate uses the first m samples of its setting), or "calibrated" (strehl_calibrated_m at delta);
    eps1: the precision parameter. Paper settings: m = 20, eps1 = 0.2 (Stock), m = 30, eps1 = 0.2 (SysAdmin,
    Diuk et al. Sec. 3.3). Its predictions are within 2 eps1 of the truth (proof of Thm 1) with probability
    >= 1 - delta only with m="calibrated" and k >= the true number of parents; with the paper's m the
    accuracy in params is nominal. Offline, the conditions are checked once with the counts of all the data
    and the lowest-index passing candidate is used (online, a learner keeps the first one accepted over
    time, which may be another, equally valid, candidate; the first-m counts freeze, so the conditions are
    monotone)."""
    if m == "calibrated":
        m = strehl_calibrated_m(dataset, k, eps1, delta)
    data = _transitions(dataset)
    factors = {}
    for i in range(len(dataset.state_vars)):
        for a, (rows, nxt) in data.items():
            factors[(i, a)] = _strehl_factor(
                dataset.B, rows, nxt[:, i], dataset.state_cards, dataset.state_cards[i], k, int(m), eps1
            )
    params = {"k": k, "m": int(m), "eps1": eps1, "delta": delta, "accuracy": 2 * eps1}
    return KwikStructure("strehl", factors, params)


# ==========================================================================================
# Diuk, Li, Leffler (ICML 2009), Alg. 1
# ==========================================================================================
def _meteorologist_predictions(rows, values, cands, cards, card, m_sub):
    """Each meteorologist's sub-algorithm, the KWIK learner of its parent subset: at every step t, its
    prediction for the input rows[t] (DONT_KNOW until it has m_sub samples of that setting, then the
    frequency of the first m_sub; it only receives the label while it says DONT_KNOW, Alg. 1 lines 7-9, so
    it never uses more). Returns known [H, N], pred [H, N, card] and the final tables of every setting."""
    N, H = len(rows), len(cands)
    known = np.zeros((H, N), dtype=bool)
    pred = np.zeros((H, N, card))
    tables = []
    onehot = np.eye(card)[values]
    for h, U in enumerate(cands):
        cu = [cards[u] for u in U]
        codes = encode(rows[:, list(U)], cu)
        n_settings = int(np.prod(cu))
        seen = np.zeros(n_settings, dtype=np.int64)
        sums = np.zeros((n_settings, card))
        for t in range(N):  # sequential: the prediction at t uses only the samples before t
            c = codes[t]
            if seen[c] >= m_sub:
                known[h, t] = True
                pred[h, t] = sums[c] / m_sub
            else:
                sums[c] += onehot[t]
                seen[c] += 1
        tables.append((cu, seen, sums))
    return known, pred, tables


def _met_factor(B, rows, values, cards, card, D, m, m_sub, eps):
    n = B.shape[1]
    cands = list(itertools.combinations(range(n), min(D, n)))
    H = len(cands)
    known, pred, tables = _meteorologist_predictions(rows, values, cands, cards, card, m_sub)
    R = list(range(H))  # line 3
    c = np.zeros((H, H), dtype=np.int64)  # line 4
    delta_sq = np.zeros((H, H))
    onehot = np.eye(card)[values]
    for t in range(len(rows)):  # line 5
        live = np.array(R)
        if not known[live, t].all():  # line 7: some meteorologist in R says "don't know"
            continue  # line 8-9 (its label was given to it in _meteorologist_predictions)
        p = pred[live, t]  # [|R|, card]
        if (p.max(0) - p.min(0)).max() <= eps + 1e-12:  # line 11: all within eps -> predict (line 12)
            continue
        z = onehot[t]  # line 14: "don't know", observe z
        err = ((p - z) ** 2).sum(1)
        removed = set()
        # line 15, pairs in index order; R is updated within the step: a pair with a member removed earlier in
        # the same step is skipped (the pseudo-code does not say; this reading cannot empty R)
        for a, b in itertools.combinations(range(len(live)), 2):
            i, j = live[a], live[b]
            if i in removed or j in removed:
                continue
            if np.abs(p[a] - p[b]).max() >= eps / 2 - 1e-12:
                c[i, j] += 1  # line 16
                delta_sq[i, j] += err[a] - err[b]  # line 17
                if c[i, j] >= m:  # line 18
                    removed.add(i if delta_sq[i, j] > TIE else j)  # line 19 (an exact tie removes j)
        R = [h for h in R if h not in removed]
    # the final predictions at every state, with the final R (lines 6-12 at input s)
    S = len(B)
    predict, ok = np.zeros((S, card)), np.zeros(S, dtype=bool)
    preds = []
    for h in R:
        cu, seen, sums = tables[h]
        at = encode(B[:, list(cands[h])], cu)
        preds.append((seen[at] >= m_sub, sums[at] / m_sub))
    all_known = np.all([k for k, _ in preds], axis=0)
    P = np.stack([p for _, p in preds])  # [|R|, S, card]
    agree = (P.max(0) - P.min(0)).max(-1) <= eps + 1e-12
    ok = all_known & agree
    # line 12: (max + min) / 2 per value; for a factor with > 2 values and >= 3 survivors it need not sum to 1
    # (each value is still within eps); the point model normalises it (scm_learning/kwik_intervals.kwik_point)
    predict[ok] = ((P.max(0) + P.min(0)) / 2)[ok]
    accepted = np.where(ok, 0, DONT_KNOW)  # "known" marker; the survivors are the structure
    return KwikFactor(cands, accepted, R, predict, ok)


def met_sub_m(dataset: Dataset, D: int, eps: float, delta: float) -> int:
    """Alg. 1, line 2: every sub-algorithm runs with accuracy eps / 8 and confidence delta / (k + 1), k the
    number of meteorologists. For the frequency learner of a parent subset ("don't know" until m samples of a
    setting, then their mean) Hoeffding gives m = ln(2 N / delta') / (2 (eps / 8)^2), N its settings x values,
    and delta' = delta / (k + 1) shared by the factors x actions learners. The explicit constant is ours; the
    paper states the order (Thm 1)."""
    cards, n = dataset.state_cards, len(dataset.state_vars)
    cands = list(itertools.combinations(range(n), min(D, n)))
    settings = max(int(np.prod([cards[u] for u in c])) for c in cands)
    values = max(1 if c == 2 else c for c in cards)
    d_sub = delta / (len(cands) + 1) / (n * len(dataset.actions))
    return math.ceil(math.log(2 * settings * values / d_sub) / (2 * (eps / 8) ** 2))


def met_rmax(
    dataset: Dataset,
    D: int = 2,
    m: int = 7,
    m_sub: int | str | None = None,
    eps: float = 0.2,
    delta: float = 0.05,
) -> KwikStructure:
    """The Adaptive k-Meteorologists Algorithm (Diuk et al., Alg. 1) with one meteorologist per subset of D
    state variables (Sec. 3.1), one learner per (factor, action), replayed on the logged transitions.
    m: the number of disagreements before a comparison is decided (Alg. 1, line 18); m_sub: the samples a
    meteorologist needs before it predicts (default m: the paper's experiments give one m, "the number of
    samples required by each meteorologist before it assumes its prediction is known", Sec. 3.3.1); eps: the
    accuracy of a prediction (lines 11, 15; the paper does not state the eps of its experiments). Paper
    settings: m = 7 (Stock), m = 50 (SysAdmin); m_sub="calibrated": met_sub_m (line 2: eps / 8 and
    delta / (k + 1)). Its predictions are eps-accurate (Thm 1) only with Thm 1's parameters (m_sub calibrated
    and m of the order eps^-2 ln(k / delta), whose constant the paper does not give); with the paper's
    experimental m the accuracy in params is nominal."""
    if m_sub == "calibrated":
        m_sub = met_sub_m(dataset, D, eps, delta)
    m_sub = m if m_sub is None else int(m_sub)
    data = _transitions(dataset)
    factors = {}
    for i in range(len(dataset.state_vars)):
        for a, (rows, nxt) in data.items():
            factors[(i, a)] = _met_factor(
                dataset.B, rows, nxt[:, i], dataset.state_cards, dataset.state_cards[i], D, m, m_sub, eps
            )
    params = {"D": D, "m": m, "m_sub": m_sub, "eps": eps, "delta": delta, "accuracy": eps}
    return KwikStructure("met_rmax", factors, params)
