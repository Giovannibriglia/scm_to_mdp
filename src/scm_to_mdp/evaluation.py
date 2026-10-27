"""Evaluation on the true model (step S9), an MDPGroundTruth (scm_to_mdp/ground_truth.py).

Values of a learned policy:
    promise     the value the method claims (for a robust method: a guaranteed lower bound);
    true value  the exact value of the policy in the true environment;
    V_star      the reference values passed by the caller (tool/evaluate.py passes V*_obs, the value
                of the best policy over the observed state; = V* when nothing is hidden).
    regret = V_star - true value;  gap = promise - true value (<= 0: the promise holds).
Each is reported at the nominal start (the environment's start distribution) and averaged over k
start states drawn uniformly among the non-terminal states.

Validity of the promise, over the non-terminal OBSERVED states of one run:
    safe              promise(o) <= V^pi(o) for every observed state o (averaged over runs: the hold rate
                      of the promise, which a valid robust method keeps >= 1 - delta);
    violation_rate    share of observed states with promise(o) > V^pi(o);
    worst_violation   max_o (promise(o) - V^pi(o))^+, in value units (0 when safe).
With a hidden variable, V^pi(o) is the true value averaged over the hidden part of o with the weights of
the observed MDP (MDPGroundTruth.hidden_prior): the promise is a bound on the process the learner sees,
not on every value of a variable it never sees (a storm-on state may lie below it). Nothing hidden: o = s.

Structure of a learned 2-slice graph (inputs at t -> state variables at t+1, and same-step edges
x_j' -> x_i') against the truth:
    SHD   structural Hamming distance = missing + extra edges (edges point forward in time, so no
          edge can be reversed). A same-step edge counts as a pair, whatever its direction: both
          directions give the same kernel (the truth's direction is a factorisation of a shared draw);
    SID   structural intervention distance (Peters & Buhlmann, Neural Computation 27(3):771-799,
          2015), computed with gadjid (Henckel, Wurtzen, Weichwald, arXiv:2402.08616): the number of
          (cause, effect) pairs whose interventional distribution is wrong when one adjusts for the
          learned parents. Here the inputs have no parents, so an extra parent is still a valid
          adjustment set: SID counts only the harm of MISSING parents (SID = 0 iff every learned
          parent set contains the true one, the structural condition for the factor intervals to bound
          the true mechanisms).
    recall      true edges found / true edges (1 - missing / true): no false negatives = recall 1
    precision   true edges found / edges found (1 - extra / found): no false positives = precision 1;
                NaN for an empty graph
    action_edges, action_missing, action_recall   the true edges from an action column to a state
          variable, how many the learned graph lacks, and the share it keeps: the edges a planner
          needs to tell actions apart (an SHD of 1 can be the only action edge of a family).

Near-optimality certificate of a robust method (Jackson, Laurenti, Frew, Lahijanian, HSCC 2021, Thm 2):
on the same interval MDP, V_opt = max over policies and kernels in the set (optimistic value
iteration). When the true kernel lies in the set, V_rob <= V^pi_rob <= V* <= V_opt, so
    V* - V^pi_rob <= V_opt - V_rob     (cert_gap; a bound computed without the truth)
    V_opt, cert_gap, cert_gap_K   at the nominal start and averaged over the k starts
    cert_holds   V*_obs - V^pi <= cert_gap in every non-terminal state (checked here on the truth)
"""

import gadjid
import numpy as np

from scm_to_mdp.core import NEXT_MARK, Dataset, FactorBounds, Graph, Solution, encode
from scm_to_mdp.ground_truth import MDPGroundTruth
from scm_to_mdp.solving import value_iteration


# ---- values -------------------------------------------------------------------------------------
def optimal_values(truth: MDPGroundTruth, gamma: float) -> np.ndarray:
    """V*[s] for every state."""
    return truth.optimal_values(gamma)


def random_starts(truth: MDPGroundTruth, k: int, seed: int) -> np.ndarray:
    """k start states drawn uniformly (without replacement) among the non-terminal states."""
    live = np.flatnonzero(~truth.reward.terminal)
    return np.random.default_rng(seed).choice(live, size=min(k, len(live)), replace=False)


def evaluate(
    truth: MDPGroundTruth,
    dataset: Dataset,
    solution: Solution,
    gamma: float,
    V_star: np.ndarray,
    starts: np.ndarray,
) -> dict:
    """Promise, true value, regret and gap at the nominal start and over `starts`; safety."""
    observed = truth.observed_index()  # full state -> the state the learner sees (identity if none hidden)
    promised = solution.V[observed]  # [S] the promise in every full state
    V_true = true_values(truth, dataset, solution, gamma)
    promise, true, best = truth.start @ promised, truth.start @ V_true, truth.start @ V_star
    promise_obs, true_obs = solution.V, observed_average(truth, V_true)  # over the observed states
    live_obs = ~np.asarray(dataset.reward.terminal, dtype=bool)
    excess = promise_obs[live_obs] - true_obs[live_obs]
    solution = Solution(promised, solution.pi[observed])  # over the full states, for what follows
    return {
        "promise": float(promise),
        "true": float(true),
        "regret": float(best - true),
        "gap": float(promise - true),
        "regret_K": float(np.mean(V_star[starts] - V_true[starts])),
        "gap_K": float(np.mean(solution.V[starts] - V_true[starts])),
        "safe": bool(np.all(excess <= 1e-6)),
        "worst_gap": float(np.max(excess)),
        "violation_rate": float(np.mean(excess > 1e-6)),
        "worst_violation": float(max(0.0, np.max(excess))),
    }


def observed_average(truth: MDPGroundTruth, V: np.ndarray) -> np.ndarray:
    """[S_obs] a value over the full states averaged over the hidden part of every observed state, with
    the observed MDP's weights (hidden_prior); nothing hidden: V itself."""
    if not truth.hidden:
        return V
    idx, w = truth.observed_index(), truth.hidden_prior()
    S_obs = truth.learner_spec.n_states
    return np.bincount(idx, weights=w * V, minlength=S_obs) / np.bincount(idx, weights=w, minlength=S_obs)


def true_values(truth: MDPGroundTruth, dataset: Dataset, solution: Solution, gamma: float) -> np.ndarray:
    """[S] the exact value of the learned policy in every full state of the true model."""
    observed = truth.observed_index()
    to_true = truth.actions_of(dataset)  # learner action index -> true action
    if solution.probs is None:
        true_policy = to_true[solution.pi[observed]]
    else:  # a stochastic policy, as probabilities over the true actions in every full state
        true_policy = np.zeros((len(observed), truth.P.shape[1]))
        true_policy[:, to_true] = solution.probs[observed]
    return value_iteration.policy_value(truth.P, truth.reward, true_policy, gamma)


# ---- structure ----------------------------------------------------------------------------------
def adjacency(parents: dict[str, tuple[str, ...]], dataset: Dataset) -> np.ndarray:
    """Adjacency matrix (row -> column) of the 2-slice graph over inputs + next-state variables."""
    nodes = dataset.inputs + dataset.next_vars
    index = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    for v, v_next in zip(dataset.state_vars, dataset.next_vars, strict=True):
        for p in parents[v]:
            A[index[p], index[v_next]] = 1
    return A


def _edges(parents: dict[str, tuple[str, ...]]) -> set:
    """Edges (parent, child); a same-step edge y' -> x as the unordered pair {y, x}."""
    return {
        frozenset((p.removesuffix(NEXT_MARK), v)) if p.endswith(NEXT_MARK) else (p, v)
        for v, ps in parents.items()
        for p in ps
    }


def structure_distances(graph: Graph, true_parents: dict[str, tuple[str, ...]], dataset: Dataset) -> dict:
    found, true = _edges(graph.parents), _edges(true_parents)
    truth, guess = adjacency(true_parents, dataset), adjacency(graph.parents, dataset)
    actions = {e for e in true if isinstance(e, tuple) and e[0] in set(dataset.action_vars)}
    kept = len(actions & found)
    hits = len(found & true)
    return {
        "shd": len(found ^ true),
        "missing": len(true - found),
        "extra": len(found - true),
        "recall": hits / len(true) if true else float("nan"),
        "precision": hits / len(found) if found else float("nan"),
        "sid": int(gadjid.sid(truth, guess, edge_direction="from row to column")[1]),
        "action_edges": len(actions),
        "action_missing": len(actions) - kept,
        "action_recall": kept / len(actions) if actions else float("nan"),
    }


# ---- near-optimality certificate (robust methods) -------------------------------------------------
CERTIFICATE_TOL = 1e-5  # V_opt: the stopping rule's distance to the fixed point (it stays an upper bound)


def optimistic_values(model, gamma: float, tol: float = CERTIFICATE_TOL) -> np.ndarray:
    """[S] V_opt: the best value over policies AND kernels of the interval MDP (a FactorBounds is first
    relaxed to its box, as box_robust_vi does; for the exact product set the box is a superset, so
    V_opt stays an upper bound): robust value iteration with a cooperative nature (solving/robust_vi.py),
    returned as a sound UPPER bound on the fixed point (solving/stopping.py). tol 1e-5 rather than the
    solvers' 1e-7: a quarter fewer sweeps, and V_opt moved by at most 4e-14 on learned Chemistry
    models (the corrected iterate is close to the fixed point when the error shrinks at one rate)."""
    from scm_to_mdp.compilation import product_bounds
    from scm_to_mdp.solving import robust_vi

    mdp = product_bounds.compile(model) if isinstance(model, FactorBounds) else model
    return robust_vi.solve(mdp, gamma, tol=tol, nature="cooperative").V


def certificate(model, truth: MDPGroundTruth, solution: Solution, V_true: np.ndarray, V_star: np.ndarray,
                gamma: float, starts: np.ndarray, V_opt_obs: np.ndarray | None = None) -> dict:  # fmt: skip
    """V_opt and the certificate V_opt - V_rob (see the module docstring); cert_holds over the observed
    states, with the true values averaged over the hidden part (as safe, in evaluate). V_opt_obs: the
    optimistic values if already computed (optimistic_values)."""
    observed = truth.observed_index()
    if V_opt_obs is None:
        V_opt_obs = optimistic_values(model, gamma)
    V_opt = V_opt_obs[observed]
    promised = solution.V[observed]
    cert = V_opt - promised
    cert_obs = V_opt_obs - solution.V
    regret_obs = observed_average(truth, V_star) - observed_average(truth, V_true)
    live = ~np.asarray(model.reward.terminal, dtype=bool)  # the observed non-terminal states
    return {
        "V_opt": float(truth.start @ V_opt),
        "cert_gap": float(truth.start @ cert),
        "cert_gap_K": float(np.mean(cert[starts])),
        "cert_holds": bool(np.all(regret_obs[live] <= cert_obs[live] + 1e-6)),
    }


# ---- the learned probability sets (S4) ----------------------------------------------------------
def factor_intervals(model, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """lo, hi [S, K, n, C]: the per-factor intervals of a learned set. A FactorBounds gives them
    directly; an IntervalMDP (joint intervals L <= P(s' | s, a) <= U, flat IMDP) is projected onto
    every factor's marginal: P(x_i' = v) = sum of P(s') over the s' with x_i' = v, so it lies in
    [max(sum L, 1 - sum_rest U), min(sum U, 1 - sum_rest L)] (the box intersected with the simplex)."""
    if isinstance(model, FactorBounds):
        return model.lo, model.hi
    L, U = model.L, model.U  # [S, K, S']
    B, cards = dataset.B, dataset.state_cards
    lo = np.zeros((*L.shape[:2], len(cards), max(cards)))
    hi = np.zeros_like(lo)
    total_L, total_U = L.sum(-1), U.sum(-1)
    for i, card in enumerate(cards):
        for v in range(card):
            mask = B[:, i] == v
            in_L, in_U = L[..., mask].sum(-1), U[..., mask].sum(-1)
            lo[:, :, i, v] = np.maximum(in_L, 1 - (total_U - in_U))
            hi[:, :, i, v] = np.minimum(in_U, 1 - (total_L - in_L))
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def true_factors(truth: MDPGroundTruth, dataset: Dataset) -> np.ndarray:
    """[S, K, n, C] P(x_i' = v | s, a_k) in the observed MDP (MDPGroundTruth.observed_kernel), at the
    dataset's states and action vectors; padding values are 0."""
    P = truth.observed_kernel()[:, truth.actions_of(dataset)]  # [S, K, S']
    B, cards = dataset.B, dataset.state_cards
    out = np.zeros((*P.shape[:2], len(cards), max(cards)))
    for i, card in enumerate(cards):
        for v in range(card):
            out[:, :, i, v] = P[..., B[:, i] == v].sum(-1)
    return out


def optimal_occupancy(truth: MDPGroundTruth, dataset: Dataset, gamma: float) -> np.ndarray:
    """[S, K] the discounted occupancy of (s, a_k) under the best policy over the observed state, in
    the observed MDP from the start distribution (0 for actions the dataset does not have)."""
    P = truth.observed_kernel()
    pi, _ = truth.observed_optimal(gamma)  # [S_obs] true action
    S = P.shape[0]
    start = np.bincount(truth.observed_index(), weights=truth.start, minlength=S)
    P_pi = P[np.arange(S), pi]  # [S, S']
    d = (1 - gamma) * np.linalg.solve((np.eye(S) - gamma * P_pi).T, start)
    out = np.zeros((S, len(dataset.actions)))
    learner = {int(a): k for k, a in enumerate(truth.actions_of(dataset))}
    for s in range(S):
        k = learner.get(int(pi[s]))
        if k is not None:
            out[s, k] = d[s]
    return out


def interval_metrics(model, dataset: Dataset, truth: MDPGroundTruth, gamma: float) -> dict:
    """The learned sets against the truth (S4):
    s4_width     mean interval width (hi - lo) over factors and values, weighted by the occupancy of
                 the best policy over the observed state (where the width matters for the value)
    s4_coverage  share of the true probabilities P(x_i' = v | s, a) inside their intervals, over
                 every state, action vector, factor and value (validity of S4 on its own)
    s4_coverage_seen   the same over the (s, a) the data visit (an unseen row is [0, 1] and covers
                 trivially, which inflates s4_coverage)
    s4_unseen    share of parent rows with no data (their interval is [0, 1])
    s4_rows      number of parent rows (every factor's family; the whole (s, a) grid for flat IMDP)
    The last two come from the learner when it reports them (FactorBounds.info)."""
    lo, hi = factor_intervals(model, dataset)
    P = true_factors(truth, dataset)
    cards = dataset.state_cards
    valid = np.arange(lo.shape[-1])[None, :] < np.array(cards)[:, None]  # [n, C]
    inside = (lo <= P + 1e-9) & (P <= hi + 1e-9)
    width = ((hi - lo) * valid).sum((-2, -1)) / valid.sum()  # [S, K]
    w = optimal_occupancy(truth, dataset, gamma)
    s_logged = dataset.state_index(dataset.df[dataset.state_vars].to_numpy())
    K = lo.shape[1]
    visits = np.bincount(s_logged * K + dataset.action_index(), minlength=lo.shape[0] * K).reshape(-1, K)
    seen = inside[visits > 0]  # [rows seen, n, C]
    out = {
        "s4_width": float((w * width).sum() / w.sum()) if w.sum() > 0 else float("nan"),
        "s4_coverage": float(inside[..., valid].mean()),
        "s4_coverage_seen": float(seen[:, valid].mean()) if len(seen) else float("nan"),
    }
    info = getattr(model, "info", {}) or {}
    if "s4_rows" in info:
        out["s4_rows"], out["s4_unseen"] = info["s4_rows"], info["s4_unseen"]
    else:  # joint rows (s, a): unseen when every entry is [0, 1]
        unseen = np.all((model.L <= 1e-12) & (model.U >= 1 - 1e-12), axis=-1)
        out["s4_rows"], out["s4_unseen"] = int(unseen.size), float(unseen.mean())
    return out


# ---- coverage of a task's rows by the data (claim B1: "structure buys coverage") ------------------
COVERAGE_COUNTS = (1, 5, 20)  # a row is "unseen" with fewer logged transitions than this


def task_occupancy(truth: MDPGroundTruth, dataset: Dataset, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """(d [S_obs], pi [S_obs]): the normalised discounted occupancy of the best policy over the observed
    state (pi: its true action in every observed state) in the observed MDP, from the start
    distribution; an episode stops on arrival in a terminal state, which is never acted in (d = 0
    there). The reward and the terminal states are the dataset's (a task's data carry its reward)."""
    key = ("task_occupancy", gamma)
    if key not in truth._cache:
        P = truth.observed_kernel()
        pi, _ = truth.observed_optimal(gamma)
        S = P.shape[0]
        live = ~np.asarray(dataset.reward.terminal, dtype=bool)
        start = np.bincount(truth.observed_index(), weights=truth.start, minlength=S)
        P_pi = P[np.arange(S), pi] * live[:, None]  # a terminal state does not go on
        mu = np.linalg.solve((np.eye(S) - gamma * P_pi).T, start) * live
        truth._cache[key] = (mu / mu.sum() if mu.sum() > 0 else mu, pi)
    return truth._cache[key]


def _family_counts(dataset: Dataset, parents: dict) -> dict:
    """{v: (columns, cards, counts per parent row)} over the logged transitions (slice-t parents)."""
    out = {}
    for v in dataset.state_vars:
        ps = [p for p in parents.get(v, ()) if not p.endswith(NEXT_MARK) and p in dataset.cards]
        cards = [dataset.cards[p] for p in ps]
        code = encode(dataset.df[ps].to_numpy(), cards) if ps else np.zeros(len(dataset.df), dtype=np.int64)
        out[v] = (ps, cards, np.bincount(code, minlength=int(np.prod(cards)) if ps else 1))
    return out


def _row_counts(truth: MDPGroundTruth, dataset: Dataset, families: dict, states, actions) -> np.ndarray:
    """[E, n] the logged count of every factor's parent row at (states[e], true action actions[e])."""
    values = {v: dataset.B[states, i] for i, v in enumerate(dataset.state_vars)}
    columns = truth.actions[actions]  # [E, n_a] the action columns of the true actions
    values |= {a: columns[:, j] for j, a in enumerate(dataset.action_vars)}
    out = np.zeros((len(states), len(dataset.state_vars)), dtype=np.int64)
    for i, v in enumerate(dataset.state_vars):
        ps, cards, counts = families[v]
        code = encode(np.stack([values[p] for p in ps], axis=1), cards) if ps else np.zeros(len(states), int)
        out[:, i] = counts[code]
    return out


def coverage(truth: MDPGroundTruth, dataset: Dataset, gamma: float, parents: dict, prefix="cov") -> dict:
    """What the data cover of the rows the task's best policy needs (method-independent with the true
    parents; with a method's learned parents, what that method can estimate). With d the policy's
    occupancy (task_occupancy) and k in COVERAGE_COUNTS:
        <prefix>_joint_k    occupancy on (s, pi(s)) with fewer than k logged transitions (a flat model's
                            unseen rows; k = 1: never seen)
        <prefix>_factor_k   occupancy on (s, pi(s)) where some factor's parent row has fewer than k
                            (the factored model's unseen rows; <= _joint_k)
        <prefix>_alt        occupancy on states where SOME action (every true action, the planner must
                            compare them) has a factor row never seen
    joint - factor is the occupancy that only a factored model can estimate from the data."""
    d, pi = task_occupancy(truth, dataset, gamma)
    to_true = truth.actions_of(dataset)
    s_logged = dataset.state_index(dataset.df[dataset.state_vars].to_numpy())
    a_logged = to_true[dataset.action_index()]
    A = truth.P.shape[1]
    joint = np.bincount(s_logged * A + a_logged, minlength=len(d) * A).reshape(len(d), A)
    states = np.flatnonzero(d > 0)
    families = _family_counts(dataset, parents)
    rows = _row_counts(truth, dataset, families, states, pi[states]).min(1)  # the least-seen factor row
    out = {}
    for k in COVERAGE_COUNTS:
        out[f"{prefix}_joint_{k}"] = float(d[states][joint[states, pi[states]] < k].sum())
        out[f"{prefix}_factor_{k}"] = float(d[states][rows < k].sum())
    every = np.repeat(np.arange(A)[None, :], len(states), axis=0)
    alt = _row_counts(truth, dataset, families, np.repeat(states, A), every.ravel()).min(1).reshape(-1, A)
    out[f"{prefix}_alt"] = float(d[states][(alt < 1).any(1)].sum())
    return out


# ---- the learned model against the true kernel (claim C, and the containment event of S4) ---------
CHUNK = 64  # states per block when a factored box is expanded over the next states


def _factor_box_rows(model: FactorBounds, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[len(s), K, S'] the box of a FactorBounds without same-step parents at states s: products of the
    tightened factor intervals (C order over the next state: the first variable is the most significant)."""
    from scm_to_mdp.compilation.product_bounds import tighten

    L = U = None
    for i, card in enumerate(model.cards):
        lo, hi = tighten(model.lo[s, :, i, :card], model.hi[s, :, i, :card])
        if L is None:
            L, U = lo, hi
        else:
            L = (L[..., :, None] * lo[..., None, :]).reshape(*L.shape[:-1], -1)
            U = (U[..., :, None] * hi[..., None, :]).reshape(*U.shape[:-1], -1)
    return L, np.minimum(U, 1.0)


def model_errors(model, truth: MDPGroundTruth, dataset: Dataset, gamma: float) -> dict:
    """The learned transition model against the true observed kernel, at every observed state and every
    action vector of the data, under three weightings: the logged visits (data: the i.i.d. setting),
    the occupancy of the best policy (task), and uniform over the non-terminal states and actions
    (uniform: a randomised intervention on the state and the actions within their logged values; the
    model answers it correctly only if its factors are the mechanisms). Point models (L == U):
        tv_<w>     total-variation distance 1/2 sum_s' |P_hat - P|
        kl_<w>     KL(P || P_hat) in nats (the point models are smoothed; inf where P_hat = 0 < P)
    Interval models:
        contain_<w>   share of rows with L <= P <= U entry-wise (the event a robust promise relies on)
        contain_all   every row contains the true one (the event E every promise relies on)
        contain_all_seen   every row with data contains the true one
        width_seen, width_unseen   mean total width sum_s' (U - L) over rows with / without data
    A FactorBounds enters as its box (tightened factor intervals, products over the factors); one with
    same-step parents only when small enough to compile. Action vectors absent from the data are not in
    the model (see actions_absent)."""
    from scm_to_mdp.compilation import product_bounds

    P_all = truth.observed_kernel()[:, truth.actions_of(dataset)]  # [S, K, S']
    S, K, _ = P_all.shape
    live = ~np.asarray(dataset.reward.terminal, dtype=bool)
    s_logged = dataset.state_index(dataset.df[dataset.state_vars].to_numpy())
    visits = np.bincount(s_logged * K + dataset.action_index(), minlength=S * K).reshape(S, K)
    d, pi = task_occupancy(truth, dataset, gamma)
    learner = {int(a): k for k, a in enumerate(truth.actions_of(dataset))}
    task = np.zeros((S, K))
    for s in np.flatnonzero(d > 0):
        if int(pi[s]) in learner:
            task[s, learner[int(pi[s])]] = d[s]
    weights = {"data": visits * live[:, None], "task": task, "uniform": np.repeat(live[:, None], K, 1) * 1.0}
    if isinstance(model, FactorBounds):
        # a point model: every factor exact (a conditional factor's lo/hi hold its hull over the
        # same-step parents' values, so only its own rows lo_i, hi_i tell)
        plain = [i for i in range(len(model.cards)) if i not in model.conditional]
        point = np.array_equal(model.lo[:, :, plain], model.hi[:, :, plain]) and all(
            np.array_equal(lo_i, hi_i) for _, lo_i, hi_i in model.conditional.values()
        )
        if model.conditional:
            if S * K * S > 2e7:
                return {}
            dense = (product_bounds.exact if point else product_bounds.compile)(model)
            rows = lambda s: (dense.L[s], dense.U[s])  # noqa: E731
        else:
            rows = lambda s: _factor_box_rows(model, s)  # noqa: E731
    else:
        rows = lambda s: (model.L[s], model.U[s])  # noqa: E731
        point = np.array_equal(model.L, model.U)
    tv, kl, inside, width = (np.zeros((S, K)) for _ in range(4))
    for start in range(0, S, CHUNK):
        s = np.arange(start, min(start + CHUNK, S))
        L, U = rows(s)
        P = P_all[s]
        if point:
            tv[s] = 0.5 * np.abs(L - P).sum(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl[s] = np.where(P > 0, P * (np.log(P) - np.log(L)), 0.0).sum(-1)
        else:
            inside[s] = np.all((L <= P + 1e-9) & (P <= U + 1e-9), axis=-1)
            width[s] = (U - L).sum(-1)
    out = {}
    for name, w in weights.items():
        total = w.sum()
        if total <= 0:
            continue
        on = w > 0  # only the weighted rows (an infinite KL at a row of weight 0 does not count)
        if point:
            out[f"tv_{name}"] = float((w[on] * tv[on]).sum() / total)
            out[f"kl_{name}"] = float((w[on] * kl[on]).sum() / total)
        else:
            out[f"contain_{name}"] = float((w[on] * inside[on]).sum() / total)
    if not point:
        on = live[:, None] & np.ones((S, K), dtype=bool)
        out["contain_all"] = bool(inside[on].all())
        seen = on & (visits > 0)
        out["contain_all_seen"] = bool(inside[seen].all()) if seen.any() else True
        out["width_seen"] = float(width[seen].mean()) if seen.any() else float("nan")
        out["width_unseen"] = float(width[on & ~seen].mean()) if (on & ~seen).any() else float("nan")
    return out


def actions_absent(truth: MDPGroundTruth, dataset: Dataset) -> float:
    """Share of the environment's action vectors that never appear in the data (no method has them)."""
    return float(1.0 - len(dataset.actions) / truth.P.shape[1])
