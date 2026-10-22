"""Evaluation on the true model (step S9), an MDPGroundTruth (scm_to_mdp/ground_truth.py).

Values of a learned policy:
    promise     the value the method claims (for a robust method: a guaranteed lower bound);
    true value  the exact value of the policy in the true environment;
    V_star      the reference values passed by the caller (tool/evaluate.py passes V*_obs, the value
                of the best policy over the observed state; = V* when nothing is hidden).
    regret = V_star - true value;  gap = promise - true value (<= 0: the promise holds).
Each is reported at the nominal start (the environment's start distribution) and averaged over k
start states drawn uniformly among the non-terminal states.

Validity of the promise, over the non-terminal states S_live of one run:
    safe              promise(s) <= V^pi(s) for every s (averaged over runs: the hold rate of the
                      promise, which a valid robust method keeps >= 1 - delta);
    violation_rate    share of states with promise(s) > V^pi(s);
    worst_violation   max_s (promise(s) - V^pi(s))^+, in value units (0 when safe).

Structure of a learned 2-slice graph (inputs at t -> state variables at t+1) against the truth:
    SHD   structural Hamming distance = missing + extra edges (edges point forward in time, so no
          edge can be reversed);
    SID   structural intervention distance (Peters & Buhlmann, Neural Computation 27(3):771-799,
          2015), computed with gadjid (Henckel, Wurtzen, Weichwald, arXiv:2402.08616): the number of
          (cause, effect) pairs whose interventional distribution is wrong when one adjusts for the
          learned parents. Here the inputs have no parents, so an extra parent is still a valid
          adjustment set: SID counts only the harm of MISSING parents (SID = 0 iff every learned
          parent set contains the true one, the structural condition for the factor intervals to bound
          the true mechanisms).
"""

import gadjid
import numpy as np

from scm_to_mdp.core import Dataset, Graph, Solution
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
    to_true = truth.actions_of(dataset)  # learner action index -> true action
    if solution.probs is None:
        true_policy = to_true[solution.pi[observed]]
    else:  # a stochastic policy, as probabilities over the true actions in every full state
        true_policy = np.zeros((len(observed), truth.P.shape[1]))
        true_policy[:, to_true] = solution.probs[observed]
    V_true = value_iteration.policy_value(truth.P, truth.reward, true_policy, gamma)
    live = ~truth.reward.terminal  # a terminal state is never acted in
    promise, true, best = truth.start @ promised, truth.start @ V_true, truth.start @ V_star
    solution = Solution(promised, solution.pi[observed])  # over the full states, for what follows
    return {
        "promise": float(promise),
        "true": float(true),
        "regret": float(best - true),
        "gap": float(promise - true),
        "regret_K": float(np.mean(V_star[starts] - V_true[starts])),
        "gap_K": float(np.mean(solution.V[starts] - V_true[starts])),
        "safe": bool(np.all(solution.V[live] <= V_true[live] + 1e-6)),
        "worst_gap": float(np.max(solution.V[live] - V_true[live])),
        "violation_rate": float(np.mean(solution.V[live] > V_true[live] + 1e-6)),
        "worst_violation": float(max(0.0, np.max(solution.V[live] - V_true[live]))),
    }


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


def structure_distances(graph: Graph, true_parents: dict[str, tuple[str, ...]], dataset: Dataset) -> dict:
    found = {(p, v) for v, ps in graph.parents.items() for p in ps}
    true = {(p, v) for v, ps in true_parents.items() for p in ps}
    truth, guess = adjacency(true_parents, dataset), adjacency(graph.parents, dataset)
    return {
        "shd": len(found ^ true),
        "missing": len(true - found),
        "extra": len(found - true),
        "sid": int(gadjid.sid(truth, guess, edge_direction="from row to column")[1]),
    }
