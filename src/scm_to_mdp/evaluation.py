"""Evaluation on the true model (step S9).

Values of a learned policy:
    promise     the value the method claims (for a robust method: a guaranteed lower bound);
    true value  the exact value of the policy in the true environment;
    V*          the optimal value (the oracle).
    regret = V* - true value;  gap = promise - true value (<= 0: the promise holds).
Each is reported at the nominal start (the environment's start distribution) and averaged over K
start states drawn uniformly among the non-terminal states.

Validity of the promise, over the non-terminal states S_live of one run:
    safe              promise(s) <= V^pi(s) for every s (averaged over runs: the coverage, which a valid
                      robust method keeps >= 1 - delta);
    violation_rate    share of states with promise(s) > V^pi(s);
    worst_violation   max_s (promise(s) - V^pi(s))^+, in value units (0 when safe).
A method is "safe" in a run when
promise <= true value in every non-terminal state.

Structure of a learned 2-slice graph (inputs at t -> state variables at t+1) against the truth:
    SHD   structural Hamming distance = missing + extra edges (edges point forward in time, so no
          edge can be reversed);
    SID   structural intervention distance (Peters & Buhlmann, Neural Computation 27(3):771-799,
          2015), computed with gadjid (Henckel, Wurtzen, Weichwald, arXiv:2402.08616): the number of
          (cause, effect) pairs whose interventional distribution is wrong when one adjusts for the
          learned parents. Here the inputs have no parents, so an extra parent is still a valid
          adjustment set: SID counts only the harm of MISSING parents (SID = 0 iff every learned
          parent set contains the true one, which is when the robust promise is valid).
"""

import gadjid
import numpy as np

from scm_to_mdp.core import Dataset, Graph, Solution
from scm_to_mdp.envs.base import Env
from scm_to_mdp.solving import value_iteration


# ---- values -------------------------------------------------------------------------------------
def optimal_values(env: Env, gamma: float) -> np.ndarray:
    """V*[s] for every state."""
    return value_iteration.solve(env.P, env.true_reward, gamma).V


def random_starts(env: Env, k: int, seed: int) -> np.ndarray:
    """k start states drawn uniformly (without replacement) among the non-terminal states."""
    live = np.flatnonzero(~env.true_reward.terminal)
    return np.random.default_rng(seed).choice(live, size=min(k, len(live)), replace=False)


def evaluate(
    env: Env, dataset: Dataset, solution: Solution, gamma: float, V_star: np.ndarray, starts: np.ndarray
) -> dict:
    """Promise, true value, regret and gap at the nominal start and over `starts`; safety."""
    env_policy = env.env_actions(dataset)[solution.pi]  # learner action index -> env action
    V_true = value_iteration.policy_value(env.P, env.true_reward, env_policy, gamma)
    live = ~env.true_reward.terminal  # a terminal state is never acted in
    promise, true, best = env.start @ solution.V, env.start @ V_true, env.start @ V_star
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
