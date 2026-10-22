"""Part 2 of the tool: evaluate what a method produced (S9).

With a ground truth (MDPGroundTruth), every value is exact:
    promise          what the method claims at the start distribution (robust methods: a lower bound)
    true             the true value of its policy
    V_star_obs       the value of the best policy over the observed state (the target; = V_star when
                     nothing is hidden; see MDPGroundTruth.observed_optimal)
    V_star           the optimal value of an agent that sees every variable (an upper reference)
    regret           V_star_obs - true;  norm_regret = (1 - gamma) * regret (values scale with
                     1 / (1 - gamma));  regret_oracle = V_star - true
    optimal_share    true / V_star_obs  (the policy is "95% optimal" when it is >= 0.95; needs
                     V_star_obs > 0);  optimal_share_oracle = true / V_star
    gap              promise - true (<= 0: the promise holds);  safe: it holds in every state
    regret_K, gap_K  the same averaged over k start states;  violation_rate, worst_violation
    shd, missing, extra, sid   the selected graph against the true one (if true_parents is known)
    V_random         the value of the uniformly random policy (how hard "95% optimal" is)
Without a ground truth only the promise is known, at the empirical start (the first state of every
rollout in the data).
"""

import numpy as np

from scm_to_mdp import evaluation
from scm_to_mdp.core import Dataset
from scm_to_mdp.ground_truth import MDPGroundTruth
from scm_to_mdp.tool.methods import Result


def empirical_start(data: Dataset) -> np.ndarray:
    """[S] the distribution of the first state of the rollouts (rows with t == 0), or of every row
    if the table has no index level t."""
    df = data.df
    if "t" in (df.index.names or []):
        df = df[df.index.get_level_values("t") == 0]
    s = data.state_index(df[data.state_vars].values)
    return np.bincount(s, minlength=len(data.B)) / len(s)


def evaluate(
    result: Result,
    data: Dataset,
    truth: MDPGroundTruth | None = None,
    gamma: float = 0.9,
    starts: np.ndarray | None = None,
) -> dict:
    """The metrics of a solved result (an empty dict if the method stopped before a solution).
    starts: the k start states (indices) of regret_K and gap_K; default every non-terminal state."""
    if result.solution is None:
        return {}
    solution = result.solution
    if truth is None:
        return {"promise": float(empirical_start(data) @ solution.V)}
    _, V_target = truth.observed_optimal(gamma)  # [S] value of the best policy over the observed state
    if starts is None:
        starts = np.flatnonzero(~truth.reward.terminal)
    metrics = evaluation.evaluate(truth, data, solution, gamma, V_target, starts)
    best, oracle = float(truth.start @ V_target), float(truth.start @ truth.optimal_values(gamma))
    metrics["optimal_share"] = metrics["true"] / best if best > 0 else float("nan")
    metrics["optimal_share_oracle"] = metrics["true"] / oracle if oracle > 0 else float("nan")
    metrics["regret_oracle"] = oracle - metrics["true"]
    metrics["norm_regret"] = (1 - gamma) * metrics["regret"]
    if result.graph is not None and truth.true_parents is not None:
        metrics.update(evaluation.structure_distances(result.graph, truth.parents_in(data), data))
    return metrics


def reference_values(truth: MDPGroundTruth, gamma: float, starts: np.ndarray) -> dict:
    """Values that do not depend on the method: V* (sees everything), V*_obs (best over the observed
    state; also over the k starts) and the random policy."""
    V_star = truth.optimal_values(gamma)
    _, V_target = truth.observed_optimal(gamma)
    return {
        "V_star": float(truth.start @ V_star),
        "V_star_obs": float(truth.start @ V_target),
        "V_star_K": float(V_target[starts].mean()),
        "V_random": float(truth.start @ truth.random_policy_values(gamma)),
    }
