"""Robustness checks of a robust method's promise (step S9), computed against the truth.

All of them rest on ONE event of step S4: E = "the true observed kernel lies in the learned set".
E depends on the transition counts only, never on the reward or the policy, so on E every check
below must hold, for every reward and every policy at once (the value bracket follows from E for
any measure: Badings et al., Robust Verification of Stochastic Systems, Lemma 3.32). A check that
fails while the set contains the truth (contain_all, evaluation.model_errors) is a bug in the
compiler, the solver or the stopping rule, not bad luck.

    bracket_holds   V_lo <= V^pi_true <= V_up in every non-terminal observed state, where V_lo and V_up
                    are the method's policy evaluated against an adversarial and a cooperative nature
                    on the same set (robust_vi.evaluate); V^pi_true is averaged over hidden variables
    bracket_width   (V_up - V_lo) at the start, as a share of V*_obs - V_rand (how loose the bracket is)
    opt_holds       V*_obs <= V_opt in every non-terminal observed state (V_opt: the optimistic value
                    of the certificate, evaluation.optimistic_values)
    cert_residual   min over non-terminal states of (T_pi V - V) for the promise V and one robust
                    backup T_pi of the returned policy. >= 0 (up to rounding) proves V <= V^pi_lo without
                    trusting the solver's convergence: T_pi is monotone and a gamma-contraction, so
                    V <= T_pi V implies V <= the fixed point ("superharmonic" check)
    pfail_up        an upper bound on the probability of reaching a failure state within `horizon`
                    steps under the method's policy, from the start: nature maximises it on the set
    pfail_true      the same probability under the true model
    pfail_holds     pfail_true <= pfail_up in every non-terminal observed state

A failure state is a terminal state that pays less on arrival than the best terminal state does
(failure_states: Frozen Lake's holes, Sepsis's death). Domains without one have no pfail_* values.
With hidden variables the true values are averaged over the hidden part (as `safe` is), and the
checks are empirical.
"""

import numpy as np

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import Dataset, FactorBounds, IntervalMDP, Reward, Solution
from scm_to_mdp.ground_truth import MDPGroundTruth
from scm_to_mdp.solving import robust_vi

TOL = 1e-6  # slack of every comparison (the solvers stop at 1e-7)


def interval_mdp(model) -> IntervalMDP:
    """The set a robust solver works on: a FactorBounds as its box, an IntervalMDP as it is."""
    return product_bounds.compile(model) if isinstance(model, FactorBounds) else model


def policy_table(solution: Solution, n_actions: int) -> np.ndarray:
    """[S, K] the action probabilities of a solution (one-hot for a deterministic policy)."""
    if solution.probs is not None:
        return solution.probs
    return np.eye(n_actions)[solution.pi]


def failure_states(reward: Reward) -> np.ndarray:
    """[S] bool: terminal states that pay less on arrival than the best terminal state."""
    terminal = np.asarray(reward.terminal, dtype=bool)
    if not terminal.any():
        return terminal
    best = reward.r_next[terminal].max()
    return terminal & (reward.r_next < best - 1e-12)


def reach_probability(step, failure: np.ndarray, terminal: np.ndarray, horizon: int) -> np.ndarray:
    """[S] P(reach a failure state within `horizon` steps), by `horizon` backward steps:
    f_0 = 1 on failure states; f_{h+1}(s) = 1 on failure states, 0 on the other terminal states, else
    step(f_h)(s), the expected f_h one step ahead under the policy."""
    f = failure.astype(float)
    for _ in range(horizon):
        f = np.where(failure, 1.0, np.where(terminal, 0.0, step(f)))
    return f


def checks(
    model,
    solution: Solution,
    truth: MDPGroundTruth,
    data: Dataset,
    gamma: float,
    V_true: np.ndarray,
    V_opt_obs: np.ndarray,
    horizon: int | None = None,
) -> dict:
    """The checks of the module docstring for one solved robust model. V_true [S_full]: the true value
    of the policy (evaluation.true_values); V_opt_obs [S_obs]: the certificate's optimistic values."""
    from scm_to_mdp.evaluation import observed_average

    full = interval_mdp(model)
    S, K = full.L.shape[:2]
    pi_full = policy_table(solution, K)
    mdp, pi = robust_vi.on_policy(full, pi_full)  # the actions the policy plays: exact, K times cheaper
    live = ~np.asarray(data.reward.terminal, dtype=bool)
    start = np.bincount(truth.observed_index(), weights=truth.start, minlength=S)  # over observed states
    _, V_star_full = truth.observed_optimal(gamma)
    V_star, V_pi = observed_average(truth, V_star_full), observed_average(truth, V_true)
    span = float(start @ V_star - start @ observed_average(truth, truth.random_policy_values(gamma)))

    # started at the solver's promise and at V_opt (both close); any start gives the same sound bounds
    V_lo = robust_vi.evaluate(mdp, pi, gamma, nature="adversarial", V0=solution.V)
    V_up = robust_vi.evaluate(mdp, pi, gamma, nature="cooperative", V0=V_opt_obs)
    backup = (
        pi * (mdp.reward.R + robust_vi.Nature(mdp).expectation(mdp.reward.backup(solution.V, gamma)))
    ).sum(1)
    out = {
        "bracket_holds": bool(np.all((V_lo[live] <= V_pi[live] + TOL) & (V_pi[live] <= V_up[live] + TOL))),
        "bracket_width": float(start @ (V_up - V_lo)) / span if span > 0 else float("nan"),
        "opt_holds": bool(np.all(V_star[live] <= V_opt_obs[live] + TOL)),
        "cert_residual": float((backup - solution.V)[live].min()) if live.any() else float("nan"),
    }
    failure = failure_states(data.reward)
    if horizon is not None and failure.any():
        terminal = np.asarray(data.reward.terminal, dtype=bool)
        worst = robust_vi.Nature(mdp, "cooperative")  # nature makes failure as likely as it can
        p_up = reach_probability(lambda f: (pi * worst.expectation(f)).sum(1), failure, terminal, horizon)
        p_true_full = _true_reach_probability(truth, data, pi_full, horizon)
        p_true = observed_average(truth, p_true_full)
        out |= {
            "pfail_up": float(start @ p_up),
            "pfail_true": float(truth.start @ p_true_full),
            "pfail_holds": bool(np.all(p_true[live] <= p_up[live] + TOL)),
        }
    return out


def _true_reach_probability(truth: MDPGroundTruth, data: Dataset, pi: np.ndarray, horizon: int) -> np.ndarray:
    """[S_full] P(reach a failure state within `horizon` steps) in the true model, under the policy pi
    [S_obs, K] lifted to the full states (it sees only the observed part)."""
    to_true = truth.actions_of(data)  # learner action index -> true action
    policy = np.zeros((truth.n_states, truth.P.shape[1]))
    policy[:, to_true] = pi[truth.observed_index()]
    P_pi = np.einsum("sa,sat->st", policy, truth.P)  # [S, S'] the true chain under the policy
    reward = truth.reward
    return reach_probability(
        lambda f: P_pi @ f, failure_states(reward), np.asarray(reward.terminal, bool), horizon
    )
