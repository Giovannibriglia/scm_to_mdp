"""Behaviour policies as exact tables, and what they imply for the data (S1; reporting only).

A behaviour policy here is a table probs [S, A]: the probability of every true action in every
FULL state (hidden variables included, so a logger may read what the learner cannot see). The
simulator samples from it (sampler), and the exact model gives what the data will look like:

    data_occupancy(truth, probs, horizon)   nu(s, a): share of the logged transitions at (s, a)
    coverage(truth, probs, horizon, gamma)  how well the data cover the target policy and all
                                            reachable (observed state, action) pairs
    logger_value(truth, probs, gamma)       the logger's own true value at the start (V_logger)
    confounding_bias(truth, probs, horizon) how far the kernel the data show is from the
                                            interventional one (0 when the logger ignores the
                                            hidden variables)

Coverage (Agarwal, Jiang, Kakade, Sun, "RL: Theory and Algorithms", draft 2026, Assumption 4.1 for
the all-policy coefficient; the single-policy coefficient is the usual C* of offline RL):
    C_star           max over (o, a) of d*(o, a) / nu(o, a), with d* the normalized discounted
                     occupancy of the target policy (the best policy over the observed state);
                     inf if the data miss a pair the target policy uses
    C_all_bound      1 / min nu(o, a) over the pairs any policy can reach: an upper bound on the
                     all-policy coefficient (inf if a reachable pair is never logged)
    unvisited_pairs  share of the reachable pairs that the data never visit

Confounding bias, per observed (o, a) with nu(o, a) > 0, in total variation:
    data kernel      P_data(o' | o, a) = sum_{s in o} nu(s, a) P(o' | s, a) / nu(o, a)
    do kernel        P_do(o' | o, a)   = sum_{s in o} nu(s) P(o' | s, a) / nu(o)
where nu(s) is the share of the data in full state s. The do kernel intervenes on the current
action only, with the hidden part distributed as in the data (nu(u | o)); when the hidden part
persists over an episode, that distribution depends on the logger, so it is the kernel of the
logged process, not one kernel for every policy. The two kernels coincide when, in the data, the
logger's action does not depend on the hidden part given o (no unmeasured confounders; Bareinboim,
Causal AI, draft v0.27, Def. 8.2.1, p. 547); in general they differ (an MDP with unobserved
confounders: Def. 9.4.4 and Eq. 9.137, p. 623). The converse does not hold: they also coincide
when the hidden part does not affect the next observed state.
    bias_max   max TV over the logged (o, a);  bias_mean   TV averaged with weights nu(o, a)
"""

import numpy as np

from scm_to_mdp.core import encode
from scm_to_mdp.ground_truth import MDPGroundTruth

ZERO = 1e-12  # occupancies below this count as "never logged"


# ---- policies ---------------------------------------------------------------------------------
def uniform(truth: MDPGroundTruth) -> np.ndarray:
    A = len(truth.actions)
    return np.full((truth.n_states, A), 1.0 / A)


def expert(truth: MDPGroundTruth, gamma: float, epsilon: float) -> np.ndarray:
    """The best policy over the observed state (truth.observed_optimal), with a uniform action
    w.p. epsilon. It reads only the observed state."""
    pi_obs, _ = truth.observed_optimal(gamma)
    A = len(truth.actions)
    probs = np.full((truth.n_states, A), epsilon / A)
    probs[np.arange(truth.n_states), pi_obs[truth.observed_index()]] += 1 - epsilon
    return probs


def sampler(probs: np.ndarray, cards: list[int]):
    """A behaviour policy for Env.sample: (states [E, n], rng) -> env actions [E], drawn from probs."""
    cdf = np.cumsum(probs, axis=1)

    def act(states, rng):
        rows = cdf[encode(states, cards)]
        return np.minimum((rng.random(len(states))[:, None] > rows).sum(1), probs.shape[1] - 1)

    return act


# ---- occupancies ------------------------------------------------------------------------------
def _goes_on(truth: MDPGroundTruth) -> np.ndarray:
    return np.where(truth.reward.terminal, 0.0, 1.0)  # an episode ends on arrival in a terminal state


def data_occupancy(truth: MDPGroundTruth, probs: np.ndarray, horizon: int) -> np.ndarray:
    """[S, A] nu(s, a): the expected share of logged transitions at (s, a), for rollouts of at most
    `horizon` steps from the start distribution that stop on arrival in a terminal state (the loop
    ends early once every rollout has ended, so a large horizon costs nothing more)."""
    P_b = np.einsum("sa,sat->st", probs, truth.P)  # [S, S'] kernel of the logger
    mu, nu = truth.start.copy(), np.zeros_like(probs)
    for _ in range(horizon):
        nu += mu[:, None] * probs
        mu = (mu @ P_b) * _goes_on(truth)
        if mu.sum() < 1e-12:  # every rollout has ended
            break
    return nu / nu.sum()


def discounted_occupancy(truth: MDPGroundTruth, probs: np.ndarray, gamma: float) -> np.ndarray:
    """[S, A] the normalized discounted occupancy of a policy table from the start distribution."""
    S = truth.n_states
    P_b = np.einsum("sa,sat->st", probs, truth.P) * _goes_on(truth)[None, :]
    mu = np.linalg.solve((np.eye(S) - gamma * P_b).T, truth.start)
    d = mu[:, None] * probs
    return d / d.sum()


def one_hot(policy: np.ndarray, n_actions: int) -> np.ndarray:
    """A deterministic policy [S] as a table [S, A]."""
    table = np.zeros((len(policy), n_actions))
    table[np.arange(len(policy)), policy] = 1.0
    return table


def to_observed(truth: MDPGroundTruth, x: np.ndarray) -> np.ndarray:
    """[S, A] -> [S_obs, A], summing over the hidden part."""
    out = np.zeros((truth.learner_spec.n_states, x.shape[1]))
    np.add.at(out, truth.observed_index(), x)
    return out


# ---- what the data cover ----------------------------------------------------------------------
def coverage(truth: MDPGroundTruth, probs: np.ndarray, horizon: int, gamma: float) -> dict:
    nu = to_observed(truth, data_occupancy(truth, probs, horizon))
    pi_obs, _ = truth.observed_optimal(gamma)
    A = len(truth.actions)
    target = one_hot(pi_obs[truth.observed_index()], A)  # the target policy, over the full states
    d_star = to_observed(truth, discounted_occupancy(truth, target, gamma))
    used = d_star > ZERO
    C_star = np.inf if (nu[used] <= ZERO).any() else float((d_star[used] / nu[used]).max())
    reach = to_observed(truth, discounted_occupancy(truth, uniform(truth), gamma)) > ZERO  # pairs any
    # policy can reach: the uniform policy reaches all of them
    missed = reach & (nu <= ZERO)
    C_all = np.inf if missed.any() else float(1.0 / nu[reach].min())
    return {"C_star": C_star, "C_all_bound": C_all, "unvisited_pairs": float(missed.sum() / reach.sum())}


# ---- what the data get wrong ------------------------------------------------------------------
def confounding_bias(truth: MDPGroundTruth, probs: np.ndarray, horizon: int) -> dict:
    if not truth.hidden:
        return {"bias_max": 0.0, "bias_mean": 0.0}
    nu = data_occupancy(truth, probs, horizon)  # [S, A]
    nu_state = nu.sum(1)  # [S]
    idx, P_next = truth.observed_index(), truth.observed_next()  # [S], [S, A, S_obs]
    S_obs = truth.learner_spec.n_states
    nu_o = to_observed(truth, nu)  # [S_obs, A]
    nu_state_o = np.bincount(idx, weights=nu_state, minlength=S_obs)  # [S_obs]
    tv = np.zeros((S_obs, len(truth.actions)))
    for a in range(len(truth.actions)):
        data, do = np.zeros((S_obs, S_obs)), np.zeros((S_obs, S_obs))
        np.add.at(data, idx, nu[:, a, None] * P_next[:, a])
        np.add.at(do, idx, nu_state[:, None] * P_next[:, a])
        with np.errstate(invalid="ignore", divide="ignore"):
            data /= nu_o[:, a, None]
            do /= nu_state_o[:, None]
        tv[:, a] = 0.5 * np.abs(data - do).sum(1)
    logged = nu_o > ZERO
    return {
        "bias_max": float(tv[logged].max()),
        "bias_mean": float((tv[logged] * nu_o[logged]).sum() / nu_o[logged].sum()),
    }


def logger_value(truth: MDPGroundTruth, probs: np.ndarray, gamma: float) -> float:
    """The true discounted value of the logger itself at the nominal start (what SPIBB improves on)."""
    from scm_to_mdp.solving.value_iteration import policy_value

    return float(truth.start @ policy_value(truth.P, truth.reward, probs, gamma))


def behaviour_metrics(truth: MDPGroundTruth, probs: np.ndarray, horizon: int, gamma: float) -> dict:
    """Coverage, confounding bias and value of a logger, from the exact model (columns of results.csv)."""
    metrics = coverage(truth, probs, horizon, gamma) | confounding_bias(truth, probs, horizon)
    return metrics | {"V_logger": logger_value(truth, probs, gamma)}
