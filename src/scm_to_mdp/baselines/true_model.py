"""The oracle (not a learner): the true MDP over the observed state, given to the solver (S8, S9).

    oracle -> MDP    value iteration on the true observed kernel: its policy is optimal over the
                     observed state, so its true value is V*_obs (the target) and its promise is exact
    oracle -> IMDP   robust value iteration on [P - tol, P + tol] (clipped to [0, 1]) around the true
                     observed kernel: the set contains the truth, so its promise is a lower bound, at
                     most about tol / (1 - gamma) below V*_obs

The kernel is the observed MDP's (MDPGroundTruth.observed_kernel: the true kernel averaged over the
hidden part of the state; with nothing hidden, the true kernel), restricted to the action vectors of
the data. It plays two parts: the upper reference of every learner, and the check of the
computational budget, since every size of a grid should be solvable by the oracle within the limits
(an interval set of tol > 0 is as dense as a learned one: its robust solve costs what theirs costs).
"""

import numpy as np

from scm_to_mdp.core import Dataset, IntervalMDP

TOL = 1e-6  # oracle -> IMDP: the half-width of its intervals


def model(kernel: np.ndarray, dataset: Dataset, tol: float = 0.0) -> IntervalMDP:
    """kernel [S_obs, K, S_obs]: the true observed kernel on the data's action vectors. tol = 0: the
    exact MDP (L == U), for value iteration; tol > 0: the interval MDP of the robust oracle."""
    if tol < 0:
        raise ValueError("tol must be >= 0")
    if kernel.shape[:2] != (len(dataset.B), len(dataset.actions)):
        raise ValueError(f"kernel {kernel.shape} does not match the data's states and actions")
    if tol == 0:
        return IntervalMDP(kernel, kernel, dataset.reward, info={"oracle_tol": 0.0})
    L = np.clip(kernel - tol, 0.0, 1.0)
    U = np.clip(kernel + tol, 0.0, 1.0)
    return IntervalMDP(L, U, dataset.reward, info={"oracle_tol": tol})
