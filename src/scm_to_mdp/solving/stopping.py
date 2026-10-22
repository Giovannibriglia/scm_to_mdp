"""When value iteration stops: tol is the distance to the fixed point, not the size of the last step.

For a gamma-contraction T with fixed point V_fix, ||V_fix - V_k|| <= gamma / (1 - gamma) *
||V_k - V_{k-1}|| in the sup norm (Kallenberg, Markov Decision Processes, lecture notes, Leiden,
2016, Thm 3.2(1), p. 45). Stopping when the step is below tol * (1 - gamma) / gamma therefore leaves
V_k within tol of the fixed point.

Within tol is not below: the promise of a robust method must be a LOWER bound. Let pi be the greedy
policy of the last update (greedy with respect to V_{k-1}), so V_k = T_pi V_{k-1}, with T_pi the
(robust) Bellman operator of pi. T_pi is itself a gamma-contraction with fixed point V^pi, so the
same bound gives
    V^pi >= V_k - gamma / (1 - gamma) * ||V_k - V_{k-1}||
at whatever step value iteration stops (for plain VI this is Kallenberg's Thm 3.7(1), p. 50; for
robust VI the argument needs only that T_pi is a gamma-contraction, which holds for a min over an
(s, a)-rectangular set of kernels). sound_lower_bound returns this, and the solvers report it as the
promise: at most tol below V_k. Comparing the step itself to tol is not enough: at gamma = 0.99 it
leaves up to 99 * tol, and made a robust promise exceed the true value by about 1e-4 in a test with
tol = 1e-6.
"""


def stop_at(tol: float, gamma: float) -> float:
    """The step size below which the iterate is within tol of the fixed point."""
    return tol * (1 - gamma) / max(gamma, 1e-12)


def sound_lower_bound(V_new, step: float, gamma: float):
    """A lower bound on the robust value of the greedy policy of the last update (see above)."""
    return V_new - gamma / (1 - gamma) * step
