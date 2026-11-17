"""One MDP out of a factored interval set, then value iteration on it (step S8): the "-> MDP" arms built from
each of Schnitzer, Abate, Parker's relaxations (AAAI 2026, arXiv:2508.00707: interval arithmetic,
McCormick, exact vertex products).

Fixing nature's choice turns a robust MDP into an ordinary MDP (Badings, Robust Verification of Stochastic
Systems, PhD thesis 2025, Sec. 3.3.2, p. 42). Two choices:

    worst   nature's choice at the ROBUST fixed point, for every action (adversarial): an MDP whose value
            iteration returns the robust value and the robust-greedy policies (its Bellman operator has the
            robust fixed point as its fixed point). With a relaxed set (IA, McCormick) the kernel may be a
            non-product distribution, outside the true model set (a valid MDP all the same).
    best    nature's choice at the OPTIMISTIC fixed point (cooperative): the best MDP in the set, whose value
            is an upper bound when the truth is in the set (Suilen et al. 2024, arXiv:2411.11451, p. 12;
            Jackson et al., HSCC 2021, Thm 2: the optimistic - robust gap bounds the error). An exploration
            device online; offline its policy has no lower-bound guarantee.

The returned Solution is value iteration's on that MDP (its V: that MDP's optimal value).

A joint interval MDP (flat IMDP: one interval per (s, a, s')) is one factor: its set is already the joint
one, so the three relaxations coincide and nature chooses within its intervals (the "ia" path).
"""

import numpy as np

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import FactorBounds, IntervalMDP, Solution
from scm_to_mdp.solving import factored_robust_vi, robust_vi, value_iteration

RELAXATIONS = ("ia", "mccormick", "exact")
CHOICES = ("worst", "best")


def ia_kernel(mdp, W: np.ndarray, nature: str) -> np.ndarray:
    """[S, K, S'] nature's distribution on an interval MDP for the vector W (robust_vi.Nature's fill)."""
    inner = robust_vi.Nature(mdp, nature)
    inner.expectation(W)  # computes the order and the extra mass of every next state
    P = mdp.L.copy()
    np.add.at(P, (slice(None), slice(None), inner._order), inner._given)
    return P


def nature_kernel(
    bounds: FactorBounds | IntervalMDP, gamma: float, relaxation: str, choice: str, tol: float = 1e-7
):
    """[S, K, S'] nature's choice at the fixed point of the robust (worst) or optimistic (best) iteration."""
    if relaxation not in RELAXATIONS:
        raise ValueError(f"relaxation must be one of {RELAXATIONS}, not {relaxation!r}")
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}, not {choice!r}")
    nature = "adversarial" if choice == "worst" else "cooperative"
    if isinstance(bounds, IntervalMDP) or relaxation == "ia":  # a joint set: every relaxation is the set
        mdp = bounds if isinstance(bounds, IntervalMDP) else product_bounds.compile(bounds)
        sol = robust_vi.solve(mdp, gamma, tol=tol, nature=nature)
        return ia_kernel(mdp, bounds.reward.backup(sol.V, gamma), nature)
    method = "vertex" if relaxation == "exact" else "mccormick"
    _, P = factored_robust_vi.solve(bounds, gamma, method, nature, tol=tol, kernel=True)
    return P


def kernel_at(bounds: FactorBounds | IntervalMDP, V: np.ndarray, gamma: float, relaxation: str, choice: str):
    """[S, K, S'] nature's choice for the value V, for every action: one inner problem per (s, a), no
    iteration. With V the fixed point of the robust (optimistic) iteration of the same relaxation, this is
    nature_kernel's kernel; with V within tol of it (the last iterate or the sound bound of that solve),
    the worst MDP's value is within about 3 tol of the robust value, from above (a fixed nature can only
    help the agent; the contraction bound gamma / (1 - gamma) times the last step, twice, plus value
    iteration's own tol: Badings, Robust Verification of Stochastic Systems, 2025, Lemma 3.32; Kallenberg,
    Markov Decision Processes, lecture notes, 2016, Thm 3.3; Meggendorfer et al., AAAI 2025,
    App. C, Lemma 4, at the exact point)."""
    nature = "adversarial" if choice == "worst" else "cooperative"
    W = bounds.reward.backup(V, gamma)
    if isinstance(bounds, IntervalMDP) or relaxation == "ia":
        mdp = bounds if isinstance(bounds, IntervalMDP) else product_bounds.compile(bounds)
        return ia_kernel(mdp, W, nature)
    if relaxation == "exact":
        _, P = factored_robust_vi.VertexNature(bounds, nature)(W, kernel=True)
    else:
        _, P = factored_robust_vi.McCormickNature(bounds, nature)(W, kernel=True)
    return P


def solve(
    bounds: FactorBounds | IntervalMDP,
    gamma: float,
    relaxation: str = "ia",
    choice: str = "worst",
    V: np.ndarray | None = None,
) -> Solution:
    """Value iteration on nature's MDP (module docstring). V: the value of the robust (or optimistic)
    solve of the same set, when it is already known (the IMDP arm's: tool/methods.py reuses it), so nature's
    kernel takes one inner problem per (s, a) instead of the whole robust iteration (kernel_at)."""
    if relaxation not in RELAXATIONS:
        raise ValueError(f"relaxation must be one of {RELAXATIONS}, not {relaxation!r}")
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}, not {choice!r}")
    if V is None:
        P = nature_kernel(bounds, gamma, relaxation, choice)
    else:
        P = kernel_at(bounds, V, gamma, relaxation, choice)
    return value_iteration.solve(P, bounds.reward, gamma)
