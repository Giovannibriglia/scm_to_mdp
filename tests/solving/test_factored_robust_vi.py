"""Schnitzer, Abate, Parker (AAAI 2026) solvers (solving/factored_robust_vi.py, solving/nature_mdp.py): exact
vertex products and the McCormick relaxation of a product of factor interval sets."""

import numpy as np
import pytest
from scipy.optimize import linprog

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import FactorBounds, Reward
from scm_to_mdp.discovery import oracle
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given
from scm_to_mdp.solving import factored_robust_vi as fr
from scm_to_mdp.solving import nature_mdp, robust_vi, vertex_robust_vi
from scm_to_mdp.utils.limits import run_limited


def _bounds(N, episodes, seed=0):
    env = SysAdmin(N, seed)
    data = env.ground_truth().learner_spec.read(env.sample(episodes, 20, "random"))
    return nbn_intervals.learn(data, given.select(oracle.fit(data, env.true_parents()), data), 0.05)


def test_factor_vertices_reach_the_lp_optimum():
    rng = np.random.default_rng(0)
    for c in (2, 3, 4):
        for _ in range(50):
            a, b = rng.random(c), rng.random(c)
            lo, hi = np.minimum(a, b) * 0.6, np.maximum(a, b)
            if lo.sum() > 1 or hi.sum() < 1:
                continue
            lo, hi = product_bounds.tighten(lo, hi)
            V = fr.factor_vertices(lo, hi)
            w = rng.normal(size=c)
            lp = linprog(
                w, A_eq=np.ones((1, c)), b_eq=[1], bounds=list(zip(lo, hi, strict=True)), method="highs"
            )
            assert (V @ w).min() == pytest.approx(lp.fun, abs=1e-9)


def test_empty_factor_set_raises():
    with pytest.raises(ValueError):
        fr.factor_vertices(np.array([0.6, 0.6]), np.array([0.7, 0.7]))


def test_vertex_equals_the_binary_solver_and_the_ladder_holds():
    b = _bounds(4, 60)
    exact = fr.solve(b, 0.9, "vertex")
    assert np.allclose(exact.V, vertex_robust_vi.solve(b, 0.9).V, atol=1e-9)
    mcc = fr.solve(b, 0.9, "mccormick", marginal_simplex=False)  # the LP as listed in App. B
    tight = fr.solve(b, 0.9, "mccormick")  # with the factors' simplex constraints (the default)
    ia = robust_vi.solve(product_bounds.compile(b), 0.9)
    assert (
        np.all(exact.V >= tight.V - 1e-6) and np.all(tight.V >= mcc.V - 1e-6) and np.all(mcc.V >= ia.V - 1e-6)
    )


def _linprog_nature(nat, W):
    """McCormick's inner values by scipy's linprog, one cold LP per (s, a): the reference."""
    S, K = nat.boxes[0][0].shape[:2]
    out = np.zeros((S, K))
    c = np.zeros(nat.n_var)
    c[nat.last] = nat.sign * W
    for s in range(S):
        for a in range(K):
            A_ub, b_ub, bounds = nat._lp(s, a)
            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=nat.A_eq, b_eq=nat.b_eq, bounds=bounds, method="highs"
            )
            out[s, a] = nat.sign * res.fun
    return out


@pytest.mark.parametrize("budget", [fr.MODEL_NNZ_BUDGET, 0])  # models kept / only their bases kept
def test_warm_started_highs_equals_cold_linprog_over_sweeps(monkeypatch, budget):
    monkeypatch.setattr(fr, "MODEL_NNZ_BUDGET", budget)
    b = _bounds(3, 40)
    rng = np.random.default_rng(0)
    for nature in ("adversarial", "cooperative"):
        nat = fr.McCormickNature(b, nature)
        W = rng.normal(size=len(b.B)) * 5
        for _ in range(4):  # sweeps: the cost changes, the warm start is reused
            assert np.allclose(nat(W), _linprog_nature(nat, W), atol=1e-7)
            W = W + rng.normal(size=len(W)) * 0.5
        assert (len(nat._models) > 0) == (budget > 0) and (len(nat._bases) > 0) == (budget == 0)


def test_mccormick_is_exact_for_two_factors():
    b = _bounds(2, 30)
    assert np.allclose(fr.solve(b, 0.9, "mccormick").V, fr.solve(b, 0.9, "vertex").V, atol=1e-6)


def test_cooperative_is_an_upper_bound_and_nature_mdps_reproduce_the_fixed_points():
    b = _bounds(3, 60)
    for relaxation in ("ia", "mccormick", "exact"):
        worst = nature_mdp.solve(b, 0.9, relaxation, "worst")
        best = nature_mdp.solve(b, 0.9, relaxation, "best")
        if relaxation == "ia":
            rob = robust_vi.solve(product_bounds.compile(b), 0.9)
            opt = robust_vi.solve(product_bounds.compile(b), 0.9, nature="cooperative")
        else:
            how = "vertex" if relaxation == "exact" else "mccormick"
            rob, opt = fr.solve(b, 0.9, how), fr.solve(b, 0.9, how, "cooperative")
        assert np.allclose(worst.V, rob.V, atol=1e-5) and np.all(opt.V >= best.V - 1e-9)
        assert np.all(best.V >= worst.V - 1e-9)


def test_vertex_memory_of_the_sepsis_problems():
    size3 = fr.vertex_memory_bytes(288, 8, [2, 3, 3, 2, 2, 2, 2])
    size4 = fr.vertex_memory_bytes(1440, 8, [2, 3, 3, 2, 5, 2, 2, 2])
    assert size3 / 1e9 == pytest.approx(0.17, abs=0.01)  # ran: 0.32 GB with the step's other memory
    assert size4 / 1e9 == pytest.approx(68, abs=0.5)  # the panic of 2026-11-06, five times over


def _wide_bounds(cards, n_actions):
    """A FactorBounds of the given factors with every interval [0, 1]: the most vertices."""
    S = int(np.prod(cards))
    states = np.stack(np.unravel_index(np.arange(S), cards), axis=1)
    C = max(cards)
    return FactorBounds(
        B=states, cards=list(cards), lo=np.zeros((S, n_actions, len(cards), C)),
        hi=np.ones((S, n_actions, len(cards), C)),
        reward=Reward(np.zeros((S, n_actions)), np.zeros(S), np.zeros(S, dtype=bool)),
    )  # fmt: skip


def _one_inner_problem(cards, n_actions):
    bounds = _wide_bounds(cards, n_actions)
    return fr.VertexNature(bounds, "adversarial")(np.zeros(int(np.prod(cards)))).shape


def test_too_large_a_vertex_table_is_a_memout_before_any_allocation():
    cards, K = (5, 5, 5), 4  # 80^3 vertex products at 125 x 4 pairs: 4 GB
    need = fr.vertex_memory_bytes(125, K, list(cards)) / 1e9
    assert need > 1
    out = run_limited(_one_inner_problem, cards, K, mem_gb=1.0)
    assert out.status == "memout" and out.peak_gb < 0.5 and out.seconds < 10, out


def test_the_check_is_a_lower_bound_on_the_real_memory():
    """So it never stops a step the monitor would have let finish."""
    cards, K = (5, 5, 3), 4
    need = fr.vertex_memory_bytes(75, K, list(cards)) / 1e9
    assert 0.2 < need < 1
    out = run_limited(_one_inner_problem, cards, K, mem_gb=4.0)
    assert out.status == "ok" and out.value == (75, K)
    assert out.peak_gb >= need, (out.peak_gb, need)
    assert out.peak_gb < need + 0.5, (out.peak_gb, need)  # and not far below it


def test_too_many_vertices_is_a_memout():
    c = 12  # 12 * 2^11 vertices > MAX_VERTICES
    with pytest.raises(MemoryError):
        fr.factor_vertices(np.full(c, 0.0), np.full(c, 1.0))
