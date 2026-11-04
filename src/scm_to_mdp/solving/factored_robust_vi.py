"""Robust value iteration over a PRODUCT of factor interval sets (step S8): the exact solution and the
McCormick relaxation of Schnitzer, Abate, Parker, "Efficient Solution and Learning of Robust Factored MDPs"
(AAAI 2026, arXiv:2508.00707). (The third, interval arithmetic, is compilation/product_bounds.py +
solving/robust_vi.py.)

At every (s, a) nature picks one distribution per factor, p_k in P_k = {p in simplex : lo_k <= p <= hi_k}
(the Clopper-Pearson box of S4, tightened to the simplex), and the kernel is their product (Eq. 2); the inner
problem is min (or max) over p_1..p_n of sum_{i1..in} W(i1..in) prod_k p_k(i_k) (App. B.2), with W the backup
r_next + gamma V of the next state with values (i1..in).

    vertex      exact (Thm 1): the optimum is at a product of extreme points of the factor polytopes. An
                extreme point of a box intersected with the simplex has every coordinate at a bound but at
                most one (Haddad & Monmege, interval MDPs, BFS characterisation), so every factor's vertices
                are enumerated as: one free coordinate, each other at its lower or upper bound, the free one
                = 1 - the rest if that lies within its bounds (card * 2^(card - 1) candidates, duplicates
                harmless). The product's value is contracted factor by factor and minimised over all vertex
                combinations: exponential in the factors and in the cards (App. C), exact.
    mccormick   the recursive McCormick relaxation (App. B.2, after Ryoo & Sahinidis 2001): for every index
                tuple (i1..in), n - 1 auxiliaries h_1 = p1_{i1} p2_{i2}, h_r = h_{r-1} p^{r+1}_{i_{r+1}}, each
                bilinear equation relaxed by its four McCormick inequalities (Eqs. 12a-12d) with the bounds
                of its terms from interval arithmetic ((lo_1, hi_1) = (lo p1 lo p2, hi p1 hi p2),
                (lo_r, hi_r) = (lo_{r-1} lo p, hi_{r-1} hi p)); box constraints on every p^k; one simplex
                constraint sum h_{n-1} = 1; objective sum W h_{n-1}; and (marginal_simplex=True, the
                default) sum_i p^k_i = 1 for every factor: the paper's factor sets lie in the simplex
                (App. B.1-B.2: P_B = {P in Delta : lo <= p <= hi}), though its LP listing writes the box
                constraints only (marginal_simplex=False). On SysAdmin N = 7 with the true kernel +- 0.1
                (the paper's widest radius) the simplex rows cut McCormick's gap to exact from 4.0% to
                0.9% (max over states). One LP per (s, a) and sweep, 4 (n - 1) prod m + 2 sum m + 1 + n
                constraints, solved by HiGHS (highspy; the paper uses Gurobi) warm-started from the last
                sweep's basis (MODEL_NNZ_BUDGET).
                Sound: a superset of the product set, so a lower bound on the exact robust value (Sec. 3.2).

nature: "adversarial" (min: the promise) or "cooperative" (max: the optimistic bound). solve(...,
kernel=True) also returns nature's choice at the fixed point for EVERY action, an [S, K, S'] kernel: the
worst-case (or best-case) MDP, solvable by plain value iteration (its VI value is the fixed point itself).
For McCormick it is the LP's joint h_{n-1}, a point of the RELAXED set: when the relaxation is strictly
looser than the product set it is not a product of factor distributions (a valid MDP, outside T(s, a)).
"""

import itertools
import time

import highspy
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_array, vstack

from scm_to_mdp.compilation.product_bounds import tighten
from scm_to_mdp.core import FactorBounds, NotApplicable, Solution
from scm_to_mdp.solving.stopping import sound_lower_bound, stop_at

METHODS = ("vertex", "mccormick")
NATURES = ("adversarial", "cooperative")
MAX_VERTICES = 4096  # per factor and (s, a): above it the vertex method refuses (App. C: exponential)

LP_METHODS = ("highs", "highs-ds", "highs-ipm")  # McCormick's last resort: scipy's linprog, each method
# McCormick: the LP of every (s, a) is kept as a HiGHS model (only its cost changes between sweeps, so each
# solve starts from the last optimal basis) while the kept models' non-zeros stay within this budget (about
# 200 bytes each in memory, so about 4 GB); beyond it, a model is rebuilt each sweep from its kept basis
MODEL_NNZ_BUDGET = 20_000_000


def factor_vertices(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """lo, hi [..., c] (tightened) -> vertices [..., V, c] of {p in simplex : lo <= p <= hi}, padded by
    repeating a vertex (V = c * 2^(c - 1) candidates)."""
    c = lo.shape[-1]
    frees = [c - 1] if c == 2 else range(c)  # c = 2: the segment's two ends (free = 0 repeats them)
    if c * 2 ** (c - 1) > MAX_VERTICES:
        raise ValueError(f"a factor with {c} values has up to {c * 2 ** (c - 1)} vertices (> {MAX_VERTICES})")
    cands = []
    for free in frees:
        others = [j for j in range(c) if j != free]
        for bits in itertools.product((0, 1), repeat=c - 1):
            p = np.zeros(lo.shape)
            for j, b in zip(others, bits, strict=True):
                p[..., j] = hi[..., j] if b else lo[..., j]
            p[..., free] = 1.0 - p.sum(-1)
            cands.append(p)
    P = np.stack(cands, axis=-2)  # [..., V, c]
    ok = (P >= lo[..., None, :] - 1e-12).all(-1) & (P <= hi[..., None, :] + 1e-12).all(-1)
    if not ok.any(-1).all():
        raise ValueError("an empty factor set (sum lo > 1 or sum hi < 1): no vertex")
    first = np.argmax(ok, axis=-1)  # a feasible vertex to pad with
    pad = np.take_along_axis(P, first[..., None, None].repeat(c, -1), axis=-2)  # [..., 1, c]
    return np.where(ok[..., None], P, pad)


def _boxes(bounds: FactorBounds):
    if bounds.conditional:
        # Schnitzer et al., Eq. 1: every factor depends on (s, a) only (no same-step arcs)
        raise NotApplicable("factored robust VI needs factors without same-step parents (use box_robust_vi)")
    out = []
    for i, c in enumerate(bounds.cards):
        lo, hi = tighten(bounds.lo[:, :, i, :c], bounds.hi[:, :, i, :c])
        out.append((lo, hi))
    return out


class VertexNature:
    """Exact inner problem by vertex products (Thm 1), vectorised over (s, a)."""

    def __init__(self, bounds: FactorBounds, nature: str):
        self.cards = bounds.cards
        self.vertices = [factor_vertices(lo, hi) for lo, hi in _boxes(bounds)]  # [S, K, V_k, c_k]
        self.sign = 1.0 if nature == "adversarial" else -1.0

    def __call__(self, W: np.ndarray, kernel: bool = False):
        T = W.reshape(self.cards)  # [c_1, ..., c_n] in the C order of B
        S, K = self.vertices[0].shape[:2]
        T = np.broadcast_to(T, (S, K) + T.shape)
        n = len(self.cards)
        for k in reversed(range(n)):  # contract the last value axis with factor k's vertices
            # T [S, K, c_1..c_k, V_{k+1}..V_n] -> [S, K, c_1..c_{k-1}, V_k..V_n]
            Vk = self.vertices[k]  # [S, K, V_k, c_k]
            T = np.einsum("sa...c,savc->sa...v", np.moveaxis(T, 2 + k, -1), Vk)
            T = np.moveaxis(T, -1, 2 + k)
        flat = self.sign * T.reshape(S, K, -1)
        best = flat.argmin(-1)
        value = self.sign * np.take_along_axis(flat, best[..., None], -1)[..., 0]
        if not kernel:
            return value
        idx = np.unravel_index(best, T.shape[2:])  # the vertex of every factor at the optimum
        P = np.ones((S, K) + tuple(self.cards))
        for k in range(n):
            p = np.take_along_axis(
                self.vertices[k], idx[k][..., None, None].repeat(self.cards[k], -1), axis=2
            )
            shape = [S, K] + [1] * n
            shape[2 + k] = self.cards[k]
            P = P * p[:, :, 0].reshape(shape)
        return value, P.reshape(S, K, -1)


class McCormickNature:
    """Recursive McCormick LP per (s, a) (App. B.2), with sparse constraint matrices: every McCormick row
    (Eqs. 12a-d) has three non-zeros, and the pattern is the same for every (s, a); only the coefficients
    (the interval ends) change, so they are filled in per (s, a) and nothing dense is kept."""

    def __init__(self, bounds: FactorBounds, nature: str, marginal_simplex: bool = True):
        self.cards = list(bounds.cards)
        self.boxes = _boxes(bounds)
        self.marginal_simplex = marginal_simplex
        self.sign = 1.0 if nature == "adversarial" else -1.0
        n, cards = len(self.cards), self.cards
        # the tuples of values in the C order of B: [n_t, n]
        self.tuples = np.array(list(itertools.product(*[range(c) for c in cards])), dtype=np.int64)
        # variables: p^k_i for every factor k and value i, then for every tuple its n - 1 auxiliaries
        self.offset = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)  # p^k_i = offset[k] + i
        self.n_p = int(sum(cards))
        self.n_aux = max(n - 1, 0)
        n_t = len(self.tuples)
        self.n_var = self.n_p + n_t * self.n_aux
        p_of = self.offset[None, :] + self.tuples  # [n_t, n] the p variable of every tuple's value
        if n > 1:
            aux = (
                self.n_p + np.arange(n_t)[:, None] * self.n_aux + np.arange(n - 1)[None, :]
            )  # h_r [n_t, n-1]
            u = np.concatenate([p_of[:, :1], aux[:, :-1]], axis=1)  # h_r = u * v: u = h_{r-1} (h_0 = p^0)
            self.h, self.u, self.v = aux.ravel(), u.ravel(), p_of[:, 1:].ravel()  # one per McCormick block
            m = len(self.h)
            self.rows = np.repeat(np.arange(4 * m), 3)  # 4 rows per block, 3 non-zeros per row
            cols = np.stack([self.h, self.u, self.v], axis=1)  # [m, 3]
            self.cols = np.repeat(cols[:, None, :], 4, axis=1).ravel()  # [m, 4, 3]
            self.last = aux[:, -1]
        else:
            self.last = p_of[:, 0]
        eq = [self.last]
        if self.marginal_simplex and n > 1:
            eq += [self.offset[k] + np.arange(c) for k, c in enumerate(cards)]
        eq_rows = np.concatenate([np.full(len(cols), r) for r, cols in enumerate(eq)])
        self.A_eq = csr_array(
            (np.ones(len(eq_rows)), (eq_rows, np.concatenate(eq))), shape=(len(eq), self.n_var)
        )
        self.b_eq = np.ones(len(eq))

    def _lp(self, s, a):
        """(A_ub, b_ub, bounds) of (s, a) (App. B.2)."""
        n = len(self.cards)
        lo_p = np.concatenate([self.boxes[k][0][s, a, : self.cards[k]] for k in range(n)])
        hi_p = np.concatenate([self.boxes[k][1][s, a, : self.cards[k]] for k in range(n)])
        bounds = np.empty((self.n_var, 2))
        bounds[: self.n_p, 0], bounds[: self.n_p, 1] = lo_p, hi_p
        if n == 1:
            return None, None, bounds
        LO, HI = (
            lo_p[self.offset[None, :] + self.tuples],
            hi_p[self.offset[None, :] + self.tuples],
        )  # [n_t, n]
        CL, CH = np.cumprod(LO, axis=1), np.cumprod(HI, axis=1)  # the bounds of h_r: products up to r
        lu, hu = CL[:, :-1].ravel(), CH[:, :-1].ravel()  # u = h_{r-1} in [lu, hu]
        lv, hv = LO[:, 1:].ravel(), HI[:, 1:].ravel()  # v = p^r in [lv, hv]
        bounds[self.h, 0], bounds[self.h, 1] = CL[:, 1:].ravel(), CH[:, 1:].ravel()
        # Eqs. 12a-12d with p = u in [lu, hu], q = v in [lv, hv], as rows of A_ub x <= b_ub:
        #   12a  h >= u lv + v lu - lu lv   ->  -h + lv u + lu v <=  lu lv
        #   12b  h >= u hv + v hu - hu hv   ->  -h + hv u + hu v <=  hu hv
        #   12c  h <= u lv + v hu - hu lv   ->   h - lv u - hu v <= -hu lv
        #   12d  h <= u hv + v lu - lu hv   ->   h - hv u - lu v <= -lu hv
        one = np.ones_like(lu)
        data = np.stack([np.stack([-one, lv, lu], 1), np.stack([-one, hv, hu], 1),
                         np.stack([one, -lv, -hu], 1), np.stack([one, -hv, -lu], 1)], axis=1)  # fmt: skip
        rhs = np.stack([lu * lv, hu * hv, -hu * lv, -lu * hv], axis=1).ravel()
        A_ub = csr_array((data.ravel(), (self.rows, self.cols)), shape=(len(rhs), self.n_var))
        return A_ub, rhs, bounds

    def _highs(self, s, a):
        """A HiGHS model of the LP of (s, a) (rows: the McCormick inequalities, then the equalities)."""
        A_ub, b_ub, bounds = self._lp(s, a)
        A = (vstack([A_ub, self.A_eq]) if A_ub is not None else self.A_eq).tocsc()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.n_var, A.shape[0]
        lp.col_cost_ = np.zeros(self.n_var)
        lp.col_lower_, lp.col_upper_ = bounds[:, 0], bounds[:, 1]
        n_ub = 0 if A_ub is None else A_ub.shape[0]
        lp.row_lower_ = np.concatenate([np.full(n_ub, -highspy.kHighsInf), self.b_eq])
        lp.row_upper_ = np.concatenate([b_ub if b_ub is not None else np.zeros(0), self.b_eq])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)  # one core per grid worker; no thread pool across the steps' forks
        h.passModel(lp)
        return h, A.nnz

    def _solve_one(self, s, a, cost):
        """(value of the scaled objective, h_{n-1}) at (s, a): the kept model warm-started from its last
        basis, or a new model given the last basis (beyond MODEL_NNZ_BUDGET); a cold restart, then scipy's
        linprog, when HiGHS does not report an optimum."""
        h = self._models.get((s, a))
        if h is None:
            h, nnz = self._highs(s, a)
            if self._nnz + nnz <= MODEL_NNZ_BUDGET:
                self._models[(s, a)], self._nnz = h, self._nnz + nnz
            elif (s, a) in self._bases:
                h.setBasis(self._bases[(s, a)])
        h.changeColsCost(len(self.last), self.last.astype(np.int32), cost)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:  # a cold restart
            self.fallbacks["cold"] += 1
            h.clearSolver()
            h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            if (s, a) not in self._models:
                self._bases[(s, a)] = h.getBasis()
            x = np.asarray(h.getSolution().col_value)
            return h.getInfo().objective_function_value, x[self.last]
        self.fallbacks["linprog"] += 1
        A_ub, b_ub, bounds = self._lp(s, a)
        c = np.zeros(self.n_var)
        c[self.last] = cost
        for method in LP_METHODS:  # scipy's HiGHS: its choice, then dual simplex, then interior point
            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=self.A_eq, b_eq=self.b_eq, bounds=bounds, method=method
            )
            if res.status == 0:
                return res.fun, res.x[self.last]
        raise RuntimeError(f"McCormick LP at (s={s}, a={a}): {res.message}")

    def __call__(self, W: np.ndarray, kernel: bool = False):
        S, K = self.boxes[0][0].shape[:2]
        value = np.zeros((S, K))
        P = np.zeros((S, K, len(W))) if kernel else None
        if not hasattr(self, "_models"):
            self._models, self._bases, self._nnz = {}, {}, 0
            self.fallbacks = {"cold": 0, "linprog": 0}
        # sum h = 1 (A_eq's first row), so W -> (W - shift) / scale changes the optimum value by the
        # affine map only, not the optimiser: a better-conditioned objective for HiGHS
        shift = float(W.mean())
        scale = float(np.abs(W - shift).max()) or 1.0
        cost = self.sign * (W - shift) / scale
        for s in range(S):
            for a in range(K):
                fun, h = self._solve_one(s, a, cost)
                value[s, a] = self.sign * fun * scale + shift
                if kernel:
                    P[s, a] = h
        return (value, P) if kernel else value


def solve(
    bounds: FactorBounds,
    gamma: float,
    method: str = "vertex",
    nature: str = "adversarial",
    tol: float = 1e-7,
    kernel: bool = False,
    marginal_simplex: bool = True,
):
    """Robust value iteration with the exact (vertex) or McCormick inner problem (module docstring). Returns
    a Solution (V: a sound lower bound for "adversarial", a sound upper bound V_k + gamma / (1 - gamma) step
    for "cooperative", as robust_vi; stopping.py), and with kernel=True also nature's [S, K, S'] kernel at the
    fixed point. marginal_simplex: McCormick only (module docstring)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if nature not in NATURES:
        raise ValueError(f"nature must be one of {NATURES}, not {nature!r}")
    if method == "vertex":
        inner = VertexNature(bounds, nature)
    else:
        inner = McCormickNature(bounds, nature, marginal_simplex)
    reward = bounds.reward
    V, start, sweeps = np.zeros(bounds.B.shape[0]), time.perf_counter(), 0
    while True:
        sweeps += 1
        Q = reward.R + inner(reward.backup(V, gamma))
        V_new = Q.max(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            seconds = (time.perf_counter() - start) / sweeps
            if nature == "adversarial":
                bound = sound_lower_bound(V_new, step, gamma)
            else:
                bound = V_new + gamma / (1 - gamma) * step
            sol = Solution(bound, Q.argmax(1), iterations=sweeps, iteration_s=seconds)
            if not kernel:
                return sol
            _, P = inner(reward.backup(V_new, gamma), kernel=True)
            return sol, P
        V = V_new
