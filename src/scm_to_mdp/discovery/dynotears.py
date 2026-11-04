"""DYNOTEARS (Pamfil, Sriwattanaworachai, Desai, Pilgerstorfer, Beaumont, Georgatzis, Aragam, "DYNOTEARS:
Structure Learning from Time-Series Data", AISTATS 2020, https://arxiv.org/abs/2002.00498), step S2.

The model is the structural VAR of the paper's Eq. (3), with autoregressive order p = 1:

    X = X W + Y A + Z,

X the slice-(t+1) variables (our next-state columns x'), Y the lagged slice-t variables (our inputs:
state and action columns), W the intra-slice weights (x'_j -> x'_i, must be acyclic) and A the
inter-slice weights (input -> x'). DYNOTEARS minimises (Eq. 5, Sec. 2.2)

    f(W, A) = 1/(2n) ||X - X W - Y A||_F^2 + lambda_W ||W||_1 + lambda_A ||A||_1   s.t.  h(W) = 0,

with h(W) = tr exp(W o W) - d (Zheng et al., NeurIPS 2018), by the augmented Lagrangian
F = f + rho/2 h^2 + alpha h (Eqs. 6-7): W = W+ - W-, A = A+ - A- with non-negative parts, each
inner problem solved by L-BFGS-B, rho and alpha updated as in NOTEARS (Zheng et al., Alg. 1). Weights
with |w| <= tau are then removed (Sec. 2.2: thresholds tau_W, tau_A).

Checked against the authors' implementation (causalnex, causalnex/structure/dynotears.py, branch develop):
the same loss, h and gradient, L1 on the positive and negative parts, zero diagonal of W, rho x 10 until h
falls below 0.25 of its last value, alpha += rho h, rho up to 1e20. causalnex is not a dependency: it pins
old versions of pandas and scikit-learn.

Our choices for discrete data (the paper's model is linear in real-valued variables):
    columns     every column is centred and scaled to unit variance, so that one lambda and one
                tau suit binary, ordinal and one-hot columns alike (the paper normalises its
                finance data the same way, Sec. 4.1); a constant column is dropped
    inputs      a state input with more than 2 values enters as its one-hot columns (all but the
                first value); it is a parent of x' when any of its columns has |A| > tau_A
    targets     x' enters as its integer value (linear in it: ordinal variables fit, categorical
                ones only approximately)
    lambda, tau lambda_W = lambda_A = 0.05 (the paper's example, Sec. 3.2); tau_W = tau_A = TAU
                (module constant, chosen on a pilot before the grids, see TAU)

Output: a class of graphs (Candidates) with directed edges only, input -> x' (from A) and
x'_j -> x'_i (from W, same-step edges); step S3 keeps both (selection/bic_best.py; a thresholded W
that closes a cycle loses the closing edge, and the report counts the kept ones as `intra_next`).
"""

import numpy as np
import scipy.linalg
import scipy.optimize

from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset

LAMBDA = 0.05  # lambda_W = lambda_A: the example of the paper (Sec. 3.2, Fig. 2)
# tau_W = tau_A: the paper's example value (Sec. 3.2). Pilot before the grids (uniform logger, 50 / 250 / 1000
# rollouts; missing / extra true parents): SysAdmin ring N = 4 and 8, tau = 0.01: 0-1 missing, 0-4 extra
# (none from 250 rollouts on); 0.05: 1 missing; 0.1: 2-6 missing. Sepsis (3 vitals): 4 of 14 missing at
# 0.01, more above. Chemistry (5 objects, density 0.4): 9-11 of 16 missing at every tau (its reaction,
# a sum of colours mod 3, is not linear). Every run took well under a second.
TAU = 0.01
H_TOL, RHO_MAX, MAX_ITER = 1e-8, 1e20, 100  # as causalnex's implementation (see below)


def _standardise(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns centred and scaled to unit variance; also the mask of non-constant columns."""
    sd = M.std(0)
    keep = sd > 1e-12
    return (M[:, keep] - M[:, keep].mean(0)) / sd[keep], keep


def _design(dataset: Dataset) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Y [n, columns] the slice-t inputs (one-hot for state inputs with > 2 values), the input each
    column belongs to, and X [n, d] the next-state values."""
    df = dataset.df
    columns, owner = [], []
    for name in dataset.inputs:
        values = df[name].to_numpy()
        card = dataset.cards[name]
        if card > 2:
            for v in range(1, card):
                columns.append((values == v).astype(float))
                owner.append(name)
        else:
            columns.append(values.astype(float))
            owner.append(name)
    Y = np.stack(columns, axis=1)
    X = df[dataset.next_vars].to_numpy().astype(float)
    return Y, owner, X


def solve(
    X: np.ndarray, Y: np.ndarray, lambda_w: float, lambda_a: float, same_step: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """W [d, d], A [m, d] minimising DYNOTEARS' objective with h(W) = 0 (no thresholding). same_step=False:
    W fixed at 0 (no same-step edge x'_j -> x'_i); then h(W) = 0 and the problem is a lasso of every x'_i
    on the inputs."""
    n, d = X.shape
    m = Y.shape[1]
    Z = np.concatenate([X, Y], axis=1)
    G, c = Z.T @ Z / n, Z.T @ X / n  # the loss only needs these: 1/(2n)||X - Z B||^2, B = [W; A]
    const = float(np.sum(X * X)) / (2 * n)
    size_w, size_a = d * d, m * d
    bounds_w = [(0, 0) if i == j or not same_step else (0, None) for i in range(d) for j in range(d)]
    bounds = bounds_w * 2 + [(0, None)] * (2 * size_a)

    def unpack(x):
        w = x[:size_w] - x[size_w : 2 * size_w]
        a = x[2 * size_w : 2 * size_w + size_a] - x[2 * size_w + size_a :]
        return w.reshape(d, d), a.reshape(m, d)

    def h_and_grad(W):
        E = scipy.linalg.expm(W * W)
        return float(np.trace(E)) - d, E.T * W * 2

    def objective(x, rho, alpha):
        W, A = unpack(x)
        B = np.concatenate([W, A], axis=0)
        GB = G @ B
        loss = const - float(np.sum(B * c)) + 0.5 * float(np.sum(B * GB))
        grad_B = GB - c  # [d + m, d]
        h, grad_h = h_and_grad(W)
        value = loss + 0.5 * rho * h * h + alpha * h + lambda_w * x[: 2 * size_w].sum()
        value += lambda_a * x[2 * size_w :].sum()
        grad_w = grad_B[:d] + (rho * h + alpha) * grad_h
        grad_a = grad_B[d:]
        g = np.concatenate([grad_w.ravel() + lambda_w, -grad_w.ravel() + lambda_w,
                            grad_a.ravel() + lambda_a, -grad_a.ravel() + lambda_a])  # fmt: skip
        return value, g

    x = np.zeros(2 * (size_w + size_a))
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(MAX_ITER):
        while rho < RHO_MAX:
            res = scipy.optimize.minimize(
                objective, x, args=(rho, alpha), jac=True, method="L-BFGS-B", bounds=bounds
            )
            h_new = h_and_grad(unpack(res.x)[0])[0]
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        x, h = res.x, h_new
        alpha += rho * h
        if h <= H_TOL or rho >= RHO_MAX:
            break
    return unpack(x)


def fit(
    dataset: Dataset,
    lambda_w: float = LAMBDA,
    lambda_a: float = LAMBDA,
    tau: float = TAU,
    same_step: bool = True,
) -> Candidates:
    """DYNOTEARS (module docstring); same_step=False: no same-step edge (W = 0, Strehl, Diuk, Littman,
    AAAI 2007, Assumption 1)."""
    Y, owner, X = _design(dataset)
    Ys, keep_y = _standardise(Y)
    Xs, keep_x = _standardise(X)
    if Xs.shape[1] == 0:  # no next-state variable varies in the data (e.g. 10 transitions): no edge
        return Candidates("graph_class", frozenset(), frozenset(), keep_vstructures=False,
                          report={"intra_weights": 0})  # fmt: skip
    W, A = solve(Xs, Ys, lambda_w, lambda_a, same_step)
    targets = [v for v, k in zip(dataset.next_vars, keep_x, strict=True) if k]
    owners = [o for o, k in zip(owner, keep_y, strict=True) if k]
    directed = set()
    for j, v_next in enumerate(targets):
        for i, name in enumerate(owners):
            if abs(A[i, j]) > tau:
                directed.add((name, v_next))
        for i, u_next in enumerate(targets):
            if abs(W[i, j]) > tau:
                directed.add((u_next, v_next))
    report = {"intra_weights": int((np.abs(W) > tau).sum())}
    assert all(b.endswith(NEXT_MARK) for _, b in directed)
    return Candidates("graph_class", frozenset(directed), frozenset(), keep_vstructures=False, report=report)
