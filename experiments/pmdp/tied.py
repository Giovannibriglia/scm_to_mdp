"""A factored MDP with 0/1 state variables as a parametric MDP: CPD rows tied into classes that share
one Bernoulli parameter (steps S4, S7, S8). The classes come from a declared template
(templates.py: SysAdmin, Stock trading); this module is domain-independent.

Declared mechanism, for every state variable x_i:

    x_i' = 1[U_i <= theta_{c_i(s, a)}],  U_i ~ Unif(0, 1) drawn afresh for every variable and step,

so the noise draws are independent across variables (a shared LAW, not a shared draw): the kernel is
still a product over the variables, and the parameters theta couple its rows across variables and
states. A pMDP with as many parameters as classes.

S4: theta's confidence box from the counts pooled over the rows of every class (NBN's Clopper-Pearson
intervals, trajectory-valid, union bound over the classes); `tying_test` checks the tying. S7: every
factor row gets its class's interval. S8: (a) the box relaxation solved by robust VI (the existing
solver); (b) for the policy of (a), a sound bound on the static worst case min over theta of
V^pi_theta (theta fixed for the whole run): Storm's parameter lifting with region refinement on the
sequentialised parametric DTMC of the policy (the reported promise), cross-checked by a Python region
refinement (a looser sound bound) and by exact evaluations at sampled theta (an upper estimate).

Why sequentialised: a row of the flat chain multiplies the factors, so it contains theta_c^2 when two
variables share a class; parameter lifting needs entries affine in each parameter of a state.
Deciding one variable per intermediate state gives every state at most one parameter and, for every
fixed theta, the same discounted value (checked with experts badings_claude, 2026-10-29).
"""

from __future__ import annotations

import contextlib
import itertools
import math
import multiprocessing
import os
import sys
import tempfile
from fractions import Fraction
from typing import Protocol

import numpy as np
import torch
from scipy.stats import chi2

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import NEXT_MARK, FactorBounds
from scm_to_mdp.solving import robust_vi


class Template(Protocol):
    """The declared tying. Classes are 0..n_classes-1; every state variable (data.state_vars order)
    gets a class at every logged transition and at every model row."""

    n_classes: int
    class_names: list[str]

    def classes_of_rows(self, df) -> np.ndarray: ...  # [rows, n] class per variable per transition

    def classes_of_model(self, data) -> np.ndarray: ...  # [S, K, n] class per variable per model row

    def untied_keys(self, df) -> np.ndarray: ...  # [rows, n] the untied row of every variable (any ints)


# ---- S4 ----------------------------------------------------------------------------------------------
def pooled_counts(data, template: Template) -> np.ndarray:
    """[C, 2] counts of x_i' = 0 / 1 per class, pooled over variables and transitions."""
    C = template.classes_of_rows(data.df)  # [rows, n]
    nxt = np.stack([data.df[v + NEXT_MARK].values for v in data.state_vars], axis=1)
    counts = np.zeros((template.n_classes, 2), dtype=np.int64)
    np.add.at(counts, (C.ravel(), nxt.ravel()), 1)
    return counts


def theta_box(counts: np.ndarray, delta: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """theta's confidence box [lo, hi] (each [C]): NBN's Clopper-Pearson intervals, trajectory-valid,
    with the union bound over the table's entries; a class without data gets [0, 1]."""
    from nbn.uncertainty.builders import interval_set_from_counts

    sets = interval_set_from_counts(
        torch.tensor(counts, dtype=torch.float64), delta=delta, method="clopper_pearson", sampling="trajectory"
    )
    lower, upper = (t.double().numpy() for t in sets.bounds())
    lo, hi = lower[:, 1].copy(), upper[:, 1].copy()
    empty = counts.sum(1) == 0
    lo[empty], hi[empty] = 0.0, 1.0
    return lo, hi


def tying_test(data, template: Template) -> dict:
    """Likelihood-ratio (G^2) test of the tied model (one Bernoulli per class) against the untied one (one
    Bernoulli per variable and untied row, template.untied_keys); df = untied rows with data - classes
    with data, chi-square reference (rough on sparse rows: a diagnostic, not a gate)."""
    C = template.classes_of_rows(data.df)
    keys = template.untied_keys(data.df)
    counts = pooled_counts(data, template)
    p_tied = (counts[:, 1] / np.maximum(counts.sum(1), 1)).clip(1e-12, 1 - 1e-12)
    loglik_tied = loglik_untied = 0.0
    rows_untied = 0
    for i, v in enumerate(data.state_vars):
        y = data.df[v + NEXT_MARK].values
        _, row = np.unique(keys[:, i], return_inverse=True)
        k = np.bincount(row, weights=y)
        n = np.bincount(row)
        p = np.clip(k / n, 1e-12, 1 - 1e-12)
        loglik_untied += float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))
        rows_untied += len(n)
        c = C[:, i]
        loglik_tied += float(np.sum(np.where(y == 1, np.log(p_tied[c]), np.log(1 - p_tied[c]))))
    dof = rows_untied - int((counts.sum(1) > 0).sum())
    g2 = max(2.0 * (loglik_untied - loglik_tied), 0.0)
    return {"tying_g2": g2, "tying_df": dof, "tying_p": float(chi2.sf(g2, dof)) if dof > 0 else float("nan")}


# ---- S7 ----------------------------------------------------------------------------------------------
def factor_bounds(data, template: Template, lo: np.ndarray, hi: np.ndarray) -> FactorBounds:
    """Every variable's row gets its class's interval for P(x_i' = 1); P(x_i' = 0) = 1 - that."""
    C = template.classes_of_model(data)  # [S, K, n]
    S, K, n = C.shape
    flo, fhi = np.zeros((S, K, n, 2)), np.zeros((S, K, n, 2))
    flo[..., 1], fhi[..., 1] = lo[C], hi[C]
    flo[..., 0], fhi[..., 0] = 1.0 - hi[C], 1.0 - lo[C]
    unseen = float(np.mean((lo <= 0.0) & (hi >= 1.0)))  # classes without data (the interval [0, 1])
    info = {"pmdp_classes": template.n_classes, "s4_rows": template.n_classes, "s4_unseen": unseen}
    return FactorBounds(flo, fhi, data.state_cards, data.B, data.reward, info=info)


# ---- S8 (b): the static worst case of a fixed policy over theta's box ---------------------------------
def _onehot(policy: np.ndarray, S: int, K: int) -> np.ndarray:
    out = np.zeros((S, K))
    out[np.arange(S), policy] = 1.0
    return out


def exact_value(data, template, theta, policy, gamma, start) -> float:
    """The policy's exact value at the start distribution under the kernel of one theta."""
    theta = np.asarray(theta, dtype=float)
    mdp = product_bounds.compile(factor_bounds(data, template, theta, theta))
    V = robust_vi.evaluate(mdp, _onehot(policy, len(data.B), len(data.actions)), gamma)
    return float(start @ V)


def region_refinement(data, template, lo, hi, policy, gamma, start, splits: int = 2, max_split: int = 4) -> float:
    """A sound lower bound on min_{theta in box} V^policy_theta(start): split the `max_split` widest
    classes' intervals into `splits` equal parts (the others stay whole), relax every sub-box to its box
    and robust-evaluate the policy there; the minimum over the sub-boxes. The sub-boxes cover the box."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    order = np.argsort(-(hi - lo))[:max_split]
    chosen = {int(c) for c in order if hi[c] > lo[c]}
    edges = [np.linspace(lo[c], hi[c], splits + 1) if c in chosen else np.array([lo[c], hi[c]])
             for c in range(len(lo))]  # fmt: skip
    onehot = _onehot(policy, len(data.B), len(data.actions))
    best, V0 = np.inf, None
    for cell in itertools.product(*[range(len(e) - 1) for e in edges]):
        rlo = np.array([edges[c][j] for c, j in enumerate(cell)])
        rhi = np.array([edges[c][j + 1] for c, j in enumerate(cell)])
        V = robust_vi.evaluate(product_bounds.compile(factor_bounds(data, template, rlo, rhi)), onehot, gamma, V0=V0)
        V0 = V
        best = min(best, float(start @ V))
    return best


def sampled_upper(data, template, lo, hi, policy, gamma, start, points: int = 64, seed: int = 0) -> float:
    """An UPPER estimate of min_{theta in box} V^policy_theta(start): exact values at the box's two
    extreme corners, at every corner when there are at most 6 classes, and at `points` uniform draws."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    thetas = [lo, hi]
    if len(lo) <= 6:
        thetas += [np.where(np.array(mask), hi, lo) for mask in itertools.product([0, 1], repeat=len(lo))]
    thetas += list(lo + (hi - lo) * rng.random((points, len(lo))))
    return min(exact_value(data, template, t, policy, gamma, start) for t in thetas)


# ---- S8 (b) with Storm: parameter lifting on the sequentialised parametric DTMC ---------------------
def prism_pdtmc(data, template, policy, gamma, start_state: int, tag: str = "th") -> str:
    """The parametric DTMC of `policy` as a PRISM program, sequentialised (see the module docstring):

        (s, j=0)       --gamma-->    (s, j=1, b=0)  and  --(1 - gamma)--> sink   (reward R[s, pi(s)])
        (s, j=i+1, b)  --theta_c-->  x_i = 1 in b   and  --(1 - theta_c)--> x_i = 0  (c: x_i's class)
        (s, j=n+1, b)  --1-->        (s'=b, j=0)

    b is the next state's index in the C order of data.B (the first variable is the most significant);
    the class is read from the stored s, so the product form is kept. Storm needs one initial state."""
    C = template.classes_of_model(data)  # [S, K, n]
    S, n = data.B.shape
    if np.any(data.reward.r_next != 0) or np.any(data.reward.terminal):
        raise ValueError("prism_pdtmc: only rewards R[s, a] and no terminal states")
    lines = ["dtmc", "", *[f"const double {tag}{c};" for c in range(template.n_classes)], "", "module m"]
    lines.append(f"  s : [0..{S}] init {int(start_state)};")  # state S is the sink
    lines.append(f"  j : [0..{n + 1}] init 0;")
    lines.append(f"  b : [0..{S - 1}] init 0;")
    lines.append(f"  [] s<{S} & j=0 -> {gamma} : (j'=1) & (b'=0) + {1 - gamma} : (s'={S});")
    for s_ in range(S):
        k = int(policy[s_])
        for i in range(n):
            c, bit = int(C[s_, k, i]), 2 ** (n - 1 - i)
            lines.append(f"  [] s={s_} & j={i + 1} & b<={S - 1 - bit} -> {tag}{c} : (b'=b+{bit}) & (j'={i + 2})"
                         f" + (1-{tag}{c}) : (j'={i + 2});")  # fmt: skip
    lines.append(f"  [] s<{S} & j={n + 1} -> 1 : (s'=b) & (j'=0) & (b'=0);")
    lines.append(f"  [] s={S} -> 1 : true;")
    lines += ["endmodule", "", f'label "sink" = s={S};', "", 'rewards "r"']
    for s_ in range(S):
        lines.append(f"  s={s_} & j=0 : {float(data.reward.R[s_, int(policy[s_])])};")
    lines.append("endrewards")
    return "\n".join(lines)


@contextlib.contextmanager
def _quiet():
    """Silence Storm's C++ warnings (printed on the process's file descriptors 1 and 2)."""
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush(), sys.stderr.flush()
        os.dup2(null, 1), os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1), os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


def _outward(x: float, up: bool, digits: int = 9) -> str:
    """x as an exact fraction rounded outward (down for a lower end, up for an upper end) in [0, 1], so
    that Storm's region contains the whole interval."""
    scale = 10**digits
    q = math.ceil(x * scale) if up else math.floor(x * scale)
    return f"{min(max(q, 0), scale)}/{scale}"


def _storm_job(queue, program_text: str, lo, hi, tag: str, precision: float) -> None:
    """One Storm computation in a fresh process (Storm keeps state across parametric models in one
    process: a second model failed with an internal IndexError). Puts (value, theta) or an error."""
    try:
        import stormpy
        import stormpy.pars as pars

        with tempfile.NamedTemporaryFile("w", suffix=".pm", delete=False) as f:
            f.write(program_text)
        try:
            program = stormpy.parse_prism_program(f.name)
        finally:
            os.unlink(f.name)
        prop = stormpy.parse_properties('R{"r"}=? [F "sink"]', program)[0]
        with _quiet():
            model = stormpy.build_model[stormpy.RationalFunction](program, [prop])
            params = model.collect_all_parameters()
            names = {v.name for v in params}
            region = ",".join(f"{_outward(lo[c], up=False)}<={tag}{c}<={_outward(hi[c], up=True)}"
                              for c in range(len(lo)) if f"{tag}{c}" in names)  # fmt: skip
            value, valuation = pars.compute_extremal_value(
                model,
                prop,
                region=pars.parse_region(region, params),
                direction=pars.OptimizationDirection.MINIMIZE,
                precision=precision,
                absolute_precision=True,
                engine=pars.RegionCheckEngine.ROBUST_PARAMETER_LIFTING,
                environment=stormpy.environments.sound(),  # sound solution methods
            )
        theta = [float(lo[c]) for c in range(len(lo))]  # a class the policy never meets: any value
        for var, q in valuation.items():
            theta[int(var.name.removeprefix(tag))] = float(Fraction(str(q)))
        queue.put(("ok", float(Fraction(str(value))), theta))
    except Exception as err:  # noqa: BLE001 - sent back to the caller
        queue.put(("error", repr(err), None))


def storm_lower_bound(data, template, lo, hi, policy, gamma, start_state: int, precision: float = 1e-3,
                      timeout_s: float = 600.0) -> dict:  # fmt: skip
    """Storm's parameter lifting (robust engine: the region may touch 0 or 1; sound solution methods)
    with region refinement on the sequentialised parametric DTMC of `policy`: the minimum over theta's
    box of the expected reward until the sink (= the discounted value), to an absolute precision.
    Returns storm_value (at the valuation found: within `precision` above the minimum), storm_lower =
    storm_value - precision (a sound lower bound), the valuation, and storm_value_check: the policy's
    exact value there computed here (equal to storm_value up to the solvers' tolerances). Runs in a
    fresh process; a run over timeout_s is stopped (storm_status "timeout")."""
    tag = "th"
    text = prism_pdtmc(data, template, policy, gamma, start_state, tag)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    args = (queue, text, list(map(float, lo)), list(map(float, hi)), tag, precision)
    proc = ctx.Process(target=_storm_job, args=args)
    proc.start()
    try:
        status, value, theta = queue.get(timeout=timeout_s)
    except Exception:  # noqa: BLE001 - queue.Empty: no answer in time
        status, value, theta = "timeout", None, None
    if proc.is_alive():
        proc.terminate()
    proc.join()
    nan = {"storm_value": np.nan, "storm_lower": np.nan, "storm_value_check": np.nan}
    if status != "ok":
        return {"storm_status": status, "storm_error": value, **nan}
    start = np.eye(len(data.B))[start_state]
    check = exact_value(data, template, theta, policy, gamma, start)
    thetas = {f"storm_theta_{name}": t for name, t in zip(template.class_names, theta, strict=True)}
    return {"storm_status": "ok", "storm_value": value, "storm_lower": value - precision, "storm_value_check": check,
            **thetas}  # fmt: skip
