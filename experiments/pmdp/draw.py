"""Picture of the learned parametric MDP: every drawn edge carries its probability as an expression in the
class parameters (a product of theta_c and 1 - theta_c, one factor per state variable) and its interval
over theta's confidence box. Built on plotting/mdp_graph.draw (the edges drawn are the most likely
outcomes under the TRUE model, as in every MDP picture of the repository).

    pmdp_all     every action (for small problems)
    pmdp_policy  only the action of the policy in every state: the parametric DTMC that option (b)
                 hands to Storm
"""

from __future__ import annotations

import numpy as np

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import IntervalMDP
from scm_to_mdp.plotting import mdp_graph

import tied

GREEK = "θ"  # theta


def short_names(class_names: list[str]) -> list[str]:
    """Readable parameter symbols: theta with the class name as a subscript (e.g. theta_up_ok)."""
    return [f"{GREEK}[{c}]" for c in class_names]


def expressions(data, template) -> np.ndarray:
    """[S, K, S'] the probability of every transition as a product of class parameters."""
    C = template.classes_of_model(data)  # [S, K, n]
    names = short_names(template.class_names)
    B = data.B
    S, K, n = C.shape
    out = np.empty((S, K, S), dtype=object)
    for s in range(S):
        for k in range(K):
            for t in range(S):
                factors = [names[C[s, k, i]] if B[t, i] == 1 else f"(1-{names[C[s, k, i]]})" for i in range(n)]
                out[s, k, t] = "·".join(factors)  # middle dot
    return out


def labels(data, template, lo, hi) -> tuple[np.ndarray, IntervalMDP]:
    """Edge labels "expression\n[L, U]" with [L, U] the box relaxation over theta's box, and that box."""
    box = product_bounds.compile(tied.factor_bounds(data, template, lo, hi))
    expr = expressions(data, template)
    lab = np.empty(expr.shape, dtype=object)
    for idx in np.ndindex(expr.shape):
        lab[idx] = f"{expr[idx]}\n[{box.L[idx]:.2f}, {box.U[idx]:.2f}]"
    return lab, box


def draw_pmdp(data, template, lo, hi, P_true, title, stem, policy=None, state_names=None, action_names=None):
    """The learned pMDP (policy None: every action) or its parametric DTMC under `policy` [S]."""
    from pathlib import Path

    stem = Path(stem)
    lab, box = labels(data, template, lo, hi)
    S, K = len(data.B), len(data.actions)
    states = state_names or ["".join(str(int(v)) for v in row) for row in data.B]
    actions = action_names or [", ".join(f"{a}={int(v)}" for a, v in zip(data.action_vars, row, strict=True))
                               for row in data.actions]  # fmt: skip
    if policy is not None:
        rows = np.arange(S)
        P_true, lab = P_true[rows, policy][:, None, :], lab[rows, policy][:, None, :]
        box = IntervalMDP(box.L[rows, policy][:, None, :], box.U[rows, policy][:, None, :], box.reward)
        actions = [f"policy's action (π)"]
    legend = "  ".join(f"{name}: {c}" for name, c in zip(short_names(template.class_names), template.class_names,
                                                           strict=True))  # fmt: skip
    mdp_graph.draw(P_true, box, states, actions, f"{title}\nstates: {', '.join(data.state_vars)}",
                   stem, threshold=0.05, max_outcomes=3, edge_labels=lab)  # fmt: skip
    return legend
