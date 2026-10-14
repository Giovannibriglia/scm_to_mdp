"""J-PCMCI+ on logged episodes, with the logger as an observed context (pipeline step S2).

J-PCMCI+ (Günther, Ninad & Runge, UAI 2023; Tigramite) pools time series from several datasets and
adds context nodes, an instance of Joint Causal Inference (Mooij, Magliacane & Claassen, 2020). Here
every episode is one dataset, the logger's id is a *space* context (constant within an episode), the
system variables are the machines ``m_i`` and the repair flags ``rep_i``, and the test is G-squared
(discrete data). Link assumptions encode the time order: machines get lag-1 parents from the system
variables; within a step, machines and the context may point into the repair flags (the policy); and
the context may point into a machine, which only confounding or a changed mechanism would produce, so
such a link is reported as a flag for the S1 gate.

Tigramite is GPL-3.0 and optional: install the ``discovery`` extra (``uv sync --extra discovery``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scm_to_mdp.data import Transitions
from scm_to_mdp.envs.sysadmin import SysAdminRing

_MISSING = -99.0


@dataclass
class JPCMCIResult:
    parents: dict[str, tuple[str, ...]]      # target m_i' -> lag-1 parents (input names)
    context_into_state: dict[str, str]       # target -> link mark from the logger context ('' = none)
    graph: np.ndarray                         # tigramite graph [N, N, tau_max + 1]
    names: list[str]


def _series(env: SysAdminRing, data: Transitions) -> dict[int, np.ndarray]:
    """One array per episode: rows (s_t, a_t, logger) for t = 0..T-1, then (s_T, missing, logger)."""
    key = data.episode.astype(np.int64) + 1_000_000 * data.regime.astype(np.int64)
    out = {}
    for k in np.unique(key):
        m = key == k
        order = np.argsort(data.t[m])
        s, a, s2 = data.state[m][order], env.flags(data.action[m][order]), data.next_state[m][order]
        rows = np.vstack([np.hstack([s, a]), np.hstack([s2[-1:], np.full((1, env.n), _MISSING)])])
        ctx = np.full((rows.shape[0], 1), float(data.regime[m][0]))
        out[int(k)] = np.hstack([rows, ctx]).astype(float)
    return out


def _link_assumptions(n: int) -> dict[int, dict[tuple[int, int], str]]:
    """``la[j][(i, -tau)]`` is the mark of ``graph[i, j, tau]``; same-step marks mirror each other.

    Nodes: machines ``0..n-1``, repair flags ``n..2n-1``, the logger context ``2n``.
    """
    ctx = 2 * n
    la: dict[int, dict[tuple[int, int], str]] = {j: {} for j in range(2 * n + 1)}
    for j in range(n):                                   # machine j at time t
        for i in range(2 * n):
            la[j][(i, -1)] = "-?>"                       # lag-1 parent from any system variable
        for i in range(n, 2 * n):
            la[j][(i, 0)] = "<?-"                        # same step: machine j -> repair flag i (policy)
        la[j][(ctx, 0)] = "-?>"                          # context -> machine: allowed, then flagged
    for j in range(n, 2 * n):                            # repair flag j at time t
        for i in range(n):
            la[j][(i, 0)] = "-?>"                        # machine i -> repair flag j (policy)
        la[j][(ctx, 0)] = "-?>"                          # context -> repair flag (the logger)
    for i in range(2 * n):
        la[ctx][(i, 0)] = "<?-"                          # the context is never a child
    return la


def jpcmci_parents(env: SysAdminRing, data: Transitions, alpha: float = 0.01) -> JPCMCIResult:
    """Run J-PCMCI+ and read the lag-1 parents of every next-state machine."""
    try:
        from tigramite.data_processing import DataFrame
        from tigramite.independence_tests.gsquared import Gsquared
        from tigramite.jpcmciplus import JPCMCIplus
    except ImportError as err:  # pragma: no cover - depends on the optional extra
        raise ImportError("J-PCMCI+ needs Tigramite: uv sync --extra discovery") from err
    n = env.n
    names = env.state_vars + env.action_flags + ["logger"]
    node_classification = {i: "system" for i in range(2 * n)}
    node_classification[2 * n] = "space_context"
    df = DataFrame(_series(env, data), analysis_mode="multiple", var_names=names, missing_flag=_MISSING)
    jp = JPCMCIplus(node_classification=node_classification, dataframe=df,
                    cond_ind_test=Gsquared(significance="analytic"), verbosity=0)
    res = jp.run_jpcmciplus(tau_min=0, tau_max=1, pc_alpha=alpha, link_assumptions=_link_assumptions(n))
    g = res["graph"]
    parents = {f"m{j}'": tuple(names[i] for i in range(2 * n) if g[i, j, 1] == "-->") for j in range(n)}
    context = {f"m{j}'": str(g[2 * n, j, 0]) for j in range(n)}
    return JPCMCIResult(parents=parents, context_into_state=context, graph=g, names=names)
