"""From a family of candidate parent sets to an interval MDP (pipeline steps S4 and S7, through NBN).

S4. For each next-state variable ``m_i'`` an arm gives a family of candidate parent sets and a mode:

- ``refute=False``: the union over the given family (NBN ``agreement_set(..., refute=False)``): one L1
  ball per row that contains every candidate's ball, each built at the node's full budget. Valid when a
  fixed candidate contains the true parents (oracle, one learned graph, flat), or, for a data-driven
  family that contains the true parents with probability ``1 - delta_G`` (ICP), with probability
  ``1 - delta_G - delta``.
- ``refute=True``: the agreement rule (``agreement_set``): candidates disagreeing with a one-variable
  extension are discarded, under the declared bound on the number of parents.

Sets are built from counts of the *estimation* half (``sampling="trajectory"``); the total budget
``delta`` is split equally over the ``N`` nodes.

S7. The per-node sets are written with ``export_factored_sets`` (schema ``nbn.factored_sets/1``, the
exact factored model with its row maps) and read back; ``joint_row_bounds`` gives each joint row's
entry-wise box (products of the factor bounds: an outer relaxation, so still sound). Rows a factor never
saw are ``[0, 1]`` for that factor.
"""
from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from nbn.structure import agreement_set, export_factored_sets, joint_row_bounds, load_factored_sets

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.structure import Candidate, family_counts


def variables(env: SysAdminRing) -> dict[str, tuple[str, int]]:
    """NBN variable specs: every machine, repair flag and next-state machine is binary."""
    names = env.state_vars + env.action_flags + [f"{v}'" for v in env.state_vars]
    return dict.fromkeys(names, ("discrete", 2))


def torch_columns(cols: Mapping[str, np.ndarray], env: SysAdminRing) -> dict[str, torch.Tensor]:
    """The observed columns as float tensors (what NBN fits from)."""
    return {k: torch.as_tensor(np.asarray(cols[k]), dtype=torch.float32) for k in variables(env)}


@dataclass
class NodeSet:
    """The S4 result for one node, with what the arm asked for."""

    node: str
    family: list[Candidate]
    refute: bool
    result: object                       # nbn.structure.AgreementSet
    empty_family: bool = False           # the arm proposed no candidate: the flat model was used


@dataclass
class IntervalModel:
    """S7 output: entry-wise boxes ``[S, A, S']`` for every feasible joint row, plus bookkeeping."""

    L: np.ndarray
    U: np.ndarray
    nodes: dict[str, NodeSet] = field(default_factory=dict)
    rows_with_set: int = 0               # (s, a) rows whose box is narrower than the simplex
    manifest: dict = field(default_factory=dict)


def node_sets(env: SysAdminRing, cols: Mapping[str, np.ndarray], families: Mapping[str, Sequence[Candidate]],
              refute: bool, delta: float, max_parents: int | None = None) -> dict[str, NodeSet]:
    """S4 for every node. ``families[node]`` is the arm's family; an empty family falls back to flat."""
    data = torch_columns(cols, env)
    spec = variables(env)
    inputs = env.state_vars + env.action_flags
    out = {}
    for node in (f"{v}'" for v in env.state_vars):
        fam = [tuple(c) for c in families.get(node, [])] if families is not None else []
        empty = refute is False and not fam
        if empty:
            fam = [tuple(inputs)]
        kw = {"candidates": fam} if fam else {"max_parents": max_parents}
        res = agreement_set(spec, data, node, inputs, delta=delta / env.n, sampling="trajectory",
                            refute=refute, **kw)
        out[node] = NodeSet(node, fam, refute, res, empty)
    return out


def planning_row(env: SysAdminRing, s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Index of ``(s, a)`` among the C-order rows over ``(m_0..m_{N-1}, rep_0..rep_{N-1})``."""
    bits = np.concatenate([env.states[s], env.flags(a)], axis=-1)
    return np.ravel_multi_index(np.moveaxis(bits, -1, 0), [2] * (2 * env.n))


def interval_model(env: SysAdminRing, sets: Mapping[str, NodeSet]) -> IntervalModel:
    """S7: export the factored sets, read them back, and build the joint boxes of every (s, a)."""
    S, A = env.n_states, env.n_actions
    L, U = np.zeros((S, A, S)), np.zeros((S, A, S))
    order = [f"{v}'" for v in env.state_vars]
    with tempfile.TemporaryDirectory() as tmp:
        path = export_factored_sets(f"{tmp}/model", {n: sets[n].result for n in order}, variables(env),
                                    env.state_vars, env.action_flags)
        ex = load_factored_sets(path)
        (hyp,) = ex.hypotheses()
        hyp = {n: hyp[n] for n in order}                  # C order of the next state = our state index
        s_idx, a_idx = np.meshgrid(np.arange(S), np.arange(A), indexing="ij")
        rows = planning_row(env, s_idx, a_idx)
        for s in range(S):
            for a in range(A):
                lo, hi, nodes = joint_row_bounds(ex, hyp, int(rows[s, a]))
                assert nodes == order
                L[s, a], U[s, a] = lo.reshape(-1), hi.reshape(-1)
        manifest = ex.manifest
    narrow = (L.sum(-1) > 1e-12) | (U.max(-1) < 1.0 - 1e-12)
    return IntervalModel(L, U, dict(sets), int(narrow.sum()), manifest)


def plug_in_kernel(env: SysAdminRing, cols: Mapping[str, np.ndarray],
                   parents: Mapping[str, Candidate]) -> np.ndarray:
    """Maximum-likelihood factored kernel ``[S, A, S']`` on given parent sets (unseen rows: 1/2)."""
    S, A = env.n_states, env.n_actions
    flags = env.flags(np.arange(A))
    p_up = np.zeros((S, A, env.n))
    for i, v in enumerate(env.state_vars):
        node, pa = f"{v}'", tuple(parents[f"{v}'"])
        counts = family_counts(cols, node, pa)                             # [R, 2]
        n = counts.sum(1)
        prob = np.where(n > 0, counts[:, 1] / np.maximum(n, 1), 0.5)
        values = {**{m: env.states[:, None, k] for k, m in enumerate(env.state_vars)},
                  **{r: flags[None, :, k] for k, r in enumerate(env.action_flags)}}
        if pa:
            idx = np.ravel_multi_index([np.broadcast_to(values[p], (S, A)) for p in pa], [2] * len(pa))
        else:
            idx = np.zeros((S, A), dtype=np.int64)
        p_up[:, :, i] = prob[idx]
    nxt = env.states[None, None, :, :]
    per = np.where(nxt == 1, p_up[:, :, None, :], 1.0 - p_up[:, :, None, :])
    return per.prod(-1)
