"""From a family of candidate parent sets to an interval MDP (pipeline steps S4 and S7, through NBN).

S4. For each next-state variable ``m_i'`` an arm gives a family of candidate parent sets and a mode:

- ``refute=False``: the union over the given family (NBN ``agreement_set(..., refute=False)``): one L1
  ball per row that contains every candidate's ball, each built at the node's full budget. Valid when a
  fixed candidate contains the true parents (oracle, one learned graph, flat), or, for a data-driven
  family that contains the true parents with probability ``1 - delta_G`` (ICP), with probability
  ``1 - delta_G - delta``.
- ``refute=True``: the agreement rule (``agreement_set``): candidates disagreeing with a one-variable
  extension are discarded, under the declared bound on the number of parents.
- ``confounded_sets``: when the S1 gate rejects, interval rows on ``P(m_i' | do(rep_i), u)`` from the
  logging regimes (NBN ``confounded_interval_set``; Manski rows intersected over regimes, or the IV
  linear program with the declared randomised regimes).

Sets are built from counts of the *estimation* half (``sampling="trajectory"``); the total budget
``delta`` is split equally over the ``N`` nodes.

S7. The per-node sets are written with ``export_factored_sets`` (schema ``nbn.factored_sets/1``, the
exact factored model with its row maps) and read back. Each joint row's box combines the factor rows:

- ``joint="product"``: ``prod lower_i <= p(x') <= prod upper_i``, sound when the machines move
  independently given ``(s, a)`` (setting 1; NBN ``joint_row_bounds``);
- ``joint="frechet"``: the Fréchet bounds ``max(0, sum_i lower_i - (N - 1)) <= p(x') <= min_i upper_i``,
  sound for *any* dependence between the machines, e.g. a hidden load that moves all of them (S-E).

Extra observed inputs that are not part of the planning state (the distractor bit ``x``) can be
parents of a factor. The planner does not control them and they are drawn afresh at every step, so
each ``(s, a)`` box is the entry-wise hull over their values: nature picks the worst value, which is
sound whatever their law.
"""
from __future__ import annotations

import itertools
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from nbn.structure import agreement_set, export_factored_sets, load_factored_sets
from nbn.uncertainty import confounded_interval_set

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.structure import Candidate, family_counts


def inputs_of(env: SysAdminRing, extra: Sequence[str] = ()) -> list[str]:
    """Every variable allowed as a parent: machines, repair flags, then extra observed inputs."""
    return env.state_vars + env.action_flags + list(extra)


def nodes_of(env: SysAdminRing) -> list[str]:
    return [f"{v}'" for v in env.state_vars]


def variables(env: SysAdminRing, extra: Sequence[str] = ()) -> dict[str, tuple[str, int]]:
    """NBN variable specs: every input and every next-state machine is binary."""
    return dict.fromkeys(inputs_of(env, extra) + nodes_of(env), ("discrete", 2))


def torch_columns(cols: Mapping[str, np.ndarray], env: SysAdminRing,
                  extra: Sequence[str] = ()) -> dict[str, torch.Tensor]:
    """The observed columns as float tensors (what NBN fits from), plus ``regime``."""
    out = {k: torch.as_tensor(np.asarray(cols[k]), dtype=torch.float32) for k in variables(env, extra)}
    out["regime"] = torch.as_tensor(np.asarray(cols["regime"]), dtype=torch.float32)
    return out


@dataclass
class NodeSet:
    """The S4 result for one node, with what the arm asked for."""

    node: str
    family: list[Candidate]
    refute: bool
    result: object                       # nbn AgreementSet or ConfoundedSet
    empty_family: bool = False           # the arm proposed no candidate: the flat model was used
    confounded: bool = False             # built from confounded rows (S1 gate rejected)

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(self.result.parents)

    @property
    def survivors(self) -> list[Candidate]:
        return list(getattr(self.result, "survivors", [self.parents]))


@dataclass
class IntervalModel:
    """S7 output: entry-wise boxes ``[S, A, S']`` for every feasible joint row, plus bookkeeping."""

    L: np.ndarray
    U: np.ndarray
    nodes: dict[str, NodeSet] = field(default_factory=dict)
    narrow: np.ndarray | None = None     # [S, A]: the row's box is narrower than the simplex
    manifest: dict = field(default_factory=dict)
    factor_rows: int = 0                 # model card: rows over all factors' parent spaces
    coupling: float = 0.0                # model card: mean joint rows sharing one factor row

    @property
    def rows_with_set(self) -> int:
        return int(self.narrow.sum())


def node_sets(env: SysAdminRing, cols: Mapping[str, np.ndarray], families: Mapping[str, Sequence[Candidate]],
              refute: bool, delta: float, max_parents: int | None = None,
              extra: Sequence[str] = ()) -> dict[str, NodeSet]:
    """S4 for every node. ``families[node]`` is the arm's family; an empty family falls back to flat."""
    data = torch_columns(cols, env, extra)
    spec = variables(env, extra)
    inputs = inputs_of(env, extra)
    out = {}
    for node in nodes_of(env):
        fam = [tuple(c) for c in families.get(node, [])] if families is not None else []
        empty = families is not None and not fam
        if empty and not refute:
            fam = [tuple(inputs)]
        kw = {"candidates": fam} if fam else {"max_parents": max_parents}
        res = agreement_set(spec, data, node, inputs, delta=delta / env.n, sampling="trajectory",
                            refute=refute, **kw)
        out[node] = NodeSet(node, fam, refute, res, empty)
    return out


@dataclass
class _DagView:
    """The two attributes ``confounded_interval_set`` reads from a model: variables and parents."""

    variables: dict
    parent_map: dict

    @property
    def dag(self):
        return self

    def parents(self, node):
        return list(self.parent_map[node])


def confounded_sets(env: SysAdminRing, cols: Mapping[str, np.ndarray], parents: Mapping[str, Candidate],
                    delta: float, declared_unconfounded: Sequence[int] = (),
                    extra: Sequence[str] = ()) -> dict[str, NodeSet]:
    """S4 when the S1 gate rejects: rows on ``P(m_i' | do(rep_i), other parents)`` for every node.

    ``parents[node]`` must contain the node's repair flag (the action). NBN validates the variables
    and reads the parents through a light view of a model (``variables`` and ``dag.parents``).
    """
    from nbn.core.variables import DiscreteVariable

    data = torch_columns(cols, env, extra)
    view = _DagView({k: DiscreteVariable(k, 2) for k in variables(env, extra)}, dict(parents))
    n_regimes = int(np.max(cols["regime"])) + 1
    out = {}
    for i, node in enumerate(nodes_of(env)):
        res = confounded_interval_set(view, data, node, f"rep{i}", "regime", method="auto",
                                      delta=delta / env.n, sampling="trajectory",
                                      declared_unconfounded=tuple(declared_unconfounded), n_regimes=n_regimes)
        out[node] = NodeSet(node, [tuple(parents[node])], False, res, confounded=True)
    return out


def planning_row(env: SysAdminRing, s: np.ndarray, a: np.ndarray, x: Sequence[int] = ()) -> np.ndarray:
    """Index of ``(s, x, a)`` among the C-order rows over ``(m_0..m_{N-1}, x..., rep_0..rep_{N-1})``.

    This is the export's planning space: state variables (machines, then extra inputs), then actions.
    """
    s, a = np.asarray(s), np.asarray(a)
    parts = [env.states[s]]
    if len(x):
        parts.append(np.broadcast_to(np.asarray(x, dtype=np.int8), s.shape + (len(x),)))
    parts.append(env.flags(a))
    bits = np.concatenate(parts, axis=-1)
    return np.ravel_multi_index(np.moveaxis(bits, -1, 0), [2] * bits.shape[-1])


def _factor_rows(ex, fid: str, row: int) -> tuple[np.ndarray, np.ndarray]:
    r = ex.factor_row(fid, row)
    lo, hi = ex.array(fid, "lower")[r], ex.array(fid, "upper")[r]
    if bool(ex.array(fid, "unseen")[r]):
        lo, hi = np.zeros_like(lo), np.ones_like(hi)
    return lo, hi


def _joint(los, his, joint: str) -> tuple[np.ndarray, np.ndarray]:
    """Joint box over ``prod_i K_i`` values (C order) from per-factor rows."""
    if joint == "product":
        lo, hi = np.ones(()), np.ones(())
        for a, b in zip(los, his, strict=True):
            lo, hi = np.multiply.outer(lo, a), np.multiply.outer(hi, b)
        return lo.reshape(-1), np.minimum(hi, 1.0).reshape(-1)
    if joint == "frechet":
        grids = np.meshgrid(*[np.arange(len(a)) for a in los], indexing="ij")
        lsum = sum(lo[g] for lo, g in zip(los, grids, strict=True))
        hmin = np.minimum.reduce([hi[g] for hi, g in zip(his, grids, strict=True)])
        return np.maximum(0.0, lsum - (len(los) - 1)).reshape(-1), np.minimum(hmin, 1.0).reshape(-1)
    raise ValueError(f"joint must be 'product' or 'frechet'; got {joint!r}")


def interval_model(env: SysAdminRing, sets: Mapping[str, NodeSet], joint: str = "product",
                   extra: Sequence[str] = ()) -> IntervalModel:
    """S7: export the factored sets, read them back, and build the joint box of every (s, a)."""
    S, A = env.n_states, env.n_actions
    L, U = np.ones((S, A, S)), np.zeros((S, A, S))
    order = nodes_of(env)
    xs = list(itertools.product((0, 1), repeat=len(extra)))
    with tempfile.TemporaryDirectory() as tmp:
        path = export_factored_sets(f"{tmp}/model", {n: sets[n].result for n in order},
                                    variables(env, extra), env.state_vars + list(extra), env.action_flags)
        ex = load_factored_sets(path)
        (hyp,) = ex.hypotheses()
        used: list[set[int]] = [set() for _ in order]
        for s, a in itertools.product(range(S), range(A)):
            for x in xs:      # nature picks the extra inputs' values: hull over them
                row = int(planning_row(env, np.array(s), np.array(a), x))
                pairs = [_factor_rows(ex, hyp[n], row) for n in order]
                for k, n in enumerate(order):
                    used[k].add(ex.factor_row(hyp[n], row))
                lo, hi = _joint([p[0] for p in pairs], [p[1] for p in pairs], joint)
                L[s, a] = np.minimum(L[s, a], lo)
                U[s, a] = np.maximum(U[s, a], hi)
        manifest = ex.manifest
    narrow = (L.sum(-1) > 1e-12) | (U.max(-1) < 1.0 - 1e-12)
    factor_rows = sum(math.prod(sets[n].result.set.center.shape[:-1]) if hasattr(sets[n].result.set, "center")
                      else int(sets[n].result.set.bounds()[0].shape[0]) for n in order)
    coupling = float(np.mean([S * A * len(xs) / max(len(u), 1) for u in used]))
    return IntervalModel(L, U, dict(sets), narrow, manifest, factor_rows, coupling)


def plug_in_kernel(env: SysAdminRing, cols: Mapping[str, np.ndarray],
                   parents: Mapping[str, Candidate], extra: Sequence[str] = ()) -> np.ndarray:
    """Maximum-likelihood factored kernel ``[S, A, S']`` on given parent sets (unseen rows: 1/2).

    An extra input among the parents is averaged out with its empirical frequency.
    """
    S, A = env.n_states, env.n_actions
    flags = env.flags(np.arange(A))
    p_up = np.zeros((S, A, env.n))
    freq = {x: float(np.mean(cols[x])) for x in extra}
    for i, v in enumerate(env.state_vars):
        node, pa = f"{v}'", tuple(parents[f"{v}'"])
        counts = family_counts(cols, node, pa)                             # [R, 2]
        n = counts.sum(1)
        prob = np.where(n > 0, counts[:, 1] / np.maximum(n, 1), 0.5)
        values = {**{m: env.states[:, None, k] for k, m in enumerate(env.state_vars)},
                  **{r: flags[None, :, k] for k, r in enumerate(env.action_flags)}}
        ex_pa = [p for p in pa if p in freq]
        for xv in itertools.product((0, 1), repeat=len(ex_pa)):
            weight = math.prod(freq[p] if b else 1 - freq[p] for p, b in zip(ex_pa, xv, strict=True))
            vals = dict(values, **{p: np.int8(b) for p, b in zip(ex_pa, xv, strict=True)})
            if pa:
                idx = np.ravel_multi_index([np.broadcast_to(vals[p], (S, A)) for p in pa], [2] * len(pa))
            else:
                idx = np.zeros((S, A), dtype=np.int64)
            p_up[:, :, i] += weight * prob[idx]
    nxt = env.states[None, None, :, :]
    per = np.where(nxt == 1, p_up[:, :, None, :], 1.0 - p_up[:, :, None, :])
    return per.prod(-1)


def candidate_bounds(env: SysAdminRing, cols: Mapping[str, np.ndarray], node: str, cand: Candidate,
                     delta: float, extra: Sequence[str] = ()) -> np.ndarray:
    """``[S, A, 2, 2]`` (lower/upper of P(node = 0/1)) of one candidate's L1 ball on every (s, a).

    Extra inputs among the parents are hulled over their values (as the planner sees them). Used by
    S5 to compare candidates row by row.
    """
    from nbn.uncertainty.builders import l1_ball_set_from_counts

    counts = torch.as_tensor(family_counts(cols, node, cand), dtype=torch.float64)
    lo, hi = (t.numpy() for t in l1_ball_set_from_counts(counts, delta=delta, sampling="trajectory").bounds())
    S, A = env.n_states, env.n_actions
    flags = env.flags(np.arange(A))
    values = {**{m: env.states[:, None, k] for k, m in enumerate(env.state_vars)},
              **{r: flags[None, :, k] for k, r in enumerate(env.action_flags)}}
    ex_pa = [p for p in cand if p in extra]
    out_lo, out_hi = np.ones((S, A, 2)), np.zeros((S, A, 2))
    for xv in itertools.product((0, 1), repeat=len(ex_pa)):
        vals = dict(values, **{p: np.int8(b) for p, b in zip(ex_pa, xv, strict=True)})
        idx = (np.ravel_multi_index([np.broadcast_to(vals[p], (S, A)) for p in cand], [2] * len(cand))
               if cand else np.zeros((S, A), dtype=np.int64))
        out_lo, out_hi = np.minimum(out_lo, lo[idx]), np.maximum(out_hi, hi[idx])
    return np.stack([out_lo, out_hi], axis=-2)
