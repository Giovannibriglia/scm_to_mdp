"""One end-to-end run of the pipeline on the SysAdmin ring, for every arm (steps S1 to S9).

S1: the data are split by episode: the discovery half feeds S2/S3 (which graphs) and the S1 gate, the
estimation half feeds S4 (the sets), so no data are used twice. S2/S3: each arm proposes a family of
parent sets per node (see ``propose``). S4/S7: ``scm_to_mdp.sets`` builds the per-node sets with NBN
and exports the factored model; joint rows become interval boxes. S6 is declared: worst case, dynamic
nature, discounted, Markov stationary policies, start state "all machines up". S8: robust value
iteration on the boxes. S9: the exact evaluator gives the true value of the returned policy, its
regret, and whether the promise held (promise <= true value).

``oracle_gated`` follows the S1 gate: if the policy-as-context test rejects, the oracle parents get
confounded rows (NBN ``confounded_interval_set``, declared randomised regimes from the loggers'
metadata) combined by Fréchet bounds (a hidden variable may move every machine); otherwise the
ordinary oracle sets.

S5 (does graph disagreement matter?), for arms with several candidates per node:

- *collapse*: per node, candidates whose sets agree within ``tau`` (largest entry-wise gap of the
  lower and upper bounds) on every row ``(s, a)`` of a state the data reached are merged
  (``K -> K'``, the product over nodes);
- *decision*: up to ``n_hyp`` hypotheses (one candidate per node) are drawn; each one's plug-in model
  gives a greedy policy; we report the largest regret among them and the share of reached states where
  they choose different actions.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.mdp import greedy, policy_evaluation, value_iteration
from scm_to_mdp.robust import robust_policy_value, robust_value_iteration
from scm_to_mdp.sets import (
    candidate_bounds,
    confounded_sets,
    inputs_of,
    interval_model,
    node_sets,
    nodes_of,
    plug_in_kernel,
)
from scm_to_mdp.structure import (
    Candidate,
    best_bic,
    candidate_sets,
    contains_superset,
    credible_family,
    icp_family,
    posterior,
)

ARMS = ("oracle", "flat", "bic", "posterior", "icp", "agreement", "icp_agreement",
        "plugin_oracle", "plugin_bic", "oracle_gated")
PLUG_IN = {"plugin_oracle", "plugin_bic"}
SINGLE_GRAPH = {"oracle", "bic", "plugin_oracle", "plugin_bic", "oracle_gated"}


@dataclass
class Proposal:
    """What S2/S3 hand to S4 for one arm: a family per node, and how S4 should treat it."""

    families: dict[str, list[Candidate]] | None   # None: the agreement rule enumerates its own candidates
    refute: bool
    level_structure: float = 0.0                  # delta_G spent by S2/S3 (ICP), added to S4's delta


@dataclass
class Context:
    """Everything a run shares across arms."""

    env: SysAdminRing
    disc_cols: Mapping[str, np.ndarray]
    disc_regime: np.ndarray
    est_cols: Mapping[str, np.ndarray]
    alpha: float = 0.1
    delta: float = 0.1
    d: int = 3
    n_hyp: int = 32
    tau: float = 0.05
    extra: tuple[str, ...] = ()
    gate_rejected: bool = False
    declared_unconfounded: tuple[int, ...] = ()


def propose(arm: str, ctx: Context) -> Proposal:
    """S2/S3 for one arm on the discovery half."""
    env, cols = ctx.env, ctx.disc_cols
    inputs = inputs_of(env, ctx.extra)
    cands = candidate_sets(inputs, ctx.d)
    nodes = nodes_of(env)
    a_t = ctx.alpha / env.n
    if arm in ("oracle", "plugin_oracle", "oracle_gated"):
        return Proposal({n: [env.true_parents()[n]] for n in nodes}, refute=False)
    if arm == "flat":
        return Proposal({n: [tuple(inputs)] for n in nodes}, refute=False)
    if arm in ("bic", "plugin_bic"):
        return Proposal({n: [best_bic(cols, n, cands)] for n in nodes}, refute=False)
    if arm == "posterior":
        return Proposal({n: credible_family(posterior(cols, n, cands), 1 - a_t) for n in nodes}, refute=False)
    if arm == "icp":
        return Proposal({n: icp_family(cols, n, cands, ctx.disc_regime, a_t) for n in nodes}, refute=False,
                        level_structure=ctx.alpha)
    if arm == "agreement":
        return Proposal(None, refute=True)
    if arm == "icp_agreement":
        return Proposal({n: icp_family(cols, n, cands, ctx.disc_regime, a_t) for n in nodes}, refute=True,
                        level_structure=ctx.alpha)
    raise ValueError(f"unknown arm {arm!r}")


@dataclass
class Truth:
    """The evaluator's exact quantities (never shown to an arm)."""

    P: np.ndarray
    R: np.ndarray
    gamma: float
    s0: int
    v_star: float

    @staticmethod
    def of(env: SysAdminRing) -> Truth:
        P, R = env.kernel(), env.reward
        V, _ = value_iteration(P, R, env.gamma)
        s0 = int(env.index(env.initial_state))
        return Truth(P, R, env.gamma, s0, float(V[s0]))

    def value(self, policy: np.ndarray) -> float:
        return float(policy_evaluation(self.P, self.R, self.gamma, policy)[self.s0])


def reached_states(env: SysAdminRing, cols: Mapping[str, np.ndarray]) -> np.ndarray:
    """``[S]`` bool: states that occur in the data (as a current or a next state)."""
    now = np.stack([cols[v] for v in env.state_vars], -1)
    nxt = np.stack([cols[f"{v}'"] for v in env.state_vars], -1)
    seen = np.zeros(env.n_states, dtype=bool)
    seen[env.index(np.concatenate([now, nxt]))] = True
    return seen


def collapse(env: SysAdminRing, cols: Mapping[str, np.ndarray], families: Mapping[str, list[Candidate]],
             delta: float, tau: float, reached: np.ndarray, extra: Sequence[str] = ()) -> dict[str, int]:
    """S5 merge: per node, the number of candidate classes whose sets agree within ``tau`` on reached rows."""
    out = {}
    for node, fam in families.items():
        reps: list[np.ndarray] = []
        for cand in fam:
            b = candidate_bounds(env, cols, node, cand, delta, extra)[reached]       # [S_r, A, 2, 2]
            if not any(np.max(np.abs(b - r)) <= tau for r in reps):
                reps.append(b)
        out[node] = len(reps)
    return out


def disagreement(env: SysAdminRing, cols: Mapping[str, np.ndarray], families: Mapping[str, list[Candidate]],
                 truth: Truth, n_hyp: int, rng: np.random.Generator, reached: np.ndarray | None = None,
                 extra: Sequence[str] = ()) -> dict[str, float]:
    """S5 decision: plug-in policies of up to ``n_hyp`` hypotheses drawn from the per-node families."""
    nodes = list(families)
    sizes = [len(families[n]) for n in nodes]
    n_total = int(np.prod(sizes)) if sizes else 0
    if n_total <= n_hyp:
        picks = list(itertools.product(*[range(k) for k in sizes]))
    else:
        picks = [tuple(int(rng.integers(k)) for k in sizes) for _ in range(n_hyp)]
    policies, regrets = [], []
    for pick in picks:
        parents = {n: families[n][i] for n, i in zip(nodes, pick, strict=True)}
        pi = greedy(value_iteration(plug_in_kernel(env, cols, parents, extra), truth.R, truth.gamma)[1])
        policies.append(pi)
        regrets.append(truth.v_star - truth.value(pi))
    acts = np.stack(policies)
    differ = acts.min(0) != acts.max(0)
    mask = np.ones(env.n_states, dtype=bool) if reached is None else reached
    return {"n_hypotheses": float(n_total), "s5_sampled": float(len(picks)),
            "s5_max_regret": float(max(regrets)), "s5_disagree_share": float(np.mean(differ[mask]))}


def shd(parents: Mapping[str, Candidate], true: Mapping[str, Candidate]) -> int:
    """Structural Hamming distance of 2-slice parent sets (edges only from slice t to t + 1)."""
    return int(sum(len(set(parents[n]) ^ set(true[n])) for n in true))


def joint_visited(env: SysAdminRing, cols: Mapping[str, np.ndarray]) -> np.ndarray:
    """``[S, A]`` bool: (s, a) rows that occur jointly in the data."""
    s = env.index(np.stack([cols[v] for v in env.state_vars], -1))
    flags = np.stack([cols[r] for r in env.action_flags], -1)
    a = np.where(flags.any(-1), flags.argmax(-1) + 1, 0)
    seen = np.zeros((env.n_states, env.n_actions), dtype=bool)
    seen[s, a] = True
    return seen


def run_arm(arm: str, ctx: Context, truth: Truth, rng: np.random.Generator) -> dict:
    """S2-S9 for one arm; returns one result row (without the run's identifiers)."""
    env = ctx.env
    prop = propose(arm, ctx)
    true_pa = env.true_parents()
    row: dict = {"arm": arm, "level": 1.0 - ctx.delta - prop.level_structure}
    if arm in PLUG_IN:
        parents = {n: fam[0] for n, fam in prop.families.items()}
        P_hat = plug_in_kernel(env, ctx.est_cols, parents, ctx.extra)
        V_hat, Q_hat = value_iteration(P_hat, truth.R, truth.gamma)
        pi = greedy(Q_hat)
        promise = float(V_hat[truth.s0])
        row.update(valid_structure=int(all(set(true_pa[n]) <= set(parents[n]) for n in parents)),
                   empty_family=0, survivors=1.0, shd=shd(parents, true_pa))
    else:
        joint = "product"
        if arm == "oracle_gated" and ctx.gate_rejected:
            sets = confounded_sets(env, ctx.est_cols, true_pa, ctx.delta, ctx.declared_unconfounded,
                                   ctx.extra)
            joint = "frechet"
        else:
            sets = node_sets(env, ctx.est_cols, prop.families, prop.refute, ctx.delta, max_parents=ctx.d,
                             extra=ctx.extra)
        model = interval_model(env, sets, joint=joint, extra=ctx.extra)
        _, _, pi = robust_value_iteration(model.L, model.U, truth.R, truth.gamma)
        promise = float(robust_policy_value(model.L, model.U, truth.R, truth.gamma, pi)[truth.s0])
        inside = bool(np.all((model.L <= truth.P + 1e-12) & (truth.P <= model.U + 1e-12)))
        survivors = {n: ns.survivors for n, ns in sets.items()}
        jv = joint_visited(env, ctx.est_cols)
        row.update(valid_structure=int(all(contains_superset(survivors[n], true_pa[n]) for n in survivors)),
                   rows_with_set=model.rows_with_set, truth_in_boxes=int(inside),
                   rows_set_joint=int((model.narrow & jv).sum()),
                   rows_set_factor_only=int((model.narrow & ~jv).sum()),
                   rows_visited_joint=int(jv.sum()),
                   empty_family=int(any(ns.empty_family for ns in sets.values())),
                   survivors=float(np.mean([len(v) for v in survivors.values()])),
                   assumption_refuted=int(any(getattr(ns.result, "assumption_refuted", False)
                                              for ns in sets.values())),
                   confounded_route=int(any(ns.confounded for ns in sets.values())),
                   model_factor_rows=model.factor_rows, model_coupling=model.coupling)
        if arm in SINGLE_GRAPH:
            row["shd"] = shd({n: survivors[n][0] for n in survivors}, true_pa)
        if max(len(v) for v in survivors.values()) > 1:
            # a node whose candidates were all refuted uses NBN's fallback (its result's parents)
            fams = {n: survivors[n] or [sets[n].parents] for n in survivors}
            reached = reached_states(env, ctx.est_cols)
            k_prime = collapse(env, ctx.est_cols, fams, ctx.delta / env.n, ctx.tau, reached, ctx.extra)
            row.update(s5_k_prime=float(np.prod(list(k_prime.values()))),
                       s5_classes=float(np.mean(list(k_prime.values()))))
            row.update(disagreement(env, ctx.est_cols, fams, truth, ctx.n_hyp, rng, reached, ctx.extra))
    true_value = truth.value(pi)
    row.update(promise=promise, true_value=true_value, v_star=truth.v_star,
               regret=truth.v_star - true_value, sound=int(promise <= true_value + 1e-9))
    return row


def logger_value(env: SysAdminRing, logger, n_episodes: int = 1000, horizon: int = 200, seed: int = 0,
                 distractor_prob: float | None = None) -> float:
    """Monte Carlo discounted value of a logger from "all machines up".

    The truncation error is at most ``gamma^horizon * R_max / (1 - gamma)``.
    """
    rng = np.random.default_rng(seed)
    disc = env.gamma ** np.arange(horizon)
    total = 0.0
    for _ in range(n_episodes):
        s = env.initial_state.copy()
        logger.reset()
        ret = 0.0
        for t in range(horizon):
            ret += disc[t] * env.reward[env.index(s)]
            w = env.sample_load(rng)
            if distractor_prob is not None:
                logger.distractor = int(rng.random() < distractor_prob)
            a = logger.act(s, rng, w)
            s2 = env.step(s, a, rng, w)
            logger.observe(s, a, s2)
            s = s2
        total += ret
    return total / n_episodes
