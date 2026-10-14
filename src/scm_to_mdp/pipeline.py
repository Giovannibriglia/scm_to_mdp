"""One end-to-end run of the pipeline on the SysAdmin ring, for every arm (steps S1 to S9).

S1: the data are split by episode: the discovery half feeds S2/S3 (which graphs), the estimation half
feeds S4 (the sets), so no data are used twice. S2/S3: each arm proposes a family of parent sets per
node (see ``ARMS``). S4/S7: ``scm_to_mdp.sets`` builds the per-node sets with NBN and exports the
factored model; the joint rows become interval boxes. S6 is declared: worst case, dynamic nature,
discounted, Markov stationary policies, start state "all machines up". S8: robust value iteration on
the boxes (``scm_to_mdp.robust``). S9: the exact evaluator gives the true value of the returned
policy, its regret, and whether the promise held (promise <= true value).

S5 (does graph disagreement matter?): for arms with several candidates per node, up to ``n_hyp``
hypotheses (one candidate per node) are drawn; each one's plug-in model gives a greedy policy, and we
report the largest regret among them and the share of states where they choose different actions.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.mdp import greedy, policy_evaluation, value_iteration
from scm_to_mdp.robust import robust_policy_value, robust_value_iteration
from scm_to_mdp.sets import interval_model, node_sets, plug_in_kernel
from scm_to_mdp.structure import (
    Candidate,
    best_bic,
    candidate_sets,
    contains_superset,
    credible_family,
    icp_family,
    posterior,
)


@dataclass
class Proposal:
    """What S2/S3 hand to S4 for one arm: a family per node, and how S4 should treat it."""

    families: dict[str, list[Candidate]] | None   # None: the agreement rule enumerates its own candidates
    refute: bool
    level_structure: float = 0.0                  # delta_G spent by S2/S3 (ICP), added to S4's delta


def propose(arm: str, env: SysAdminRing, cols: Mapping[str, np.ndarray], regime: np.ndarray,
            alpha: float, d: int) -> Proposal:
    """S2/S3 for one arm on the discovery half."""
    inputs = env.state_vars + env.action_flags
    cands = candidate_sets(inputs, d)
    nodes = [f"{v}'" for v in env.state_vars]
    a_t = alpha / env.n
    if arm in ("oracle", "plugin_oracle"):
        return Proposal({n: [env.true_parents()[n]] for n in nodes}, refute=False)
    if arm == "flat":
        return Proposal({n: [tuple(inputs)] for n in nodes}, refute=False)
    if arm in ("bic", "plugin_bic"):
        return Proposal({n: [best_bic(cols, n, cands)] for n in nodes}, refute=False)
    if arm == "posterior":
        return Proposal({n: credible_family(posterior(cols, n, cands), 1 - a_t) for n in nodes}, refute=False)
    if arm == "icp":
        return Proposal({n: icp_family(cols, n, cands, regime, a_t) for n in nodes}, refute=False,
                        level_structure=alpha)
    if arm == "agreement":
        return Proposal(None, refute=True)
    if arm == "icp_agreement":
        return Proposal({n: icp_family(cols, n, cands, regime, a_t) for n in nodes}, refute=True,
                        level_structure=alpha)
    raise ValueError(f"unknown arm {arm!r}")


ARMS = ("oracle", "flat", "bic", "posterior", "icp", "agreement", "icp_agreement",
        "plugin_oracle", "plugin_bic")
PLUG_IN = {"plugin_oracle", "plugin_bic"}


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


def disagreement(env: SysAdminRing, cols: Mapping[str, np.ndarray], families: Mapping[str, list[Candidate]],
                 truth: Truth, n_hyp: int, rng: np.random.Generator) -> dict[str, float]:
    """S5: plug-in policies of up to ``n_hyp`` hypotheses drawn from the per-node families."""
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
        pi = greedy(value_iteration(plug_in_kernel(env, cols, parents), truth.R, truth.gamma)[1])
        policies.append(pi)
        regrets.append(truth.v_star - truth.value(pi))
    acts = np.stack(policies)
    return {"n_hypotheses": float(n_total), "s5_sampled": float(len(picks)),
            "s5_max_regret": float(max(regrets)),
            "s5_disagree_share": float(np.mean(acts.min(0) != acts.max(0)))}


def run_arm(arm: str, env: SysAdminRing, disc_cols, disc_regime, est_cols, truth: Truth, alpha: float,
            delta: float, d: int, n_hyp: int, rng: np.random.Generator,
            timer: Callable[[], float] | None = None) -> dict:
    """S2-S9 for one arm; returns one result row (without the run's identifiers)."""
    prop = propose(arm, env, disc_cols, disc_regime, alpha, d)
    true_pa = env.true_parents()
    row: dict = {"arm": arm, "level": 1.0 - delta - prop.level_structure}
    if arm in PLUG_IN:
        parents = {n: fam[0] for n, fam in prop.families.items()}
        P_hat = plug_in_kernel(env, est_cols, parents)
        V_hat, Q_hat = value_iteration(P_hat, truth.R, truth.gamma)
        pi = greedy(Q_hat)
        promise = float(V_hat[truth.s0])
        row.update(valid_structure=int(all(set(true_pa[n]) <= set(parents[n]) for n in parents)),
                   rows_with_set=np.nan, truth_in_boxes=np.nan, empty_family=0, survivors=1.0)
    else:
        sets = node_sets(env, est_cols, prop.families, prop.refute, delta, max_parents=d)
        model = interval_model(env, sets)
        _, _, pi = robust_value_iteration(model.L, model.U, truth.R, truth.gamma)
        promise = float(robust_policy_value(model.L, model.U, truth.R, truth.gamma, pi)[truth.s0])
        inside = bool(np.all((model.L <= truth.P + 1e-12) & (truth.P <= model.U + 1e-12)))
        survivors = {n: list(ns.result.survivors) for n, ns in sets.items()}
        row.update(valid_structure=int(all(contains_superset(survivors[n], true_pa[n]) for n in survivors)),
                   rows_with_set=model.rows_with_set, truth_in_boxes=int(inside),
                   empty_family=int(any(ns.empty_family for ns in sets.values())),
                   survivors=float(np.mean([len(v) for v in survivors.values()])),
                   assumption_refuted=int(any(ns.result.assumption_refuted for ns in sets.values())))
        if max(len(v) for v in survivors.values()) > 1:
            row.update(disagreement(env, est_cols, survivors, truth, n_hyp, rng))
    true_value = truth.value(pi)
    row.update(promise=promise, true_value=true_value, v_star=truth.v_star,
               regret=truth.v_star - true_value, sound=int(promise <= true_value + 1e-9))
    return row
