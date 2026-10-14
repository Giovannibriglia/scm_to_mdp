"""E1 stage 3: the whole pipeline (S1-S9) for every arm on the SysAdmin ring.

For every (logger suite, episodes per logger, edge strength, exploration, seed): simulate the suite,
split it by episode (discovery half for the S1 gate and S2/S3, estimation half for S4), and run each
arm of ``scm_to_mdp.pipeline.ARMS``: S2/S3 propose parent sets, S4/S7 build per-node sets with NBN and
export the factored model, S8 runs robust value iteration on the joint boxes, S9 evaluates the
returned policy on the true MDP. ``oracle_gated`` follows the S1 gate.

Suites (``suites(n)``; regions are the two halves of the ring):

- ``S-A`` aliasing (offset -1) alone; ``S-B`` aliasing -1 + aliasing +1; ``S-C`` softmax over Q* +
  lazy; ``S-D`` the two region specialists;
- ``S-E`` specialist + confounded logger, with a hidden load that moves the machines
  (``--load-effect``): the S1 gate should reject (not a setting-1 result);
- ``S-X`` (ablation) a specialist that acts only when an observed distractor bit is 1 + the other
  specialist; the bit is offered to S2/S3 as a candidate parent;
- with ``--alone`` (default), every multi-logger suite is also run with each of its loggers alone, at
  the same *total* number of episodes: does adding a logger help beyond adding data?

Each logger's own value (Monte Carlo, from "all machines up") is recorded as a reference.
Runs are spread over ``--workers`` processes; a progress bar shows the time left.

    uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scm_to_mdp.assumptions import confounding_gate, episode_split
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.pipeline import ARMS, Context, Truth, logger_value, run_arm

EDGES = {"strong": 0.60, "weak": 0.85}
HORIZON = 30
DISTRACTOR_PROB = 0.5


def suites(n: int) -> dict[str, list[tuple[str, dict]]]:
    """Logger suites for a ring of ``n`` machines: name -> list of (family, kwargs)."""
    left, right = tuple(range(n // 2)), tuple(range(n // 2, n))
    return {
        "S-A aliasing": [("aliasing", {"offset": -1})],
        "S-B two aliasing": [("aliasing", {"offset": -1}), ("aliasing", {"offset": 1})],
        "S-C softmax+lazy": [("softmax_q", {"temperature": 0.5}), ("lazy", {"p_noop": 0.9})],
        "S-D two specialists": [("specialist", {"region": left}), ("specialist", {"region": right})],
        "S-E specialist+confounded": [("specialist", {"region": left}), ("confounded", {})],
        "S-X distractor": [("distractor", {"region": left}), ("specialist", {"region": right})],
    }


def expand_alone(names: list[str], table: dict) -> list[tuple[str, str, int]]:
    """``(suite, label, logger index or -1)``: every suite, plus each logger of a multi-logger suite alone."""
    out = []
    for name in names:
        out.append((name, name, -1))
        if len(table[name]) > 1:
            out += [(name, f"{name} | alone: {fam}{k + 1}", k) for k, (fam, _) in enumerate(table[name])]
    return out


def make_env(n: int, edge: str, suite: str, load_effect: float) -> SysAdminRing:
    return SysAdminRing(n=n, p_bad=EDGES[edge], load_effect=load_effect if suite.startswith("S-E") else 0.0)


def build_loggers(env, specs, eps):
    return [make_logger(f, env, **{**kw, "eps": eps}) for f, kw in specs]


def run_one(job: dict) -> list[dict]:
    """One run (all arms on the same data)."""
    import torch

    torch.set_num_threads(1)
    keys = ("n", "suite", "label", "only", "per", "edge", "eps", "seed")
    n, suite, label, only, per, edge, eps, seed = (job[k] for k in keys)
    env = make_env(n, edge, suite, job["load_effect"])
    specs = suites(n)[suite]
    total = per * len(specs)                         # the suite's total data, also for a logger alone
    if only >= 0:
        specs = [specs[only]]
    loggers = build_loggers(env, specs, eps)
    distractor = suite.startswith("S-X")
    extra = ("x",) if distractor else ()
    data = collect(env, loggers, [total // len(specs)] * len(specs), horizon=HORIZON, seed=seed,
                   distractor_prob=DISTRACTOR_PROB if distractor else None)
    disc, est = episode_split(data, 0.5, seed=seed)
    gate = confounding_gate(env, disc) if len(specs) > 1 else {}
    gate_p = min(gate.values()) if gate else np.nan
    rejected = bool(gate) and gate_p <= job["alpha_gate"] / env.n
    declared = tuple(k for k, lg in enumerate(loggers) if not lg.reads_hidden)
    ctx = Context(env, disc.columns(env, extra), disc.regime, est.columns(env, extra), alpha=job["alpha"],
                  delta=job["delta"], d=job["d"], n_hyp=job["n_hyp"], tau=job["tau"], extra=extra,
                  gate_rejected=rejected, declared_unconfounded=declared)
    truth = Truth.of(env)
    ids = {"suite": label, "base_suite": suite, "alone": int(only >= 0), "n": n, "n_loggers": len(specs),
           "per_logger": per, "episodes": total, "edge": edge, "eps": eps, "seed": seed, "d": job["d"],
           "alpha": job["alpha"], "delta": job["delta"], "tau": job["tau"], "gate_p": gate_p,
           "gate_rejected": int(rejected), "load_effect": env.load_effect,
           "logger_value_best": max(job["logger_values"]),
           "logger_value_mean": float(np.mean(job["logger_values"]))}
    rows = []
    for arm in job["arms"]:
        rng = np.random.default_rng([seed, ARMS.index(arm)])
        rows.append({**ids, **run_arm(arm, ctx, truth, rng)})
    return rows


def _logger_value_job(args) -> tuple[tuple, float]:
    key, n, edge, suite, load_effect, family, kw, eps = args
    env = make_env(n, edge, suite, load_effect)
    lg = make_logger(family, env, **{**kw, "eps": eps})
    distractor = DISTRACTOR_PROB if family == "distractor" else None
    return key, logger_value(env, lg, n_episodes=500, horizon=150, seed=0, distractor_prob=distractor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4, help="machines on the ring")
    ap.add_argument("--per-logger", type=int, nargs="+", default=[20, 50, 200, 1000],
                    help="episodes per logger (a logger alone gets the suite's total)")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.2], help="exploration of every logger")
    ap.add_argument("--edges", nargs="+", default=list(EDGES), choices=list(EDGES))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1, help="ICP family-wise level over the targets")
    ap.add_argument("--alpha-gate", type=float, default=0.1, help="S1 gate family-wise level")
    ap.add_argument("--delta", type=float, default=0.1, help="S4 budget, split over the targets")
    ap.add_argument("--d", type=int, default=3, help="declared in-degree bound")
    ap.add_argument("--n-hyp", type=int, default=32, help="S5: hypotheses drawn per run")
    ap.add_argument("--tau", type=float, default=0.05, help="S5: merge tolerance")
    ap.add_argument("--load-effect", type=float, default=0.4, help="hidden load effect in S-E")
    ap.add_argument("--suites", nargs="+", default=[s for s in suites(4) if not s.startswith("S-X")])
    ap.add_argument("--no-alone", action="store_true", help="skip the loggers-alone baselines")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", type=Path, default=Path("results/e1/pipeline_arms.csv"))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    table = suites(args.n)
    unknown = [s for s in args.suites if s not in table]
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; known: {list(table)}")
    plan = [(s, s, -1) for s in args.suites] if args.no_alone else expand_alone(args.suites, table)

    # each logger's own value, once per (suite, edge, eps, logger)
    vkeys = {}
    for suite, edge, eps in itertools.product(args.suites, args.edges, args.eps):
        for k, (fam, kw) in enumerate(table[suite]):
            vkeys[(suite, edge, eps, k)] = (args.n, edge, suite, args.load_effect, fam, kw, eps)
    values = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_logger_value_job, (key, *v)) for key, v in vkeys.items()]
        for f in tqdm(as_completed(futs), total=len(futs), desc="logger values", disable=args.no_progress):
            key, v = f.result()
            values[key] = v

    jobs = []
    for (suite, label, only), per, edge, eps, seed in itertools.product(
            plan, args.per_logger, args.edges, args.eps, range(args.seeds)):
        idx = range(len(table[suite])) if only < 0 else [only]
        jobs.append({"n": args.n, "suite": suite, "label": label, "only": only, "per": per, "edge": edge,
                     "eps": eps, "seed": seed, "alpha": args.alpha, "alpha_gate": args.alpha_gate,
                     "delta": args.delta, "d": args.d, "n_hyp": args.n_hyp, "tau": args.tau,
                     "load_effect": args.load_effect, "arms": args.arms,
                     "logger_values": [values[(suite, edge, eps, k)] for k in idx]})
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run_one, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="pipeline", unit="run",
                      disable=args.no_progress):
            rows += f.result()
    rows.sort(key=lambda r: (r["suite"], r["per_logger"], r["edge"], r["eps"], r["seed"],
                             ARMS.index(r["arm"])))
    columns = list(dict.fromkeys(k for r in rows for k in r))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(jobs)} runs x {len(args.arms)} arms on {args.workers} workers; wrote {args.out}")


if __name__ == "__main__":
    main()
