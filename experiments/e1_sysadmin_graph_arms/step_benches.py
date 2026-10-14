"""E1 per-step benches: vary one pipeline step, keep every other step fixed, on the same data.

For every (suite, episodes per logger, edge, seed) one dataset is simulated and split by episode, then:

- ``s1`` (the split): discovery fraction 0.3 / 0.5 / 0.7 for the S1 gate and S2/S3, the rest for S4;
  arms ``bic``, ``icp_agreement`` and ``oracle`` of the pipeline (everything else as in stage 3);
- ``s4`` (the set builder): true parents fixed; L1 balls (Weissman), KL, Hoeffding and Clopper–Pearson
  intervals, a Dirichlet posterior box (no frequentist guarantee); product boxes; robust VI;
- ``s78`` (compilation and solving): true parents and L1 sets fixed; product box + interval robust VI
  (the pipeline), Fréchet box + interval robust VI, and the exact factored inner problem (corners of
  the factor intervals);
- ``s6`` (the declared criterion): true parents, L1 sets and product boxes fixed; worst case
  (pessimism 1), Hurwicz 0.5, optimistic (0), and the nominal maximum-likelihood model.

Each row records the variant's promise (its own value at "all machines up"), the true value of its
policy, the regret, whether the promise held, and step-specific columns (coverage of the true factor
probabilities and width for s4, solve time for s78). Results: ``results/e1/bench_<step>.csv``.

    uv run python experiments/e1_sysadmin_graph_arms/step_benches.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scm_to_mdp.assumptions import confounding_gate, episode_split
from scm_to_mdp.benches import (
    BUILDERS,
    boxes_from_factors,
    criterion_vi,
    factor_bounds,
    factored_policy_value,
    factored_robust_vi,
    q_intervals,
)
from scm_to_mdp.data import collect
from scm_to_mdp.mdp import greedy, value_iteration
from scm_to_mdp.pipeline import Context, Truth, run_arm
from scm_to_mdp.robust import robust_policy_value, robust_value_iteration
from scm_to_mdp.sets import plug_in_kernel

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_arms import build_loggers, make_env, suites  # noqa: E402

BENCHES = ("s1", "s4", "s78", "s6")
SPLITS = (0.3, 0.5, 0.7)
S1_ARMS = ("bic", "icp_agreement", "oracle")


def _inside(L, U, P) -> int:
    return int(np.all(L <= P + 1e-12) and np.all(P <= U + 1e-12))


def _outcome(truth: Truth, promise: float, pi: np.ndarray) -> dict:
    true = truth.value(pi)
    return {
        "promise": promise,
        "true_value": true,
        "regret": truth.v_star - true,
        "sound": int(promise <= true + 1e-9),
    }


def run_job(job: dict) -> dict[str, list[dict]]:
    import torch

    torch.set_num_threads(1)
    env = make_env(4, job["edge"], job["suite"], 0.0)
    specs = suites(4)[job["suite"]]
    loggers = build_loggers(env, specs, job["eps"])
    data = collect(env, loggers, [job["per"]] * len(specs), horizon=30, seed=job["seed"])
    truth = Truth.of(env)
    ids = {k: job[k] for k in ("suite", "per", "edge", "eps", "seed")}
    out: dict[str, list[dict]] = {b: [] for b in job["benches"]}
    delta = job["delta"]
    parents = env.true_parents()

    if "s1" in job["benches"]:
        for frac in SPLITS:
            disc, est = episode_split(data, frac, seed=job["seed"])
            gate = confounding_gate(env, disc)
            ctx = Context(
                env,
                disc.columns(env),
                disc.regime,
                est.columns(env),
                delta=delta,
                gate_rejected=min(gate.values()) <= 0.1 / env.n,
            )
            for arm in S1_ARMS:
                row = run_arm(arm, ctx, truth, np.random.default_rng([job["seed"], S1_ARMS.index(arm)]))
                out["s1"].append(
                    {
                        **ids,
                        "bench": "s1",
                        "variant": f"{frac:.1f}",
                        "arm": arm,
                        "gate_rejected": int(ctx.gate_rejected),
                        **{
                            k: row.get(k)
                            for k in (
                                "promise",
                                "true_value",
                                "regret",
                                "sound",
                                "truth_in_boxes",
                                "valid_structure",
                                "rows_with_set",
                            )
                        },
                    }
                )

    _, est = episode_split(data, 0.5, seed=job["seed"])
    cols = est.columns(env)
    q_true = env.factor_probs()  # [S, A, N]

    if "s4" in job["benches"]:
        for b in BUILDERS:
            fb = factor_bounds(env, cols, parents, b, delta)
            qa, qb = q_intervals(env, parents, fb)
            L, U = boxes_from_factors(env, qa, qb)
            _, _, pi = robust_value_iteration(L, U, truth.R, truth.gamma)
            promise = float(robust_policy_value(L, U, truth.R, truth.gamma, pi)[truth.s0])
            seen = [(bb - aa)[bb - aa < 1.0] for aa, bb in fb.values()]
            width = float(np.mean(np.concatenate(seen))) if sum(len(s) for s in seen) else 1.0
            out["s4"].append(
                {
                    **ids,
                    "bench": "s4",
                    "variant": b,
                    **_outcome(truth, promise, pi),
                    "truth_in_boxes": _inside(L, U, truth.P),
                    "factor_coverage": int(np.all((qa <= q_true + 1e-12) & (q_true <= qb + 1e-12))),
                    "width": width,
                    "rows_with_set": int(((L.sum(-1) > 1e-12) | (U.max(-1) < 1 - 1e-12)).sum()),
                }
            )

    fb = factor_bounds(env, cols, parents, "l1", delta)
    qa, qb = q_intervals(env, parents, fb)
    L, U = boxes_from_factors(env, qa, qb)

    if "s78" in job["benches"]:
        for name in ("box (pipeline)", "Frechet box", "exact factored"):
            t0 = time.perf_counter()
            if name == "exact factored":
                _, pi = factored_robust_vi(env, qa, qb, truth.R, truth.gamma)
                promise = float(factored_policy_value(env, qa, qb, truth.R, truth.gamma, pi)[truth.s0])
            else:
                Lj, Uj = (L, U) if name.startswith("box") else boxes_from_factors(env, qa, qb, "frechet")
                _, _, pi = robust_value_iteration(Lj, Uj, truth.R, truth.gamma)
                promise = float(robust_policy_value(Lj, Uj, truth.R, truth.gamma, pi)[truth.s0])
            out["s78"].append(
                {
                    **ids,
                    "bench": "s78",
                    "variant": name,
                    **_outcome(truth, promise, pi),
                    "seconds": time.perf_counter() - t0,
                }
            )

    if "s6" in job["benches"]:
        for name, pess in (("worst case", 1.0), ("Hurwicz 0.5", 0.5), ("optimistic", 0.0)):
            V, pi = criterion_vi(L, U, truth.R, truth.gamma, pess)
            out["s6"].append(
                {**ids, "bench": "s6", "variant": name, **_outcome(truth, float(V[truth.s0]), pi)}
            )
        V, Q = value_iteration(plug_in_kernel(env, cols, parents), truth.R, truth.gamma)
        out["s6"].append(
            {
                **ids,
                "bench": "s6",
                "variant": "nominal (MLE)",
                **_outcome(truth, float(V[truth.s0]), greedy(Q)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benches", nargs="+", default=list(BENCHES), choices=BENCHES)
    ap.add_argument("--suites", nargs="+", default=["S-B two aliasing", "S-D two specialists"])
    ap.add_argument("--per-logger", type=int, nargs="+", default=[20, 50, 200, 1000])
    ap.add_argument("--edges", nargs="+", default=["strong", "weak"])
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out-dir", type=Path, default=Path("results/e1"))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    jobs = [
        {
            "suite": s,
            "per": p,
            "edge": e,
            "eps": args.eps,
            "seed": k,
            "delta": args.delta,
            "benches": args.benches,
        }
        for s, p, e, k in itertools.product(args.suites, args.per_logger, args.edges, range(args.seeds))
    ]
    rows: dict[str, list[dict]] = {b: [] for b in args.benches}
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run_job, j) for j in jobs]
        for f in tqdm(
            as_completed(futs), total=len(futs), desc="benches", unit="run", disable=args.no_progress
        ):
            for b, rs in f.result().items():
                rows[b] += rs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for b, rs in rows.items():
        rs.sort(key=lambda r: (r["suite"], r["per"], r["edge"], r["seed"], r["variant"], r.get("arm", "")))
        path = args.out_dir / f"bench_{b}.csv"
        cols = list(dict.fromkeys(k for r in rs for k in r))
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(rs)
        print(f"{b}: {len(rs)} rows -> {path}")


if __name__ == "__main__":
    main()
