"""E1 stage 3: the whole pipeline (S1-S9) for every arm on the SysAdmin ring.

For every (logger suite, total episodes, edge strength, seed): simulate the suite, split it by episode
(discovery half for S2/S3, estimation half for S4), and run each arm of ``scm_to_mdp.pipeline.ARMS``:
S2/S3 propose parent sets, S4/S7 build per-node sets with NBN and export the factored model, S8 runs
robust value iteration on the joint boxes, S9 evaluates the returned policy on the true MDP.

Columns: the run's identifiers, ``arm``, ``promise`` (robust value of the returned policy at "all
machines up"), ``true_value``, ``v_star``, ``regret``, ``sound`` (promise <= true value),
``truth_in_boxes``, ``rows_with_set`` (of 80 feasible rows), ``valid_structure`` (every node's surviving
family contains a superset of its true parents), ``survivors`` (mean per node), ``level`` (the
promised joint level 1 - delta - delta_G), and the S5 columns for arms with several candidates.

    uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scm_to_mdp.assumptions import episode_split
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.pipeline import ARMS, Truth, run_arm

sys.path.insert(0, str(Path(__file__).parent))
from structure_arms import EDGES, SUITES  # noqa: E402  (same suites and edges as stage 2)

COLUMNS = ["suite", "episodes", "edge", "seed", "arm", "promise", "true_value", "v_star", "regret",
           "sound", "truth_in_boxes", "rows_with_set", "valid_structure", "survivors", "empty_family",
           "assumption_refuted", "level", "n_hypotheses", "s5_sampled", "s5_max_regret",
           "s5_disagree_share", "alpha", "delta", "d"]


def run_one(suite: str, episodes: int, edge: str, seed: int, arms, alpha: float, delta: float, d: int,
            n_hyp: int) -> list[dict]:
    env = SysAdminRing(n=4, p_bad=EDGES[edge])
    specs = SUITES[suite]
    loggers = [make_logger(f, env, **kw) for f, kw in specs]
    data = collect(env, loggers, [episodes // len(specs)] * len(specs), horizon=30, seed=seed)
    disc, est = episode_split(data, 0.5, seed=seed)
    disc_cols, est_cols = disc.columns(env), est.columns(env)
    truth = Truth.of(env)
    rows = []
    for arm in arms:
        rng = np.random.default_rng([seed, ARMS.index(arm)])
        row = run_arm(arm, env, disc_cols, disc.regime, est_cols, truth, alpha, delta, d, n_hyp, rng)
        rows.append({"suite": suite, "episodes": episodes, "edge": edge, "seed": seed, **row,
                     "alpha": alpha, "delta": delta, "d": d})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, nargs="+", default=[40, 100, 400], help="total per suite")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1, help="ICP family-wise level over the targets")
    ap.add_argument("--delta", type=float, default=0.1, help="S4 budget, split over the targets")
    ap.add_argument("--d", type=int, default=3, help="declared in-degree bound")
    ap.add_argument("--n-hyp", type=int, default=32, help="S5: hypotheses drawn per run")
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--out", type=Path, default=Path("results/e1/pipeline_arms.csv"))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    jobs = list(itertools.product(args.suites, args.episodes, EDGES, range(args.seeds)))
    rows = []
    bar = tqdm(jobs, desc="pipeline", unit="run", disable=args.no_progress)
    for suite, episodes, edge, seed in bar:
        bar.set_postfix_str(f"{suite}, {episodes} ep, {edge}, seed {seed}")
        rows += run_one(suite, episodes, edge, seed, args.arms, args.alpha, args.delta, args.d, args.n_hyp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(jobs)} runs x {len(args.arms)} arms; wrote {args.out}")


if __name__ == "__main__":
    main()
