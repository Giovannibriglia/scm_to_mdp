"""E1 pilot: one dataset per logger family on the SysAdmin ring, and what each dataset can tell.

Pipeline steps S0 (environment) and S1 (loggers, profiling). No learning yet: for every logger family
this prints how much of the state-action space the data cover (jointly and factor by factor), how
strongly a repair flag is tied to a candidate parent (aliasing), how far the data are from the
optimal policy's visits (single-policy concentrability), and whether the data show the interventional kernel
(confounding). Results are written to ``results/e1/pilot_loggers.csv``.

    uv run python experiments/e1_sysadmin_graph_arms/pilot_loggers.py --seeds 5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scm_to_mdp.data import rollout
from scm_to_mdp.diagnostics import observational_gap, profile
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger

# One configuration per logger family (the confounded one also needs a load that moves the dynamics).
LOGGERS = [
    ("uniform", {}),
    ("specialist", {"region": (0, 1), "eps": 0.05}),
    ("aliasing", {"offset": -1, "eps": 0.05}),
    ("softmax_q", {"temperature": 0.5}),
    ("lazy", {"p_noop": 0.9}),
    ("recent_failure", {"eps": 0.05}),
    ("confounded", {"p_high": 0.9, "p_low": 0.1}),
]
COLUMNS = ["logger", "seed", "transitions", "joint_rows", "factor_rows", "factor_only_rows",
           "min_count_visited", "aliasing_max", "aliasing_empty_cells", "concentrability_pi_star",
           "confounding_excess"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4, help="machines on the ring")
    ap.add_argument("--episodes", type=int, default=100, help="episodes per dataset")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--load-effect", type=float, default=0.3,
                    help="how much the hidden load lowers survival and repair success")
    ap.add_argument("--out", type=Path, default=Path("results/e1/pilot_loggers.csv"))
    args = ap.parse_args()

    env = SysAdminRing(n=args.n, load_effect=args.load_effect)
    rows = []
    for family, kwargs in LOGGERS:
        for seed in range(args.seeds):
            logger = make_logger(family, env, **kwargs)
            data = rollout(env, logger, args.episodes, args.horizon, np.random.default_rng(seed))
            prof = profile(env, data)
            rows.append({"logger": family, "seed": seed, **prof,
                         "confounding_excess": observational_gap(env, data)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"SysAdmin ring N={args.n}, {args.episodes} episodes x {args.horizon} steps, "
          f"{args.seeds} seeds (mean over seeds; feasible rows = {env.n_states * env.n_actions})")
    head = f"{'logger':15s} {'joint':>6s} {'factor':>7s} {'f-only':>7s} {'min n':>6s} " \
           f"{'alias':>6s} {'conc.':>8s} {'confound':>9s}"
    print(head)
    print("-" * len(head))
    for family, _ in LOGGERS:
        sub = [r for r in rows if r["logger"] == family]

        def mean(key, sub=sub):
            return float(np.mean([r[key] for r in sub]))

        conc = [r["concentrability_pi_star"] for r in sub]
        conc_txt = "inf" if any(np.isinf(conc)) else f"{np.mean(conc):8.1f}"
        print(f"{family:15s} {mean('joint_rows'):6.2f} {mean('factor_rows'):7.2f} "
              f"{mean('factor_only_rows'):7.2f} {mean('min_count_visited'):6.1f} "
              f"{mean('aliasing_max'):6.2f} {conc_txt:>8s} {mean('confounding_excess'):9.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
