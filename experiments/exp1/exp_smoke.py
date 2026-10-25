"""Smoke test of experiment 1: run once after cloning, to see that everything is installed and works.

    .venv/bin/python experiments/exp1/exp_smoke.py    (about 15 minutes on a recent laptop)

It checks that every dependency imports, prints the machine and the plan the real runs would use
(workers, training device: utils/compute.py), then runs every complete grid (*_complete.py, one per
domain) on one tiny setting with every behaviour policy and every method (offline DQN and CQL included,
briefly trained), the dynamics shifts and the reward transfer, with that plan, in parallel worker
processes as the real runs do, and writes the report. It ends with a PASS / FAIL line per grid.
--workers N and --device D override the plan, as in the grid scripts.

Output: results/exp1/smoke_<date>_<time>/<grid>/ (safe to delete). Only offline DQN / CQL can use a
GPU (CUDA, when a test step runs on it); everything else runs on the CPU.
"""

import dataclasses
import importlib
import os
import platform
import sys
import time
import traceback
import warnings
from importlib import metadata
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

# one tiny setting per grid: (script, size, knob, numbers of rollouts)
TINY = [
    ("sysadmin_complete", 3, "ring+crash", [15, 30]),
    ("sepsis_complete", 2, "hidden", [50, 100]),
    ("chemistry_complete", 4, "0.4", [15, 30]),
    ("aircraft_complete", "8x6", "on", [15, 30]),
    ("drone_complete", "3-1", "gust", [15, 30]),
    ("stock_complete", "2x2", 0.8, [15, 30]),
    ("frozenlake_complete", "4x4-1", "wind", [15, 30]),
    ("coffee_complete", "coffee", "original", [15, 30]),
    ("taxi_complete", "deterministic", "original", [25, 50]),
]
WORKERS = "auto"  # as the grids: what the machine allows


def check_imports() -> bool:
    ok = True
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "torch",
        "nbn",
        "gadjid",
        "psutil",
        "pgmpy",
        "tigramite",
    ):
        try:
            importlib.import_module(name)
            print(f"  {name:11s} {metadata.version(name)}")
        except Exception as err:  # noqa: BLE001 - reported
            ok = False
            print(f"  {name:11s} MISSING: {err!r}  (run: uv sync --extra discovery)")
    return ok


def machine():
    import psutil

    from scm_to_mdp.utils import compute

    sys.path.insert(0, str(HERE))
    import sepsis_complete

    ram = psutil.virtual_memory()
    print(f"  Python {platform.python_version()}, {ram.available / 1e9:.0f} GB RAM free")
    mem_gb = sepsis_complete.LIMITS.mem_gb
    p = compute.plan(mem_gb, workers=os.environ.get(compute.ENV_WORKERS) or WORKERS)
    print("  " + p.describe().replace("\n", "\n  "))
    print(f"  (the full grids: up to {mem_gb:g} GB per step, so {p.workers} x {mem_gb:g} GB at most)")


def run_one(script: str, size, knob, n_rollouts, out_dir: Path) -> tuple[bool, str]:
    import pandas as pd

    from scm_to_mdp.tool import Method

    m = importlib.import_module(script)
    quick = {"steps": 300, "hidden": 32}  # offline DQN / CQL: only that they train, not how well
    assert {"offline DQN", "offline CQL"} <= set(m.GRID.methods), "the grid should run offline DQN and CQL"
    methods = dict(m.GRID.methods) | {  # the same learners, trained briefly
        "offline DQN": Method("offline_dqn", "greedy_q", solver_options=quick),
        "offline CQL": Method("offline_cql", "greedy_q", solver_options=quick),
    }
    grid = dataclasses.replace(
        m.GRID,
        sizes=[size],
        knobs=[knob],
        n_rollouts=n_rollouts,
        seeds=[0],
        workers=WORKERS,
        out_dir=out_dir,
        methods=methods,
    )
    m.run_grid(grid)
    m.report_grid(grid.out_dir, delta=m.DELTA)
    runs = pd.read_csv(out_dir / "summary" / "runs.csv")
    bad = runs[runs["status"] != "ok"]
    problems = []
    if len(bad):
        problems.append(
            f"{len(bad)} runs not ok: " + "; ".join(sorted(set(bad["method"] + ": " + bad["status"])))
        )
    for f in (
        "tables.md",
        "README.md",
        "figures/sample_efficiency_grid.png",
        "figures/promise_heatmap.png",
        "validity.csv",
        "coverage.csv",
    ):
        if not (out_dir / "summary" / f).exists():
            problems.append(f"summary/{f} missing")
    if not list(out_dir.glob("*/*/models/*/n=*/*/model.npz")):
        problems.append("no saved model")
    if m.GRID.reward_transfer and not list(out_dir.glob("*/*/results_transfer.csv")):
        problems.append("no results_transfer.csv")
    if m.GRID.shifts is not None and not list(out_dir.glob("*/*/results_shift.csv")):
        problems.append("no results_shift.csv")
    expected = len(runs["policy"].unique()) * len(n_rollouts) * runs["method"].nunique()
    n_methods, n_policies = runs["method"].nunique(), runs["policy"].nunique()
    note = f"{len(runs)} runs ({expected} expected), {n_methods} methods, {n_policies} policies"
    return not problems and len(runs) == expected, note if not problems else note + " | " + " | ".join(
        problems
    )


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Smoke test of experiment 1.")
    parser.add_argument("--workers", metavar="N", help="parallel jobs: a number or 'auto'")
    parser.add_argument("--device", metavar="D", help="training device: auto, cpu, cuda, cuda:N, mps")
    args = parser.parse_args()
    if args.workers:
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    if args.device:
        os.environ["SCM_TO_MDP_DEVICE"] = args.device
    print("1. Dependencies")
    if not check_imports():
        sys.exit("FAIL: install the missing packages first")
    print("2. Machine")
    machine()
    out = ROOT / "results" / "exp1" / f"smoke_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    print(f"3. Tiny grids -> {out}")
    summary = []
    for script, size, knob, n in TINY:
        start = time.monotonic()
        print(f"\n--- {script}: size {size}, knob {knob}, rollouts {n}")
        try:
            ok, note = run_one(script, size, knob, n, out / script)
        except Exception:  # noqa: BLE001 - reported
            ok, note = False, traceback.format_exc(limit=3).strip().splitlines()[-1]
        summary.append((script, ok, time.monotonic() - start, note))
    print("\n4. Result")
    for script, ok, seconds, note in summary:
        print(f"  {'PASS' if ok else 'FAIL'}  {script:20s} {seconds:5.0f} s  {note}")
    if all(ok for _, ok, _, _ in summary):
        print("\nEverything works. Run the full grids with: .venv/bin/python experiments/exp1/<grid>.py")
    else:
        sys.exit("\nSome grids failed: see the messages above (and the results.csv of the failing setting).")


if __name__ == "__main__":
    main()
