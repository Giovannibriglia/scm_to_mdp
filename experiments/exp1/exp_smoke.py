"""Smoke test of experiment 1: run once after cloning, to see that everything is installed and works.

    .venv/bin/python experiments/exp1/exp_smoke.py    (2 minutes on a recent laptop, 8 on an older server)

It checks that every dependency imports, prints the machine (CPUs, RAM, Python), then runs each
complete grid (sysadmin_complete.py, sepsis_complete.py, chemistry_complete.py) on one tiny setting
with every behaviour policy and every method, with 2 worker processes (as the real runs do), and
writes the report. It ends with a PASS / FAIL line per grid.

Output: results/exp1/smoke_<date>_<time>/<grid>/ (safe to delete). Nothing runs on the GPU: the
pipeline uses NumPy and PyTorch on the CPU, so a GPU is neither needed nor used.
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
    ("sysadmin_complete", 3, "bidirectional", [15, 30]),
    ("sepsis_complete", 2, "hidden", [50, 100]),
    ("chemistry_complete", 4, "0.4", [15, 30]),
]
WORKERS = 2


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

    ram = psutil.virtual_memory()
    print(f"  Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"  {os.cpu_count()} CPUs, {ram.total / 1e9:.0f} GB RAM ({ram.available / 1e9:.0f} GB free)")
    print("  The full grids use workers=3 and up to 8 GB per step (LIMITS in each script): about 24 GB.")


def run_one(script: str, size, knob, n_rollouts, out_dir: Path) -> tuple[bool, str]:
    import pandas as pd

    m = importlib.import_module(script)
    grid = dataclasses.replace(
        m.GRID, sizes=[size], knobs=[knob], n_rollouts=n_rollouts, seeds=[0], workers=WORKERS, out_dir=out_dir
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
    for f in ("tables.md", "README.md", "figures/sample_efficiency_grid.png", "figures/promise_heatmap.png"):
        if not (out_dir / "summary" / f).exists():
            problems.append(f"summary/{f} missing")
    if not list(out_dir.glob("*/*/models/*/n=*/*/model.npz")):
        problems.append("no saved model")
    expected = len(runs["policy"].unique()) * len(n_rollouts) * runs["method"].nunique()
    n_methods, n_policies = runs["method"].nunique(), runs["policy"].nunique()
    note = f"{len(runs)} runs ({expected} expected), {n_methods} methods, {n_policies} policies"
    return not problems and len(runs) == expected, note if not problems else note + " | " + " | ".join(
        problems
    )


def main():
    from datetime import datetime

    print("1. Dependencies")
    if not check_imports():
        sys.exit("FAIL: install the missing packages first")
    print("2. Machine")
    machine()
    out = ROOT / "results" / "exp1" / f"smoke_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    print(f"3. Tiny grids ({WORKERS} workers) -> {out}")
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
