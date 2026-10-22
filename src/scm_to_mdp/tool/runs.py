"""One folder per run: <base>_<date>_<time>, so that no run overwrites another.

    out_dir = run_folder(OUT_DIR)       # in an experiment script, from the command line:

    (no option)              a new folder  <base>_2026-10-22_15-30-12
    --resume [FOLDER]        continue a stopped run: FOLDER, or the latest <base>_* folder
    --report-only [FOLDER]   only the tables and figures of FOLDER, or of the latest <base>_* folder
    --workers N|auto         parallel jobs (default: the script's, "auto" = what the machine allows)
    --device auto|cpu|cuda|cuda:N|mps   where offline DQN / CQL train (default auto: CUDA if usable)

--workers and --device set SCM_TO_MDP_WORKERS / SCM_TO_MDP_DEVICE for the run (utils/compute.py);
they never change the results' configuration, so a run may resume with other values.

A run resumes only in its own folder (results.csv and meta.json are there), so a stopped run must be
continued with --resume, not started again.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

STAMP = "%Y-%m-%d_%H-%M-%S"


def runs_of(base: Path) -> list[Path]:
    """The run folders of base, oldest first."""
    base = Path(base)
    if not base.parent.exists():
        return []
    found = []
    for path in base.parent.glob(f"{base.name}_*"):
        try:
            datetime.strptime(path.name[len(base.name) + 1 :], STAMP)
        except ValueError:
            continue
        if path.is_dir():
            found.append(path)
    return sorted(found)


def run_folder(base, argv: list[str] | None = None) -> tuple[Path, bool]:
    """(folder, report_only) from the command line (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Run (or resume, or report) an experiment.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", nargs="?", const="latest", metavar="FOLDER", help="continue a stopped run")
    group.add_argument(
        "--report-only", nargs="?", const="latest", metavar="FOLDER", help="tables and figures only"
    )
    parser.add_argument("--workers", metavar="N", help="parallel jobs: a number or 'auto'")
    parser.add_argument("--device", metavar="D", help="training device: auto, cpu, cuda, cuda:N, mps")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    if args.device is not None:
        os.environ["SCM_TO_MDP_DEVICE"] = args.device
    chosen = args.resume or args.report_only
    if chosen is None:
        return Path(f"{base}_{datetime.now().strftime(STAMP)}"), False
    if chosen == "latest":
        found = runs_of(base)
        if not found:
            raise SystemExit(
                f"no run of {base} to {'resume' if args.resume else 'report'} (no {Path(base).name}_* folder)"
            )
        folder = found[-1]
    else:
        folder = Path(chosen)
        if not folder.is_dir():
            raise SystemExit(f"{folder} is not a folder")
    print(f"{'resuming' if args.resume else 'reporting'} {folder}")
    return folder, args.report_only is not None
