"""Run one experiment script from the terminal, by name (any working directory).

    python experiments/run.py --list                       # every script, by experiment
    python experiments/run.py exp1 sepsis imdp --show      # what it would run: sizes, data, methods
    python experiments/run.py exp1 sepsis imdp             # run it (a new run folder)
    python experiments/run.py exp1 sepsis imdp --resume    # continue its latest run
    python experiments/run.py exp2 taxi mdp --report-only  # tables and figures of its latest run
    python experiments/run.py calibration sepsis           # the 1 - delta calibration of one domain

Every other flag (--resume [FOLDER], --report-only [FOLDER], --workers N, --device D) goes to the
script unchanged. Each script also runs on its own:
    .venv/bin/python experiments/exp1/sepsis_imdp.py [flags]
(use the project's Python: .venv/bin/python, or `uv run python` from the repository root).
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ("exp1", "exp2")
KINDS = ("mdp", "imdp")


def scripts(experiment: str) -> list[str]:
    return sorted(p.stem for p in (HERE / experiment).glob("*_*.py"))


def show(experiment: str, stem: str) -> None:
    """Print the grid of a script without running it."""
    sys.path.insert(0, str(HERE / experiment))
    grid = importlib.import_module(stem).GRID
    sizes = grid.sizes if isinstance(grid.sizes, dict) else {k: grid.sizes for k in grid.knobs}
    print(f"{experiment}/{stem}.py: {grid.name}")
    for knob, values in sizes.items():
        print(f"  {grid.knob_name} = {knob}: {grid.size_name} in {list(values)}")
    print(
        f"  episodes {grid.n_rollouts} of {grid.horizon} steps, {len(grid.seeds)} seeds, gamma {grid.gamma}"
    )
    logger = [p.label for p in grid.policies]
    print(f"  logger {logger}, evaluation at the start and from {grid.n_starts} random starts")
    print(f"  methods: {', '.join(grid.methods)}")
    if grid.reward_family is not None:
        f = grid.reward_family
        at = list(f.n_rollouts) if f.n_rollouts else "every number of"
        print(f"  random rewards: {f.per_family} per family {list(f.families)}, at {at} episodes")
    print(f"  alternative rewards: {'yes' if grid.reward_transfer else 'no'}")
    print(
        f"  limits: {grid.limits}; solve <= {grid.solve_factor} x the oracle (floor {grid.solve_floor_s:g} s)"
    )
    print(f"  output: {grid.out_dir}_<date>_<time>/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="this help, or a script's own (after its name)"
    )
    parser.add_argument("experiment", nargs="?", choices=EXPERIMENTS + ("calibration",))
    parser.add_argument("domain", nargs="?")
    parser.add_argument("kind", nargs="?", choices=KINDS)
    parser.add_argument("--list", action="store_true", help="list every script")
    parser.add_argument("--show", action="store_true", help="print what the script would run, then stop")
    args, rest = parser.parse_known_args()
    if args.help and args.experiment is None:
        parser.print_help()
        return
    rest += ["--help"] if args.help else []  # after a script's name: the script's own help
    if args.list or args.experiment is None:
        for experiment in EXPERIMENTS:
            print(f"{experiment}: {' '.join(scripts(experiment))}")
        print("calibration: one domain at a time (python experiments/run.py calibration <domain>)")
        return
    if args.experiment == "calibration":
        target = [str(HERE / "calibration.py"), *([args.domain] if args.domain else [])]
    else:
        if args.domain is None or args.kind is None:
            parser.error(f"{args.experiment} needs a domain and a kind (mdp | imdp); see --list")
        stem = f"{args.domain}_{args.kind}"
        if stem not in scripts(args.experiment):
            parser.error(f"no script {args.experiment}/{stem}.py; see --list")
        if args.show:
            show(args.experiment, stem)
            return
        target = [str(HERE / args.experiment / f"{stem}.py")]
    os.execv(sys.executable, [sys.executable, *target, *rest])  # the script itself, with the other flags


if __name__ == "__main__":
    main()
