"""Join two runs of the same grids that ran different methods, e.g. experiment 1 and its sparse-graph arms
added later (experiments/exp1_sparse), into new run folders with every method, and rebuild their tables
and figures. The two runs are left untouched.

    .venv/bin/python experiments/merge_runs.py results/exp1 results/exp1_sparse --out results/exp1_merged
    .venv/bin/python experiments/merge_runs.py results/exp1 results/exp1_sparse --out ... sepsis_imdp

Each <domain>_<kind> of the first experiment folder is paired with the same script of the second (the
latest run of each). A pair is joined only if it is the same experiment on other methods:
    same grid       grid.json identical (sizes, knobs, data budgets, seeds, horizon, settings)
    same protocol   meta.json's configuration identical apart from the methods (limits, evaluation)
    same data       a method both runs share (the oracle) has the same reference values at every
                    (setting, size, seed, budget): V*, V*_obs, the random policy's value, transitions
Then, per setting: the first run's rows, plus the second run's rows of the methods the first does not
have (shared methods keep the first run's rows), for results.csv and results_transfer.csv; the second
run's saved models of those methods; meta.json with every method and where each came from.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROWS = ("results.csv", "results_transfer.csv")  # the per-setting tables of a sweep (tool/sweep.py)
SAME_DATA = (
    "n_states",
    "transitions",
    "episodes",
    "V_star",
    "V_star_obs",
    "V_star_K",
    "V_random",
    "V_random_K",
)
KEYS = ("size", "seed", "n_transitions")


RUN_NAME = re.compile(r"^(?P<script>.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")  # <script>_<date>_<time>


def latest_runs(folder: Path) -> dict[str, Path]:
    """{<domain>_<kind>: its latest run folder} in an experiment folder (<domain>_<kind>_<date>_<time>; the
    time may start with 20, so the name is matched as a whole, not split at "_20")."""
    runs = {}
    for path in sorted(folder.glob("*_*_20*")):
        match = RUN_NAME.match(path.name)
        if match and (path / "grid.json").exists():
            runs[match["script"]] = path  # sorted by date: the latest wins
    return runs


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


def check_same_experiment(a: Path, b: Path) -> list[str]:
    """The reasons a and b cannot be joined (empty: they can)."""
    problems = []
    grid_a, grid_b = (json.loads((r / "grid.json").read_text()) for r in (a, b))
    if grid_a != grid_b:
        problems.append(f"grid.json differs in {sorted(k for k in grid_a if grid_a.get(k) != grid_b.get(k))}")
        return problems
    for setting in grid_a["settings"]:
        meta_a, meta_b = (
            json.loads((r / setting["folder"] / "meta.json").read_text())["config"] for r in (a, b)
        )
        differ = sorted(k for k in meta_a if k not in ("methods", "name") and meta_a[k] != meta_b.get(k))
        if differ:
            problems.append(f"{setting['folder']}: the configuration differs in {differ}")
        rows_a, rows_b = (_read(r / setting["folder"] / "results.csv") for r in (a, b))
        shared = sorted(set(rows_a["method"]) & set(rows_b["method"]))
        if not shared:
            problems.append(f"{setting['folder']}: no method in common to check that the data are the same")
            continue
        cols = [c for c in SAME_DATA if c in rows_a and c in rows_b]
        for method in shared:
            x = rows_a[rows_a["method"] == method].set_index(list(KEYS))[cols]
            y = rows_b[rows_b["method"] == method].set_index(list(KEYS))[cols]
            common = x.index.intersection(y.index)
            close = (x.loc[common] - y.loc[common]).abs().le(1e-9 + 1e-9 * x.loc[common].abs())
            if not close.fillna(x.loc[common].isna() & y.loc[common].isna()).all().all():
                problems.append(f"{setting['folder']}: {method} has other reference values (other data?)")
    return problems


def merge(a: Path, b: Path, out: Path) -> None:
    """Write a's run with b's extra methods to out (a copy of a, plus b's rows and models)."""
    shutil.copytree(a, out, ignore=shutil.ignore_patterns("summary", "figures", "figure_data"))
    grid = json.loads((a / "grid.json").read_text())
    for setting in grid["settings"]:
        folder = setting["folder"]
        own = set(_read(a / folder / "results.csv")["method"])
        added = sorted(set(_read(b / folder / "results.csv")["method"]) - own)
        for name in ROWS:
            rows_a, rows_b = _read(a / folder / name), _read(b / folder / name)
            if rows_b is None:
                continue
            rows_b = rows_b[rows_b["method"].isin(added)]
            joined = rows_b if rows_a is None else pd.concat([rows_a, rows_b], ignore_index=True)
            joined.to_csv(out / folder / name, index=False)
        for model in (b / folder / "models").glob("*/*/*"):  # models/<size>/n=<n>/<method>/
            target = out / folder / model.relative_to(b / folder)
            if model.is_dir() and not target.exists():
                shutil.copytree(model, target)
        meta_a, meta_b = (json.loads((r / folder / "meta.json").read_text()) for r in (a, b))
        meta_a["config"]["methods"] |= {m: meta_b["config"]["methods"][m] for m in added}
        meta_a["merged"] = {
            "methods from the first run": sorted(own),
            "methods from the second run": added,
            "first run": {"folder": str(a.resolve()), "git_commit": meta_a.get("git_commit")},
            "second run": {"folder": str(b.resolve()), "git_commit": meta_b.get("git_commit")},
        }
        (out / folder / "meta.json").write_text(json.dumps(meta_a, indent=2))
        print(f"  {folder}: + {', '.join(added)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("first", type=Path, help="experiment folder whose rows win for shared methods")
    parser.add_argument("second", type=Path, help="experiment folder with the extra methods")
    parser.add_argument("--out", type=Path, required=True, help="new experiment folder for the joined runs")
    parser.add_argument("only", nargs="*", help="<domain>_<kind> to join (default: every one in both)")
    parser.add_argument("--no-report", action="store_true", help="skip rebuilding tables and figures")
    args = parser.parse_args(argv)
    first, second = latest_runs(args.first), latest_runs(args.second)
    names = args.only or sorted(set(first) & set(second))
    missing = [n for n in names if n not in first or n not in second]
    if missing:
        print(f"not in both experiments: {missing}", file=sys.stderr)
        return 1
    failed = []
    for name in names:
        a, b = first[name], second[name]
        out = args.out / a.name
        problems = check_same_experiment(a, b)
        if problems:
            print(f"{name}: NOT joined\n  " + "\n  ".join(problems), file=sys.stderr)
            failed.append(name)
            continue
        if out.exists():
            print(f"{name}: {out} exists, skipped", file=sys.stderr)
            continue
        print(f"{name}: {a} + {b} -> {out}")
        merge(a, b, out)
        if not args.no_report:
            from scm_to_mdp.tool.report_grid import report_grid

            report_grid(out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
