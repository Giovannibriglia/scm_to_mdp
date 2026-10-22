"""The experiment scripts on a tiny grid, each in a fresh process (the steps fork): the three complete
grids of experiments/exp1/ and the single experiments experiments/{coffee,taxi}.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

GRID_CODE = """
import dataclasses, sys
sys.path.insert(0, "experiments/exp1")
import {module} as m
grid = dataclasses.replace(m.GRID, sizes=[{size!r}], knobs=[{knob!r}], n_rollouts=[15, 30], seeds=[1],
                           workers=1, out_dir=__import__("pathlib").Path("{out}"))
m.run_grid(grid)
m.report_grid(grid.out_dir)
"""

EXPERIMENT_CODE = """
import dataclasses, sys
sys.path.insert(0, "experiments")
import {module} as m
exp = dataclasses.replace(m.EXPERIMENT, sizes=[{size!r}], n_rollouts=[{n}], seeds=[1], workers=1,
                          out_dir=__import__("pathlib").Path("{out}"), pictures_max_states=0)
m.run_experiment(exp)
m.report(exp.out_dir)
"""


def run(code: str) -> None:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]


@pytest.mark.parametrize(
    "module, size, knob",
    [
        ("sysadmin_complete", 3, "bidirectional"),
        ("sepsis_complete", 2, "hidden"),
        ("chemistry_complete", 4, "0.4"),
    ],
)
def test_complete_grids_on_a_tiny_grid(tmp_path, module, size, knob):
    run(GRID_CODE.format(module=module, size=size, knob=knob, out=tmp_path))
    summary = tmp_path / "summary"
    runs = pd.read_csv(summary / "runs.csv")
    assert set(runs["status"]) == {"ok"}, runs[["method", "status", "failed_step", "error"]]
    assert runs["policy"].nunique() >= 2
    for file in ("edges", "behaviour", "cells", "rollouts_needed", "rollouts_needed_by_seed", "promise_held",
                 "best_method", "method_rank"):  # fmt: skip
        assert (summary / f"{file}.csv").exists(), file
    for figure in ("sample_efficiency_grid", "promise_heatmap", "rollouts_vs_model_size"):
        assert (summary / "figures" / f"{figure}.pdf").exists()
        assert (summary / "figure_data" / f"{figure}.csv").exists()
    assert (summary / "tables.md").exists() and (summary / "README.md").exists()
    edges = pd.read_csv(summary / "edges.csv")
    # every edge count in runs.csv is recomputed from edges.csv
    learned = runs[runs["parents"].notna()]
    key = ["knob", "policy", "size", "seed", "n_rollouts", "method"]
    missing = edges[(edges["in_true"] == 1) & (edges["in_learned"] == 0)].groupby(key).size()
    merged = learned.set_index(key)["missing"].to_frame().join(missing.rename("from_edges")).fillna(0)
    assert (merged["missing"] == merged["from_edges"]).all()


@pytest.mark.parametrize("module, size, n", [("coffee", "coffee", 30), ("taxi", "deterministic", 30)])
def test_experiments_on_a_tiny_grid(tmp_path, module, size, n):
    run(EXPERIMENT_CODE.format(module=module, size=size, n=n, out=tmp_path))
    res = pd.read_csv(tmp_path / "results.csv")
    # Taxi's robust solves take ~40 s on a recent laptop and ~2 min on an older server CPU: there a
    # step may stop at the script's time limit, which is the limit working, not an error
    allowed = {"ok", "timeout"} if module == "taxi" else {"ok"}
    statuses = set(res["status"])
    assert statuses <= allowed and "ok" in statuses, res[["method", "status", "failed_step"]]
    res = res[res["status"] == "ok"]
    valid = res[res["method"].isin(["Schnitzer IA", "flat RMDP"])]  # true graph or no graph
    assert (valid["promise"] <= valid["true"] + 1e-6).all()
    assert (res["true"] <= res["V_star"] + 1e-6).all()
    assert (tmp_path / "tables.md").exists()
