"""The experiment scripts: every complete grid of experiments/exp1/ follows the shared protocol
(protocol.py), and three of them run on a tiny grid, each in a fresh process (the steps fork)."""

import importlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]  # tests/tool/ -> the repository
EXP1 = ROOT / "experiments" / "exp1"
SCRIPTS = sorted(p.stem for p in EXP1.glob("*_complete.py"))

GRID_CODE = """
import dataclasses, sys
sys.path.insert(0, "experiments/exp1")
import {module} as m
from scm_to_mdp.tool import Method
quick = {{"steps": 300, "hidden": 32}}  # offline DQN / CQL: only that they train inside the grid
methods = dict(m.GRID.methods)
for name, learner in (("offline DQN", "offline_dqn"), ("offline CQL", "offline_cql")):
    assert name in methods  # the grid runs them
    methods[name] = Method(learner, "greedy_q", solver_options=quick)
grid = dataclasses.replace(m.GRID, sizes=[{size!r}], knobs=[{knob!r}], n_rollouts=[15, 30], seeds=[1],
                           workers=1, out_dir=__import__("pathlib").Path("{out}"), methods=methods)
m.run_grid(grid)
m.report_grid(grid.out_dir)
"""


def run(code: str) -> None:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]


@pytest.mark.parametrize("module", SCRIPTS)
def test_every_grid_follows_the_protocol(module):
    """5 seeds, the uniform and both expert loggers, reward transfer, the controls and the deep
    baselines; a hidden variable means a confounded logger and a dynamics shift."""
    sys.path.insert(0, str(EXP1))
    m = importlib.import_module(module)
    protocol = importlib.import_module("protocol")
    grid = m.GRID
    assert grid.seeds == protocol.SEEDS and grid.reward_transfer
    labels = {p.label for p in grid.policies}
    assert {"uniform", "expert ε=0.05", "expert ε=0"} <= labels
    assert {"logger-Markov graph -> IMDP", "random matched graph -> IMDP", "true graph + 1 -> IMDP"} <= set(
        grid.methods
    )
    assert {"offline DQN", "offline CQL"} <= set(grid.methods)
    env = grid.make_env(grid.sizes[0], grid.knobs[-1], 0)
    if env.hidden_vars:
        assert any(p.name == "confounded" for p in grid.policies) and grid.shifts is not None
    assert env.alternative_rewards()


@pytest.mark.parametrize(
    "module, size, knob",
    [
        ("sysadmin_complete", 3, "ring+crash"),
        ("sepsis_complete", 2, "hidden"),
        ("frozenlake_complete", "4x4-1", "wind"),
    ],
)
def test_complete_grids_on_a_tiny_grid(tmp_path, module, size, knob):
    run(GRID_CODE.format(module=module, size=size, knob=knob, out=tmp_path))
    summary = tmp_path / "summary"
    runs = pd.read_csv(summary / "runs.csv")
    assert set(runs["status"]) == {"ok"}, runs[["method", "status", "failed_step", "error"]]
    assert runs["policy"].nunique() >= 2
    for file in ("edges", "behaviour", "cells", "rollouts_needed", "rollouts_needed_by_seed", "promise_held",
                 "validity", "coverage", "box_looseness", "best_method", "method_rank"):  # fmt: skip
        assert (summary / f"{file}.csv").exists(), file
    for figure in ("sample_efficiency_grid", "promise_heatmap", "rollouts_vs_model_size"):
        assert (summary / "figures" / f"{figure}.pdf").exists()
        assert (summary / "figure_data" / f"{figure}.csv").exists()
    assert (summary / "tables.md").exists() and (summary / "README.md").exists()
    assert list(tmp_path.glob("*/*/results_transfer.csv")) and list(tmp_path.glob("*/*/results_shift.csv"))
    ok = runs[runs["role"].isin(["reference", "baseline"]) & runs["guarantee"].astype(bool)]
    assert ok["cov_joint_1"].notna().all() and ok["contain_uniform"].notna().all()
    edges = pd.read_csv(summary / "edges.csv")
    # every edge count in runs.csv is recomputed from edges.csv
    learned = runs[runs["parents"].notna()]
    key = ["knob", "policy", "size", "seed", "n_rollouts", "method"]
    missing = edges[(edges["in_true"] == 1) & (edges["in_learned"] == 0)].groupby(key).size()
    merged = learned.set_index(key)["missing"].to_frame().join(missing.rename("from_edges")).fillna(0)
    assert (merged["missing"] == merged["from_edges"]).all()
