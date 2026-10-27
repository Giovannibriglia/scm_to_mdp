"""The experiment scripts: every script of experiments/exp1 and experiments/exp2 follows the shared
protocol (experiments/common.py, protocol.py, domains.py), and two of them run on a tiny grid, each in a
fresh process (the steps fork)."""

import importlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]  # tests/tool/ -> the repository
DOMAINS = ["sysadmin", "chemistry", "aircraft", "drone", "frozenlake", "stock", "sepsis", "coffee", "taxi"]
SCRIPTS = [(e, f"{d}_{k}") for e in ("exp1", "exp2") for d in DOMAINS for k in ("mdp", "imdp")]

GRID_CODE = """
import dataclasses, sys
sys.path.insert(0, "experiments/{experiment}")
import {module} as m
from common import TARGET
from scm_to_mdp.tool import Method, report_grid, run_grid
quick = {{"steps": 300, "hidden": 32}}  # offline DQN: only that it trains inside the grid
methods = dict(m.GRID.methods)
if "offline DQN" in methods:
    methods["offline DQN"] = Method("offline_dqn", "greedy_q", solver_options=quick)
grid = dataclasses.replace(m.GRID, sizes=[{size!r}], knobs=[{knob!r}], n_rollouts=[2, 5], seeds=[1],
                           workers=1, out_dir=__import__("pathlib").Path("{out}"), methods=methods)
run_grid(grid)
report_grid(grid.out_dir, target=TARGET)
"""


def run(code: str) -> None:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]


def load(experiment: str, module: str):
    sys.path.insert(0, str(ROOT / "experiments" / experiment))
    try:
        for name in ("protocol", "common", module):
            sys.modules.pop(name, None)
        return (
            importlib.import_module(module),
            importlib.import_module("protocol"),
            importlib.import_module("common"),
        )
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("experiment, module", SCRIPTS)
def test_every_script_follows_the_protocol(experiment, module):
    """Uniform logger only, 10 seeds, 1 ... 10k episodes of 256 steps, 20 random starts, the same MDP
    and reward (no transfer, no shift), the methods of its experiment and kind, and an oracle."""
    m, protocol, common = load(experiment, module)
    grid, kind = m.GRID, module.rsplit("_", 1)[1]
    assert [p.label for p in grid.policies] == ["uniform"]
    assert grid.seeds == list(range(10)) and grid.horizon == 256 and grid.n_starts == 20
    assert grid.n_rollouts == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    assert grid.shifts is None
    rewards = experiment == "exp1" and module.startswith("sepsis")  # many rewards: exp1, Sepsis only
    assert grid.reward_transfer == rewards and (grid.reward_family is not None) == rewards
    assert tuple(grid.methods) == protocol.METHODS[kind]
    assert f"oracle -> {kind.upper()}" in grid.methods
    assert grid.out_dir == common.ROOT / "results" / experiment / module


def test_experiment_1_compares_discovery_with_the_full_factored_graph():
    _, protocol, _ = load("exp1", "sysadmin_mdp")
    for kind, tail in (("mdp", "MDP"), ("imdp", "IMDP")):
        names = set(protocol.METHODS[kind])
        assert {f"{x} -> {tail}" for x in ("PCMCI+", "DYNOTEARS", "full-fact", "true graph")} <= names
        assert f"flat {tail}" in names


def test_experiment_2_compares_with_the_state_of_the_art():
    _, protocol, _ = load("exp2", "sysadmin_mdp")
    for kind in ("mdp", "imdp"):
        assert {"DiFaRL", "FOCUS", "SL-SPIBB", "offline DQN"} <= set(protocol.METHODS[kind])


@pytest.mark.parametrize(
    "experiment, module, size, knob",
    [("exp1", "sysadmin_imdp", 3, "ring"), ("exp2", "sepsis_mdp", 2, "observed")],
)
def test_scripts_on_a_tiny_grid(tmp_path, experiment, module, size, knob):
    run(GRID_CODE.format(experiment=experiment, module=module, size=size, knob=knob, out=tmp_path))
    summary = tmp_path / "summary"
    runs = pd.read_csv(summary / "runs.csv")
    assert set(runs["status"]) == {"ok"}, runs[["method", "status", "failed_step", "error"]]
    learned = runs[runs["parents"].notna() & (runs["role"] != "oracle")]
    assert learned["recall"].between(0, 1).all() and learned["precision"].dropna().between(0, 1).all()
    if experiment == "exp1":  # the full factored graph contains the truth (no same-step edges here)
        assert (runs.loc[runs["method"] == "full-fact -> IMDP", "recall"] == 1).all()
    for figure in ("sample_efficiency_grid", "promise_heatmap"):
        assert (summary / "figures" / f"{figure}.pdf").exists()
    assert (summary / "tables.md").exists()


def test_every_script_runs_by_name_from_the_launcher(tmp_path):
    """experiments/run.py lists the scripts, shows one without running it, and refuses a wrong name;
    from any working directory."""
    launcher = [sys.executable, str(ROOT / "experiments" / "run.py")]
    listed = subprocess.run([*launcher, "--list"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert listed.returncode == 0 and all(f"{d}_imdp" in listed.stdout for d in DOMAINS)
    shown = subprocess.run([*launcher, "exp1", "sepsis", "imdp", "--show"], cwd=tmp_path, capture_output=True,
                           text=True, timeout=300)  # fmt: skip
    assert shown.returncode == 0, shown.stderr[-2000:]
    assert "full-fact -> IMDP" in shown.stdout and "random rewards" in shown.stdout
    wrong = subprocess.run(
        [*launcher, "exp1", "nowhere", "mdp"], cwd=tmp_path, capture_output=True, text=True
    )
    assert wrong.returncode != 0 and "no script" in wrong.stderr
