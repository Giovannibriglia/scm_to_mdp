"""The ablation experiments/clear27/ablations/ablation_rewards: the ladder of rewards as rules in N, and a
small run with its figures."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "experiments" / "clear27" / "ablations" / "ablation_rewards"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"ablation_rewards_{name}", FOLDER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


rewards = load("rewards")


def test_the_ladder_grows_with_the_size():
    assert [len(rewards.names(n)) for n in range(3, 11)] == [10, 11, 19, 19, 19, 19, 19, 19]
    assert rewards.levels_of(4) == [1, 2, 3, 4] and rewards.levels_of(5) == [1, 2, 3, 4, 5, 6]
    assert list(rewards.subsets(3)) == ["m0", "m1", "half"]  # at N=3 the half is all but the last
    assert rewards.subsets(6)["half"] == [0, 1, 2] and rewards.subsets(6)["all_but_one"] == [0, 1, 2, 3, 4]
    assert rewards.thresholds(6) == [2, 4, 6] and rewards.thresholds(10) == [4, 7, 10]
    assert rewards.crashes(6) == [3, 4] and rewards.crashes(10) == [5, 7]
    assert rewards.names(6)[0] == (1, "additive") and rewards.names(6)[-1] == (6, "rare_3")


def test_the_rewards_are_what_their_names_say():
    machines = 5
    B = np.array([[(s >> (machines - 1 - i)) & 1 for i in range(machines)] for s in range(2**machines)])
    actions = np.arange(machines + 1)[:, None]  # reboot: 0 = nothing, j > 0 = a repair
    terminal = np.zeros(len(B), dtype=bool)
    visits = np.arange(len(B))[::-1].copy()  # the last state is the least visited
    ladder = {rung.name: rung for rung in rewards.ladder(B, actions, terminal, visits)}
    assert list(ladder) == [name for _, name in rewards.names(machines)]
    up = B.sum(1)
    additive = ladder["additive"].task
    assert np.allclose(additive.R[:, 0], up) and np.allclose(additive.R[:, 1], up - 0.4)  # a repair costs
    assert np.allclose(ladder["cost_2"].task.R[:, 3], up - 2.0)
    assert np.allclose(ladder["subset_m0"].task.R[:, 0], machines * B[:, 0])  # the same maximum, N
    assert np.allclose(ladder["threshold_4"].task.R[:, 0], machines * (up >= 4))
    goal = ladder["goal_all_up"].task
    assert goal.r_next.sum() == 1 and goal.r_next[-1] == 1 and goal.terminal.sum() == 1 and not goal.R.any()
    assert ladder["goal_all_down"].task.r_next[0] == 1
    crash = ladder["crash_3"].task
    assert (crash.terminal == ((machines - up) >= 3)).all() and (crash.r_next[crash.terminal] == -10).all()
    assert np.flatnonzero(ladder["rare_1"].task.r_next).tolist() == [len(B) - 1]
    assert sorted(np.flatnonzero(ladder["rare_3"].task.r_next)) == [len(B) - 3, len(B) - 2, len(B) - 1]
    assert {rung.level for rung in ladder.values()} == {1, 2, 3, 4, 5, 6}


def test_a_run_writes_one_row_per_policy_its_time_and_the_figures(tmp_path):
    """N=3, one seed, two numbers of transitions, through the scripts as a user launches them."""
    run = [sys.executable, str(FOLDER / "run.py"), "--machines", "3", "--seeds", "1", "--transitions", "100",
           "1000", "--workers", "1", "--results", str(tmp_path)]  # fmt: skip
    done = subprocess.run(run, capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "time per size" in done.stdout and "N=3: 10 rewards (levels 1-4)" in done.stdout
    (folder,) = tmp_path.glob("sysadmin_*")
    config = json.loads((folder / "config.json").read_text())
    assert config["seeds"] == [0] and len(config["rewards"]["3"]) == 10 and len(config["methods"]) == 7
    assert config["reference"] == "true graph"  # six compared methods and the true-graph reference
    runs = pd.read_csv(folder / "N=3" / "runs" / "seed=0.csv")
    assert len(runs) == 2 * 7 * 10 and set(runs["status"]) == {"ok"}  # data sizes x methods x rewards
    assert set(runs["method"]) == set(config["methods"]) and set(runs["level"]) == {1, 2, 3, 4}
    assert (runs["learn_s"] >= 0).all() and (runs["replan_s"] > 0).all()
    one = runs[(runs["method"] == "PCMCI+") & (runs["n_transitions"] == 1000)]
    assert one["learn_s"].nunique() == 1  # one model, re-planned under every reward
    assert (one[one["held"]]["promise"] <= one[one["held"]]["value"] + 1e-9).all()
    (timing,) = pd.read_csv(folder / "timing.csv").to_dict("records")
    assert timing["machines"] == 3 and timing["rewards"] == 10 and timing["seconds"] >= timing["replan_s"] > 0

    again = subprocess.run([*run, "--resume"], capture_output=True, text=True, cwd=ROOT)
    assert again.returncode == 0 and "0 of 1 (size, seed) jobs to run" in again.stdout
    plot = [
        sys.executable,
        str(FOLDER / "plot.py"),
        "--results",
        str(tmp_path),
        "--out",
        str(tmp_path / "out"),
    ]
    drawn = subprocess.run(plot, capture_output=True, text=True, cwd=ROOT)
    assert drawn.returncode == 0, drawn.stderr[-2000:]
    for name in ("value", "promise", "held", "solved"):
        assert (tmp_path / "out" / "N=3" / f"{name}_by_level.pdf").stat().st_size > 0
    for name in ("value_by_size_n=1000", "solved_by_level", "time"):
        assert (tmp_path / "out" / "summary" / f"{name}.png").stat().st_size > 0
    assert (tmp_path / "out" / "data" / "summary_time.csv").stat().st_size > 0
