"""The ablation experiments/clear27/ablations/ablation_rewards: the ladders of rewards of the three domains,
a small run of each with its figures, a run that extends another, and the re-planning of a learned model
(with HALT pairs; SPIBB's and FOCUS's) under another reward."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def grid_world(cards, names, terminal_at=()):
    """Every state of the variables `names` (the last one varies fastest), a World with no reward."""
    B = np.array(np.meshgrid(*[range(c) for c in cards], indexing="ij")).reshape(len(cards), -1).T
    terminal = np.zeros(len(B), dtype=bool)
    terminal[list(terminal_at)] = True
    return B, terminal


def test_the_sepsis_ladder_is_what_its_names_say():
    names = ["diabetic", "hr", "sysbp", "o2", "abx_on", "vaso_on", "vent_on"]
    B, _ = grid_world([2, 3, 3, 2, 2, 2, 2], names)
    normal = np.stack([B[:, 1] == 1, B[:, 2] == 1, B[:, 3] == 1], axis=1)
    flags = B[:, 4:]
    death = (~normal).sum(1) >= 3
    discharge = normal.all(1) & (flags == 0).all(1)
    r_next = np.where(death, -1.0, np.where(discharge, 1.0, 0.0))
    actions = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    visits = np.arange(len(B))  # the first live state is the least visited
    w = rewards.World(B, names, actions, death | discharge, r_next, visits)
    ladder = {rung.name: rung for rung in rewards.sepsis_ladder(w)}
    assert list(ladder) == [name for _, name in rewards.sepsis_names(3)]
    assert [len(rewards.sepsis_names(v)) for v in (2, 3, 4)] == [16, 17, 17]
    assert rewards.names_of("sepsis", 3) == rewards.sepsis_names(3)
    original = ladder["original"].task
    assert (
        np.allclose(original.r_next, r_next)
        and not original.R.any()
        and (original.terminal == w.terminal).all()
    )
    assert np.allclose(ladder["subset_hr"].task.r_next, normal[:, 0]) and np.allclose(
        ladder["subset_all"].task.r_next, normal.mean(1)
    )
    cost = ladder["cost_0.05"].task
    assert np.allclose(cost.R[0], -0.05 * actions.sum(1)) and np.allclose(cost.r_next, r_next)
    assert np.allclose(ladder["threshold_2"].task.r_next, normal.sum(1) >= 2)
    goal = ladder["goal_on_abx"].task  # every vital normal, antibiotics only: one state per value of diabetic
    at = np.flatnonzero(goal.r_next)
    assert len(at) == 2 and normal[at].all() and (flags[at] == [1, 0, 0]).all()
    assert (goal.terminal == (w.terminal | (goal.r_next == 1))).all()
    high = np.flatnonzero(ladder["goal_hr_high"].task.r_next)
    assert (B[high, 1] == 2).all() and (flags[high] == 0).all() and not w.terminal[high].any()
    assert (ladder["death_10"].task.r_next[death] == -10).all() and ladder["death_10"].task.r_next[
        discharge
    ].min() == 1
    live = np.flatnonzero(~w.terminal & (visits > 0))
    assert np.flatnonzero(ladder["rare_1"].task.r_next).tolist() == [live[0]]
    assert np.flatnonzero(ladder["rare_3"].task.r_next).tolist() == live[:3].tolist()
    for rung in ladder.values():  # every reward keeps the original terminal states
        assert rung.task.terminal[w.terminal].all() and rung.task.R.shape == (len(B), len(actions))


def test_the_drone_ladder_is_what_its_names_say():
    side = 5  # the target and "high" / "low" are the last / first two cells
    B, _ = grid_world([side] * 3, ["x0", "y0", "z0"])
    x, y, z = B.T
    target = (x >= 3) & (y <= 1) & (z >= 3)
    wall = (x == 2) & ((z == 0) | (y == 0) | (y >= 2))
    visits = np.arange(len(B))[::-1] + 1  # the last live state is the least visited
    w = rewards.World(
        B, ["x0", "y0", "z0"], np.arange(6)[:, None], target | wall, target.astype(float), visits
    )
    ladder = {rung.name: rung for rung in rewards.drone_ladder(w)}
    assert list(ladder) == [name for _, name in rewards.drone_names()] and len(ladder) == 15
    assert rewards.names_of("drone", 7) == rewards.drone_names()
    assert np.allclose(ladder["original"].task.r_next, target)
    slab = ladder["subset_x"].task
    assert (slab.r_next == ((x >= 3) & ~wall)).all() and (slab.terminal == (w.terminal | (x >= 3))).all()
    assert (ladder["subset_xz"].task.r_next == ((x >= 3) & (z >= 3))).all()
    assert (ladder["cost_0.03"].task.R == -0.03).all() and np.allclose(
        ladder["cost_0.03"].task.r_next, target
    )
    region = ladder["threshold_yz"].task
    assert (region.r_next == ((y >= 3) & (z >= 3) & ~wall)).all() and (region.terminal == w.terminal).all()
    goal = ladder["goal_x0y0z1"].task  # another corner; the old target still ends the episode, with nothing
    assert (goal.r_next == ((x <= 1) & (y <= 1) & (z >= 3))).all() and goal.terminal[target].all()
    assert (ladder["wall_10"].task.r_next[wall] == -10).all() and (
        ladder["wall_10"].task.r_next[target] == 1
    ).all()
    live = np.flatnonzero(~w.terminal)
    assert np.flatnonzero(ladder["rare_1"].task.r_next).tolist() == [live[-1]]
    for rung in ladder.values():
        assert rung.task.terminal[w.terminal].all()


def test_a_model_with_halt_pairs_is_replanned_under_the_new_reward():
    """SLF-Rmax's point model sends "don't know" pairs to HALT with the worst value of the reward: under
    another reward the re-planned policy must be the one of a model learned with that reward."""
    run = load("run")
    from scm_to_mdp import reward_family as rf
    from scm_to_mdp.data import first_transitions
    from scm_to_mdp.tool.methods import learn, replan, solve

    s = run.setup("sysadmin")
    exp, method = s["exp"], s["methods"]["SLF-Rmax"]
    assert method.discovery_options == {"k": 3, "m": 30, "eps1": 0.2, "delta": 0.05}  # the papers'
    assert s["methods"]["Met-Rmax"].discovery_options == {
        "D": 3,
        "m": 50,
        "m_sub": 50,
        "eps": 0.2,
        "delta": 0.05,
    }
    env = exp.make_env(4, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(first_transitions(exp.collect(env, 2000), 2000))
    base = solve(learn(data, method, exp.limits, seed=0), data, exp.gamma, exp.limits, seed=0)
    halt = base.model.info["_halt"]
    assert base.ok and 0 < halt.mean() < 1  # some pairs are known, some are not
    terminal = np.asarray(data.reward.terminal, dtype=bool)
    visits = np.ones(len(data.B), dtype=int)
    ladder = {r.name: r for r in rewards.ladder(data.B, np.asarray(truth.actions), terminal, visits)}
    other = rf.with_task(data, ladder["cost_2"].task, truth.actions_of(data))
    again = replan(base, other, exp.gamma, exp.limits, seed=0)
    fresh = solve(learn(other, method, exp.limits, seed=0), other, exp.gamma, exp.limits, seed=0)
    assert again.ok and fresh.ok and np.allclose(again.solution.V, fresh.solution.V)
    assert not np.allclose(again.solution.V, base.solution.V)  # the reward did change the values


@pytest.mark.parametrize("label", ["FOCUS", "flat SPIBB", "SL-SPIBB", "DYNOTEARS SPIBB"])
def test_spibb_and_focus_are_replanned_under_the_new_reward(label):
    """A SPIBB model carries its reward, and FOCUS's reward carries its uncertainty penalty (on the
    reward's scale): re-planned under another reward, the model gives the values of one learned with it."""
    run = load("run")
    from scm_to_mdp import reward_family as rf
    from scm_to_mdp.data import first_transitions
    from scm_to_mdp.tool.methods import learn, replan, solve

    s = run.setup("sysadmin", (label,))
    exp, method = s["exp"], s["methods"][label]
    assert list(s["methods"]) == [label]
    env = exp.make_env(4, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(first_transitions(exp.collect(env, 2000), 2000))
    base = solve(learn(data, method, exp.limits, seed=0), data, exp.gamma, exp.limits, seed=0)
    assert base.ok
    terminal = np.asarray(data.reward.terminal, dtype=bool)
    visits = np.ones(len(data.B), dtype=int)
    ladder = {r.name: r for r in rewards.ladder(data.B, np.asarray(truth.actions), terminal, visits)}
    for name in ("cost_2", "subset_m0", "threshold_2"):  # another scale of the reward, too (FOCUS's penalty)
        other = rf.with_task(data, ladder[name].task, truth.actions_of(data))
        again = replan(base, other, exp.gamma, exp.limits, seed=0)
        fresh = solve(learn(other, method, exp.limits, seed=0), other, exp.gamma, exp.limits, seed=0)
        assert again.ok and fresh.ok and np.allclose(again.solution.V, fresh.solution.V), name
        assert np.array_equal(again.solution.pi, fresh.solution.pi), name
        assert not np.allclose(again.solution.V, base.solution.V)  # the reward did change the values


def test_every_method_has_a_panel_and_a_row_shows_the_methods_of_the_run():
    run, plot = load("run"), load("plot")
    drawn = [m for methods in plot.PANELS.values() for m in methods]
    assert sorted(drawn) == sorted(run.METHODS) and len(drawn) == 20 and set(drawn) <= set(plot.STYLE)
    assert set(run.CERTIFIED) == set(next(iter(plot.PANELS.values())))  # the row of the promise
    new = pd.DataFrame({"method": ["sparse", "FOCUS", "flat SPIBB"]})
    assert list(plot.panels_of("value", new).values()) == [["sparse"], ["flat SPIBB", "FOCUS"]]
    assert list(plot.panels_of("promise", new).values()) == [["sparse"]]
    assert plot.panels_of("held", pd.DataFrame({"method": ["FOCUS", "flat MDP"]})) == {}  # no promise
    assert len(plot.panels_of("value", pd.DataFrame({"method": list(run.METHODS)}))) == 3


def test_a_run_of_more_methods_extends_the_latest_run_and_is_drawn_with_it(tmp_path):
    """--methods on a few labels, then other labels with --extends: the second run records the first,
    a resume keeps its methods, and plot.py draws the methods of both."""
    common = [sys.executable, str(FOLDER / "run.py"), "--machines", "3", "--seeds", "1", "--transitions",
              "100", "--workers", "1", "--results", str(tmp_path)]  # fmt: skip
    first = subprocess.run([*common, "--methods", "true graph", "flat MDP"], capture_output=True, text=True,
                           cwd=ROOT)  # fmt: skip
    assert first.returncode == 0, first.stderr[-2000:]
    assert "2 methods (1 with a promise)" in first.stdout and "(extends" not in first.stdout
    (base,) = (tmp_path / "sysadmin").glob("sysadmin_*")
    assert json.loads((base / "config.json").read_text())["extends"] is None
    more = [*common, "--methods", "FOCUS", "sparse", "flat SPIBB"]
    refused = subprocess.run([*more, "--resume"], capture_output=True, text=True, cwd=ROOT)
    assert refused.returncode == 2 and "has the methods true graph, flat MDP" in refused.stderr
    refused = subprocess.run([*common, "--resume", "--extends"], capture_output=True, text=True, cwd=ROOT)
    assert refused.returncode == 2 and "extends no run" in refused.stderr
    second = subprocess.run([*more, "--extends"], capture_output=True, text=True, cwd=ROOT)
    assert second.returncode == 0, second.stderr[-2000:]
    assert f"(extends {base.name})" in second.stdout and "3 methods (1 with a promise)" in second.stdout
    base, extension = sorted((tmp_path / "sysadmin").glob("sysadmin_*"))
    config = json.loads((extension / "config.json").read_text())
    assert config["extends"] == base.name and config["certified"] == ["sparse"]
    assert list(config["methods"]) == ["sparse", "flat SPIBB", "FOCUS"]  # in the order of METHODS
    runs = pd.read_csv(extension / "N=3" / "runs" / "seed=0.csv")
    assert len(runs) == 3 * 10 and set(runs["status"]) == {"ok"} and runs["value"].notna().all()
    assert set(runs["method"]) == set(config["methods"])
    assert set(pd.read_csv(base / "N=3" / "runs" / "seed=0.csv")["method"]) == {"true graph", "flat MDP"}
    # a resume without --methods (or with the same) keeps the run's methods and the run it extends
    again = subprocess.run([*common, "--resume"], capture_output=True, text=True, cwd=ROOT)
    assert again.returncode == 0 and "0 of 1 (size, seed) jobs to run" in again.stdout
    assert f"3 methods (1 with a promise) along the ladder of rewards -> {extension} (extends" in again.stdout
    assert json.loads((extension / "config.json").read_text())["methods"] == config["methods"]

    plot = [sys.executable, str(FOLDER / "plot.py"), "--results", str(tmp_path), "--out",
            str(tmp_path / "out")]  # fmt: skip
    drawn = subprocess.run(plot, capture_output=True, text=True, cwd=ROOT)
    assert drawn.returncode == 0, drawn.stderr[-2000:]
    assert f"{extension} with {base.name}" in drawn.stdout
    assert "50 policies" in drawn.stdout and "100.0% finished" in drawn.stdout  # 5 methods x 10 rewards
    out = tmp_path / "out" / "sysadmin"
    both = {"true graph", "flat MDP", "sparse", "flat SPIBB", "FOCUS"}
    assert set(pd.read_csv(out / "data" / "N=3_value_by_level.csv")["method"]) == both
    assert set(pd.read_csv(out / "data" / "summary_value_by_size_n=100.csv")["method"]) == both
    assert (out / "N=3" / "value_by_level.png").stat().st_size > 0
    assert (out / "N=3" / "promise_by_level.png").stat().st_size > 0
    times = pd.read_csv(out / "data" / "summary_time.csv")
    assert both <= set(times["method"].dropna())
    (total,) = times[times["step"] == "job totals"].to_dict("records")  # the jobs of the two runs
    seconds = sum(
        json.loads((r / "N=3" / "runs" / "seed=0.json").read_text())["seconds"] for r in (base, extension)
    )
    assert total["seconds"] == pytest.approx(seconds, abs=0.02)
    # the base alone (--run): its two rows of panels, nothing of the extension
    alone = subprocess.run([*plot, "--run", str(base), "--out", str(tmp_path / "alone")],
                           capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    assert alone.returncode == 0, alone.stderr[-2000:]
    data = tmp_path / "alone" / "sysadmin" / "data"
    assert set(pd.read_csv(data / "N=3_value_by_level.csv")["method"]) == {"true graph", "flat MDP"}


@pytest.mark.parametrize(("domain", "size", "folder", "n_rewards"), [("sepsis", 2, "vitals=2", 16),
                                                                    ("drone", 3, "side=3", 15)])  # fmt: skip
def test_sepsis_and_drone_run_and_draw(tmp_path, domain, size, folder, n_rewards):
    """The smallest size of the domain, one seed and one data size, through the scripts."""
    run = [sys.executable, str(FOLDER / "run.py"), "--domain", domain, "--sizes", str(size), "--seeds", "1",
           "--transitions", "1000", "--workers", "1", "--results", str(tmp_path)]  # fmt: skip
    done = subprocess.run(run, capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    (result,) = (tmp_path / domain).glob(f"{domain}_*")
    runs = pd.read_csv(result / folder / "runs" / "seed=0.csv")
    assert len(runs) == 20 * n_rewards and set(runs["status"]) == {"ok"} and set(runs["size"]) == {size}
    assert (
        runs["value"].notna().all() and (runs["V_star_obs"] > runs["V_random"]).all()
    )  # every reward is a task
    plot = [sys.executable, str(FOLDER / "plot.py"), "--domain", domain, "--results", str(tmp_path),
            "--out", str(tmp_path / "out")]  # fmt: skip
    drawn = subprocess.run(plot, capture_output=True, text=True, cwd=ROOT)
    assert drawn.returncode == 0, drawn.stderr[-2000:]
    assert (tmp_path / "out" / domain / folder / "value_by_level.pdf").stat().st_size > 0
    assert (tmp_path / "out" / domain / "summary" / "time.png").stat().st_size > 0


def test_a_run_writes_one_row_per_policy_its_time_and_the_figures(tmp_path):
    """N=3, one seed, two numbers of transitions, through the scripts as a user launches them."""
    run = [sys.executable, str(FOLDER / "run.py"), "--machines", "3", "--seeds", "1", "--transitions", "100",
           "1000", "--workers", "1", "--results", str(tmp_path)]  # fmt: skip
    done = subprocess.run(run, capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "time per size" in done.stdout and "N=3: 10 rewards (levels 1-4)" in done.stdout
    (folder,) = (tmp_path / "sysadmin").glob("sysadmin_*")  # a folder per domain
    config = json.loads((folder / "config.json").read_text())
    assert config["seeds"] == [0] and len(config["rewards"]["3"]) == 10 and len(config["methods"]) == 20
    assert config["reference"] == "true graph" and config["domain"] == "sysadmin" and config["sizes"] == [3]
    assert config["certified"] == ["PCMCI+", "DYNOTEARS", "full graph", "flat", "true graph", "sparse"]
    assert config["extends"] is None
    assert config["kwik_paper"]["SLF-Rmax"]["m"] == 30 and config["kwik_paper"]["Met-Rmax"]["m_sub"] == 50
    runs = pd.read_csv(folder / "N=3" / "runs" / "seed=0.csv")
    assert len(runs) == 2 * 20 * 10 and set(runs["status"]) == {"ok"}  # data sizes x methods x rewards
    assert set(runs["method"]) == set(config["methods"]) and set(runs["level"]) == {1, 2, 3, 4}
    assert (runs["learn_s"] >= 0).all() and (runs["replan_s"] > 0).all()
    one = runs[(runs["method"] == "PCMCI+") & (runs["n_transitions"] == 1000)]
    assert one["learn_s"].nunique() == 1  # one model, re-planned under every reward
    steps = ["discovery_s", "selection_s", "learning_s", "relax_s", "solve_s"]  # the time of every step
    assert (one[steps] > 0).all().all() and one["discovery_s"].nunique() == 1 and one["solve_s"].nunique() > 1
    assert (one["discovery_s"] + one["selection_s"] + one["learning_s"] - one["learn_s"]).abs().max() < 1e-9
    assert (one["relax_s"] + one["solve_s"] <= one["replan_s"]).all()
    flat = runs[runs["method"] == "flat"]  # no graph to discover and no box to compile
    assert flat["discovery_s"].isna().all() and flat["relax_s"].isna().all() and (flat["solve_s"] > 0).all()
    assert (one[one["held"]]["promise"] <= one[one["held"]]["value"] + 1e-9).all()
    (timing,) = pd.read_csv(folder / "timing.csv").to_dict("records")
    assert timing["size"] == 3 and timing["rewards"] == 10 and timing["seconds"] >= timing["replan_s"] > 0
    assert all(timing[f"{step}_s"] > 0 for step in ("discovery", "selection", "learning", "relax", "solve"))

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
    out = (
        tmp_path / "out" / "sysadmin"
    )  # what an earlier run left: a size and a data size this run does not have
    stale = [
        out / "N=9" / "value_by_level.pdf",
        out / "summary" / "value_by_size_n=7.png",
        out / "data" / "N=9_value_by_level.csv",
        out / "data" / "summary_value_by_size_n=7.csv",
    ]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    (out / "notes.txt").write_text("not a figure: stays")
    drawn = subprocess.run(plot, capture_output=True, text=True, cwd=ROOT)
    assert drawn.returncode == 0, drawn.stderr[-2000:]
    assert "removed 4 stale file(s)" in drawn.stdout and not any(path.exists() for path in stale)
    assert not (out / "N=9").exists() and (out / "notes.txt").exists()
    for name in ("value", "promise", "held", "solved", "promised"):
        assert (out / "N=3" / f"{name}_by_level.pdf").stat().st_size > 0
    for name in (
        "value_by_size_n=100",
        "value_by_size_n=1000",
        "solved_by_level",
        "promised_by_level",
        "time",
    ):  # every data size
        assert (out / "summary" / f"{name}.png").stat().st_size > 0
    assert (out / "data" / "summary_time.csv").stat().st_size > 0
    # the pooled figures (clear27's plot.py), in their two sets: 10 instances (rewards) per data size
    assert load("plot").SOLVED == 0.9  # one threshold, clear27's
    assert "up to 10 instances per data size" in drawn.stdout
    assert "skipped paper/vs_baselines/solved_vs_time: no run at 100,000 transitions" in drawn.stdout
    for group in ("paper", "appendix"):
        for name in ("imdp_by_graph/solved_vs_transitions", "mdp_by_graph/solved_vs_transitions", "tradeoff",
                     "tradeoff_pareto", "tradeoff_data_needed", "solved_and_tradeoff", "reported_vs_true",
                     "reported_vs_true_common"):  # fmt: skip
            assert (out / group / f"{name}.png").stat().st_size > 0
    # reported_vs_true: one dot per run of every reward, a panel per method the ablation has (none model-free)
    scatter = pd.read_csv(out / "data" / "paper_reported_vs_true.csv")
    order = ["DYNOTEARS", "DYNOTEARS MDP", "flat MDP", "full graph MDP", "FOCUS", "SL-SPIBB", "flat SPIBB",
             "SLF-Rmax", "Met-Rmax"]  # fmt: skip
    assert list(scatter["method"]) == [m for m in order if m in set(scatter["method"])] and len(scatter)
    assert not {"DiFaRL", "offline DQN", "offline CQL"} & set(scatter["method"])
    assert list(scatter["panel"]) == list(range(1, len(scatter) + 1))
    assert set(scatter["domain"]) == {"sysadmin"} and (scatter["runs"] > 0).all()
    assert scatter["reported_le_true"].between(0, 100).all()
    assert scatter.set_index("method").loc["DYNOTEARS", "reported_le_true"] == 100  # its lower bound held
    pooled = pd.read_csv(out / "data" / "paper_vs_baselines_solved_vs_transitions.csv")
    assert set(pooled["instances"]) == {10} and {"DYNOTEARS", "DYNOTEARS MDP", "FOCUS"} <= set(
        pooled["method"]
    )
    assert "CD-IMDP (ours)" in set(pooled["label"]) and not pooled["method"].str.contains("PCMCI").any()
    both = pd.read_csv(out / "data" / "appendix_vs_baselines_solved_vs_transitions.csv")
    assert {"PCMCI+-IMDP (ours)", "PCMCI+-MDP (ours)", "DYNOTEARS-IMDP (ours)"} <= set(both["label"])
    again = subprocess.run(plot, capture_output=True, text=True, cwd=ROOT)  # they are not stale files
    assert again.returncode == 0 and "stale" not in again.stdout
    assert (out / "data" / "paper_tradeoff.csv").exists() and (out / "appendix" / "tradeoff.pdf").exists()
    everywhere = subprocess.run([*plot, "--domain", "all_domains"], capture_output=True, text=True, cwd=ROOT)
    assert everywhere.returncode == 2 and "needs a finished run of drone" in everywhere.stderr


def test_the_colours_of_the_ablation_are_those_of_the_registry_of_clear27():
    plot = load("plot")
    for method, style in plot.STYLE.items():
        assert style == plot.paper.look_of(method)[:2], method
    assert plot.STYLE["DYNOTEARS"] == (plot.paper.BLUE, "D")
    assert plot.STYLE["DYNOTEARS MDP"] == (plot.paper.CORAL, "D")  # its own colour (paper.OWN_COLOUR)
    assert plot.STYLE["flat MDP"][0] == plot.paper.MAGENTA != plot.STYLE["flat"][0]
    assert plot.STYLE["full graph MDP"][0] == plot.paper.ROSE != plot.STYLE["full graph"][0]
    assert plot.STYLE["PCMCI+ MDP"] == plot.STYLE["PCMCI+"]  # the other plug-in MDPs: as their structure


def fake_ablation_run(folder: Path, stamp: str, sizes: list, transitions: list, methods: list,
                      finished: bool = True) -> Path:  # fmt: skip
    """A run folder of the ablation on SysAdmin as run.py writes it: one csv and json per (size, seed);
    a job that did not finish has a .partial file only."""
    run = folder / "sysadmin" / f"sysadmin_{stamp}"
    for size in sizes:
        (run / f"N={size}" / "runs").mkdir(parents=True)
        for seed in range(2):
            rows = [
                {"size": size, "seed": seed, "n_transitions": n, "method": method, "level": level,
                 "reward": reward, "status": "ok", "failed_step": None, "value": 0.95, "promise": 0.5,
                 "held": n < 1000, "true": 9.5 + size, "promise_raw": 5.0, "V_star_obs": 10.0 + size,
                 "V_random": float(size), "learn_s": 3.0, "replan_s": 3.0, "discovery_s": 1.0,
                 "selection_s": None, "learning_s": 2.0, "relax_s": 1.0, "solve_s": 2.0}
                for n in transitions for method in methods
                for level, reward in ((1, "additive"), (3, "cost_1"))
            ]  # fmt: skip
            name = run / f"N={size}" / "runs" / f"seed={seed}"
            if finished:
                pd.DataFrame(rows).to_csv(name.with_suffix(".csv"), index=False)
                name.with_suffix(".json").write_text(json.dumps({"size": size, "seed": seed, "seconds": 6.0}))
            else:
                pd.DataFrame(rows[:1]).to_csv(f"{name}.csv.partial", index=False)
    (run / "config.json").write_text(json.dumps({
        "domain": "sysadmin", "extends": None, "commit": "abc1234",
        "kwik_paper": {"SLF-Rmax": {"m": 30, "eps1": 0.2}, "Met-Rmax": {"m": 50, "eps": 0.2}},
    }))  # fmt: skip
    return run


def test_the_runs_of_a_domain_are_drawn_together_without_the_sizes_left_out(tmp_path):
    """More sizes and more numbers of transitions come as run folders of their own: plot.py reads every
    folder of the domain together, a job that did not finish is not read, and the sizes of LEFT_OUT
    (SysAdmin N = 10 and 11) are in no figure. --run reads one folder."""
    plot = load("plot")
    assert plot.LEFT_OUT == {"sysadmin": [10, 11]}
    methods = ["true graph", "DYNOTEARS", "SLF-Rmax"]
    first = fake_ablation_run(tmp_path, "2026-11-13_15-30-09", [3], [100, 1000], methods)
    large = fake_ablation_run(tmp_path, "2026-11-14_16-37-35", [9, 11], [100, 1000], methods)
    fake_ablation_run(tmp_path, "2026-11-14_16-40-00", [10], [100, 1000], methods, finished=False)
    more_data = fake_ablation_run(tmp_path, "2026-11-14_21-42-06", [3, 9], [200_000], methods)
    folders = plot.runs_of(tmp_path / "sysadmin", "sysadmin")
    assert folders == [
        first,
        large,
        more_data,
    ]  # the earliest first; the folder with no finished job: not one
    runs, timing = plot.load(folders, "sysadmin")
    assert set(runs["size"]) == {3, 9} and set(timing["size"]) == {3, 9}
    assert set(runs["n_transitions"]) == {100, 1000, 200_000}
    assert len(runs) == 2 * 2 * 3 * 3 * 2  # sizes x seeds x data sizes x methods x rewards
    assert set(plot.load(folders)[0]["size"]) == {3, 9, 11}  # LEFT_OUT is the domain's
    out = tmp_path / "out"
    arguments = ["--results", str(tmp_path), "--out", str(out)]
    assert plot.main(arguments) == 0
    drawn = pd.read_csv(out / "sysadmin" / "data" / "summary_solved_by_level.csv")
    assert set(drawn["n_transitions"]) == {100, 1000, 200_000}
    assert (out / "sysadmin" / "N=9" / "value_by_level.png").exists()
    assert not (out / "sysadmin" / "N=11").exists() and not (out / "sysadmin" / "N=10").exists()
    pooled = pd.read_csv(out / "sysadmin" / "data" / "paper_imdp_by_graph_solved_vs_transitions.csv")
    assert set(pooled["instances"]) == {2 * 2 * 2}  # sizes x seeds x rewards, at every data size
    assert plot.main([*arguments, "--run", str(first), "--out", str(tmp_path / "alone")]) == 0
    alone = pd.read_csv(tmp_path / "alone" / "sysadmin" / "data" / "summary_solved_by_level.csv")
    assert set(alone["n_transitions"]) == {100, 1000}
