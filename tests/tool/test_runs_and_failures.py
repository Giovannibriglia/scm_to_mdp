"""Run folders (tool/runs.py), sweeps that go on after a failure, and results.csv read back safely
(tool/sweep.py)."""

import time

import pandas as pd
import pytest

from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Experiment, Limits, Method, run_experiment, run_folder
from scm_to_mdp.tool.sweep import COLUMNS, read_results


# ---- one folder per run ---------------------------------------------------------------------------
def test_a_new_run_gets_a_new_folder_and_resume_finds_the_latest(tmp_path):
    base = tmp_path / "grid"
    first, report_only = run_folder(base, [])
    assert first.name.startswith("grid_") and not report_only
    first.mkdir()
    time.sleep(1.1)
    second, _ = run_folder(base, [])
    assert second != first
    second.mkdir()
    (tmp_path / "grid_notes").mkdir()  # not a run folder: no date and time
    assert run_folder(base, ["--resume"]) == (second, False)
    assert run_folder(base, ["--report-only"]) == (second, True)
    assert run_folder(base, ["--resume", str(first)]) == (first, False)
    with pytest.raises(SystemExit):
        run_folder(tmp_path / "other", ["--resume"])


# ---- a failure does not stop a sweep ---------------------------------------------------------------
def make_env(size, seed):
    if size == 3:
        raise RuntimeError("this size is broken")
    return SysAdmin(size, seed)


def collect(env, n_rollouts):
    return env.sample(n_rollouts, 10, "random")


def test_a_failed_setup_becomes_error_rows(tmp_path):
    exp = Experiment(
        name="failing", make_env=make_env, collect=collect, sizes=[2, 3], n_rollouts=[20, 40], seeds=[0],
        methods={"flat MDP": Method("flat_mdp", "vi")}, limits=Limits(mem_gb=2), out_dir=tmp_path,
        save_models="none",
    )  # fmt: skip
    res = run_experiment(exp)
    broken = res[res["size"] == "3"]
    assert len(broken) == 2 and set(broken["status"]) == {"error"} and set(broken["failed_step"]) == {"setup"}
    assert broken["error"].str.contains("this size is broken").all()
    assert set(res.loc[res["size"] == "2", "status"]) == {"ok"}
    assert len(run_experiment(exp)) == 4  # the error rows count as finished: nothing runs again


def test_a_changed_configuration_is_refused(tmp_path):
    exp = Experiment(
        name="tiny", make_env=make_env, collect=collect, sizes=[2], n_rollouts=[20], seeds=[0],
        methods={"flat MDP": Method("flat_mdp", "vi")}, limits=Limits(mem_gb=2), out_dir=tmp_path,
        settings={"horizon": 10}, save_models="none",
    )  # fmt: skip
    run_experiment(exp)
    import dataclasses

    with pytest.raises(ValueError, match="different configuration"):
        run_experiment(dataclasses.replace(exp, settings={"horizon": 30}))
    with pytest.raises(ValueError, match="different configuration"):
        run_experiment(dataclasses.replace(exp, n_rollouts=[20, 40]))  # a new largest table
    run_experiment(dataclasses.replace(exp, seeds=[0, 1]))  # more seeds: fine


# ---- results.csv ------------------------------------------------------------------------------------
def test_labels_are_read_back_as_written_and_a_cut_line_is_dropped(tmp_path):
    rows = pd.DataFrame(
        [{"size": "0.0", "knob": "None", "policy": "NA", "seed": 0, "n_rollouts": 10, "method": "m",
          "status": "ok", "promise": 1.5}] * 2
    ).reindex(columns=COLUMNS)  # fmt: skip
    text = rows.to_csv(index=False)
    (tmp_path / "results.csv").write_text(text + text.splitlines()[-1][:25])  # a run killed mid-line
    res = read_results(tmp_path)
    assert len(res) == 2
    assert (
        list(res["size"]) == ["0.0", "0.0"] and set(res["knob"]) == {"None"} and set(res["policy"]) == {"NA"}
    )
    assert res["promise"].dtype == float and res["shd"].isna().all()
