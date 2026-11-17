"""experiments/clear27/ablations/ablation_kwik_thresholds: SLF-Rmax and Met-Rmax at their theorems' thresholds
are run by the main protocol's code into the ablation's folder, and drawn with their twins of the main runs
(the papers' thresholds): the % of all instances solved and reliable."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "clear27" / "ablations" / "ablation_kwik_thresholds"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"kwik_thresholds_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run, plot = load("run"), load("plot")
paper = plot.paper
ARMS = {"strehl": "Strehl-KWIK -> MDP (plug-in)", "met": "Met-Rmax-KWIK -> MDP (plug-in)"}


def test_the_main_runs_have_the_papers_thresholds_and_the_ablation_the_theorems(tmp_path):
    protocol = run.protocol
    domain = protocol.DOMAINS["sysadmin"]
    k = domain.sl_spibb_parents
    assert protocol.kwik_options("strehl", domain) == {"k": k, "m": 30, "eps1": 0.2, "delta": protocol.DELTA}
    assert protocol.kwik_options("met", domain) == {"D": k, "m": 50, "m_sub": 50, "eps": 0.2,
                                                    "delta": protocol.DELTA}  # fmt: skip
    assert protocol.kwik_options("strehl", domain, "theorem")["m"] == "calibrated"
    assert protocol.kwik_options("met", domain, "theorem")["m_sub"] == "calibrated"
    with pytest.raises(ValueError, match="level"):
        protocol.kwik_options("met", domain, "other")
    main = protocol.experiment_grid("sysadmin", "strehl")
    assert main.methods[ARMS["strehl"]].discovery_options["m"] == 30
    assert run.SOURCES == ("strehl_theorem", "met_theorem") and run.DOMAINS == ("sysadmin", "drone", "sepsis")
    g = run.ablation_grid("drone", "met_theorem", tmp_path)
    assert list(g.methods) == ["oracle -> MDP", ARMS["met"]]
    assert g.methods[ARMS["met"]].discovery_options["m_sub"] == "calibrated"
    twin = protocol.experiment_grid("drone", "met")
    assert (g.sizes, g.n_transitions, g.seeds, g.limits) == (twin.sizes, twin.n_transitions, twin.seeds,
                                                             twin.limits)  # fmt: skip
    assert g.out_dir == tmp_path / "drone" / "drone_met_theorem" and g.save_models == "none"
    every = run.ablation_grid("sysadmin", "strehl_theorem", tmp_path, every_compilation=True)
    assert list(every.methods) == list(protocol.experiment_grid("sysadmin", "strehl_theorem").methods)
    with pytest.raises(ValueError, match="source must be one of"):
        run.ablation_grid("sysadmin", "strehl", tmp_path)  # a source of the main runs


def test_the_launch_script_runs_the_three_domains():
    script = (HERE / "run_all.sh").read_text()
    assert "run.py" in script and "sysadmin drone sepsis" in script
    assert subprocess.run(["bash", "-n", str(HERE / "run_all.sh")]).returncode == 0
    rewards = ROOT / "experiments" / "clear27" / "ablations" / "ablation_rewards" / "run_all.sh"
    assert "for domain in sysadmin drone sepsis" in rewards.read_text()
    assert subprocess.run(["bash", "-n", str(rewards)]).returncode == 0


def fake_run(folder: Path, domain: str, source: str, level: str, value_at: dict, unsafe_from=None) -> None:
    """A run folder of clear27 of one KWIK source: two sizes x two seeds, the oracle and the learner's
    plug-in MDP; its meta.json has the learner's thresholds at `level`."""
    learner = source.removesuffix("_theorem")
    arm = ARMS[learner]
    rows = []
    for size, states in (("3", 8), ("4", 16)):
        for seed in (0, 1):
            for n, value in value_at.items():
                for method in ("oracle -> MDP", arm):
                    rows.append({
                        "size": size, "size_label": f"N={size}", "n_states": states, "n_observed_states": states,
                        "seed": seed, "n_transitions": n, "method": method, "status": "ok", "promise": 2.0,
                        "true": 2.0 + 3.0 * value, "V_star_obs": 5.0, "V_star": 5.0, "V_random": 2.0,
                        "safe": not (unsafe_from is not None and n >= unsafe_from), "discovery_s": 0.1,
                        "selection_s": 0.1, "learning_s": 0.1, "solve_s": 1.0, "missing": 0, "extra": 0,
                        "n_true_edges": 4, "true_parents": json.dumps({"x": ["x"], "y": ["x", "y"]}),
                    })  # fmt: skip
    out = folder / domain / f"{domain}_{source}_2026-11-17_10-00-00" / "knob=a" / "uniform"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    m = {"paper": 30, "theorem": "calibrated"}[level]
    meta = {"config": {"methods": {arm: {"learner": "kwik", "discovery_options": {"k": 2, "m": m, "m_sub": m}}}}}
    (out / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def results(tmp_path):
    """The papers' thresholds solve everything from 1,000 transitions; the theorems' never solve, and the
    theorems' Met-Rmax is never reliable."""
    main, ablation = tmp_path / "main", tmp_path / "ablation"
    for domain in plot.DOMAINS:
        fake_run(main, domain, "strehl", "paper", {100: 0.5, 1000: 1.0})
        fake_run(main, domain, "met", "paper", {100: 0.5, 1000: 1.0})
        fake_run(ablation, domain, "strehl_theorem", "theorem", {100: 0.1, 1000: 0.2})
        fake_run(ablation, domain, "met_theorem", "theorem", {100: 0.1, 1000: 0.2}, unsafe_from=100)
    return main, ablation


def test_the_four_learners_are_counted_over_every_instance(results):
    main, ablation = results
    instances = plot.instances_of(plot.load(main, ablation, "sysadmin"))
    table = plot.shares(instances).set_index(["learner", "n_transitions"])
    assert set(table["instances"]) == {4}
    assert table.loc[("SLF-Rmax, papers' thresholds", 1000), "solved"] == 100
    assert table.loc[("SLF-Rmax, theorems' thresholds", 1000), "solved"] == 0
    assert table.loc[("Met-Rmax, theorems' thresholds", 1000), "safe"] == 0
    assert table.loc[("Met-Rmax, papers' thresholds", 100), "safe"] == 100


def test_the_figures_are_written_in_three_forms(results, tmp_path):
    main, ablation = results
    out = tmp_path / "outputs"
    assert plot.main(["sysadmin", "all_domains", "--main", str(main), "--results", str(ablation),
                      "--out", str(out)]) == 0  # fmt: skip
    for where in ("sysadmin", "all_domains"):
        for group in paper.SETS:
            for suffix in ("_a", "_b", "_legend", "_a_full", "_b_full"):
                assert (out / where / group / f"kwik_thresholds{suffix}.pdf").stat().st_size > 0
    numbers = pd.read_csv(out / "all_domains" / "data" / "paper_kwik_thresholds.csv")
    assert set(numbers["instances"]) == {12} and numbers["learner"].nunique() == 4
