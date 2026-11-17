"""experiments/clear27/ablations/ablation_box_exact: the interval MDP of the true graph and of DYNOTEARS with
the exact set is run by the main protocol's code into the ablation's folder, and drawn with its twin of the
main runs (the box): the % of all instances solved and reliable, the time and the lower bound."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "clear27" / "ablations" / "ablation_box_exact"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"box_exact_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run, plot = load("run"), load("plot")
paper = plot.paper


def test_the_main_runs_have_the_box_and_the_ablation_the_exact_set(tmp_path):
    protocol = run.protocol
    assert list(protocol.RELAXATIONS) == ["IA", "exact"] and protocol.MAIN_RELAXATIONS == ("IA",)
    main = protocol.experiment_grid("sysadmin", "truegraph")
    assert "true graph -> IMDP (IA)" in main.methods and "true graph -> IMDP (exact)" not in main.methods
    assert not any("McCormick" in m for m in main.methods)
    g = run.ablation_grid("drone", "dynotears_lagged_exact", tmp_path)
    assert list(g.methods) == ["oracle -> MDP", "DYNOTEARS (lagged) -> IMDP (exact)"]
    assert g.methods["DYNOTEARS (lagged) -> IMDP (exact)"].solver == "exact_robust_vi"
    twin = protocol.experiment_grid("drone", "dynotears_lagged")
    assert (g.sizes, g.n_transitions, g.seeds, g.limits) == (twin.sizes, twin.n_transitions, twin.seeds,
                                                             twin.limits)  # fmt: skip
    with pytest.raises(ValueError, match="source must be one of"):
        run.ablation_grid("sysadmin", "truegraph", tmp_path)  # a source of the main runs
    assert subprocess.run(["bash", "-n", str(HERE / "run_all.sh")]).returncode == 0
    assert "sysadmin drone sepsis" in (HERE / "run_all.sh").read_text()


def fake_run(folder: Path, domain: str, source: str, model: str, solver: str, stopped_from=None) -> None:
    rows = []
    for size, states in (("3", 8), ("4", 16)):
        for seed in (0, 1):
            for n in (1000, 10_000):
                for method in ("oracle -> MDP", f"{model} -> IMDP ({solver})"):
                    stop = stopped_from is not None and size >= stopped_from and "IMDP" in method
                    rows.append({
                        "size": size, "size_label": f"N={size}", "n_states": states, "n_observed_states": states,
                        "seed": seed, "n_transitions": n, "method": method,
                        "status": "timeout" if stop else "ok", "promise": 2.5 if solver == "exact" else 2.2,
                        "true": 5.0, "V_star_obs": 5.0, "V_star": 5.0, "V_random": 2.0, "safe": True,
                        "discovery_s": 0.1, "selection_s": 0.1, "learning_s": 0.1, "relax_s": 0.1,
                        "solve_s": 5.0 if solver == "exact" else 0.5, "missing": 0, "extra": 0,
                        "n_true_edges": 4, "true_parents": json.dumps({"x": ["x"], "y": ["x", "y"]}),
                    })  # fmt: skip
    out = folder / domain / f"{domain}_{source}_2026-11-17_10-00-00" / "knob=a" / "uniform"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


@pytest.fixture
def results(tmp_path):
    main, ablation = tmp_path / "main", tmp_path / "ablation"
    for domain in plot.DOMAINS:
        fake_run(main, domain, "truegraph", "true graph", "IA")
        fake_run(main, domain, "dynotears_lagged", "DYNOTEARS (lagged)", "IA")
        fake_run(ablation, domain, "truegraph_exact", "true graph", "exact", stopped_from="4")
        fake_run(ablation, domain, "dynotears_lagged_exact", "DYNOTEARS (lagged)", "exact", stopped_from="4")
    return main, ablation


def test_the_exact_set_counts_its_stops_and_is_drawn_in_the_convention_of_the_paper(results, tmp_path):
    main, ablation = results
    instances = plot.instances_of(plot.load(main, ablation, "sysadmin"))
    table = plot.shares(instances).set_index(["learner", "n_transitions"])
    assert table.loc[("CD-IMDP, box", 1000), "solved"] == 100
    assert table.loc[("CD-IMDP, exact", 1000), "solved"] == 50  # the larger size stopped: not solved
    assert table.loc[("True-IMDP, exact", 1000), "bound"] == pytest.approx(0.5 / 3)  # (2.5 - 2) / (5 - 2)
    cost = plot.costs(instances).set_index(["learner", "states"])["seconds"]
    assert cost["CD-IMDP, exact", 8] == pytest.approx(5.1) and ("CD-IMDP, exact", 16) not in cost.index
    out = tmp_path / "outputs"
    assert plot.main(["sysadmin", "all_domains", "--main", str(main), "--results", str(ablation),
                      "--out", str(out)]) == 0  # fmt: skip
    for where in ("sysadmin", "all_domains"):
        for group in paper.SETS:
            for name in ("box_exact", "box_exact_cost"):
                for suffix in ("_a", "_b", "_legend", "_a_full", "_b_full"):
                    assert (out / where / group / f"{name}{suffix}.pdf").stat().st_size > 0
