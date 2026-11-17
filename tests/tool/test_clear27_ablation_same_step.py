"""experiments/clear27/ablations/ablation_same_step: PCMCI+ and DYNOTEARS with same-step arrows allowed
are run by the main protocol's code into the ablation's folder, and drawn with their twins of the main
runs: the % of all instances solved safely, and the errors of the graphs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "clear27" / "ablations" / "ablation_same_step"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"same_step_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run, plot = load("run"), load("plot")
paper = plot.paper
MODELS = {"dynotears_lagged": "DYNOTEARS (lagged)", "pcmci_lagged": "PCMCI+ (lagged)",
          "dynotears": "DYNOTEARS", "pcmci": "PCMCI+", "truegraph": "true graph"}  # fmt: skip


def test_the_grid_is_the_main_protocols_with_the_two_forms(tmp_path):
    protocol = run.protocol
    assert run.SOURCES == ("dynotears", "pcmci") and run.DOMAINS == ("sysadmin", "drone", "sepsis")
    main = protocol.experiment_grid("drone", "pcmci")
    g = run.ablation_grid("drone", "pcmci", tmp_path)
    assert list(g.methods) == ["oracle -> MDP", "PCMCI+ -> IMDP (IA)", "PCMCI+ -> MDP (plug-in)"]
    assert all(g.methods[name] == main.methods[name] for name in g.methods)  # the same methods, not copies
    assert g.methods["PCMCI+ -> IMDP (IA)"].learning_options["n_candidates"] == "inputs+next"
    assert "same_step" not in g.methods["PCMCI+ -> IMDP (IA)"].discovery_options  # same-step arrows allowed
    assert (g.sizes, g.n_transitions, g.seeds, g.limits) == (main.sizes, main.n_transitions, main.seeds,
                                                             main.limits)  # fmt: skip
    assert (g.gamma, g.horizon, g.n_starts, g.policies) == (main.gamma, main.horizon, main.n_starts,
                                                            main.policies)  # fmt: skip
    assert g.out_dir == tmp_path / "drone" / "drone_pcmci" and g.save_models == "none"
    assert run.RESULTS == ROOT / "results" / "clear27" / "ablations" / "ablation_same_step"
    every = run.ablation_grid("sysadmin", "dynotears", tmp_path, every_compilation=True)
    assert list(every.methods) == list(protocol.experiment_grid("sysadmin", "dynotears").methods)
    small = run.ablation_grid("sysadmin", "dynotears", tmp_path, ["3"], ["100", "500000"], ["0", "1"])
    assert (small.sizes, small.n_transitions, small.seeds) == ([3], [100, 500_000], [0, 1])
    with pytest.raises(ValueError, match="--sizes"):
        run.ablation_grid("sysadmin", "dynotears", tmp_path, ["99"])
    with pytest.raises(ValueError, match="source must be one of"):
        run.ablation_grid("sysadmin", "dynotears_lagged", tmp_path)  # a source of the main runs
    assert run.arm("dynotears", "MDP") == "DYNOTEARS -> MDP (plug-in)"


def test_a_run_folder_is_new_or_the_latest_and_has_its_provenance(tmp_path):
    base = tmp_path / "sepsis" / "sepsis_pcmci"
    first = run.run_folder(base, resume=True)  # nothing yet: a new folder
    assert first.name.startswith("sepsis_pcmci_2") and not first.exists()
    grid = run.ablation_grid("sepsis", "pcmci", tmp_path, ["2"], ["100"], ["0"])
    run.write_provenance(first, "sepsis", "pcmci", grid, ["sepsis", "--sizes", "2"])
    record = json.loads((first / "provenance.json").read_text())
    assert record["origin"] == "run" and record["source"] == "pcmci" and record["sizes"] == ["2"]
    assert record["methods"] == list(grid.methods) and "sepsis_pcmci_lagged" in record["lagged_twin"]
    assert run.run_folder(base, resume=True) == first  # the latest run of the source
    older = tmp_path / "sepsis" / "sepsis_pcmci_2026-11-05_22-52-41"
    older.mkdir()
    assert run.run_folder(base, resume=True) == first and run.run_folder(base, resume=False) != older
    (first / "provenance.json").write_text("{}")
    run.write_provenance(first, "sepsis", "pcmci", grid, [])  # a resumed run keeps its record
    assert (first / "provenance.json").read_text() == "{}"


def test_the_launch_scripts_run_the_ablation_then_its_figures():
    every = (HERE / "run_all.sh").read_text()
    assert "run.py" in every and "sysadmin drone sepsis" in every
    assert subprocess.run(["bash", "-n", str(HERE / "run_all.sh")]).returncode == 0
    runner = (HERE / "run_domain.sh").read_text()
    assert "run.py" in runner and "plot.py" in runner and "dynotears_lagged pcmci_lagged" in runner
    for domain in run.DOMAINS:
        script = HERE / f"run_{domain}.sh"
        assert f'run_domain.sh" {domain}' in script.read_text()
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(HERE / "run_domain.sh")]).returncode == 0


def fake_run(folder: Path, domain: str, source: str, value_at: dict, unsafe_from: int | None = None,
             stopped_sizes=(), missing=0, extra=0) -> None:  # fmt: skip
    """A results.csv of one source of clear27, two sizes x two seeds: value_at = {transitions: normalised
    value}; unsafe_from: the first number of transitions whose promise fails; stopped_sizes: the sizes
    whose interval MDP hits a limit (its plug-in MDP finishes)."""
    rows = []
    model = MODELS[source]
    for size, states in (("3", 8), ("4", 16)):
        for seed in (0, 1):
            for n, value in value_at.items():
                for method in ("oracle -> MDP", f"{model} -> IMDP (IA)", f"{model} -> MDP (plug-in)"):
                    stopped = size in stopped_sizes and method.endswith("IMDP (IA)")
                    rows.append({
                        "size": size, "size_label": f"N={size}", "n_states": states,
                        "n_observed_states": states, "seed": seed, "n_transitions": n, "method": method,
                        "status": "timeout" if stopped else "ok", "promise": 2.0,
                        "true": np.nan if stopped else 2.0 + 3.0 * value, "V_star_obs": 5.0, "V_star": 5.0,
                        "V_random": 2.0, "safe": not (unsafe_from is not None and n >= unsafe_from),
                        "discovery_s": 0.1, "selection_s": 0.1, "learning_s": 0.1, "solve_s": 1.0,
                        "missing": missing, "extra": extra, "n_true_edges": 4, "shd": missing + extra,
                        "intra_next": extra, "true_parents": json.dumps({"x": ["x"], "y": ["x", "y"]}),
                    })  # fmt: skip
    out = folder / domain / f"{domain}_{source}_2026-11-05_22-52-41" / "knob=a" / "uniform"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


@pytest.fixture
def results(tmp_path):
    """Main results (the lagged learners, and the true graph) and the ablation's (the twins) of the three
    domains. With 100,000 transitions every learner reaches the value 1; with 1,000 only DYNOTEARS does.
    The twin of PCMCI+ breaks its promise at 100,000 and its interval MDP stops at the larger size."""
    main, twins = tmp_path / "main", tmp_path / "ablation"
    good, late = {1000: 0.95, 100_000: 1.0}, {1000: 0.5, 100_000: 1.0}
    for domain in plot.DOMAINS:
        fake_run(main, domain, "truegraph", good)
        fake_run(main, domain, "dynotears_lagged", good)
        fake_run(main, domain, "pcmci_lagged", late, missing=1)
        fake_run(twins, domain, "dynotears", late, extra=2)
        fake_run(twins, domain, "pcmci", late, unsafe_from=100_000, stopped_sizes=("4",), missing=1, extra=2)
    return main, twins


def test_solved_safely_counts_every_instance_and_needs_both_conditions(results):
    main, twins = results
    runs = plot.load(main, twins, "sysadmin")
    assert set(runs["source"]) == {"truegraph", "dynotears_lagged", "pcmci_lagged", "dynotears", "pcmci",
                                   "oracle"}  # fmt: skip
    instances = plot.instances_of(runs)
    assert set(instances["method"]) == {"DYNOTEARS", "DYNOTEARS same-step", "PCMCI+", "PCMCI+ same-step"}
    imdp = plot.shares(instances, "IMDP").set_index(["method", "n_transitions"])
    assert set(imdp["instances"]) == {4}  # 2 sizes x 2 seeds, also where a run did not finish
    assert imdp.loc[("PCMCI+ same-step", 1000), "finished"] == 2
    at = imdp["solved_safely"]
    assert at["DYNOTEARS", 1000] == 100 and at["DYNOTEARS", 100_000] == 100
    assert at["DYNOTEARS same-step", 1000] == 0 and at["DYNOTEARS same-step", 100_000] == 100  # solved late
    assert at["PCMCI+", 100_000] == 100
    assert at["PCMCI+ same-step", 100_000] == 0  # solved at one size, but its promise failed
    assert imdp.loc[("PCMCI+ same-step", 100_000), "solved"] == 50  # the size that did not finish counts
    assert imdp.loc[("PCMCI+ same-step", 1000), "safe"] == 50
    mdp = plot.shares(instances, "MDP").set_index(["method", "n_transitions"])
    assert mdp.loc[("PCMCI+ same-step", 100_000), "finished"] == 4  # the plug-in MDP finished everywhere
    assert mdp.loc[("PCMCI+ same-step", 100_000), "solved"] == 100
    assert mdp.loc[("PCMCI+ same-step", 100_000), "solved_safely"] == 0
    table = plot.summary(runs, at=(100_000,))
    assert len(table) == 8 and set(table["form"]) == {"IMDP", "MDP"}


def test_the_figures_of_a_domain_and_of_all_domains_are_written_with_their_numbers(results, tmp_path, capsys):
    main, twins = results
    out = tmp_path / "outputs"
    arguments = ["--main", str(main), "--results", str(twins), "--out", str(out)]
    assert plot.main(["sepsis", "all_domains", *arguments]) == 0
    printed = capsys.readouterr().out
    assert "12 instances (drone: 4, sepsis: 4, sysadmin: 4)" in printed
    for where in ("sepsis", "all_domains"):
        for group in paper.SETS:
            for name in plot.FIGURES:
                assert (out / where / group / f"{name}.pdf").stat().st_size > 0
                assert (out / where / group / f"{name}.png").stat().st_size > 0
            for name in ("solved_safely", "graph_errors_split"):  # the three forms of the appendix
                for suffix in ("_a", "_b", "_legend", "_a_full", "_b_full"):
                    assert (out / where / group / f"{name}{suffix}.pdf").stat().st_size > 0
    data = out / "all_domains" / "data"
    drawn = pd.read_csv(data / "paper_imdp_solved_safely_vs_transitions.csv")
    assert set(drawn["instances"]) == {12} and set(drawn["form"]) == {"IMDP"}
    assert set(drawn["label"]) == {"DYNOTEARS", "DYNOTEARS, same-step arrows", "PCMCI+",
                                   "PCMCI+, same-step arrows"}  # fmt: skip
    both = pd.read_csv(data / "appendix_mdp_solved_safely_vs_transitions.csv")
    assert set(both["label"]) == {"DYNOTEARS-MDP", "DYNOTEARS-MDP, same-step arrows",
                                  "PCMCI+-MDP", "PCMCI+-MDP, same-step arrows"}  # fmt: skip
    graphs = pd.read_csv(data / "appendix_graph_errors.csv")
    assert set(graphs["label"]) == {"DYNOTEARS", "DYNOTEARS, same-step arrows", "PCMCI+",
                                    "PCMCI+, same-step arrows"}  # fmt: skip
    missed = graphs[graphs["metric"] == "missing_share"].set_index(["method", "n_transitions"])["mean"]
    extra = graphs[graphs["metric"] == "extra_per_var"].set_index(["method", "n_transitions"])["mean"]
    assert missed["PCMCI+", 1000] == pytest.approx(0.25) and missed["DYNOTEARS", 1000] == 0
    assert extra["DYNOTEARS same-step", 1000] == pytest.approx(1.0) and extra["DYNOTEARS", 1000] == 0
    assert set(graphs["instances"]) == {12} and set(graphs["graphs"]) == {12}  # every discovery finished
    assert plot.main(["drone", "--only", "paper", *arguments]) == 0
    assert (out / "drone" / "paper" / "graph_errors.pdf").exists()
    assert not (out / "drone" / "appendix").exists()
    assert plot.main(["sepsis", "--main", str(tmp_path / "nowhere"), "--results", str(twins),
                      "--out", str(out)]) == 0  # fmt: skip
    assert "nothing to draw" in capsys.readouterr().out


def test_a_twin_has_the_colour_and_marker_of_its_method_a_broken_line_and_a_hollow_marker():
    to = paper.FigureSet("paper", paper.SETS["paper"], Path("."))
    both = paper.FigureSet("appendix", paper.SETS["appendix"], Path("."))
    for method in paper.DISCOVERY:
        _, marker, _ = paper.METHODS[method]
        for form in ("IMDP", "MDP"):
            color = paper.look_of(method if form == "IMDP" else f"{method} {form}")[0]  # its method's
            lagged, twin = plot.line_of(method, False, form, to), plot.line_of(method, True, form, to)
            assert lagged["color"] == twin["color"] == twin["mec"] == color
            assert lagged["marker"] == twin["marker"] == marker and twin["mfc"] == "white"
            assert twin["ls"] == paper.SAME_STEP_LINES[form][1] != lagged["ls"]
            assert "(ours)" not in twin["label"]
            assert "(ours)" not in plot.line_of(method, True, form, both)["label"]
            assert plot.line_of(method, False, form, both)["label"] == f"{method}-{form}"
            assert "(ours)" not in lagged["label"]  # no learner is "ours" in this ablation
            assert lagged["lw"] > twin["lw"]  # without same-step arrows: thick, for both methods
    first, second = plot.line_of("DYNOTEARS", False, "IMDP", to), plot.line_of("PCMCI+", False, "IMDP", to)
    assert first["label"] == "DYNOTEARS" and second["label"] == "PCMCI+" and first["lw"] == second["lw"]
    assert plot.line_of("PCMCI+", True, "IMDP", to, label_form=None)["label"] == "PCMCI+, same-step arrows"
    assert [plot.model_of(m, twin) for m, twin in plot.LEARNERS] == ["DYNOTEARS (lagged)", "DYNOTEARS",
                                                                     "PCMCI+ (lagged)", "PCMCI+"]  # fmt: skip
    assert set(paper.SAME_STEP) == set(paper.DISCOVERY)  # every discovery method of the registry has a twin
    assert [source for source, _ in paper.SAME_STEP.values()] == list(run.SOURCES)
