"""experiments/clear27/plot.py: the runs of a domain are read from its latest run folders, the sizes and
panels come from the data, and a figure is written with the numbers it draws."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "clear27"))
plot = importlib.import_module("plot")
sys.path.pop(0)

SOURCES = {"truegraph": "true graph", "pcmci_lagged": "PCMCI+ (lagged)",
           "dynotears_lagged": "DYNOTEARS (lagged)",
           "sparse": "sparse", "fullfact": "full-fact", "flat": None}  # fmt: skip


def fake_run(folder: Path, domain: str, source: str, stamp: str, sizes: dict, stop_from=None) -> None:
    """A results.csv of one source: sizes = {size: observed states}; stop_from: the first size whose
    certified runs all hit a limit."""
    rows = []
    methods = ["oracle -> MDP", "flat IMDP"] if SOURCES[source] is None else [
        "oracle -> MDP", f"{SOURCES[source]} -> IMDP (exact)", f"{SOURCES[source]} -> IMDP (IA)"]  # fmt: skip
    for k, (size, states) in enumerate(sizes.items()):
        for seed in range(4):
            for n in (10, 1000, 100_000):
                for method in methods:
                    stopped = stop_from is not None and k >= stop_from and method.endswith("(exact)")
                    true = 2.0 + min(1.0, n / 1000) * 3.0 - 0.1 * seed
                    rows.append({
                        "size": size, "size_label": f"side={size}", "n_states": 2 * states,
                        "n_observed_states": states, "seed": seed, "n_transitions": n, "method": method,
                        "status": "timeout" if stopped else "ok", "promise": true - 1.0,
                        "true": np.nan if stopped else true, "V_star_obs": 5.0, "V_star": 5.0,
                        "V_random": 2.0, "safe": True, "contain_all": n > 10,
                        "discovery_s": 0.1, "selection_s": 0.1, "learning_s": 0.1, "solve_s": 1.0,
                        "missing": 1, "extra": 2, "n_true_edges": 4, "shd": 3,
                        "true_parents": json.dumps({"x": ["x"], "y": ["x", "y"]}),
                    })  # fmt: skip
    out = folder / domain / f"{domain}_{source}_{stamp}" / "knob=a" / "uniform"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


def test_the_latest_run_of_every_source_is_read(tmp_path):
    sizes = {"3-1": 27, "4-1": 64, "10-1": 1000}
    fake_run(tmp_path, "grid", "truegraph", "2026-11-01_10-00-00", {"3-1": 27})  # an older run: not read
    for source in SOURCES:
        fake_run(tmp_path, "grid", source, "2026-11-05_22-52-41", sizes, stop_from=2)
    found = plot.latest_runs(tmp_path / "grid", "grid")
    assert set(found) == set(SOURCES) and found["truegraph"].name.endswith("2026-11-05_22-52-41")
    assert found["dynotears_lagged"].name.startswith("grid_dynotears_lagged_")  # not the source "dynotears"
    runs = plot.load(tmp_path, "grid")
    assert set(runs["source"]) == set(SOURCES) | {"oracle"}
    assert (runs["model"] == "oracle").sum() == 3 * 4 * 3  # one copy of the oracle per size, seed and budget
    flat = runs[runs["method"] == "flat IMDP"]
    assert set(flat["model"]) == {"flat"} and set(flat["solver"]) == {"IMDP"}
    best = runs[
        (runs["model"] == "true graph")
        & (runs["n_transitions"] == 100_000)
        & runs["ok"]
        & (runs["seed"] == 0)
    ]
    assert best["value"].iloc[0] == pytest.approx(1.0)  # (5 - 2) / (5 - 2)
    assert runs["extra_per_var"].dropna().iloc[0] == pytest.approx(1.0)  # 2 extra edges, 2 variables
    assert plot.load(tmp_path, "nowhere").empty


def test_sizes_panels_and_labels_come_from_the_data(tmp_path):
    sizes = {"10-1": 1000, "3-1": 27, "4-1": 64}  # not in order, and "10-1" sorts before "3-1" as text
    for source in SOURCES:
        fake_run(tmp_path, "grid", source, "2026-11-05_22-52-41", sizes, stop_from=None)
    runs = plot.load(tmp_path, "grid")
    dom = plot.describe("grid", runs, "IMDP (exact)", tmp_path / "out")
    assert dom.sizes == ["3-1", "4-1", "10-1"] == dom.kept  # by observed states
    assert dom.panels == dom.kept and dom.graph_size == "10-1" and dom.title("4-1") == "side = 4-1"
    stopped = runs.copy()  # the reference stops at the largest size: that size is left out
    large = (stopped["size"] == "10-1") & (stopped["solver"] == "IMDP (exact)")
    stopped.loc[large, "ok"] = False
    dom = plot.describe("grid", stopped, "IMDP (exact)", tmp_path / "out")
    assert dom.kept == ["3-1", "4-1"] and dom.sizes == ["3-1", "4-1", "10-1"]
    assert plot.describe("grid", stopped, "IMDP (IA)", tmp_path / "out").kept == ["3-1", "4-1", "10-1"]
    assert plot.spread(list(range(8)), 5) == [0, 2, 4, 5, 7] and plot.spread([1, 2], 5) == [1, 2]
    sysadmin = plot.describe(
        "sysadmin", runs.assign(size_label="N=" + runs["size"]), "IMDP (exact)", tmp_path
    )
    assert sysadmin.title("3-1") == "$N = 3-1$"  # a one-letter name is a symbol
    assert sysadmin.panels == dom.sizes  # the hand-made panels of SysAdmin are not sizes of these runs


def test_figures_are_written_with_their_numbers(tmp_path, capsys):
    sizes = {"3-1": 27, "4-1": 64}
    for source in SOURCES:
        fake_run(tmp_path / "results", "grid", source, "2026-11-05_22-52-41", sizes)
    out = tmp_path / "outputs"
    written = plot.plot_domain(
        "grid", tmp_path / "results", out, "IMDP (exact)", only=["value", "coverage", "baselines"]
    )
    assert written == ["value", "coverage"]  # baselines: no run of the baselines (sota) yet
    printed = capsys.readouterr().out
    assert "skipped paper/baselines: no run of sota yet" in printed
    assert "skipped appendix/baselines: no run of sota yet" in printed
    for name in written:  # in the two sets of paper figures
        for where in plot.SETS:
            assert (out / "grid" / where / f"{name}.pdf").stat().st_size > 0
            assert (out / "grid" / where / f"{name}.png").stat().st_size > 0
    drawn = pd.read_csv(out / "grid" / "data" / "paper_value.csv")  # ours is DYNOTEARS ("CD"), no PCMCI+
    assert set(drawn["method"]) == {"True-IMDP", "CD-IMDP (ours)", "Sparse-IMDP", "Full-IMDP", "Flat-IMDP"}
    assert drawn[drawn["n_transitions"] == 100_000]["mean"].between(0.85, 1.0).all()
    both = pd.read_csv(out / "grid" / "data" / "appendix_value.csv")  # the two discovery methods as ours
    assert set(both["method"]) - set(drawn["method"]) == {"DYNOTEARS-IMDP (ours)", "PCMCI+-IMDP (ours)"}
    assert plot.plot_domain("grid", tmp_path / "results", out, "IMDP (exact)", only=["graph_errors"]) == [
        "graph_errors"
    ]
    for where, ours in (
        ("paper", {"CD-IMDP (ours)"}),
        ("appendix", {"DYNOTEARS-IMDP (ours)", "PCMCI+-IMDP (ours)"}),
    ):
        errors = pd.read_csv(out / "grid" / "data" / f"{where}_graph_errors.csv")  # no true graph: no error
        assert set(errors["method"]) == ours | {"Sparse-IMDP", "Full-IMDP"}
        assert set(errors["metric"]) == {"value", "missing_share", "extra_per_var"}
    one = plot.plot_domain("grid", tmp_path / "results", tmp_path / "one", only=["value", "appendix"])
    assert one == ["value"] and (tmp_path / "one" / "grid" / "appendix" / "value.pdf").exists()
    assert not (tmp_path / "one" / "grid" / "paper").exists()  # --only appendix: that set only
    assert plot.main(["--list"]) == 0 and "9_transitions_needed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        plot.main(["nowhere", "--results", str(tmp_path / "results")])
    with pytest.raises(SystemExit):
        plot.main(["grid", "--results", str(tmp_path / "results"), "--only", "no_such_figure"])


def test_every_domain_is_drawn_with_the_box(tmp_path, capsys):
    """The exact solver stops beyond the smallest sizes, so the certified solver is the box and every
    size is kept; --solver (the solver argument) still overrides the domain's choice."""
    assert plot.DOMAIN_SETTINGS["sepsis"]["solver"] == plot.DOMAIN_SETTINGS["drone"]["solver"] == "IMDP (IA)"
    assert "aircraft" not in plot.DOMAIN_SETTINGS  # not a domain of clear27
    assert plot.DOMAIN_SETTINGS["drone"]["panels"] == ["3-1", "5-1", "7-1", "10-1", "12-1"]
    assert plot.DOMAIN_SETTINGS["sysadmin"]["solver"] == "IMDP (IA)"  # the box in every domain
    sizes = {"3-1": 27, "5-1": 125, "7-1": 343, "10-1": 1000, "12-1": 1728}
    for source in SOURCES:
        fake_run(tmp_path, "drone", source, "2026-11-05_22-52-41", sizes, stop_from=1)  # exact: 3-1 only
    plot.plot_domain("drone", tmp_path, tmp_path / "out", only=["value"])
    assert "certified solver: IMDP (IA)" in capsys.readouterr().out
    drawn = pd.read_csv(tmp_path / "out" / "drone" / "data" / "paper_value.csv")
    assert set(drawn["size"]) == set(sizes)
    plot.plot_domain("drone", tmp_path, tmp_path / "exact", solver="IMDP (exact)", only=["value"])
    exact = pd.read_csv(tmp_path / "exact" / "drone" / "data" / "paper_value.csv")
    assert set(exact[exact["method"] == "True-IMDP"]["size"]) == {"3-1"}


def test_a_size_of_left_out_is_dropped_when_the_runs_are_loaded(tmp_path, monkeypatch):
    """LEFT_OUT: no size is left out now (SysAdmin N = 2 is drawn); a size listed there is dropped when
    the domain's runs are loaded, for every figure."""
    assert plot.LEFT_OUT == {} and "2" in plot.DOMAIN_SETTINGS["sysadmin"]["panels"]
    for domain in ("sysadmin", "sepsis"):
        for source in SOURCES:
            fake_run(tmp_path, domain, source, "2026-11-05_22-52-41", {"2": 4, "3": 8})
    assert set(plot.load(tmp_path, "sysadmin")["size"]) == {"2", "3"}
    monkeypatch.setattr(plot, "LEFT_OUT", {"sysadmin": ["2"]})
    assert set(plot.load(tmp_path, "sysadmin")["size"]) == {"3"}
    assert set(plot.load(tmp_path, "sepsis")["size"]) == {"2", "3"}


def test_the_pooled_figures_of_a_domain_and_of_all_domains(tmp_path, capsys):
    """--only pooled draws the thirteen pooled figures of a domain from its runs (the box, the lagged graphs)
    in the two sets, and all_domains draws them over the instances of SysAdmin, Drone and Sepsis together."""
    results, out = tmp_path / "results", tmp_path / "out"
    for domain in plot.POOLED_DOMAINS:
        for source in SOURCES:
            fake_run(results, domain, source, "2026-11-05_22-52-41", {"2": 4, "3": 8})
    runs = plot.load(results, "sepsis")
    table = plot.instances_of(runs, "sepsis")
    assert set(table["method"]) == set(plot.GRAPHS)  # these runs have the interval MDPs only
    assert len(table) == 6 * 2 * 4 * 3 and table["ok"].all() and table["safe"].all()
    assert table["seconds"].iloc[0] == pytest.approx(1.3)  # the steps of the run, a missing one as 0
    for method in plot.MAIN_METHODS.values():
        plot.split_form(method)  # a method of the registry, or a form of a graph source
    arguments = ["--results", str(results), "--out", str(out), "--only", "pooled"]
    assert plot.main(["sepsis", "all_domains", *arguments]) == 0
    printed = capsys.readouterr().out
    assert "up to 24 instances per data size" in printed  # 2 sizes in each of the three domains, 4 seeds
    assert "skipped paper/mdp_by_graph/solved_vs_transitions: no run of its methods" in printed
    # imdp_by_graph/solved_vs_optimality is drawn at OPTIMALITY_N = 500,000 transitions, not in these runs
    assert "skipped paper/imdp_by_graph/solved_vs_optimality: no run at 500,000 transitions" in printed
    figures = ("imdp_by_graph/solved_vs_transitions", "vs_baselines/solved_vs_time", "tradeoff", "tradeoff_pareto",
               "tradeoff_data_needed", "solved_and_tradeoff")  # fmt: skip
    for where in ("sepsis", "all_domains"):
        for group in plot.SETS:
            for name in figures:
                assert (out / where / group / f"{name}.pdf").stat().st_size > 0
            assert not (out / where / group / "value.pdf").exists()  # --only pooled
    data = out / "all_domains" / "data"
    drawn = pd.read_csv(data / "paper_imdp_by_graph_solved_vs_transitions.csv")
    assert set(drawn["instances"]) == {24}
    assert set(drawn["method"]) == set(plot.GRAPHS) - {"PCMCI+", "sparse"}  # the sparse graph: appendix only
    wide = pd.read_csv(data / "appendix_imdp_by_graph_solved_vs_transitions.csv")
    assert set(wide["method"]) == set(plot.GRAPHS)
    assert "CD-IMDP (ours)" in set(drawn["label"])
    at = drawn.set_index(["method", "n_transitions"])["solved"]
    assert at["DYNOTEARS", 100_000] == 100 and at["DYNOTEARS", 10] == 0  # the values of fake_run
    for file in data.glob("paper_*.csv"):  # PCMCI+ is in no figure of the paper set
        assert "PCMCI" not in file.read_text(), file.name
    both = pd.read_csv(data / "appendix_imdp_by_graph_solved_vs_transitions.csv")
    assert {"DYNOTEARS-IMDP (ours)", "PCMCI+-IMDP (ours)"} <= set(both["label"])
    assert not list(data.glob("pooled_*.csv"))
    with pytest.raises(SystemExit):  # all_domains needs the three domains
        plot.main(["all_domains", "--results", str(tmp_path / "nowhere")])
    assert plot.main(["--list"]) == 0 and "vs_baselines/solved_vs_time" in capsys.readouterr().out
    only = ["--results", str(results), "--out", str(tmp_path / "one"), "--only", "paper", "tradeoff"]
    assert plot.main(["all_domains", *only]) == 0  # one set, one figure
    assert sorted(p.name for p in (tmp_path / "one" / "all_domains").iterdir()) == ["data", "paper"]
    assert sorted(p.name for p in (tmp_path / "one" / "all_domains" / "paper").iterdir()) == [
        "tradeoff.pdf",
        "tradeoff.png",
    ]


def test_every_method_has_one_colour_one_marker_and_a_line_of_its_own_where_colours_are_close():
    """The registry: 16 methods and 3 plug-in MDPs with a colour of their own (OWN_COLOUR), 19 pairwise
    distinct colours and markers, used by every figure; ours is the thickest line. The two forms of a
    graph have different colours for DYNOTEARS (ours), the full graph and the flat model; for the others
    the plug-in MDP has the colour of its graph and is told by its line."""
    assert len(plot.METHODS) == 16 and set(plot.GRAPHS) | set(plot.BASELINES) <= set(plot.METHODS)
    assert list(plot.OWN_COLOUR) == ["true graph MDP", "DYNOTEARS MDP", "full graph MDP", "flat MDP"]
    every = [*plot.METHODS, *(m for m in plot.OWN_COLOUR if m not in plot.REFERENCES)]
    colours = [plot.look_of(method)[0].lower() for method in every]
    markers = [plot.look_of(method)[1] for method in every]
    assert len(set(colours)) == 19 and len(set(markers)) == 18  # ours: one marker for its two forms
    # the two references: the gold of the true graph, hollow, dotted, under ours, told by their markers
    for ours in plot.SETS.values():
        first, second = (plot.method_line(m, ours) for m in plot.REFERENCES)
        assert first["color"] == second["color"] == plot.GOLD and first["mfc"] == second["mfc"] == "white"
        assert first["marker"] != second["marker"] and first["ls"] != second["ls"]
        assert max(first["zorder"], second["zorder"]) < min(
            plot.method_line(m, ours)["zorder"] for m in ("DYNOTEARS", "DYNOTEARS MDP")
        )
        assert max(first["lw"], second["lw"]) < plot.method_line("DYNOTEARS", ours)["lw"]
    with plot.references_named():  # their names in the pooled figures
        assert [plot.label_of(m, plot.SETS["paper"]) for m in plot.REFERENCES] == [
            "True-IMDP (reference)", "True-MDP (reference)"]  # fmt: skip
        assert plot.label_of("flat", plot.SETS["paper"]) == "Flat-IMDP"
    assert plot.look_of("DYNOTEARS")[1] == plot.look_of("DYNOTEARS MDP")[1] == "D"
    assert plot.METHODS["DYNOTEARS"][0] == plot.BLUE and plot.METHODS["true graph"][0] == plot.GOLD
    # the pairs of colours that are close (under 15 for normal vision or under 8 with a colour-vision
    # deficiency, OKLab x 100; measured with the validator of the palette): marker and line differ
    close = [("true graph", "sparse"), ("true graph", "SLF-Rmax"), ("true graph", "DYNOTEARS MDP"),
             ("DYNOTEARS", "flat"), ("DYNOTEARS", "DiFaRL"), ("DYNOTEARS", "flat MDP"), ("PCMCI+", "DiFaRL"),
             ("PCMCI+", "offline DQN"), ("PCMCI+", "full graph MDP"), ("sparse", "full graph"),
             ("sparse", "DYNOTEARS MDP"), ("full graph", "DiFaRL"), ("full graph", "DYNOTEARS MDP"),
             ("full graph", "full graph MDP"), ("PCMCI+ SPIBB", "flat MDP"), ("DiFaRL", "offline DQN"),
             ("offline CQL", "flat MDP"), ("full graph MDP", "flat MDP"),
             # and the two forms of a graph, whatever their colours
             ("flat", "flat MDP"), ("full graph", "full graph MDP")]  # fmt: skip
    for ours in plot.SETS.values():
        for a, b in close:
            la, lb = plot.method_line(a, ours), plot.method_line(b, ours)
            assert la["ls"] != lb["ls"] and la["marker"] != lb["marker"], (a, b)
    paper, appendix = plot.SETS["paper"], plot.SETS["appendix"]
    imdp, mdp = plot.method_line("DYNOTEARS", paper), plot.method_line("DYNOTEARS MDP", paper)
    assert imdp["color"] == plot.BLUE and mdp["color"] == plot.CORAL  # the two colours of ours
    assert imdp["ls"] == mdp["ls"] == "-" and imdp["marker"] == mdp["marker"] == "D" and "mfc" not in mdp
    assert plot.look_of("DYNOTEARS MDP") == (plot.CORAL, "D", "-")
    other = plot.method_line("PCMCI+ MDP", appendix)  # the other plug-in MDP of ours: the colour of its graph
    assert other["color"] == plot.RED and other["ls"] == plot.DASHDOT
    full, flat = plot.method_line("full graph MDP", paper), plot.method_line("flat MDP", paper)
    assert full["color"] == plot.ROSE != plot.method_line("full graph", paper)["color"] == plot.TEAL
    assert flat["color"] == plot.MAGENTA != plot.method_line("flat", paper)["color"] == plot.PURPLE
    assert full["ls"] == plot.DASHDOTDOT and flat["ls"] == plot.DASH3DOT and full["lw"] < mdp["lw"]
    assert plot.method_line("true graph MDP", paper)["color"] == plot.GOLD  # the others: their graph's
    assert plot.method_line("DYNOTEARS MDP", appendix)["color"] == plot.CORAL
    assert plot.method_line("DYNOTEARS", appendix)["color"] == plot.BLUE
    used = {plot.look_of(method)[0].lower() for method in every}  # a ring of tradeoff_pareto marks no method
    assert plot.SAFEST != plot.MOST_SOLVED and not {plot.SAFEST.lower(), plot.MOST_SOLVED.lower()} - {
        plot.INK.lower(), plot.INK2.lower()} and plot.MOST_SOLVED.lower() not in used  # fmt: skip
    assert imdp["lw"] == mdp["lw"] == max(plot.method_line(m, paper)["lw"] for m in plot.METHODS)
    assert (imdp["label"], mdp["label"]) == ("CD-IMDP (ours)", "CD-MDP (ours)")
    assert plot.method_line("PCMCI+ MDP", appendix)["label"] == "PCMCI+-MDP (ours)"
    assert plot.method_line("DYNOTEARS", appendix)["label"] == "DYNOTEARS-IMDP (ours)"
    assert plot.method_line("PCMCI+", appendix)["lw"] == imdp["lw"]  # both discovery methods as ours
    names = [plot.label_of(m, paper) for m in ("true graph", "true graph MDP", "sparse", "full graph", "flat",
                                               "flat MDP", "FOCUS", "flat SPIBB")]  # fmt: skip
    assert names == ["True-IMDP", "True-MDP", "Sparse-IMDP", "Full-IMDP", "Flat-IMDP", "Flat-MDP", "FOCUS",
                     "flat SPIBB"]  # fmt: skip
    assert plot.drawn(plot.GRAPHS, paper) == [g for g in plot.GRAPHS if g != "PCMCI+"]
    assert plot.drawn(plot.GRAPHS, appendix) == plot.GRAPHS
    for model, method in plot.EXPLORED.items():  # the exploratory figures: the same colours
        assert plot.COLOR[model] == plot.METHODS[method][0]
    to = plot.FigureSet("paper", paper, Path("."))  # the families of the pooled figures, in the two sets
    wide = plot.FigureSet("appendix", appendix, Path("."))
    assert "sparse" not in to.methods(plot.IMDP_BY_GRAPH) and "sparse" in wide.methods(plot.IMDP_BY_GRAPH)
    for family in (plot.MDP_BY_GRAPH, plot.VS_BASELINES, plot.TRADEOFF):
        assert "full graph MDP" in to.methods(family) and "full graph MDP" in wide.methods(family)
    assert {"flat", "flat MDP"} <= set(to.methods(plot.VS_BASELINES)) and "flat MDP" in plot.MDP_BY_GRAPH
    assert "flat MDP" in plot.TRADEOFF and plot.TRADEOFF == plot.VS_BASELINES
    assert plot.VS_BASELINES[:2] == plot.REFERENCES == ["true graph", "true graph MDP"]
    assert plot.PAIR_PANELS == {"a": plot.VS_BASELINES, "b": plot.TRADEOFF}  # one setting per panel
    assert "full graph" in to.methods(plot.VS_BASELINES) and "full graph" in to.methods(plot.TRADEOFF)


def test_spibb_on_a_learned_graph_is_a_baseline_with_a_colour_of_its_own():
    """CD-SPIBB is not ours: its own colour and marker, a thin line, a name without "(ours)", and it is
    left out with its discovery method."""
    paper, appendix = plot.SETS["paper"], plot.SETS["appendix"]
    ours = plot.method_line("DYNOTEARS", paper)
    line = plot.method_line("DYNOTEARS SPIBB", paper)
    assert line["label"] == "CD-SPIBB" and line["color"] not in (plot.BLUE, plot.RED)
    assert line["lw"] < ours["lw"] and line["lw"] == plot.method_line("SL-SPIBB", paper)["lw"]
    assert line["marker"] != ours["marker"]
    names = [plot.label_of(m, appendix) for m in plot.SPIBB_ON]
    assert names == ["DYNOTEARS-SPIBB", "PCMCI+-SPIBB"]
    assert plot.split_form("DYNOTEARS SPIBB") == ("DYNOTEARS SPIBB", None)  # a baseline: no form of a graph
    for name, discovery in plot.SETS.items():
        to = plot.FigureSet(name, discovery, Path("."))
        assert not any(to.is_ours(m) for m in plot.SPIBB_ON)
    assert plot.drawn(list(plot.SPIBB_ON), paper) == ["DYNOTEARS SPIBB"]
    assert plot.drawn(list(plot.SPIBB_ON), appendix) == list(plot.SPIBB_ON)
    assert plot.label_of("true graph SPIBB", paper) == "True-SPIBB"  # on the true graph: a form of it


def test_slf_rmax_and_met_rmax_are_read_at_the_papers_thresholds_only(tmp_path):
    """The pooled figures take SLF-Rmax and Met-Rmax from the plug-in MDP of the sources strehl and met when
    the run used the thresholds of the papers' experiments (protocol.KWIK_PAPER); a run at the theorems'
    thresholds (m "calibrated" in its meta.json) gives no row."""
    fake_run(tmp_path, "sepsis", "truegraph", "2026-11-05_22-52-41", {"2": 4, "3": 8})
    arms = {"strehl": "Strehl-KWIK -> MDP (plug-in)", "met": "Met-Rmax-KWIK -> MDP (plug-in)"}
    options = {
        "strehl": {"paper": {"k": 2, "m": 30, "eps1": 0.2}, "theorem": {"k": 2, "m": "calibrated"}},
        "met": {"paper": {"D": 2, "m": 50, "m_sub": 50}, "theorem": {"D": 2, "m": 50, "m_sub": "calibrated"}},
    }
    stamp = "2026-11-17_09-00-00"
    for source, level in (("strehl", "paper"), ("met", "theorem")):
        rows = [{"size": "2", "size_label": "vitals=2", "n_states": 8, "n_observed_states": 4, "seed": seed,
                 "n_transitions": 1000, "method": arms[source], "status": "ok", "promise": 1.0, "true": 4.9,
                 "V_star_obs": 5.0, "V_random": 2.0, "safe": False, "discovery_s": 1.0, "solve_s": 2.0}
                for seed in range(4)]  # fmt: skip
        out = tmp_path / "sepsis" / f"sepsis_{source}_{stamp}" / "knob=a" / "uniform"
        out.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        meta = {"config": {"methods": {arms[source]: {"learner": "kwik",
                                                      "discovery_options": options[source][level]}}}}
        (out / "meta.json").write_text(json.dumps(meta))
    runs = plot.load(tmp_path, "sepsis")
    assert set(runs.loc[runs["method"].isin(plot.KWIK_ARMS), "kwik_level"]) == {"paper", "theorem"}
    table = plot.instances_of(runs, "sepsis")
    kwik = table[table["method"].isin(plot.KWIK_METHODS)]
    assert set(kwik["method"]) == {"SLF-Rmax"} and len(kwik) == 4  # Met-Rmax ran at the theorems' level
    assert kwik["optimality"].round(6).eq(round(2.9 / 3, 6)).all() and not kwik["safe"].any()
    # rows named SLF-Rmax in another source (the old copies of the reward ablation) are not instances
    stray = runs.assign(method=runs["method"].replace(arms["strehl"], "SLF-Rmax"))
    assert not plot.instances_of(stray, "sepsis")["method"].isin(plot.KWIK_METHODS).any()

