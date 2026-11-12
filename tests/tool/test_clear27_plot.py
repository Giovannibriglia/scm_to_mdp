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
    assert "skipped paper/baselines: no run of sota yet" in capsys.readouterr().out
    for name in written:
        assert (out / "grid" / "paper" / f"{name}.pdf").stat().st_size > 0
        assert (out / "grid" / "paper" / f"{name}.png").stat().st_size > 0
    drawn = pd.read_csv(out / "grid" / "data" / "value.csv")
    assert set(drawn["method"]) == {label for _, _, label, _, _ in plot.SERIES}
    assert drawn[drawn["n_transitions"] == 100_000]["mean"].between(0.85, 1.0).all()
    assert plot.main(["--list"]) == 0 and "9_transitions_needed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        plot.main(["nowhere", "--results", str(tmp_path / "results")])
    with pytest.raises(SystemExit):
        plot.main(["grid", "--results", str(tmp_path / "results"), "--only", "no_such_figure"])


def test_sepsis_and_drone_are_drawn_with_the_box(tmp_path, capsys):
    """The exact solver stops beyond their smallest sizes, so their certified solver is the box and every
    size is kept; --solver (the solver argument) still overrides the domain's choice."""
    assert plot.DOMAIN_SETTINGS["sepsis"]["solver"] == plot.DOMAIN_SETTINGS["drone"]["solver"] == "IMDP (IA)"
    assert plot.DOMAIN_SETTINGS["drone"]["panels"] == ["3-1", "5-1", "7-1", "10-1", "12-1"]
    assert "solver" not in plot.DOMAIN_SETTINGS["sysadmin"]  # the default: exact
    sizes = {"3-1": 27, "5-1": 125, "7-1": 343, "10-1": 1000, "12-1": 1728}
    for source in SOURCES:
        fake_run(tmp_path, "drone", source, "2026-11-05_22-52-41", sizes, stop_from=1)  # exact: 3-1 only
    plot.plot_domain("drone", tmp_path, tmp_path / "out", only=["value"])
    assert "certified solver: IMDP (IA)" in capsys.readouterr().out
    drawn = pd.read_csv(tmp_path / "out" / "drone" / "data" / "value.csv")
    assert set(drawn["size"]) == set(sizes)
    plot.plot_domain("drone", tmp_path, tmp_path / "exact", solver="IMDP (exact)", only=["value"])
    exact = pd.read_csv(tmp_path / "exact" / "drone" / "data" / "value.csv")
    assert set(exact[exact["method"] == "true graph"]["size"]) == {"3-1"}
