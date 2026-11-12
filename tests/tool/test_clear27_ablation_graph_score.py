"""The ablation experiments/clear27/ablations/ablation_graph_score: every graph our constraints allow has a
number, its BIC is the sum of its families', and a run writes one row per graph and its figures."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "experiments" / "clear27" / "ablations" / "ablation_graph_score"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"ablation_graph_score_{name}", FOLDER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# run.py's jobs run in worker processes, which import its function by module name: the folder stays on
# the path, under a name no other module has (plot.py is loaded by its file: clear27 has a plot.py too)
sys.path.insert(0, str(FOLDER))
run = importlib.import_module("run")


def test_a_graph_is_a_number_and_back():
    assert [run.n_graphs(n) for n in (2, 3, 4)] == [64, 4096, 1_048_576]  # 2^(N (N+1))
    state_vars, inputs = ["m0", "m1"], ["m0", "m1", "reboot"]
    assert run.edge_list(state_vars, inputs)[:4] == [
        ("m0", "m0"),
        ("m0", "m1"),
        ("m0", "reboot"),
        ("m1", "m0"),
    ]
    assert run.describe_edges(2) == [list(e) for e in run.edge_list(state_vars, inputs)]
    assert run.parents_of(0, state_vars, inputs) == {"m0": (), "m1": ()}
    assert run.parents_of(63, state_vars, inputs) == {"m0": tuple(inputs), "m1": tuple(inputs)}
    assert run.parents_of(0b100_001, state_vars, inputs) == {"m0": ("m0",), "m1": ("reboot",)}
    seen = {run.graph_of(run.parents_of(g, state_vars, inputs), state_vars, inputs) for g in range(64)}
    assert seen == set(range(64))
    with_same_step = {"m0": ("m0", "m1'"), "m1": ()}  # a same-step parent is not an edge of these graphs
    assert run.graph_of(with_same_step, state_vars, inputs) == 1


def test_a_run_writes_every_graph_and_its_figures(tmp_path, capsys):
    """N=2, one seed, one number of transitions: 64 rows, the marks, a resume that has nothing to do, and
    the figures with their numbers."""
    options = ["--machines", "2", "--transitions", "200", "--seeds", "0", "--workers", "1"]
    assert run.main([*options, "--results", str(tmp_path)]) == 0
    (folder,) = tmp_path.glob("sysadmin_*")
    config = json.loads((folder / "config.json").read_text())
    assert config["machines"] == [2] and len(config["edges"]["2"]) == 6
    graphs = pd.read_csv(folder / "N=2" / "graphs" / "seed=0_n=200.csv")
    assert list(graphs["graph"]) == list(range(64)) and set(graphs["status"]) == {"ok"}
    assert list(graphs.columns) == run.GRAPH_COLUMNS
    (dataset,) = pd.read_csv(folder / "N=2" / "datasets.csv").to_dict("records")
    assert dataset["graphs"] == 64 and dataset["true_graph"] == 63  # a ring of 2: every input is a parent
    assert (dataset["empty_graph"], dataset["full_graph"]) == (0, 63)
    for score in run.SCORES:  # BIC, AIC, BDeu: each has its best graph, and a value for every graph
        assert dataset[f"best_{score}_graph"] == graphs.loc[graphs[score].idxmax(), "graph"]
        assert graphs[score].notna().all()
    assert (graphs["aic"] - (graphs["loglik"] - graphs["n_params"])).abs().max() < 1e-9
    assert (graphs["bdeu"] < 0).all()  # a log marginal likelihood of discrete data
    assert dataset["pcmci_status"] == dataset["dynotears_status"] == "ok"
    # more parents never lower the likelihood, and the BIC is the likelihood minus its penalty
    by_graph = graphs.set_index("graph")
    assert by_graph.loc[63, "loglik"] == pytest.approx(graphs["loglik"].max())
    assert by_graph.loc[0, "loglik"] == pytest.approx(graphs["loglik"].min())
    assert (graphs["loglik"] - graphs["bic"] > 0).all() and by_graph.loc[0, "n_params"] == 2
    assert (
        by_graph.loc[63, "missing"] == 0 and by_graph.loc[0, "missing"] == 6 and (graphs["extra"] == 0).all()
    )
    true_graph = by_graph.loc[63]
    assert (
        true_graph["covered"] and true_graph["held"] and true_graph["promise"] <= true_graph["value"] + 1e-9
    )

    capsys.readouterr()
    assert run.main([*options, "--results", str(tmp_path), "--resume"]) == 0  # nothing left to run
    assert "0 of 1 datasets to run" in capsys.readouterr().out and len(list(tmp_path.glob("sysadmin_*"))) == 1

    plot = load("plot")
    assert plot.main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    for name in plot.FIGURES:
        assert (tmp_path / "out" / "N=2" / f"{name}.pdf").stat().st_size > 0
        assert (tmp_path / "out" / "data" / f"N=2_{name}.csv").stat().st_size > 0
    with pytest.raises(SystemExit):
        plot.main(["--results", str(tmp_path), "--machines", "5"])  # no such size in the run
