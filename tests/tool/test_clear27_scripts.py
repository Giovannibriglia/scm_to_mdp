"""The clear27 scripts (experiments/clear27): one per (domain, graph source), each with the oracle and the
same data protocol; every graph source gets the same compilations (point 2)."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "clear27"))
protocol = importlib.import_module("protocol")
sys.path.pop(0)

DOMAINS = ("sysadmin", "sepsis", "drone", "aircraft")
POINT_2 = (
    [f"IMDP ({r})" for r in ("IA", "McCormick", "exact")]
    + [f"MDP ({c}, {r})" for c in ("worst", "best") for r in ("IA", "McCormick", "exact")]
    + ["MDP (plug-in)", "Factored SPIBB"]
)


def test_every_script_exists_and_names_its_grid():
    for d in DOMAINS:
        for s in protocol.SOURCES:
            text = (ROOT / "experiments" / "clear27" / f"{d}_{s}.py").read_text()
            assert f'experiment_grid("{d}", "{s}")' in text


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_data_protocol(domain):
    g = protocol.experiment_grid(domain, "truegraph")
    assert g.n_transitions[0] == 1 and g.n_transitions[-1] == 500_000 and len(g.n_transitions) == 18
    assert g.seeds == list(range(10)) and [p.label for p in g.policies] == ["uniform"]
    assert g.out_dir == ROOT / "results" / "clear27" / f"{domain}_truegraph"
    assert "11" not in map(str, g.sizes) and "18x13" not in map(str, g.sizes)


@pytest.mark.parametrize("source", [s for s in protocol.SOURCES if s != "sparse"])
def test_every_source_has_the_oracle_and_point_2(source):
    methods = protocol.experiment_grid("sysadmin", source).methods
    assert next(iter(methods)) == "oracle -> MDP"
    names = {
        "truegraph": "true graph", "dynotears": "DYNOTEARS", "dynotears_lagged": "DYNOTEARS (lagged)",
        "pcmci": "PCMCI+", "pcmci_lagged": "PCMCI+ (lagged)", "fullfact": "full-fact",
    }  # fmt: skip
    if source in names:
        assert set(methods) == {"oracle -> MDP"} | {f"{names[source]} -> {c}" for c in POINT_2}
    if source in ("strehl", "met"):
        title = "Strehl" if source == "strehl" else "Met-Rmax"
        for hull in ("fixed", "accepted"):  # without KWIK: no point model, so the set arms only
            assert sum(n.startswith(f"{title} ({hull} hull)") for n in methods) == 9
        assert sum(n.startswith(f"{title}-KWIK") for n in methods) == 11


def test_learned_graphs_pay_the_union_bound_and_lagged_drops_same_step():
    m = protocol.experiment_grid("sysadmin", "pcmci").methods
    assert m["PCMCI+ -> IMDP (IA)"].learning_options["n_candidates"] == "inputs+next"
    lag = protocol.experiment_grid("sysadmin", "pcmci_lagged").methods
    assert lag["PCMCI+ (lagged) -> IMDP (IA)"].learning_options["n_candidates"] == "inputs"
    assert lag["PCMCI+ (lagged) -> IMDP (IA)"].discovery_options["same_step"] is False
    true = protocol.experiment_grid("sysadmin", "truegraph").methods
    assert true["true graph -> IMDP (IA)"].learning_options["n_candidates"] == 1
