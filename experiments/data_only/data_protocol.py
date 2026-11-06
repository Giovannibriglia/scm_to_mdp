"""data_only: the pipeline with only the data, against clear27's configuration (SysAdmin, a small grid).

"data-only" (learning_options data_only=True, scm_learning/data_only.py): no value domains, no support and no
terminal states are given (and no done / terminated / truncated flags); the reward function is known and enters
only value iteration. Values never seen and states never left are sinks at V_min = min(r_min, 0) / (1 - gamma),
an unseen parent row sends its pair to a sink, a value not observed at a seen row gets only the residual mass;
the promise stays a lower bound (badings_claude, agarwal_claude, 2026-11-06). The same rules for every graph:

    true graph (reference), full-fact, sparse (fixed in advance), DYNOTEARS (lagged), PCMCI+ (lagged), flat,
    Strehl et al. (2007) and Met-Rmax with our S4 (fixed and accepted hulls; their own KWIK modes need the
    value domains, as in their papers, and are left out)

Arms per graph: IMDP (IA), IMDP (exact), MDP (plug-in), each as in clear27 and "(..., data-only)"; flat: its
IMDP and plug-in MDP. Data, limits and evaluation: clear27 (experiments/clear27/protocol.py), on SysAdmin
N = 2..8, 10 seeds, 10 .. 100k transitions. Run: .venv/bin/python experiments/data_only/run.py [--resume].
Output: results/data_only/sysadmin_<source>_<date>_<time>/.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clear27"))  # clear27's protocol

import protocol as clear27  # noqa: E402

common = clear27.common
EXPERIMENT = "data_only"
DOMAIN = "sysadmin"
SIZES = [2, 3, 4, 5, 6, 7, 8]
SEEDS = list(range(10))
N_TRANSITIONS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
SOURCES = ("dynotears_lagged", "pcmci_lagged", "fullfact", "sparse", "flat", "truegraph", "strehl", "met")
KEEP = ("IMDP (IA)", "IMDP (exact)", "MDP (plug-in)")
FLAT_KEEP = ("flat IMDP", "flat -> MDP (plug-in)")
LABEL = "data-only"


def data_only_name(name: str) -> str:
    """'X -> IMDP (IA)' -> 'X -> IMDP (IA, data-only)'; 'flat IMDP' -> 'flat IMDP (data-only)'."""
    return f"{name[:-1]}, {LABEL})" if name.endswith(")") else f"{name} ({LABEL})"


def methods(source: str) -> dict:
    """The oracle, then every kept arm of the source and its data-only twin."""
    every = clear27.source_methods(source, clear27.DOMAINS[DOMAIN])
    if source == "flat":
        kept = {n: every[n] for n in FLAT_KEEP}
    else:
        kept = {n: m for n, m in every.items() if any(n.endswith(f"-> {k}") for k in KEEP) and "-KWIK" not in n}
    out = {"oracle -> MDP": every["oracle -> MDP"]}
    for name, m in kept.items():
        out[name] = m
        out[data_only_name(name)] = dataclasses.replace(m, learning_options={**m.learning_options, "data_only": True})
    return out


def experiment_grid(source: str):
    base = clear27.experiment_grid(DOMAIN, source)
    smoke = clear27.SMOKE
    return dataclasses.replace(
        base,
        sizes=SIZES[:1] if smoke else SIZES,
        seeds=[0] if smoke else SEEDS,
        n_transitions=clear27.SMOKE_TRANSITIONS if smoke else N_TRANSITIONS,
        methods=methods(source),
        save_models="none",
        out_dir=common.ROOT / "results" / (EXPERIMENT + ("_smoke" if smoke else "")) / f"{DOMAIN}_{source}",
    )
