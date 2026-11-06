"""seen_only: does causal discovery remove correlations from the MDP it builds? (SysAdmin, a small grid.)

With "seen pairs only" (solver_options seen_only=True, tool/methods.py) the max over actions runs only over
the pairs (s, a) whose every factor's parent row has data, and a state with none takes the value floor
r_min / (1 - gamma) (solving/stopping.greedy: the promise stays a sound lower bound). A factor with
spurious parents has more rows, so fewer pairs are seen at the same data: the policy has fewer actions and
more states sit at the floor. The graphs compared, each with and without the rule:

    DYNOTEARS (lagged), PCMCI+ (lagged)   discovery, no same-step links (clear27's sources)
    full-fact, sparse                     graphs fixed in advance (every input; a fixed sparse set)
    flat                                  no structure: a row is the pair (s, a) itself
    true graph                            the reference

Arms per graph: IMDP (IA), IMDP (exact), MDP (plug-in), each also "(..., seen only)"; flat: its IMDP
and plug-in MDP. Data, limits and evaluation: clear27 (experiments/clear27/protocol.py), on SysAdmin
N = 2..6, 5 seeds, 10 .. 100k transitions. Run: .venv/bin/python experiments/seen_only/run.py [--resume].
Output: results/seen_only/sysadmin_<source>_<date>_<time>/.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clear27"))  # clear27's protocol

import protocol as clear27  # noqa: E402

common = clear27.common
EXPERIMENT = "seen_only"
DOMAIN = "sysadmin"
SIZES = [2, 3, 4, 5, 6]
SEEDS = list(range(5))
N_TRANSITIONS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
SOURCES = ("dynotears_lagged", "pcmci_lagged", "fullfact", "sparse", "flat", "truegraph")
KEEP = ("IMDP (IA)", "IMDP (exact)", "MDP (plug-in)")  # the arms of a graph source
FLAT_KEEP = ("flat IMDP", "flat -> MDP (plug-in)")
SEEN = {"seen_only": True}


def seen_name(name: str) -> str:
    """'X -> IMDP (IA)' -> 'X -> IMDP (IA, seen only)'; 'flat IMDP' -> 'flat IMDP (seen only)'."""
    return f"{name[:-1]}, seen only)" if name.endswith(")") else f"{name} (seen only)"


def methods(source: str) -> dict:
    """The oracle, then every kept arm of the source and its seen-only twin."""
    every = clear27.source_methods(source, clear27.DOMAINS[DOMAIN])
    if source == "flat":
        kept = {n: every[n] for n in FLAT_KEEP}
    else:
        kept = {n: m for n, m in every.items() if any(n.endswith(f"-> {k}") for k in KEEP)}
    out = {"oracle -> MDP": every["oracle -> MDP"]}
    for name, m in kept.items():
        out[name] = m
        out[seen_name(name)] = dataclasses.replace(m, solver_options={**m.solver_options, **SEEN})
    return out


def experiment_grid(source: str):
    base = clear27.experiment_grid(DOMAIN, source)  # clear27's data, limits and evaluation
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
