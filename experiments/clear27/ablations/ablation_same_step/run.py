"""Ablation: what does forbidding arrows inside a step buy? PCMCI+ and DYNOTEARS with and without
same-step arrows, on SysAdmin, Drone and Sepsis (S2, S8, S9).

The setting of the paper assumes that every parent of a next-state variable is a variable of the current
step (no arrow between two next-state variables), and the main runs give that assumption to the two
discovery methods (clear27's sources dynotears_lagged and pcmci_lagged). This ablation runs their twins
without it (protocol.SAME_STEP_SOURCES: dynotears, pcmci): the discovery may then return arrows inside
a step, and the union bound of the intervals is over the larger family of parent sets.

    learners  four: DYNOTEARS and PCMCI+, each without same-step arrows (the main runs) and with them
              allowed (this ablation)
    forms     each learned graph as the box interval MDP ("IMDP (IA)") and as the plug-in MDP
              ("MDP (plug-in)"); --every-compilation: every compilation of clear27 instead (protocol.py,
              "Point 2"), as the main runs had
    data, limits, seeds, sizes and numbers of transitions: clear27's (protocol.py), so that a run of
              this script repeats the rows of the main protocol (checked on 2026-11-15 against the
              runs of 2026-11-05, on a slice of each domain: the same status, graph, promise and
              safety in every row, and the same value in all but one)

The methods, the data and the evaluation are built by protocol.py (experiment_grid): this script only
chooses the sources, keeps the two forms and sets the folder. The two sources run in one pool of workers
(tool/sweep.py: a free worker takes the next (size, seed) job of either).

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_same_step/run.py sysadmin
    .venv/bin/python experiments/clear27/ablations/ablation_same_step/run.py sysadmin drone sepsis
    .venv/bin/python experiments/clear27/ablations/ablation_same_step/run.py drone --sources pcmci
    --sizes 3-1 4-1   --transitions 100 1000 500000   --seeds 0 1    a smaller grid (values of the paper's)
                      The data of a (size, seed) are collected once, at the largest number of transitions
                      of the run, and a smaller number is their prefix (tool/sweep.py): a smaller grid has
                      the paper's datasets only if it keeps the paper's largest number, 500,000
    --resume          continue the latest run of every source
    --workers N|auto  --core-share P   as in clear27 (SCM_TO_MDP_WORKERS, SCM_TO_MDP_CORE_SHARE)
One launch script per domain (the run, then the figures): run_sysadmin.sh, run_drone.sh, run_sepsis.sh.
The twins without same-step arrows are sources of the main runs:
    bash experiments/clear27/run_domain.sh <domain> dynotears_lagged pcmci_lagged
Then the figures: .venv/bin/python experiments/clear27/ablations/ablation_same_step/plot.py

Output: results/clear27/ablations/ablation_same_step/<domain>/<domain>_<source>_<date>_<time>/, a run
folder of clear27 (grid.json, <knob>/uniform/results.csv and meta.json) with a provenance.json: how the
folder was made (by this script, or moved from the main results, where the two sources ran until
2026-11-15).
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEAR27 = HERE.parents[1]
sys.path.insert(0, str(CLEAR27))  # experiments/clear27/: protocol.py
sys.path.insert(0, str(CLEAR27.parent))  # experiments/: common.py, domains.py

import common  # noqa: E402
import protocol  # noqa: E402

from scm_to_mdp.tool.runs import STAMP, add_core_share, runs_of, set_core_share  # noqa: E402

NAME = "ablation_same_step"
RESULTS = common.ROOT / "results" / protocol.EXPERIMENT / "ablations" / NAME
DOMAINS = ("sysadmin", "drone", "sepsis")
SOURCES = tuple(protocol.SAME_STEP_SOURCES)  # dynotears, pcmci: with same-step arrows allowed
ORACLE = "oracle -> MDP"
# source -> the name of its graph in the methods of clear27 ("<graph> -> IMDP (IA)")
GRAPH = {"dynotears": "DYNOTEARS", "pcmci": "PCMCI+",
         "dynotears_lagged": "DYNOTEARS (lagged)", "pcmci_lagged": "PCMCI+ (lagged)"}  # fmt: skip
FORMS = {"IMDP": "IMDP (IA)", "MDP": "MDP (plug-in)"}  # form -> its compilation in clear27


def arm(source: str, form: str) -> str:
    """The method of clear27 of a source in a form: arm("pcmci", "IMDP") is "PCMCI+ -> IMDP (IA)"."""
    return f"{GRAPH[source]} -> {FORMS[form]}"


def chosen(given, allowed, what: str) -> list:
    """The grid's own values named by the user (compared as text: a size may be 5 or "5-1")."""
    if not given:
        return list(allowed)
    by_text = {str(x): x for x in allowed}
    wrong = [g for g in given if str(g) not in by_text]
    if wrong:
        raise ValueError(f"{what} must be among {list(by_text)}, not {wrong}")
    return [by_text[str(g)] for g in given]


def ablation_grid(domain: str, source: str, results: Path = RESULTS, sizes=None, n_transitions=None,
                  seeds=None, every_compilation: bool = False):  # fmt: skip
    """clear27's grid of a same-step source (protocol.experiment_grid), with the oracle and the two forms
    (every_compilation: all its methods), in the ablation's folder; the models are not saved."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    base = protocol.experiment_grid(domain, source)
    names = list(base.methods) if every_compilation else [ORACLE, *(arm(source, form) for form in FORMS)]
    return dataclasses.replace(
        base,
        methods={name: base.methods[name] for name in names},
        sizes=chosen(sizes, base.sizes, "--sizes"),
        n_transitions=chosen(n_transitions, base.n_transitions, "--transitions"),
        seeds=chosen(seeds, base.seeds, "--seeds"),
        save_models="none",
        out_dir=results / domain / f"{domain}_{source}",
    )


def run_folder(base: Path, resume: bool) -> Path:
    """The latest <base>_<date>_<time> folder on a resume (when there is one), else a new one."""
    found = runs_of(base) if resume else []
    if found:
        print(f"resuming {found[-1]}")
        return found[-1]
    new = Path(f"{base}_{datetime.now().strftime(STAMP)}")
    print(f"starting {new}")
    return new


def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=common.ROOT, check=False)  # fmt: skip
        return out.stdout.strip() or "?"
    except OSError:
        return "?"


def write_provenance(folder: Path, domain: str, source: str, grid, argv: list) -> None:
    """provenance.json of a run made by this script (a resumed run keeps the one it has)."""
    path = folder / "provenance.json"
    if path.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    twin = protocol.SAME_STEP_SOURCES[source]  # the source of the main runs without same-step arrows
    record = {
        "ablation": NAME, "domain": domain, "source": source, "origin": "run",
        "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "commit": _commit(),
        "command": " ".join(["experiments/clear27/ablations/ablation_same_step/run.py", *argv]),
        "methods": list(grid.methods), "sizes": [str(s) for s in grid.sizes],
        "n_transitions": list(grid.n_transitions), "seeds": list(grid.seeds),
        "lagged_twin": f"results/{protocol.EXPERIMENT}/{domain}/{domain}_{twin}_*",
    }  # fmt: skip
    path.write_text(json.dumps(record, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="clear27 ablation: PCMCI+ and DYNOTEARS with same-step arrows allowed."
    )
    parser.add_argument("domains", nargs="+", choices=list(DOMAINS))
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--sizes", nargs="+", metavar="S", help="sizes of the domain (default: the paper's)")
    parser.add_argument("--transitions", nargs="+", metavar="N", help="data sizes (default: the paper's)")
    parser.add_argument("--seeds", nargs="+", metavar="SEED", help="seeds (default: the paper's 0 .. 9)")
    parser.add_argument("--every-compilation", action="store_true",
                        help="every compilation of clear27 on the two graphs, not the two forms")  # fmt: skip
    parser.add_argument("--resume", action="store_true", help="continue the latest run of every source")
    parser.add_argument("--workers", metavar="N", help="parallel jobs: a number or 'auto'")
    add_core_share(parser)
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    set_core_share(parser, args.core_share)
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    if len(args.domains) > 1 and args.sizes:
        parser.error("--sizes names the sizes of one domain: give one domain")

    from scm_to_mdp.tool.grid import experiments, write_grid_file
    from scm_to_mdp.tool.sweep import run_sweeps

    grids = []
    for domain in args.domains:
        for source in args.sources:
            try:
                base = ablation_grid(domain, source, args.results, args.sizes, args.transitions, args.seeds,
                                     args.every_compilation)  # fmt: skip
            except ValueError as err:
                parser.error(str(err))
            folder = run_folder(base.out_dir, args.resume)
            grids.append((domain, source, dataclasses.replace(base, out_dir=folder)))
    exps, ranks = [], []
    for domain, source, grid in grids:
        write_provenance(grid.out_dir, domain, source, grid, list(sys.argv[1:] if argv is None else argv))
        write_grid_file(grid)
        mine = experiments(grid)
        exps += mine
        ranks += [args.domains.index(domain)] * len(mine)
    title = f"{NAME}: {' + '.join(args.domains)}"
    print(f"\n===== {title}: {len(grids)} sources ({', '.join(args.sources)}), {len(exps)} settings =====")
    # each source in its own folder; a free worker takes a job of the first domain that has one waiting
    run_sweeps(exps, resume=True, title=f"{title} ({len(grids)} sources)", ranks=ranks)
    for _, _, grid in grids:
        print(f"results: {grid.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
