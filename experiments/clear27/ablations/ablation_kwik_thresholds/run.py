"""Ablation: the thresholds of the two KWIK learners, SLF-Rmax (Strehl, Diuk, Littman, AAAI 2007) and
Met-Rmax (Diuk, Li, Leffler, ICML 2009), on SysAdmin, Drone and Sepsis (S2, S8, S9).

The main runs give the two learners the thresholds of their papers' experiments (protocol.KWIK_PAPER:
SLF-Rmax m = 30, eps1 = 0.2; Met-Rmax m = m_sub = 50, eps = 0.2; sources strehl and met). This ablation
runs them at their theorems' level (protocol.KWIK_THEOREM_SOURCES: strehl_theorem, met_theorem): m
calibrated at delta, which leaves almost every pair "don't know" at the data sizes of the experiments.

    learners  four: SLF-Rmax and Met-Rmax, each with the papers' thresholds (the main runs) and the
              theorems' (this ablation)
    form      the learner's own structure and predictions, planned by value iteration ("<learner>-KWIK ->
              MDP (plug-in)": SLF-Rmax and Met-Rmax in the figures of the paper); --every-compilation:
              every arm of the learner in clear27 (protocol.kwik_arms)
    data, limits, seeds, sizes and numbers of transitions: clear27's (protocol.py)

Run from the repository root:
    bash experiments/clear27/ablations/ablation_kwik_thresholds/run_all.sh      # the three domains
    .venv/bin/python experiments/clear27/ablations/ablation_kwik_thresholds/run.py sysadmin drone sepsis
    .venv/bin/python experiments/clear27/ablations/ablation_kwik_thresholds/run.py drone --sources met_theorem
    --sizes 3-1 4-1   --transitions 100 1000 500000   --seeds 0 1    a smaller grid (values of the paper's)
    --resume          continue the latest run of every source
    --workers N|auto  --core-share P   as in clear27 (SCM_TO_MDP_WORKERS, SCM_TO_MDP_CORE_SHARE)
The twins with the papers' thresholds are sources of the main runs (strehl, met).
Then the figures: .venv/bin/python experiments/clear27/ablations/ablation_kwik_thresholds/plot.py

Output: results/clear27/ablations/ablation_kwik_thresholds/<domain>/<domain>_<source>_<date>_<time>/, a run
folder of clear27 (grid.json, <knob>/uniform/results.csv and meta.json) with a provenance.json.
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

NAME = "ablation_kwik_thresholds"
RESULTS = common.ROOT / "results" / protocol.EXPERIMENT / "ablations" / NAME
DOMAINS = ("sysadmin", "drone", "sepsis")
SOURCES = tuple(protocol.KWIK_THEOREM_SOURCES)  # strehl_theorem, met_theorem: the theorems' thresholds
ORACLE = "oracle -> MDP"
# source -> the method of clear27 drawn as the learner in the figures (its own predictions, planned by VI)
STREHL, MET = "Strehl-KWIK -> MDP (plug-in)", "Met-Rmax-KWIK -> MDP (plug-in)"
ARM = {"strehl": STREHL, "met": MET, "strehl_theorem": STREHL, "met_theorem": MET}


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
    """clear27's grid of a source at the theorems' thresholds (protocol.experiment_grid), with the oracle and
    the plug-in MDP of the learner (every_compilation: all its methods), in the ablation's folder; the
    models are not saved."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    base = protocol.experiment_grid(domain, source)
    names = list(base.methods) if every_compilation else [ORACLE, ARM[source]]
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
    twin = protocol.KWIK_THEOREM_SOURCES[source]  # the source of the main runs, at the papers' thresholds
    record = {
        "ablation": NAME, "domain": domain, "source": source, "origin": "run",
        "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "commit": _commit(),
        "command": " ".join(["experiments/clear27/ablations/ablation_kwik_thresholds/run.py", *argv]),
        "methods": list(grid.methods), "sizes": [str(s) for s in grid.sizes],
        "n_transitions": list(grid.n_transitions), "seeds": list(grid.seeds),
        "papers_twin": f"results/{protocol.EXPERIMENT}/{domain}/{domain}_{twin}_*",
    }  # fmt: skip
    path.write_text(json.dumps(record, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="clear27 ablation: SLF-Rmax and Met-Rmax at their theorems' thresholds."
    )
    parser.add_argument("domains", nargs="+", choices=list(DOMAINS))
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--sizes", nargs="+", metavar="S", help="sizes of the domain (default: the paper's)")
    parser.add_argument("--transitions", nargs="+", metavar="N", help="data sizes (default: the paper's)")
    parser.add_argument("--seeds", nargs="+", metavar="SEED", help="seeds (default: the paper's 0 .. 9)")
    parser.add_argument("--every-compilation", action="store_true",
                        help="every arm of the learner in clear27, not only its plug-in MDP")  # fmt: skip
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
