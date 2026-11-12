"""clear27: ONE method on one domain, with the oracle as yardstick.

The method, the data, the limits and the evaluation are the paper's (protocol.py): the method is taken
from the graph source that owns it, and the grid can be made smaller (sizes, numbers of transitions,
seeds). A progress bar counts the (size, seed) jobs, and a step that stops is printed as it happens.

Run from the repository root (or: bash experiments/clear27/run_single.sh ...):
    .venv/bin/python experiments/clear27/run_single.py drone --list                # the method names
    .venv/bin/python experiments/clear27/run_single.py drone "PCMCI+ -> IMDP (IA)"  # the paper's whole grid
    .venv/bin/python experiments/clear27/run_single.py drone "PCMCI+ -> IMDP (IA)" --size 5-1 --n 1000 5000 \
        --seeds 0 1
    --save all|largest|none   the models of the first seed (tool/store.py; default all)
    --workers N|auto          as in the scripts (SCM_TO_MDP_WORKERS)
    --core-share P            percent of the physical cores to use, 1-100 (default 100; utils/compute.py)
    --resume                  continue the latest run of this method, with the same --size, --n and --seeds

Output: results/clear27_single/<domain>/<method>_<date>_<time>/ (results.csv, meta.json, grid.json, models/,
summary/), apart from results/clear27/, so a single run is never mixed into the paper's results.

A worst MDP ("... -> MDP (worst, IA)") takes the robust value of its IMDP arm when both run together
(tool/methods.py, solve); alone it computes that value itself: the same result, another solve time.
"""

import argparse
import dataclasses
import os
import sys
from datetime import datetime
from pathlib import Path

import protocol  # also puts experiments/ on the path (common.py, domains.py)

import common  # noqa: E402

from scm_to_mdp.tool import report_grid  # noqa: E402
from scm_to_mdp.tool.grid import run_grid  # noqa: E402
from scm_to_mdp.tool.runs import STAMP, add_core_share, runs_of, set_core_share  # noqa: E402
from scm_to_mdp.tool.store import method_folder  # noqa: E402

ORACLE = "oracle -> MDP"
EXPERIMENT = protocol.EXPERIMENT + "_single"


def methods_of(domain: str) -> dict[str, str]:
    """method name -> the graph source that owns it (the first one: the oracle is in every source)."""
    out: dict[str, str] = {}
    for source in protocol.SOURCES:
        for name in protocol.source_methods(source, protocol.DOMAINS[domain]):
            out.setdefault(name, source)
    return out


def _chosen(given, allowed, what: str) -> list:
    """The grid's own values named by the user (compared as text: a size may be 5 or "5-1")."""
    if not given:
        return list(allowed)
    by_text = {str(x): x for x in allowed}
    wrong = [g for g in given if str(g) not in by_text]
    if wrong:
        raise ValueError(f"{what} must be among {list(by_text)}, not {wrong}")
    return [by_text[str(g)] for g in given]


def single_grid(domain: str, method: str, sizes=None, n_transitions=None, seeds=None, save: str = "all"):
    """The paper's grid of the method's source, with the oracle and this method only."""
    owners = methods_of(domain)
    if method not in owners:
        names = "\n  ".join(owners)
        raise ValueError(f"no method {method!r} in {domain}; the methods are:\n  {names}")
    base = protocol.experiment_grid(domain, owners[method])
    return dataclasses.replace(
        base,
        methods={name: base.methods[name] for name in dict.fromkeys((ORACLE, method))},
        sizes=_chosen(sizes, base.sizes, "--size"),
        n_transitions=_chosen(n_transitions, base.n_transitions, "--n"),
        seeds=_chosen(seeds, base.seeds, "--seeds"),
        save_models=save,
        out_dir=common.ROOT / "results" / EXPERIMENT / domain / method_folder(method),
    )


def folder(base: Path, resume: bool) -> Path:
    """The latest <base>_<date>_<time> folder on a resume (when there is one), else a new one."""
    found = runs_of(base) if resume else []
    if found:
        print(f"resuming {found[-1]}")
        return found[-1]
    return Path(f"{base}_{datetime.now().strftime(STAMP)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27: one method on one domain, with the oracle.")
    parser.add_argument("domain", choices=list(protocol.SIZES))
    parser.add_argument("method", nargs="?", help="the method's name as in results.csv (see --list)")
    parser.add_argument("--list", action="store_true", help="print the method names of the domain")
    parser.add_argument("--size", nargs="+", metavar="S", help="sizes of the domain (default: the paper's)")
    parser.add_argument("--n", nargs="+", metavar="N", help="numbers of transitions (default: the paper's)")
    parser.add_argument("--seeds", nargs="+", metavar="SEED", help="seeds (default: the paper's)")
    parser.add_argument("--save", choices=["all", "largest", "none"], default="all",
                        help="the models of the first seed: every n, the largest n, or none")  # fmt: skip
    parser.add_argument("--resume", action="store_true", help="continue the latest run of this method")
    parser.add_argument("--workers", metavar="N", help="parallel jobs: a number or 'auto'")
    add_core_share(parser)
    args = parser.parse_args(argv)
    set_core_share(parser, args.core_share)
    if args.list:
        for name, source in methods_of(args.domain).items():
            print(f"{name:<45} ({source})")
        print(f"sizes: {' '.join(map(str, protocol.SIZES[args.domain]))}")
        return 0
    if args.method is None:
        parser.error("give a method name (--list prints them)")
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    try:
        base = single_grid(args.domain, args.method, args.size, args.n, args.seeds, args.save)
    except ValueError as err:
        parser.error(str(err))
    grid = dataclasses.replace(base, out_dir=folder(base.out_dir, args.resume))
    run_grid(grid)  # prints the settings and shows the progress bar (tool/sweep.py)
    report_grid(grid.out_dir, target=common.TARGET, delta=common.DELTA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
