"""clear27: the graph sources of one or more domains in ONE pool of workers.

A source's script (<domain>/<domain>_<source>.py) runs its (size, seed) jobs in waves by size: a size starts
when every seed of the smaller size is done, because a method stopped by a limit there is skipped at the
larger sizes (tool/sweep.py, "Rules"). So one script keeps at most len(seeds) = 10 jobs running, and a
machine with more workers idles. Here every source is a setting of one run_sweeps call: a free worker takes
the next job of any source, each source keeps its own waves and stops, and each source writes to its
own folder exactly as its script would (results/clear27/<domain>/<domain>_<source>_<date>_<time>/), so a
source can also be resumed or reported by its own script.

Run from the repository root:
    .venv/bin/python experiments/clear27/run_pool.py sysadmin                        # every source, new runs
    .venv/bin/python experiments/clear27/run_pool.py sysadmin --resume               # continue the latest runs
    .venv/bin/python experiments/clear27/run_pool.py sysadmin drone sepsis --resume  # one pool for all:
                                                     # a free worker takes a job of the first domain that has one
    .venv/bin/python experiments/clear27/run_pool.py sysadmin --sources flat pcmci   # only these sources
    --workers N|auto and --core-share P as in the scripts (SCM_TO_MDP_WORKERS, SCM_TO_MDP_CORE_SHARE:
    utils/compute.py); SMOKE=1: the tiny grids (protocol.py).
Afterwards every source's tables and figures are written (report_grid), as its script does; a failed
report is printed and gives exit code 1, the other reports still run. The start and the end of the pool
are appended to results/clear27[_smoke]/logs/chain.log.
"""

import argparse
import dataclasses
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import protocol  # also puts experiments/ on the path (common.py, domains.py)

import common  # noqa: E402

from scm_to_mdp.tool import report_grid  # noqa: E402
from scm_to_mdp.tool.grid import experiments, write_grid_file  # noqa: E402
from scm_to_mdp.tool.runs import STAMP, add_core_share, runs_of, set_core_share  # noqa: E402
from scm_to_mdp.tool.sweep import run_sweeps  # noqa: E402


def folder(base: Path, resume: bool) -> Path:
    """The latest <base>_<date>_<time> folder on a resume (when there is one), else a new one."""
    found = runs_of(base) if resume else []
    if found:
        print(f"resuming {found[-1]}")
        return found[-1]
    new = Path(f"{base}_{datetime.now().strftime(STAMP)}")
    print(f"starting {new}")
    return new


def pool_grids(domains, sources, resume: bool) -> list[tuple[str, object]]:
    """(name, grid with its run folder) for every domain x source."""
    out = []
    for domain in domains:
        for source in sources:
            base = protocol.experiment_grid(domain, source)
            out.append((f"{domain}_{source}", dataclasses.replace(base, out_dir=folder(base.out_dir, resume))))
    return out


def log_dir() -> Path:
    return common.ROOT / "results" / (protocol.EXPERIMENT + ("_smoke" if protocol.SMOKE else "")) / "logs"


def _chain(line: str) -> None:
    path = log_dir() / "chain.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {line}"
    print(stamped, flush=True)
    with open(path, "a") as f:
        f.write(stamped + "\n")


def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=common.ROOT, check=False)  # fmt: skip
        return out.stdout.strip() or "?"
    except OSError:
        return "?"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27: several sources in one pool of workers.")
    parser.add_argument("domains", nargs="+", choices=list(protocol.SIZES))
    parser.add_argument("--sources", nargs="+", choices=list(protocol.SOURCES), default=list(protocol.SOURCES))
    parser.add_argument("--resume", action="store_true", help="continue the latest run of every source")
    parser.add_argument("--workers", metavar="N", help="parallel jobs: a number or 'auto'")
    add_core_share(parser)
    args = parser.parse_args(argv)
    set_core_share(parser, args.core_share)
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    name = f"pool {'+'.join(args.domains)}"
    _chain(f"start {name}: {' '.join(args.sources)} (host {os.uname().nodename}, commit {_commit()})")
    start, failed = time.time(), []
    try:
        grids = pool_grids(args.domains, args.sources, args.resume)
        exps, ranks = [], []
        for source, g in grids:
            write_grid_file(g)
            mine = experiments(g)
            exps += mine
            ranks += [args.domains.index(source.split("_")[0])] * len(mine)
        print(f"\n===== {protocol.EXPERIMENT} {name}: {len(grids)} sources, {len(exps)} settings =====")
        # each source in its own folder; the domains in the order given: a free worker takes a job of the
        # first domain that has one waiting, so that domain finishes first and no worker idles
        run_sweeps(exps, resume=True, title=f"{name} ({len(grids)} sources)", ranks=ranks)
        for source, g in grids:
            try:
                report_grid(g.out_dir, target=common.TARGET, delta=common.DELTA)
            except Exception:  # noqa: BLE001 - one report must not stop the others
                traceback.print_exc()
                failed.append(source)
                _chain(f"report failed: {source}")
    except BaseException:
        traceback.print_exc()
        failed.append("pool")
        raise
    finally:
        rc = 1 if failed else 0
        _chain(f"end   {name} rc={rc} ({int(time.time() - start) // 60} min)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
