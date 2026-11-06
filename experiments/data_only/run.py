"""data_only: every source in ONE pool of workers (as clear27's run_pool.py), then each source's report.

    .venv/bin/python experiments/data_only/run.py             # new runs
    .venv/bin/python experiments/data_only/run.py --resume    # continue the latest run of every source
    SMOKE=1 ...                                               # the tiny grid; --workers N as the scripts
"""

import argparse
import dataclasses
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_protocol as protocol  # noqa: E402

from scm_to_mdp.tool import report_grid  # noqa: E402
from scm_to_mdp.tool.grid import experiments, write_grid_file  # noqa: E402
from scm_to_mdp.tool.runs import STAMP, runs_of  # noqa: E402
from scm_to_mdp.tool.sweep import run_sweeps  # noqa: E402


def folder(base: Path, resume: bool) -> Path:
    found = runs_of(base) if resume else []
    if found:
        print(f"resuming {found[-1]}")
        return found[-1]
    new = Path(f"{base}_{datetime.now().strftime(STAMP)}")
    print(f"starting {new}")
    return new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="data_only: every source in one pool of workers.")
    parser.add_argument("--sources", nargs="+", choices=list(protocol.SOURCES), default=list(protocol.SOURCES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", metavar="N")
    args = parser.parse_args(argv)
    if args.workers is not None:
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    grids = []
    for source in args.sources:
        g = protocol.experiment_grid(source)
        grids.append((source, dataclasses.replace(g, out_dir=folder(g.out_dir, args.resume))))
    exps = []
    for _, g in grids:
        write_grid_file(g)
        exps += experiments(g)
    run_sweeps(exps, resume=True, title=f"{protocol.EXPERIMENT} ({len(grids)} sources)")
    failed = []
    for source, g in grids:
        try:
            report_grid(g.out_dir, target=protocol.common.TARGET, delta=protocol.common.DELTA)
        except Exception:  # noqa: BLE001 - one report must not stop the others
            traceback.print_exc()
            failed.append(source)
    print("reports failed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
