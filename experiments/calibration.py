"""Calibration: does the promise hold with probability at least 1 - delta? (S4, S9)

Ten seeds cannot test delta = 0.05: with no failure in 10 runs the 95% Clopper-Pearson upper bound
on the failure rate is still 0.26, and a bound below 0.05 needs at least 59 runs without a failure.
This script runs the methods with a promise (experiment 1's -> IMDP arms) on the smallest sizes of
every domain with SEEDS seeds, everything else as in experiments/common.py (uniform logger, 256-step
episodes, the robustness checks; no random rewards). Its report (summary/validity.csv) gives, per
cell, the share of runs whose promise held with a Clopper-Pearson interval and the verdict against
1 - delta ("confirmed", "refuted", "undecided"); summary/robustness.csv adds the other checks.

    .venv/bin/python experiments/calibration.py                   # every domain, in turn
    .venv/bin/python experiments/calibration.py sysadmin taxi     # only these
    .venv/bin/python experiments/calibration.py sepsis --resume   # continue the latest run
Output: results/calibration/<domain>_imdp_<date>_<time>/.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "exp1"))  # experiment 1's methods

from common import ROOT, main  # noqa: E402
from domains import DOMAINS  # noqa: E402
from protocol import experiment_grid  # noqa: E402

# >= 59 seeds: then no failure puts the 95% upper bound on the failure rate below 0.05
SEEDS = list(range(60))
SMALLEST = 2  # the smallest sizes of every domain (per knob)
N_EPISODES = [10, 100, 1000]  # of 256 steps


def calibration_grid(domain: str):
    base = experiment_grid(domain, "imdp")
    sizes = (
        {k: v[:SMALLEST] for k, v in base.sizes.items()}
        if isinstance(base.sizes, dict)
        else base.sizes[:SMALLEST]
    )
    return dataclasses.replace(
        base,
        sizes=sizes,
        seeds=SEEDS,
        n_rollouts=N_EPISODES,
        save_models="none",
        reward_transfer=False,  # the calibration asks about 1 - delta only
        reward_family=None,
        out_dir=ROOT / "results" / "calibration" / f"{domain}_imdp",
    )


if __name__ == "__main__":
    domains = [a for a in sys.argv[1:] if not a.startswith("-")] or list(DOMAINS)
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    for domain in domains:
        sys.argv = [sys.argv[0], *flags]  # run_folder reads --resume / --report-only / --workers
        main(calibration_grid(domain))
