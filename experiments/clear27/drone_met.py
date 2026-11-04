"""clear27, Drone: pool 3, Met-Rmax (2009) with and without KWIK; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/drone_met.py                # a new run
    .venv/bin/python experiments/clear27/drone_met.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/drone_met.py --report-only  # tables and figures
Output: results/clear27/drone_met_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("drone", "met")

if __name__ == "__main__":
    main(GRID)
