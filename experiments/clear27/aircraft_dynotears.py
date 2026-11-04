"""clear27, Aircraft: pool 3, DYNOTEARS -> point 2; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/aircraft_dynotears.py                # a new run
    .venv/bin/python experiments/clear27/aircraft_dynotears.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/aircraft_dynotears.py --report-only  # tables and figures
Output: results/clear27/aircraft_dynotears_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("aircraft", "dynotears")

if __name__ == "__main__":
    main(GRID)
