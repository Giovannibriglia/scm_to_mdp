"""clear27, Aircraft: pool 3, Met-Rmax (2009) with and without KWIK; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/aircraft_met.py                # a new run
    .venv/bin/python experiments/clear27/aircraft_met.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/aircraft_met.py --report-only  # tables and figures
Output: results/clear27/aircraft_met_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("aircraft", "met")

if __name__ == "__main__":
    main(GRID)
