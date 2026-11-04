"""clear27, Aircraft: pool 4, flat -> IMDP and MDPs; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/aircraft_flat.py                # a new run
    .venv/bin/python experiments/clear27/aircraft_flat.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/aircraft_flat.py --report-only  # tables and figures
Output: results/clear27/aircraft_flat_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("aircraft", "flat")

if __name__ == "__main__":
    main(GRID)
