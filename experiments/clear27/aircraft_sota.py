"""clear27, Aircraft: pool 5, the state of the art; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/aircraft_sota.py                # a new run
    .venv/bin/python experiments/clear27/aircraft_sota.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/aircraft_sota.py --report-only  # tables and figures
Output: results/clear27/aircraft_sota_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("aircraft", "sota")

if __name__ == "__main__":
    main(GRID)
