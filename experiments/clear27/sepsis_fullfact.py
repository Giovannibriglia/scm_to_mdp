"""clear27, Sepsis: pool 4, the full factored graph -> point 2; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/sepsis_fullfact.py                # a new run
    .venv/bin/python experiments/clear27/sepsis_fullfact.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/sepsis_fullfact.py --report-only  # tables and figures
Output: results/clear27/sepsis_fullfact_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("sepsis", "fullfact")

if __name__ == "__main__":
    main(GRID)
