"""clear27, Sepsis: pool 3, Strehl et al. (2007) with and without KWIK; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/sepsis_strehl.py                # a new run
    .venv/bin/python experiments/clear27/sepsis_strehl.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/sepsis_strehl.py --report-only  # tables and figures
Output: results/clear27/sepsis_strehl_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("sepsis", "strehl")

if __name__ == "__main__":
    main(GRID)
