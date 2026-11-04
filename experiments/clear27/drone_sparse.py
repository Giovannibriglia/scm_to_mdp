"""clear27, Drone: pool 4, the sparse graph -> point 2; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/drone_sparse.py                # a new run
    .venv/bin/python experiments/clear27/drone_sparse.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/drone_sparse.py --report-only  # tables and figures
Output: results/clear27/drone_sparse_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("drone", "sparse")

if __name__ == "__main__":
    main(GRID)
