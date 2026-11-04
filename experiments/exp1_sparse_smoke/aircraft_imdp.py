"""Smoke run of exp1_sparse/aircraft_imdp.py: the same methods on a tiny grid (experiments/smoke.py).

    .venv/bin/python experiments/run.py exp1_sparse_smoke aircraft imdp
Output: results/exp1_sparse_smoke/aircraft_imdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("aircraft", "imdp")

if __name__ == "__main__":
    main(GRID)
