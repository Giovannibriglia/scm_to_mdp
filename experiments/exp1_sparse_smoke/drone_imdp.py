"""Smoke run of exp1_sparse/drone_imdp.py: the same methods on a tiny grid (experiments/smoke.py).

    .venv/bin/python experiments/run.py exp1_sparse_smoke drone imdp
Output: results/exp1_sparse_smoke/drone_imdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("drone", "imdp")

if __name__ == "__main__":
    main(GRID)
