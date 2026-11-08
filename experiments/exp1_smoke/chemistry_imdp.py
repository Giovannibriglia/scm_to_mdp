"""Smoke run of exp1/chemistry_imdp.py: the same methods on a tiny grid (experiments/smoke.py), in minutes.

    .venv/bin/python experiments/run.py exp1_smoke chemistry imdp
Output: results/exp1_smoke/chemistry_imdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("chemistry", "imdp")

if __name__ == "__main__":
    main(GRID)
