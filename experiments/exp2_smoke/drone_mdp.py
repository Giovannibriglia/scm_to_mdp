"""Smoke run of exp2/drone_mdp.py: the same methods on a tiny grid (experiments/smoke.py), in minutes.

    .venv/bin/python experiments/run.py exp2_smoke drone mdp
Output: results/exp2_smoke/drone_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("drone", "mdp")

if __name__ == "__main__":
    main(GRID)
