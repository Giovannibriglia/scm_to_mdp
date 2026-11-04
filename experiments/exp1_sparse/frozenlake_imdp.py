"""Experiment 1 (sparse graph), Frozen Lake: sparse -> IMDP (and the oracle) on experiment 1's data;
experiments/merge_runs.py joins it to experiment 1's run.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1_sparse/frozenlake_imdp.py                # a new run
    .venv/bin/python experiments/exp1_sparse/frozenlake_imdp.py --resume       # continue the latest
    .venv/bin/python experiments/exp1_sparse/frozenlake_imdp.py --report-only  # tables and figures
    .venv/bin/python experiments/exp1_sparse/frozenlake_imdp.py --workers 6    # own plan
Output: results/exp1_sparse/frozenlake_imdp_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("frozenlake", "imdp")

if __name__ == "__main__":
    main(GRID)
