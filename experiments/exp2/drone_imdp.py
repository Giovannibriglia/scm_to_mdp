"""Experiment 2, Drone: ours (PCMCI+ / DYNOTEARS) vs DiFaRL, FOCUS, SL-SPIBB and offline DQN;
our arms with interval sets and robust value iteration (-> IMDP, a promise).

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp2/drone_imdp.py                # a new run
    .venv/bin/python experiments/exp2/drone_imdp.py --resume       # continue the latest
    .venv/bin/python experiments/exp2/drone_imdp.py --report-only  # tables and figures
    .venv/bin/python experiments/exp2/drone_imdp.py --workers 6    # own plan
Output: results/exp2/drone_imdp_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("drone", "imdp")

if __name__ == "__main__":
    main(GRID)
