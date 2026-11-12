"""clear27, Drone: pool 3, DYNOTEARS -> point 2; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/drone/drone_dynotears.py                # a new run
    .venv/bin/python experiments/clear27/drone/drone_dynotears.py --resume       # continue it
    .venv/bin/python experiments/clear27/drone/drone_dynotears.py --report-only  # the report
Output: results/clear27/drone/drone_dynotears_<date>_<time>/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/clear27/: protocol.py

from protocol import experiment_grid, main  # noqa: E402

GRID = experiment_grid("drone", "dynotears")

if __name__ == "__main__":
    main(GRID)
