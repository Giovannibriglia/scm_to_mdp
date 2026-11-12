"""clear27, Sepsis: pool 4, flat -> IMDP and MDPs; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/sepsis/sepsis_flat.py                # a new run
    .venv/bin/python experiments/clear27/sepsis/sepsis_flat.py --resume       # continue it
    .venv/bin/python experiments/clear27/sepsis/sepsis_flat.py --report-only  # the report
Output: results/clear27/sepsis/sepsis_flat_<date>_<time>/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/clear27/: protocol.py

from protocol import experiment_grid, main  # noqa: E402

GRID = experiment_grid("sepsis", "flat")

if __name__ == "__main__":
    main(GRID)
