"""Bootstrap summaries and the E1 analysis outputs."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scm_to_mdp.stats import bootstrap_mean_ci

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_interval_contains_the_mean_and_shrinks_with_data():
    rng = np.random.default_rng(0)
    small = bootstrap_mean_ci(rng.normal(size=10))
    large = bootstrap_mean_ci(rng.normal(size=1000))
    assert small[1] <= small[0] <= small[2] and large[1] <= large[0] <= large[2]
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_bootstrap_edge_cases():
    assert bootstrap_mean_ci([2.0]) == (2.0, 2.0, 2.0)
    assert all(math.isinf(v) for v in bootstrap_mean_ci([1.0, float("inf")]))
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, float("nan")])


def test_pilot_and_analysis_write_figure_and_tables(tmp_path):
    exp = ROOT / "experiments" / "e1_sysadmin_graph_arms"
    run = [sys.executable, str(exp / "pilot_loggers.py"), "--seeds", "2", "--episodes", "10",
           "--out", str(tmp_path / "pilot_loggers.csv")]
    subprocess.run(run, check=True, capture_output=True, cwd=ROOT)
    subprocess.run([sys.executable, str(exp / "analyze.py"), "--results", str(tmp_path)],
                   check=True, capture_output=True, cwd=ROOT)
    for rel in ("figures/pilot_loggers.pdf", "figures/pilot_loggers.png",
                "tables/pilot_loggers.csv", "tables/pilot_loggers.tex"):
        assert (tmp_path / rel).stat().st_size > 0
    assert "\\begin{tabular}" in (tmp_path / "tables" / "pilot_loggers.tex").read_text()
