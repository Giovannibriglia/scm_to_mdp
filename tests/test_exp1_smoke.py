"""exp1 scripts on their smallest instance, in a fresh process (the pipeline forks its steps)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = """
import json, sys
sys.path.insert(0, "experiments/exp1")
import sysadmin
print(json.dumps(sysadmin.run(3, 1), default=float))  # seed 1: no pictures
"""


def test_exp1_sysadmin_small():
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    row = json.loads(out.stdout.strip().splitlines()[-1])
    statuses = {k: v for k, v in row.items() if k.endswith("_status")}
    assert statuses and all(v == "ok" for v in statuses.values()), statuses
    for arm in ["BIC -> RMDP", "PCMCI+ -> RMDP", "Schnitzer IA", "Schnitzer exact", "flat RMDP"]:
        assert row[f"{arm}|promise"] <= row[f"{arm}|true"] + 1e-6
        assert row[f"{arm}|true"] <= row["V_star"] + 1e-6


SEPSIS = """
import json, sys
sys.path.insert(0, "experiments/exp1")
import sepsis
print(json.dumps(sepsis.run("small", 1), default=float))
"""


def test_exp1_sepsis_small():
    out = subprocess.run(
        [sys.executable, "-c", SEPSIS], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    row = json.loads(out.stdout.strip().splitlines()[-1])
    statuses = {k: v for k, v in row.items() if k.endswith("_status")}
    assert statuses and all(v == "ok" for v in statuses.values()), statuses
    for arm in ["Schnitzer IA", "flat RMDP"]:  # true graph or no graph: the promise must hold
        assert row[f"{arm}|promise"] <= row[f"{arm}|true"] + 1e-6
    assert all(row[f"{a}|true"] <= row["V_star"] + 1e-6 for a in ["BIC -> RMDP", "Schnitzer IA", "flat MDP"])
