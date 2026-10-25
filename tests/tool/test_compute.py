"""How a run uses the machine (utils/compute.py, utils/parallel.py, utils/limits.py): the plan of
workers and device, a GPU only when it works, no orphan processes after the runner dies, and a
grid run in parallel giving the same rows as one run serially."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from scm_to_mdp.utils import compute

ROOT = Path(__file__).resolve().parents[1]


# ---- the plan --------------------------------------------------------------------------------
def test_workers_follow_cores_and_the_ram_rule():
    assert compute._workers("auto", 8.0, cores=10, ram_gb=34)[0] == 3  # 0.8 x 34 / 8 = 3.4
    assert compute._workers("auto", 8.0, cores=10, ram_gb=135)[0] == 10  # the cores bind
    assert compute._workers(None, None, cores=6, ram_gb=8)[0] == 6  # no memory limit
    assert compute._workers("auto", 64.0, cores=4, ram_gb=16)[0] == 1  # at least one
    assert compute._workers("5", 8.0, cores=2, ram_gb=8) == (5, "set by the user")
    with pytest.raises(ValueError):
        compute._workers("0", 8.0, cores=2, ram_gb=8)


def test_auto_takes_cuda_never_mps_and_explicit_devices_must_work():
    cuda = {"cuda": [{"index": 0, "name": "GPU", "memory_gb": 12.0, "capability": "8.0"}], "mps": False}
    assert compute._device("auto", cuda)[0] == "cuda"
    assert compute._device(None, {"cuda": [], "mps": True})[0] == "cpu"  # MPS is slower here
    assert compute._device("auto", {"cuda": [], "mps": False, "cuda_error": "old driver. More"}) == (
        "cpu",
        "CUDA unusable: old driver",
    )
    assert compute._device("cpu", cuda) == ("cpu", "set by the user")
    assert compute._device("cuda:0", cuda)[0] == "cuda:0"
    with pytest.raises(RuntimeError, match="CUDA cannot train"):
        compute._device("cuda", {"cuda": [], "mps": True})
    with pytest.raises(RuntimeError, match="MPS cannot train"):
        compute._device("mps", {"cuda": [], "mps": False})
    with pytest.raises(ValueError, match="unknown device"):
        compute._device("tpu", cuda)


def test_plan_describes_this_machine(monkeypatch):
    monkeypatch.delenv(compute.ENV_WORKERS, raising=False)
    p = compute.plan(8.0, workers="auto")
    assert 1 <= p.workers <= compute.physical_cores() and p.threads == 1
    assert p.device in ("cpu", "cuda")  # never MPS by default
    assert "physical cores" in p.describe() and p.machine["ram_gb"] > 0
    assert compute.plan(8.0, workers="auto", n_jobs=1).workers == 1  # never more workers than jobs
    monkeypatch.setenv(compute.ENV_WORKERS, "2")
    assert compute.plan(8.0).workers == 2


def test_training_records_its_device():
    from scm_to_mdp.baselines import offline_q
    from scm_to_mdp.envs import SysAdmin

    env = SysAdmin(2, 0)
    data = env.ground_truth().learner_spec.read(env.sample(30, 10, "random"))
    q = offline_q.learn(data, 0.9, steps=50, hidden=16, device="cpu")
    assert q.info["device"] == "cpu" and q.info["cuda_graph"] is False
    assert q.Q.shape == (4, len(data.actions)) and q.info["final_loss"] >= 0


# ---- no orphans ------------------------------------------------------------------------------
ORPHAN = """
import sys, time
from scm_to_mdp.utils.limits import run_limited
from scm_to_mdp.utils.parallel import run_dynamic

def nap(seconds):
    time.sleep(seconds)
    return seconds

def step(seconds):  # a worker job that runs a memory-limited child, as run_point does
    return run_limited(nap, seconds, mem_gb=2, start_method=sys.argv[1]).value

if __name__ == "__main__":
    for _ in run_dynamic(step, [(1,), (600,)], workers=2):  # one worker idles after a second
        pass
"""


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_stopping_the_runner_stops_workers_and_steps(tmp_path, start_method):
    script = tmp_path / "orphan.py"
    script.write_text(ORPHAN)
    runner = subprocess.Popen([sys.executable, str(script), start_method], cwd=ROOT)
    proc = psutil.Process(runner.pid)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # forkserver, 2 workers and their 2 steps (+ trackers)
        if len(proc.children(recursive=True)) >= 4:
            break
        time.sleep(0.5)
    time.sleep(2.5)  # the first job is done: one worker busy, one idle
    family = proc.children(recursive=True)
    assert len(family) >= 3, "the run did not start its workers and steps"
    os.kill(runner.pid, signal.SIGKILL)  # as the IDE's stop button, or a crash
    runner.wait()
    gone, alive = psutil.wait_procs(family, timeout=15)
    assert not alive, f"left behind: {[p.cmdline()[-2:] for p in alive]}"


# ---- a parallel grid gives the same rows as a serial one ---------------------------------------
GRID = """
import sys
from pathlib import Path
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Grid, Limits, Method, Policy, run_grid

def make_env(n, topology, seed):
    return SysAdmin(int(n), seed, topology=topology)

grid = Grid(
    name="g", make_env=make_env, sizes=[2, 3], knobs=["ring", "star"],
    policies=[Policy("uniform"), Policy("expert")], n_rollouts=[10, 30], seeds=[0, 1], horizon=8,
    methods={"BIC -> RMDP": Method("causal", "box_robust_vi", discovery="bic", selection="bic_best"),
             "flat MDP": Method("flat_mdp", "vi")},
    limits=Limits(mem_gb=1), workers=sys.argv[2], out_dir=Path(sys.argv[1]), save_models="none",
)
if __name__ == "__main__":
    run_grid(grid)
"""


def test_a_parallel_grid_gives_the_rows_of_a_serial_one(tmp_path):
    import pandas as pd

    script = tmp_path / "grid.py"
    script.write_text(textwrap.dedent(GRID))
    tables = {}
    for workers in ("1", "3"):
        out = tmp_path / f"w{workers}"
        done = subprocess.run(
            [sys.executable, str(script), str(out), workers], cwd=ROOT, capture_output=True, text=True,
            timeout=600, env={k: v for k, v in os.environ.items() if not k.startswith("SCM_TO_MDP_")},
        )  # fmt: skip
        assert done.returncode == 0, done.stderr[-3000:]
        assert f"plan:    {workers} workers" in done.stdout
        files = sorted(out.glob("*/*/results.csv"))
        assert len(files) == 4  # 2 knobs x 2 policies, each in its own folder
        rows = pd.concat([pd.read_csv(f) for f in files])
        timing = [c for c in rows.columns if c.endswith(("_s", "_gb"))]
        keys = ["knob", "policy", "size", "seed", "n_rollouts", "method"]
        tables[workers] = rows.drop(columns=timing).sort_values(keys).reset_index(drop=True)
    assert len(tables["1"]) == 4 * 2 * 2 * 2 * 2 and set(tables["1"]["status"]) == {"ok"}
    pd.testing.assert_frame_equal(tables["1"], tables["3"])


def test_cuda_training_uses_a_graph_and_is_reproducible():
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from scm_to_mdp.baselines import offline_q
    from scm_to_mdp.envs import SysAdmin

    env = SysAdmin(2, 0)
    data = env.ground_truth().learner_spec.read(env.sample(30, 10, "random"))
    a = offline_q.learn(data, 0.9, steps=300, hidden=16, device="cuda")
    b = offline_q.learn(data, 0.9, steps=300, hidden=16, device="cuda")
    assert a.info["cuda_graph"] and a.info["device"].startswith("cuda")
    assert np.array_equal(a.Q, b.Q)
    eager = offline_q.learn(data, 0.9, steps=300, hidden=16, device="cuda", cuda_graph=False)
    assert not eager.info["cuda_graph"] and np.isfinite(eager.Q).all()


# ---- the same string hashing in every process of a run ------------------------------------------
HASHING = """
import sys
from scm_to_mdp.utils import compute
from scm_to_mdp.utils.parallel import run_dynamic

def name_hash(text):
    return hash(text)

if __name__ == "__main__":
    compute.apply(compute.plan(None, workers="2", device="cpu"))
    print(sorted({r for _, r in run_dynamic(name_hash, [("vasopressors",)] * 4, workers=2)}))
"""


def test_workers_hash_strings_with_the_fixed_seed(tmp_path):
    script = tmp_path / "hashing.py"
    script.write_text(HASHING)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}  # the runner's own hash: random
    runs = [
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True)
        for _ in range(2)
    ]
    seen = [r.stdout.strip().splitlines()[-1] for r in runs]
    fixed = subprocess.run(
        [sys.executable, "-c", "print(hash('vasopressors'))"],
        env=env | {"PYTHONHASHSEED": compute.HASH_SEED}, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert seen[0] == seen[1] == f"[{fixed}]"
