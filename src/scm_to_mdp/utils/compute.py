"""How a run uses the machine: worker processes, threads and the device that trains the networks.

An experiment script runs as is on a laptop or a server: at start the runner looks at the machine
and chooses, then prints the choices and saves them in meta.json (key "machine").

    workers   parallel (size, seed) jobs = min(core share x physical cores, RAM rule), at least 1.
              Core share: the part of the physical cores a run may take, 100% unless set (below), e.g.
              to leave half of a shared machine free.
              RAM rule: workers x Limits.mem_gb <= 95% of the RAM (RAM_SHARE), so that a step is never
              killed because other steps took the memory; a "memout" then means the same on every
              machine. 5% stays for the system and the parent processes (Linux server, 135 GB: 21
              workers of 6 GB; at N=11 of SysAdmin 17 workers used about 70 GB in all).
    hashing   PYTHONHASHSEED=0 in every process the run starts (HASH_SEED): set orders, and so
              results, do not change from one run to the next.
    threads   one per process (torch and the BLAS libraries of numpy / scipy): more threads made a
              training slower (1 / 2 / 4 threads: 2.7 / 3.5 / 3.9 s on an M5), and workers x threads
              must not exceed the cores (an oversubscribed machine makes the time limits fire
              earlier than an idle one would).
    device    where offline DQN / CQL train (most of a run's time). "auto" (the default) takes CUDA
              when a test step runs on it, else the CPU; never Apple's MPS (DEVICE_POLICY). Every
              other step runs on the CPU (the NBN steps are counting work, not tensor algebra).

Overrides (environment, or the experiment script's command line: --workers N --device D):

    SCM_TO_MDP_WORKERS = N | auto         (a number wins over the core share)
    SCM_TO_MDP_CORE_SHARE = P     (--core-share P: percent of the physical cores, 1..100)
    SCM_TO_MDP_DEVICE  = auto | cpu | cuda | cuda:1 | mps
    SCM_TO_MDP_THREADS = N        (torch threads of one CPU training)

The GPU test runs in a separate Python process: initialising CUDA or MPS in the runner itself would
break the processes it forks later (utils/limits.py).
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field

import psutil

ENV_WORKERS = "SCM_TO_MDP_WORKERS"
ENV_CORE_SHARE = "SCM_TO_MDP_CORE_SHARE"
ENV_DEVICE = "SCM_TO_MDP_DEVICE"
ENV_THREADS = "SCM_TO_MDP_THREADS"
BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
RAM_SHARE = 0.95  # of the RAM for workers x Limits.mem_gb (the plan and sweep._check_memory)
# String hashing is randomised per Python process, so the order of a set of names changes between
# runs, and a library that breaks ties by that order gives different results (pgmpy's GES did,
# discovery/ges.py). Every process a run starts gets this fixed seed (the runner's own hash is set when
# it starts, so every step runs in a worker, also with one worker: tool/sweep.py).
HASH_SEED = "0"

# "auto" -> which GPU backends may train the networks. Measured 2026-10-22, offline DQN on Sepsis
# (2 x 256 MLP, batch 256, 4,000 steps, two datasets): Apple M5, CPU 4.7 s vs MPS 6.1 s (MPS slower);
# Xeon E5-2640 v4 + TITAN X (Pascal), 10 trainings at once, CPU 60 s vs CUDA 38 s (README).
DEVICE_POLICY = {"cuda": True, "mps": False}


def physical_cores() -> int:
    """Cores this process may use (CPU affinity on Linux), without hyper-threads."""
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    ratio = max(1, (psutil.cpu_count(logical=True) or logical) // physical)  # threads per core
    return max(1, min(physical, logical // ratio))


_PROBE = r"""
import json, sys
out = {"torch": None, "cuda": [], "cuda_error": None, "mps": False, "mps_error": None}
try:
    import torch
except Exception as err:
    print(json.dumps(out | {"torch_error": repr(err)})); sys.exit()
out["torch"] = torch.__version__
def works(dev):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)).to(dev)
    loss = net(torch.randn(4, 8, device=dev)).logsumexp(1).mean()
    loss.backward()
    return float(loss.detach().cpu())
try:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        works(f"cuda:{i}")
        out["cuda"].append({"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1e9, 1),
                            "capability": f"{p.major}.{p.minor}"})
except Exception as err:
    out["cuda"], out["cuda_error"] = [], repr(err).splitlines()[0][:300]
try:
    if torch.backends.mps.is_available():
        works("mps")
        out["mps"] = True
except Exception as err:
    out["mps_error"] = repr(err).splitlines()[0][:300]
print(json.dumps(out))
"""

_probed: dict | None = None
_applied: str | None = None  # the device this process chose last (apply): not a user's choice


def probe_gpus() -> dict:
    """Which GPUs torch can really train on (a test step on each), from a fresh Python process."""
    global _probed
    if _probed is None:
        try:
            done = subprocess.run(
                [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=180, check=False
            )
            _probed = json.loads(done.stdout.strip().splitlines()[-1])
        except Exception as err:  # noqa: BLE001 - no GPU then; reported in the machine description
            _probed = {"torch": None, "cuda": [], "mps": False, "probe_error": repr(err)}
    return _probed


def core_share(spec=None) -> float:
    """The percent of the physical cores a run may take (spec, else SCM_TO_MDP_CORE_SHARE, else 100)."""
    spec = os.environ.get(ENV_CORE_SHARE) if spec in (None, "") else spec
    if spec in (None, ""):
        return 100.0
    try:
        share = float(spec)
    except (TypeError, ValueError):
        share = math.nan
    if not 1 <= share <= 100:
        raise ValueError(f"the core share is a percent from 1 to 100, not {spec!r}")
    return share


def _workers(spec, mem_gb: float | None, cores: int, ram_gb: float, share: float = 100.0) -> tuple[int, str]:
    if spec not in (None, "", "auto"):
        n = int(spec)
        if n < 1:
            raise ValueError(f"workers must be >= 1 or 'auto', not {spec!r}")
        return n, "set by the user"
    allowed = max(1, int(cores * share / 100 + 1e-9))  # the cores this run may take
    by_ram = int(RAM_SHARE * ram_gb // mem_gb) if mem_gb else allowed
    n = max(1, min(allowed, by_ram))
    if n == allowed and share < 100:
        return n, f"{share:g}% of the {cores} physical cores"
    if n == allowed:
        return n, f"one per physical core ({cores})"
    ram_rule = f"RAM rule: {RAM_SHARE:g} x {ram_gb:.0f} GB / {mem_gb:g} GB per step"
    return n, f"{ram_rule}; {allowed} of {cores} cores"


def _device(spec: str | None, gpus: dict) -> tuple[str, str]:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if gpus.get("cuda") and DEVICE_POLICY["cuda"]:
            return "cuda", f"CUDA: {', '.join(g['name'] for g in gpus['cuda'])}"
        if gpus.get("mps") and DEVICE_POLICY["mps"]:
            return "mps", "Apple GPU (MPS)"
        reason = "no usable GPU"
        if gpus.get("mps"):
            reason = "the Apple GPU (MPS) is slower than the CPU for these small networks"
        elif gpus.get("cuda_error"):
            reason = f"CUDA unusable: {gpus['cuda_error'].split('. ')[0]}"
        return "cpu", reason
    if spec.startswith("cuda") and not gpus.get("cuda"):
        raise RuntimeError(f"device {spec!r} requested but CUDA cannot train here: {gpus.get('cuda_error')}")
    if spec == "mps" and not gpus.get("mps"):
        raise RuntimeError(f"device 'mps' requested but MPS cannot train here: {gpus.get('mps_error')}")
    if spec != "cpu" and not spec.startswith(("cuda", "mps")):
        raise ValueError(f"unknown device {spec!r} (auto, cpu, cuda, cuda:N, mps)")
    return spec, "set by the user"


@dataclass
class Plan:
    workers: int
    threads: int  # torch threads of one process
    device: str  # training device: cpu, cuda (spread over the GPUs), cuda:N, mps
    machine: dict = field(default_factory=dict)
    why: dict = field(default_factory=dict)

    def describe(self) -> str:
        m = self.machine
        gpus = ", ".join(f"{g['name']} ({g['memory_gb']:g} GB)" for g in m.get("cuda", [])) or (
            "Apple GPU (MPS)" if m.get("mps") else "none usable"
        )
        return (
            f"machine: {m['system']} {m['arch']}, {m['physical_cores']} physical cores, "
            f"{m['ram_gb']:.0f} GB RAM, GPU: {gpus}\n"
            f"plan:    {self.workers} workers ({self.why['workers']}); networks train on {self.device} "
            f"({self.why['device']}); {self.threads} thread(s) per process"
        )


def plan(mem_gb: float | None, workers=None, device: str | None = None, n_jobs: int | None = None) -> Plan:
    """The machine and the choices for a run (see the module docstring). Arguments left as None are
    read from the environment, else chosen automatically."""
    cores, ram_gb = physical_cores(), psutil.virtual_memory().total / 1e9
    share = core_share()
    n, why_workers = _workers(workers or os.environ.get(ENV_WORKERS), mem_gb, cores, ram_gb, share)
    if n_jobs is not None and n_jobs < n:
        n, why_workers = max(1, n_jobs), f"{n_jobs} job(s) to run; the machine allows {n}"
    gpus = probe_gpus()
    env = os.environ.get(ENV_DEVICE)
    dev, why_device = _device(device or (None if env == _applied else env), gpus)
    threads = int(os.environ.get(ENV_THREADS) or 1)
    machine = {
        "system": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "physical_cores": cores,
        "core_share_percent": share,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(ram_gb, 1),
        "torch": gpus.get("torch"),
        "cuda": gpus.get("cuda", []),
        "mps": gpus.get("mps", False),
    }
    return Plan(n, threads, dev, machine, {"workers": why_workers, "device": why_device})


def apply(p: Plan) -> None:
    """Make the plan visible to every process the run starts (they inherit the environment)."""
    global _applied
    os.environ[ENV_DEVICE] = _applied = p.device
    os.environ[ENV_THREADS] = str(p.threads)
    os.environ["PYTHONHASHSEED"] = HASH_SEED  # fresh interpreters (pool workers, trainings) read it
    for var in BLAS_VARS:
        os.environ.setdefault(var, "1")


def as_record(p: Plan) -> dict:
    return asdict(p)


# ---- inside a training process ------------------------------------------------------------------
def training_device(spec: str | None = None):
    """The torch device a network trains on: spec, else SCM_TO_MDP_DEVICE, else the CPU. "cuda" with
    several GPUs picks one per process (by process id), so parallel trainings share the GPUs."""
    import torch

    spec = (spec or os.environ.get(ENV_DEVICE) or "cpu").lower()
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() and DEVICE_POLICY["cuda"] else "cpu"
    if spec == "cuda":
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("device 'cuda' requested but torch sees no CUDA device")
        return torch.device("cuda", os.getpid() % count)
    return torch.device(spec)


def training_threads() -> int:
    return max(1, int(os.environ.get(ENV_THREADS) or 1))
