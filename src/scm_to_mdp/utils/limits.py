"""Run one pipeline step with a memory limit and a time limit.

The step runs in a forked child process. A monitor in the parent reads the child's memory (resident
set size, with the child's own children) and kills it above the limit ("memout"), or when it runs
longer than the time limit ("timeout"). macOS does not enforce RLIMIT_AS, hence the monitor. A child
killed by another process with SIGKILL (on Linux, the out-of-memory killer) also counts as "memout".
Time is time.monotonic(), which does not advance while the computer sleeps (macOS and Linux): a
laptop going to sleep in the middle of a step must not turn that step into a timeout.

Fork is used because it starts at once and the child shares the parent's data. It is safe only if
the parent never ran PyTorch computations itself: forking a process whose PyTorch thread pool was
used makes large torch operations fail in the child ("Invalid thread pool"). So every step that
uses NBN / torch must run through run_limited, and the parent must only prepare data.
"""

import multiprocessing
import os
import resource
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class StepResult:
    value: Any  # the step's output (or the error message when status == "error")
    status: str  # "ok", "memout", "timeout" or "error"
    peak_gb: float  # peak memory (includes pages shared with the parent: an upper bound)
    seconds: float


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9  # bytes on macOS, KiB on Linux


def _child(conn, fn, args):
    os.environ["TQDM_DISABLE"] = "1"  # silence the progress bars of the libraries
    try:
        conn.send(("ok", fn(*args), _peak_rss_gb()))
    except MemoryError:
        conn.send(("memout", None, _peak_rss_gb()))
    except Exception as err:  # noqa: BLE001 - reported, not hidden
        conn.send(("error", repr(err), _peak_rss_gb()))


def _memory(proc: psutil.Process) -> int:
    try:
        return proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
    except psutil.NoSuchProcess:
        return 0


def _kill(proc: psutil.Process):
    try:
        for c in proc.children(recursive=True):
            c.kill()
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def run_limited(fn, *args, mem_gb: float, timeout_s: float | None = None, poll: float = 0.02) -> StepResult:
    """fn(*args) in a child process, stopped above mem_gb of memory or after timeout_s seconds."""
    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child, args=(sender, fn, args))
    start = time.monotonic()
    child.start()
    watch = psutil.Process(child.pid)
    peak, status, value = 0.0, None, None
    while status is None:
        if receiver.poll(poll):
            status, value, child_peak = receiver.recv()
            peak = max(peak, child_peak * 1e9)  # the child's own peak (steps shorter than poll)
            break
        rss = _memory(watch)
        peak = max(peak, rss)
        if rss > mem_gb * 1e9:
            _kill(watch)
            status = "memout"
        elif timeout_s is not None and time.monotonic() - start > timeout_s:
            _kill(watch)
            status = "timeout"
        elif not child.is_alive() and not receiver.poll():
            if child.exitcode == -signal.SIGKILL:  # killed by someone else: on Linux, the OOM killer
                status, value = "memout", "killed by the operating system (SIGKILL: out of memory?)"
            else:
                status, value = "error", f"child exited with code {child.exitcode}"
    child.join()
    return StepResult(value, status, peak / 1e9, time.monotonic() - start)
