"""Run one pipeline step with a memory limit and a time limit.

The step runs in a forked child process. A monitor in the parent reads the child's memory (with the
child's own children) and kills it above the limit ("memout"), or when it runs longer than the time
limit ("timeout"). Memory is the proportional set size on Linux (PSS: private pages in full, each page
shared with other processes divided among them) and the resident set size elsewhere (RSS; MEMORY, or the
environment variable SCM_TO_MDP_MEMORY = "pss" | "uss" | "rss"). On Linux a forked child's RSS counts in
full every page it shares with its parent (the data and the true model), so RSS overstates each step by
the parent's size: on the server at SysAdmin N = 11, 4.3 GB for a step that uses under 1 GB itself. The
sum of the PSS of all processes is the memory the machine uses, so workers x the limit <= the RAM still
holds. macOS has no PSS, and its USS counts a parent's page as private once the child has read it, so
macOS keeps RSS (which there counts only the pages the child touched). Reading PSS walks the process's
memory map (1-10 ms for a process of a few GB), done every poll seconds.
macOS does not enforce RLIMIT_AS, hence the monitor. A child
killed by another process with SIGKILL (on Linux, the out-of-memory killer) also counts as "memout".
Time is time.monotonic(), which does not advance while the computer sleeps (macOS and Linux): a
laptop going to sleep in the middle of a step must not turn that step into a timeout.

Fork is used because it starts at once and the child shares the parent's data. It is safe only if
the parent never ran PyTorch computations itself: forking a process whose PyTorch thread pool was
used makes large torch operations fail in the child ("Invalid thread pool"). So every step that
uses NBN / torch must run through run_limited, and the parent must only prepare data. Training a
neural network (offline DQN / CQL) needs more: on macOS a forked child that trains is killed by the
Objective-C runtime once the parent has started initialising PyTorch's Metal backend ("MPSGraph ...
initialize may have been in progress in another thread when fork() was called"), so those steps
use start_method="spawn" (a fresh process; fn and its arguments must then be picklable).

The child exits by itself when the process that started it is gone (a daemon thread: the child
never forks), and the guard of utils/parallel.py stops the pool workers, so stopping a run (IDE
stop button, Ctrl+C, kill) never leaves a step computing in the background.
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

from scm_to_mdp.core import NotApplicable


@dataclass
class StepResult:
    value: Any  # the step's output (or the error message when status == "error")
    status: str  # "ok", "memout", "timeout", "not_applicable" or "error"
    peak_gb: float  # peak memory in the MEMORY measure (sampled every poll seconds)
    seconds: float


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9  # bytes on macOS, KiB on Linux


def _child(conn, fn, args, parent: int):
    from scm_to_mdp.utils.parallel import exit_with

    exit_with(parent)
    os.environ["TQDM_DISABLE"] = "1"  # silence the progress bars of the libraries
    try:
        conn.send(("ok", fn(*args), _peak_rss_gb()))
    except MemoryError:
        conn.send(("memout", None, _peak_rss_gb()))
    except NotApplicable as err:  # the method does not apply to this model (core.NotApplicable)
        conn.send(("not_applicable", str(err), _peak_rss_gb()))
    except Exception as err:  # noqa: BLE001 - reported, not hidden
        conn.send(("error", repr(err), _peak_rss_gb()))


def _measure() -> str:
    asked = os.environ.get("SCM_TO_MDP_MEMORY", "").lower()
    if asked in ("pss", "uss", "rss"):
        return asked
    return "pss" if sys.platform.startswith("linux") else "rss"


MEMORY = _measure()


def _bytes(proc: psutil.Process) -> int:
    """proc's memory in the MEMORY measure (RSS when the full information cannot be read)."""
    if MEMORY != "rss":
        try:
            info = proc.memory_full_info()
            value = getattr(info, MEMORY, None)
            if value is not None:
                return int(value)
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return proc.memory_info().rss


def _memory(proc: psutil.Process, family: list[psutil.Process]) -> int:
    """Memory of proc and of its children (family: listed by the caller, see run_limited), in MEMORY."""
    try:
        total = _bytes(proc)
    except psutil.NoSuchProcess:
        return 0
    for c in family:
        try:
            total += _bytes(c)
        except psutil.NoSuchProcess:
            pass
    return total


def _family(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _kill(proc: psutil.Process):
    try:
        for c in proc.children(recursive=True):
            c.kill()
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def run_limited(
    fn,
    *args,
    mem_gb: float,
    timeout_s: float | None = None,
    poll: float = 0.02,
    slow_poll: float = 0.2,
    start_method: str = "fork",
) -> StepResult:
    """fn(*args) in a child process, stopped above mem_gb of memory or after timeout_s seconds.
    The monitor reads the memory every `poll` seconds during the first second, then every
    `slow_poll` seconds, and lists the child's own children once a second: listing them scans every
    process of the machine on Linux, which at 50 times a second cost each worker half a core."""
    ctx = multiprocessing.get_context(start_method)
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child, args=(sender, fn, args, os.getpid()))
    start = time.monotonic()
    child.start()
    watch = psutil.Process(child.pid)
    family, listed = [], start
    peak, status, value = 0.0, None, None
    while status is None:
        now = time.monotonic()
        if receiver.poll(poll if now - start < 1.0 else slow_poll):
            status, value, child_peak = receiver.recv()
            if MEMORY == "rss":  # the child's own peak (between two readings too); RSS only
                peak = max(peak, child_peak * 1e9)
            break
        if time.monotonic() - listed > 1.0:
            family, listed = _family(watch), time.monotonic()
        used = _memory(watch, family)
        peak = max(peak, used)
        if used > mem_gb * 1e9:
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
