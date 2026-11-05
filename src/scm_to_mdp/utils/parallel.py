"""Run independent jobs (e.g. one per (N, seed)) in a pool of worker processes, with a progress bar.

Workers are started with "forkserver": clean processes that have not used PyTorch, so they can fork
their own memory-limited steps (utils/limits.py). If a worker dies (e.g. killed by the OS), the pool
breaks and every unfinished job fails with it: we start a new pool and retry those jobs once. A job
that breaks the pool twice, or whose fn raises, is reported as {"crashed": True, "error": ...}.

run_dynamic keeps the pool busy across several sweeps: after each finished job, `then` may hand in
new jobs (e.g. the next wave of sizes of the setting that job belonged to), so the workers of a big
machine are not idle while one setting waits for its slowest seed. With `priority` (job -> a sortable
rank), at most `workers` jobs are handed to the pool at a time, and a free worker takes the waiting job
of the lowest rank (the oldest among equal ranks): e.g. one domain's jobs before another's, while the
workers that domain cannot fill take the others' jobs.

When the process that started the run is gone (stopped from the IDE, killed, crashed), its workers
are stopped too: a small guard process notes the run's process tree every half second and kills it
once the runner is gone. (Idle pool workers would otherwise wait for work forever, and a thread in
the workers is not an option: they fork their steps, and forking a process with threads can
deadlock the child.)
"""

import itertools
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import psutil
from tqdm import tqdm

GUARD = r"""
import sys, time, psutil
main, me = psutil.Process(int(sys.argv[1])), psutil.Process()
def alive(p):
    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False
tree = []
while alive(main):
    try:
        tree = [c for c in main.children(recursive=True) if c.pid != me.pid]
    except psutil.Error:
        pass
    time.sleep(0.5)
for p in tree:
    try:
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except psutil.Error:
        pass
"""


def start_guard() -> subprocess.Popen:
    """The guard of this process's tree (see the module docstring); terminate() it when done."""
    return subprocess.Popen([sys.executable, "-c", GUARD, str(os.getpid())], stdin=subprocess.DEVNULL)


def exit_with(pid: int, every: float = 1.0) -> None:
    """Exit this process (from a daemon thread) as soon as process pid is gone. For processes that
    do not fork afterwards (the steps of utils/limits.py)."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        os._exit(1)

    def watch():
        while True:
            time.sleep(every)
            try:
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    os._exit(1)
            except psutil.NoSuchProcess:
                os._exit(1)

    threading.Thread(target=watch, daemon=True, name=f"exit-with-{pid}").start()


def _kill_workers(pool: ProcessPoolExecutor) -> None:
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):
                child.kill()
            proc.kill()
        except (psutil.Error, OSError):
            pass


def _quiet(fn, job, env: dict):
    os.environ.update(env)  # the run's choices (utils/compute.py), also when the forkserver is older
    os.environ["TQDM_DISABLE"] = "1"  # the libraries' bars would mix with the main bar
    return fn(*job)


def _env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith("SCM_TO_MDP_")}


def run_dynamic(
    fn: Callable,
    jobs: list[tuple],
    workers: int,
    then: Callable[[tuple, object], list[tuple]] | None = None,
    bar: tqdm | None = None,
    priority: Callable[[tuple], object] | None = None,
) -> Iterator[tuple[tuple, object]]:
    """Yield (job, fn(*job)) as jobs finish; then(job, result) -> new jobs to run (optional);
    priority(job) -> rank, lowest first (optional; module docstring). fn must be importable (defined at
    module level)."""
    keys = itertools.count()
    queue = [(next(keys), job) for job in jobs]  # (key, job) not submitted yet
    attempts: dict[int, int] = {}  # key -> times the pool broke while the job ran
    env = _env()
    ctx = multiprocessing.get_context("forkserver")
    guard = start_guard()
    try:
        yield from _drive(fn, queue, workers, then, bar, keys, attempts, env, ctx, priority)
    finally:
        guard.terminate()
        guard.wait()


def _next(queue: list, priority) -> int:
    """The index in queue of the job to submit: the first, or the lowest (rank, key)."""
    if priority is None:
        return 0
    return min(range(len(queue)), key=lambda i: (priority(queue[i][1]), queue[i][0]))


def _drive(fn, queue, workers, then, bar, keys, attempts, env, ctx, priority=None):
    while queue:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        running: dict = {}
        broken = False
        try:
            while queue or running:
                while queue and not broken and (priority is None or len(running) < workers):
                    key, job = queue.pop(_next(queue, priority))
                    running[pool.submit(_quiet, fn, job, env)] = key, job
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    key, job = running.pop(future)
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        broken = True
                        attempts[key] = attempts.get(key, 0) + 1
                        if attempts[key] <= 1:
                            queue.append((key, job))
                            continue
                        if bar is not None:
                            bar.write("a job broke the worker pool twice: skipped")
                        result = {"crashed": True}
                    except Exception as err:  # noqa: BLE001 - fn raised: report it, keep the other jobs
                        result = {"crashed": True, "error": repr(err)}
                    if bar is not None:
                        bar.update(1)
                    yield job, result
                    if then is not None:
                        queue.extend((next(keys), new) for new in then(job, result))
                if broken and not running:
                    break
        except BaseException:  # Ctrl+C, or the caller stopped iterating: do not wait for the jobs
            _kill_workers(pool)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if broken and queue and bar is not None:
            bar.write(f"a worker died; retrying {len(queue)} unfinished job(s) in a new pool")


def run_jobs(
    fn: Callable, jobs: list[tuple], workers: int, desc: str = "runs", bar: tqdm | None = None
) -> Iterator[tuple[tuple, dict]]:
    """Yield (job, fn(*job)) as jobs finish. fn must be importable (defined at module level).
    bar: an existing progress bar to advance (e.g. one bar over several calls); else a new one."""
    own_bar = bar is None
    if own_bar:
        bar = tqdm(total=len(jobs), desc=f"{desc} ({workers} workers)", unit="run", dynamic_ncols=True)
    yield from run_dynamic(fn, jobs, workers, bar=bar)
    if own_bar:
        bar.close()
