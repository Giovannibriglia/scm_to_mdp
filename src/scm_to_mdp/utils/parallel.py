"""Run independent jobs (e.g. one per (N, seed)) in a pool of worker processes, with a progress bar.

Workers are started with "forkserver": clean processes that have not used PyTorch, so they can fork
their own memory-limited steps (utils/limits.py). If a worker dies (e.g. killed by the OS), the pool
breaks and every unfinished job fails with it: we start a new pool and retry those jobs once. A job
that breaks the pool twice is reported as {"crashed": True}.
"""

import multiprocessing
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from tqdm import tqdm


def _quiet(fn, job):
    os.environ["TQDM_DISABLE"] = "1"  # the libraries' bars would mix with the main bar
    return fn(*job)


def run_jobs(
    fn: Callable, jobs: list[tuple], workers: int, desc: str = "runs", bar: tqdm | None = None
) -> Iterator[tuple[tuple, dict]]:
    """Yield (job, fn(*job)) as jobs finish. fn must be importable (defined at module level).
    bar: an existing progress bar to advance (e.g. one bar over several calls); else a new one."""
    own_bar = bar is None
    if own_bar:
        bar = tqdm(total=len(jobs), desc=f"{desc} ({workers} workers)", unit="run", dynamic_ncols=True)
    pending, attempts = list(jobs), dict.fromkeys(jobs, 0)
    while pending:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("forkserver"))
        futures = {pool.submit(_quiet, fn, job): job for job in pending}
        unfinished, broken = set(pending), False
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except BrokenProcessPool:
                broken = True
                continue
            unfinished.discard(job)
            bar.update(1)
            yield job, result
        pool.shutdown(cancel_futures=True)
        pending = []
        if broken:
            for job in sorted(unfinished):
                attempts[job] += 1
                if attempts[job] <= 1:
                    pending.append(job)
                else:
                    bar.write(f"{job}: worker died twice, job skipped")
                    bar.update(1)
                    yield job, {"crashed": True}
            if pending:
                bar.write(f"a worker died; retrying {len(pending)} unfinished job(s) in a new pool")
    if own_bar:
        bar.close()
