"""Run independent jobs (e.g. one per (N, seed)) in a pool of worker processes, with a progress bar.

Workers are started with "forkserver": clean processes that have not used PyTorch, so they can fork
their own memory-limited steps (utils/limits.py). If a worker dies (e.g. killed by the OS), the pool
breaks and every unfinished job fails with it: we start a new pool and retry those jobs once. A job
that breaks the pool twice, or whose fn raises, is reported as {"crashed": True, "error": ...}.
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
    pending, attempts = list(range(len(jobs))), [0] * len(jobs)  # jobs by position (need not be hashable)
    while pending:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("forkserver"))
        futures = {pool.submit(_quiet, fn, jobs[i]): i for i in pending}
        unfinished, broken = set(pending), False
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
            except BrokenProcessPool:
                broken = True
                continue
            except Exception as err:  # noqa: BLE001 - fn raised: report it, keep the other jobs
                result = {"crashed": True, "error": repr(err)}
            unfinished.discard(i)
            bar.update(1)
            yield jobs[i], result
        pool.shutdown(cancel_futures=True)
        pending = []
        if broken:
            for i in sorted(unfinished):
                attempts[i] += 1
                if attempts[i] <= 1:
                    pending.append(i)
                else:
                    bar.write(f"job {i}: worker died twice, job skipped")
                    bar.update(1)
                    yield jobs[i], {"crashed": True}
            if pending:
                bar.write(f"a worker died; retrying {len(pending)} unfinished job(s) in a new pool")
    if own_bar:
        bar.close()
