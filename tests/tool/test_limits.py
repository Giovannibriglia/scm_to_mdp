import time

import numpy as np

from scm_to_mdp.utils.limits import run_limited


def add(a, b):
    return a + b


def sleep(seconds):
    time.sleep(seconds)


def fail():
    raise ValueError("boom")


def allocate(gb):
    x = np.ones(int(gb * 1e9 / 8))
    time.sleep(1)
    return x.sum()


def test_ok():
    out = run_limited(add, 1, 2, mem_gb=2)
    assert (out.status, out.value) == ("ok", 3)


def test_timeout():
    out = run_limited(sleep, 10, mem_gb=2, timeout_s=0.5)
    assert out.status == "timeout" and out.seconds < 5


def test_error():
    out = run_limited(fail, mem_gb=2)
    assert out.status == "error" and "boom" in out.value


def test_memout():
    out = run_limited(allocate, 1.0, mem_gb=0.4)
    assert out.status == "memout"


def allocate_late(gb):
    time.sleep(1.5)  # past the first second: the monitor reads the memory less often
    x = np.ones(int(gb * 1e9 / 8))
    time.sleep(1)
    return x.sum()


def test_memout_after_the_first_second():
    out = run_limited(allocate_late, 1.0, mem_gb=0.4)
    assert out.status == "memout"


SHARED = None


def read_shared():
    """Reads (does not write) the parent's array: its pages stay shared with the parent."""
    total = float(SHARED.sum())
    time.sleep(1)
    return total


def test_pages_shared_with_the_parent_do_not_count_against_the_step():
    """PSS (Linux): a forked step that only reads 0.8 GB of its parent's data is charged its share of those
    pages (about half, shared with the parent), while memory it allocates itself still counts in full
    (and is stopped above the limit)."""
    from scm_to_mdp.utils import limits

    global SHARED
    SHARED = np.ones(int(0.8e9 / 8))
    try:
        out = run_limited(read_shared, mem_gb=2)
        assert out.status == "ok"
        if limits.MEMORY == "pss":
            assert out.peak_gb < 0.6, out.peak_gb
        own = run_limited(allocate, 0.8, mem_gb=2)
        assert own.status == "ok" and own.peak_gb > 0.6, own.peak_gb
        assert run_limited(allocate, 1.5, mem_gb=1).status == "memout"
    finally:
        SHARED = None
