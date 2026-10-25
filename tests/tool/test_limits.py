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
