import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from scm_to_mdp.utils import limits
from scm_to_mdp.utils.limits import run_limited

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="macOS's physical footprint")


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


def budget_of_the_step():
    return limits.STEP_MEM_GB


def ask_before_allocating(need_gb):
    limits.require_memory(int(need_gb * 1e9), "this table")
    return "allocated"


def test_a_step_knows_its_limit_with_either_start_method():
    assert limits.STEP_MEM_GB is None  # outside a step
    assert run_limited(budget_of_the_step, mem_gb=1.5).value == 1.5
    assert run_limited(budget_of_the_step, mem_gb=1.5, start_method="spawn").value == 1.5


def test_a_size_known_in_advance_is_a_memout_before_allocating():
    out = run_limited(ask_before_allocating, 1000.0, mem_gb=2)
    assert out.status == "memout" and out.peak_gb < 1.0 and out.seconds < 5
    assert run_limited(ask_before_allocating, 1.0, mem_gb=2).value == "allocated"


def test_outside_a_step_the_limit_is_the_machines_ram():
    with pytest.raises(MemoryError, match="this table needs"):
        limits.require_memory(100 * psutil.virtual_memory().total, "this table")
    limits.require_memory(1000, "this table")


@darwin
def test_the_default_on_macos_is_the_footprint():
    assert limits.MEMORY == "footprint"


@darwin
def test_the_footprint_follows_what_a_process_allocates():
    before = limits._rusage(os.getpid()).ri_phys_footprint
    block = np.ones(int(0.4e9 / 8))
    grown = limits._rusage(os.getpid())
    assert grown.ri_phys_footprint - before > 0.35e9
    assert grown.ri_lifetime_max_phys_footprint >= grown.ri_phys_footprint
    del block


@darwin
def test_a_step_that_rss_cannot_see_is_still_stopped(monkeypatch):
    """The panic of 2026-11-06: most of a step's memory was compressed, which RSS leaves out. Here RSS reports
    nothing at all: the old measure lets the step run to its end; the footprint stops it."""
    tiny = SimpleNamespace(rss=1)
    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: tiny)
    monkeypatch.setattr(limits, "MEMORY", "rss")
    assert run_limited(allocate, 1.0, mem_gb=0.4).status == "ok"
    monkeypatch.setattr(limits, "MEMORY", "footprint")
    assert run_limited(allocate, 1.0, mem_gb=0.4).status == "memout"


@darwin
def test_the_footprint_does_not_charge_a_step_for_a_parent_it_did_not_touch():
    global SHARED
    SHARED = np.ones(int(0.8e9 / 8))
    try:
        out = run_limited(add, 1, 2, mem_gb=2)
        assert out.status == "ok" and out.peak_gb < 0.2, out.peak_gb
    finally:
        SHARED = None


@pytest.mark.skipif(sys.platform == "darwin", reason="the fallback of the other systems")
def test_elsewhere_the_footprint_falls_back_to_rss(monkeypatch):
    monkeypatch.setenv("SCM_TO_MDP_MEMORY", "footprint")
    assert limits._measure() == "rss"


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
