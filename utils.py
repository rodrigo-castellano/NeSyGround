"""Shared utilities for grounder-based profiling and benchmarking."""

from __future__ import annotations

import time
from typing import Callable

import torch


def timed_warmup(warmup_fn: Callable[[], None]) -> float:
    """Run warmup, detect cold inductor cache, re-run if needed.

    Returns the warm warmup time in seconds.

    Uses ``torch._dynamo.utils.counters`` to detect cache misses.
    If the first run has cache misses (cold inductor cache), re-runs
    to get a warm measurement.
    """
    from torch._dynamo.utils import counters

    counters.clear()
    t0 = time.perf_counter()
    warmup_fn()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - t0

    misses = counters.get("inductor", {}).get("fxgraph_cache_miss", 0)
    if misses == 0:
        return first_s  # cache was warm

    # Cold cache — first run populated it, measure warm
    print(f"  Cold inductor cache ({misses} misses, {first_s:.1f}s). Re-measuring warm...")
    counters.clear()
    t0 = time.perf_counter()
    warmup_fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0
