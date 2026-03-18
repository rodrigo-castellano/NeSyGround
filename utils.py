"""Shared utilities for grounder-based profiling and benchmarking."""

from __future__ import annotations

import time
from typing import Callable, Tuple

import torch


def timed_warmup(warmup_fn: Callable[[], None]) -> Tuple[float, float]:
    """Run warmup twice, return (cold_s, warm_s).

    Run 1 (cold): warms all caches — inductor FX graphs, triton kernel
    compilation, CUDA graph capture (reduce-overhead mode).

    Run 2 (warm): measures steady-state performance with all caches hot.

    Returns:
        (cold_s, warm_s) — cold start time and warm replay time in seconds.
    """
    # Run 1: cold — populates all caches
    t0 = time.perf_counter()
    warmup_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0

    # Run 2: warm — all caches hot
    t0 = time.perf_counter()
    warmup_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0

    return cold_s, warm_s
