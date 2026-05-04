"""Generic speed profiler for grounders.

Two entry points:

  :func:`time_runner` — time an arbitrary callable: ``n_warmup`` untimed
  warmups, then ``n_timed`` timed calls, with optional CUDA sync per
  iteration. Returns aggregate ms statistics (median / min / max /
  mean) plus the raw sample list.

  :func:`time_grounder` — convenience wrapper around :func:`time_runner`
  for torch grounders. Auto-detects CUDA from ``queries.device`` and
  installs ``torch.cuda.synchronize`` as the per-iteration sync.

Both default to ``n_warmup=1, n_timed=1`` — the precommit budget. Set
``n_timed >= 3`` for stable median measurements when collecting
release-quality numbers.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, Optional

import torch


__all__ = ["time_runner", "time_grounder"]


def time_runner(
    fn: Callable[[], Any],
    *,
    n_warmup: int = 1,
    n_timed: int = 1,
    sync_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Time a callable.

    Args:
        fn: zero-arg callable to invoke.
        n_warmup: number of untimed warmup iterations (default 1).
        n_timed: number of timed iterations (default 1).
        sync_fn: optional barrier called after each iteration (e.g.
            ``torch.cuda.synchronize``). Wall-clock is captured between
            ``perf_counter`` calls *bracketing* the sync.

    Returns:
        ``{ms_median, ms_min, ms_max, ms_mean, samples_ms, n_timed,
        n_warmup}``. With ``n_timed=1`` all summary stats are equal
        and the single sample is in ``samples_ms[0]``.
    """
    for _ in range(n_warmup):
        fn()
    if sync_fn is not None:
        sync_fn()

    samples: list = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        if sync_fn is not None:
            sync_fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "ms_median": float(statistics.median(samples)),
        "ms_min": float(min(samples)),
        "ms_max": float(max(samples)),
        "ms_mean": float(statistics.mean(samples)),
        "samples_ms": samples,
        "n_timed": n_timed,
        "n_warmup": n_warmup,
    }


def time_grounder(
    grounder,
    queries: torch.Tensor,
    qmask: torch.Tensor,
    fwd_kwargs: Optional[Dict[str, Any]] = None,
    *,
    n_warmup: int = 1,
    n_timed: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Time a torch grounder's forward pass.

    Calls ``grounder(queries, qmask, **fwd_kwargs)`` under
    ``torch.no_grad()``. CUDA sync is applied automatically when the
    grounder runs on a cuda device (inferred from ``queries.device``
    unless ``device`` is given explicitly).
    """
    fwd_kwargs = fwd_kwargs or {}
    if device is None:
        device = queries.device
    is_cuda = (isinstance(device, torch.device) and device.type == "cuda") \
        or (isinstance(device, str) and device.startswith("cuda"))
    sync_fn = torch.cuda.synchronize if is_cuda else None

    def _call():
        with torch.no_grad():
            grounder(queries, qmask, **fwd_kwargs)

    return time_runner(_call, n_warmup=n_warmup, n_timed=n_timed,
                       sync_fn=sync_fn)
