"""Compiler — the single owner of every ``torch.compile`` call in the grounder.

Phase 2b of the restructure: the four scattered compile sites (the outer
SLD/RTF per-batch compile in ``bc/forward.py``, the per-step enum compile in
``bc/compiled_step.py``, the closure resolver in ``resolution/closure.py``,
and the optional standardizer in ``resolution/standardization.py``) all route
through :func:`Compiler.wrap`. Algorithm modules no longer call
``torch.compile`` directly, so compilation policy lives in exactly one place.

This step is behavior-neutral: ``wrap`` forwards the same ``mode`` /
``fullgraph`` / ``dynamic`` each site already used, and owns the one-time
``recompile_limit`` bump the per-step enum path needs. Later phases layer a
single ``execution`` flag (compiled vs dynamic) and the per-step single-graph
policy on top of this seam.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch


class Compiler:
    """Owns the grounder's ``torch.compile`` invocations."""

    @staticmethod
    def wrap(
        fn: Callable[..., Any],
        *,
        mode: Optional[str] = None,
        fullgraph: bool = True,
        dynamic: Optional[bool] = None,
        bump_recompile_limit: bool = False,
    ) -> Callable[..., Any]:
        """Return a ``torch.compile``-wrapped ``fn``.

        Args mirror the call sites verbatim (no policy change):

        - ``mode``: the compile mode (``grounder.compile_mode`` for the
          SLD/RTF/enum/standardize sites, hardcoded ``"reduce-overhead"``
          for closure). ``None`` = torch's default mode.
        - ``fullgraph``: True for SLD/RTF/enum/standardize, False for closure.
        - ``dynamic``: ``None`` (torch default / auto) for the outer SLD/RTF
          and standardize sites; ``False`` (static shapes) for the per-step
          enum and closure sites.
        - ``bump_recompile_limit``: raise ``torch._dynamo.config.recompile_
          limit`` to >=64 before compiling (the per-step enum path keys a
          handful of graph variants and must not trip the default limit).
        """
        if bump_recompile_limit:
            import torch._dynamo as _dynamo
            if getattr(_dynamo.config, "recompile_limit", 0) < 64:
                _dynamo.config.recompile_limit = 64
        return torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


__all__ = ["Compiler"]
