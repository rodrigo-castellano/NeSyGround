"""Compiler — THE ONLY ``torch.compile`` call site in the library.

No other module may call ``torch.compile``; everything routes here so compile
policy lives in one place (enforced by
``contracts/test_execution.py::test_single_compile_site``). ``Compiler.wrap``
reads a granular ``CompileSpec`` (backend/dynamic/fullgraph) — the SINGLE call.
"""
from __future__ import annotations

from typing import Callable

import torch

from grounder.execution.capability import CompileSpec, Integration
from grounder.vocab.shapes import Shapes


def _bump_recompile_limit() -> None:
    """The per-step path keys a few graph variants; lift the limit once."""
    import torch._dynamo as dynamo
    if getattr(dynamo.config, "recompile_limit", 0) < 64:
        dynamo.config.recompile_limit = 64


class Compiler:
    @staticmethod
    def wrap(fn: Callable, *, spec: CompileSpec, shapes: Shapes) -> Callable:
        """Return ``fn`` compiled per ``spec`` (or ``fn`` unchanged when eager).
        THE ONLY appearance of ``torch.compile`` in the library."""
        if spec.eager:
            return fn
        if spec.integration is Integration.STEP:
            _bump_recompile_limit()
        return torch.compile(fn, mode=spec.backend.torch_mode(),
                             fullgraph=spec.fullgraph, dynamic=spec.dynamic)


__all__ = ["Compiler"]
