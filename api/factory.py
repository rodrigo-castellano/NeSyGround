"""Grounder factory — ONE construction entry: ``make_grounder(kb, config, …)``.

Dispatches on the config TYPE (no string grammar, no BC/FC fork):
  Backward(resolution=PBC|SLD, …) -> BackwardGrounder
  Forward(depth, …)                   -> ForwardGrounder
Exec knobs (layout/compile/chunk_size) ride here; everything else
lives on the typed config.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from grounder.api.config import Backward, Forward
from grounder.data.kb import KB
from grounder.api.backward import BackwardGrounder


def make_grounder(
    kb: KB,
    config,
    *,
    layout: str = "auto",
    compile: str = "off",
    chunk_size: Optional[int] = None,
) -> nn.Module:
    """Build the family grounder over ``kb`` by dispatching on ``type(config)``."""
    if isinstance(config, Backward):
        return BackwardGrounder(kb, config, layout=layout, compile=compile,
                                chunk_size=chunk_size)
    if isinstance(config, Forward):
        from grounder.api.forward import ForwardGrounder
        return ForwardGrounder(kb, method=config.method, depth=config.depth,
                               join_algo=config.join_algo)
    raise TypeError(f"Unknown grounder config type: {type(config).__name__}")


__all__ = ["make_grounder"]
