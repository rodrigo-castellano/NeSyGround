"""Grounder — base class for all grounder algorithms.

Holds a reference to a KB (knowledge base). Subclasses implement forward().
"""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from grounder.kb import KB
from grounder.types import GrounderOutput


class Grounder(nn.Module):
    """Base grounder — owns a KB reference and defines the grounding interface.

    All KB data (facts, rules, indices, metadata) is accessed via ``self.kb``.
    Subclasses implement ``forward()`` with their specific algorithm.
    """

    def __init__(self, kb: KB) -> None:
        super().__init__()
        self.kb = kb

    @abstractmethod
    def forward(self, queries: Tensor, query_mask: Tensor, **kwargs) -> GrounderOutput:
        """Ground queries against the KB.

        Args:
            queries:    [B, 3] query atoms (pred, arg0, arg1)
            query_mask: [B] boolean mask (True = active query)

        Returns:
            GrounderOutput with state and evidence.
        """
        ...
