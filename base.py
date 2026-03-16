"""Grounder — base class for all grounder algorithms.

Holds a reference to a KB (knowledge base). Subclasses add algorithm logic.
"""

from __future__ import annotations

import torch.nn as nn

from grounder.kb import KB


class Grounder(nn.Module):
    """Base grounder — owns a KB reference.

    All KB data (facts, rules, indices, metadata) is accessed via ``self.kb``.
    Subclasses add algorithm-specific logic (backward chaining, etc.).
    """

    def __init__(self, kb: KB) -> None:
        super().__init__()
        self.kb = kb
