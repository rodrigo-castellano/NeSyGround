"""ForwardGrounder — forward chaining as a grounder (closure to fixpoint).

Runs ``run_forward_chaining`` over the KB's rules+facts and returns the derived
closure as a ``Closure`` (hashes / n / E) via the single verb ``ground``; the
closure triples are read off the result (``kb.with_closure(g.ground().facts())``).
FC operates over the REAL predicate range (max id in facts/rules + 1), not the
loader's inflated ``predicate_no`` (which would build thousands of empty matrices).

It shares only the execution Cell/CapabilityRow vocabulary: one ``Cell(SPARSE, EAGER)``,
producing a ``Closure``, not tiers (``producible_tiers() == frozenset()``).
"""
from __future__ import annotations

from typing import FrozenSet

import torch
import torch.nn as nn

from grounder.core import GroundRequest, Tier
from grounder.data.kb import KB
from grounder.data.rule_index import compile_rules
from grounder.execution.capability import EAGER, CapabilityRow, Cell
from grounder.forward.api import Closure
from grounder.forward.router import run_forward_chaining
from grounder.base.types import Layout


class ForwardGrounder(nn.Module):
    """Forward-chaining closure grounder (method = 'spmm' | 'staged')."""

    def __init__(self, kb: KB, *, method: str = "spmm", depth: int = 10,
                 join_algo: str = "staged") -> None:
        super().__init__()
        self.kb = kb
        self.method = method
        self.depth = depth
        self.join_algo = join_algo
        self._rules = compile_rules(kb.rules_heads_idx, kb.rules_bodies_idx,
                                    kb.rule_lens, kb.constant_no)
        self._num_entities = kb.constant_no + 1          # 1-based: ids 0..constant_no
        self._num_predicates = kb.real_num_predicates()

    @torch.no_grad()
    def ground(self, request: GroundRequest = GroundRequest()) -> Closure:
        """The runtime verb — run the closure to fixpoint → ``Closure`` (FC reads only
        ``closure_depth``). Triples/hashes are read off the result (``.facts()`` / ``.hashes``)."""
        depth = request.closure_depth if request.closure_depth is not None else self.depth
        hashes, n = run_forward_chaining(
            self._rules, self.kb.fact_index.facts_idx, self._num_entities,
            self._num_predicates, depth=depth, device=str(self.kb.device_),
            method=self.method, join_algo=self.join_algo)
        return Closure(hashes=hashes, n_provable=n, num_entities=self._num_entities)

    def capability_row(self) -> CapabilityRow:
        """One sparse-eager cell (vocab-share, no iter_chunks claim)."""
        return CapabilityRow(frozenset({Cell(Layout.SPARSE, EAGER)}))

    def producible_tiers(self) -> FrozenSet[Tier]:
        """FC produces a ``Closure``, not BC tiers."""
        return frozenset()

    def rebound(self, kb: KB) -> "ForwardGrounder":
        """Re-snapshot over a rewritten KB (transforms); same method/depth/join."""
        return ForwardGrounder(kb, method=self.method, depth=self.depth,
                               join_algo=self.join_algo)


__all__ = ["ForwardGrounder"]
