"""LazyGrounder — predicate-filtered wrapper over BCGrounder.

Computes predicate reachability via BFS on the rule head-to-body predicate
graph, filters rules to those with reachable head predicates, and passes
the filtered rule set to a BCGrounder.

Compatible with torch.compile(fullgraph=True, mode='reduce-overhead').
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Set

import torch
import torch.nn as nn
from torch import Tensor

from grounder.bc.bc import BCGrounder
from grounder.types import GroundingResult


class LazyGrounder(nn.Module):
    """Predicate-filtered grounder — wraps BCGrounder with fewer rules.

    Builds a predicate reachability graph from rules (head -> body predicates),
    performs BFS from query predicates, and filters rules to those with
    reachable head predicates before passing to the inner grounder.

    Args:
        facts_idx:           [F, 3] fact triples
        rules_heads_idx:     [R, 3] rule head atoms
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded)
        rule_lens:           [R] number of body atoms per rule
        constant_no:         highest constant index
        padding_idx:         padding value
        device:              target device
        query_predicates:    optional set of query predicate indices for
                             filtering (None = all rule-head predicates)
        **kwargs:            forwarded to BCGrounder
    """

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        *,
        query_predicates: Optional[Set[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        reachable = self._compute_reachable_predicates(
            rules_heads_idx, rules_bodies_idx, rule_lens,
            query_predicates,
        )

        # Filter rules to reachable head predicates
        R = rules_heads_idx.shape[0]
        keep_mask = torch.zeros(R, dtype=torch.bool)
        for i in range(R):
            if int(rules_heads_idx[i, 0].item()) in reachable:
                keep_mask[i] = True

        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        n_orig = R
        n_filt = keep_indices.shape[0]

        if n_filt > 0:
            filtered_heads = rules_heads_idx[keep_indices]
            filtered_bodies = rules_bodies_idx[keep_indices]
            filtered_lens = rule_lens[keep_indices]
        else:
            Bmax = rules_bodies_idx.shape[1] if rules_bodies_idx.dim() > 1 else 1
            filtered_heads = rules_heads_idx.new_zeros(0, 3)
            filtered_bodies = rules_bodies_idx.new_zeros(0, Bmax, 3)
            filtered_lens = rule_lens.new_zeros(0)

        print(
            f"  LazyGrounder: {n_orig} rules -> {n_filt} rules "
            f"({n_orig - n_filt} filtered, "
            f"{len(reachable)} reachable predicates)"
        )

        self._inner = BCGrounder(
            facts_idx, filtered_heads, filtered_bodies, filtered_lens,
            constant_no, padding_idx, device,
            **kwargs,
        )

        self.effective_total_G: int = self._inner.effective_total_G

    @property
    def max_body_atoms(self) -> int:
        return self._inner.M

    @staticmethod
    def _compute_reachable_predicates(
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        query_predicates: Optional[Set[int]] = None,
    ) -> Set[int]:
        """BFS on head->body predicate graph to find reachable predicates."""
        R = rules_heads_idx.shape[0]
        if R == 0:
            return set()

        head_to_body: dict = {}
        all_head_preds: Set[int] = set()

        for i in range(R):
            head_pred = int(rules_heads_idx[i, 0].item())
            all_head_preds.add(head_pred)
            body_len = int(rule_lens[i].item())
            body_preds: Set[int] = set()
            for j in range(body_len):
                bp = int(rules_bodies_idx[i, j, 0].item())
                body_preds.add(bp)
            head_to_body.setdefault(head_pred, set()).update(body_preds)

        seeds = (query_predicates & all_head_preds
                 if query_predicates is not None else all_head_preds)

        reachable: Set[int] = set(seeds)
        frontier = deque(seeds)
        while frontier:
            pred = frontier.popleft()
            for body_pred in head_to_body.get(pred, set()):
                if body_pred not in reachable:
                    reachable.add(body_pred)
                    frontier.append(body_pred)

        return reachable

    def forward(
        self, queries: Tensor, query_mask: Tensor,
    ) -> GroundingResult:
        return self._inner(queries, query_mask)

    def __repr__(self) -> str:
        return f"LazyGrounder(inner={self._inner!r})"
