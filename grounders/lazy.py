"""LazyGrounder — predicate-filtered wrapper over ParametrizedBCGrounder.

Computes predicate reachability via BFS on the rule head-to-body predicate
graph, filters rules to those with reachable head predicates, and passes
the filtered rule set to a ParametrizedBCGrounder.

This results in a smaller provable set and fewer rules to process, while
producing identical groundings for reachable queries.

Grounder naming: lazy_W_D — same width/depth semantics as backward_W_D,
but with smaller provable set due to predicate filtering.

Compatible with torch.compile(fullgraph=True, mode='reduce-overhead').
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Set

import torch
import torch.nn as nn
from torch import Tensor

from grounder.grounders.parametrized import ParametrizedBCGrounder
from grounder.types import ForwardResult
from grounder.compilation import compile_rules


class LazyGrounder(nn.Module):
    """Predicate-filtered grounder — wraps ParametrizedBCGrounder.

    Builds a predicate reachability graph from rules (head -> body predicates),
    performs BFS from query predicates, and filters rules to those with
    reachable head predicates before passing to the inner grounder.

    This results in a smaller provable set and fewer rules to process,
    while producing identical groundings for reachable queries.

    Constructor takes the same raw tensor args as ParametrizedBCGrounder,
    plus ``query_predicates`` controlling which predicates seed the BFS.

    Note: ``query_predicates`` is ``Set[int]`` (predicate indices, not names).
    The reachability BFS operates on predicate indices from the rule tensors.

    Args:
        facts_idx:           [F, 3] fact triples (pred, arg0, arg1)
        rules_heads_idx:     [R, 3] rule head atoms
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded)
        rule_lens:           [R] number of body atoms per rule
        constant_no:         highest constant index
        padding_idx:         padding value
        device:              target device
        query_predicates:    optional set of query predicate indices for
                             aggressive filtering. Defaults to all rule-head
                             predicates (conservative -- no filtering).
        **kwargs:            forwarded to ParametrizedBCGrounder
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

        # Compute reachable predicates and filter rules
        reachable = self._compute_reachable_predicates(
            rules_heads_idx, rules_bodies_idx, rule_lens,
            query_predicates,
        )

        # Filter rule tensors to keep only rules whose head pred is reachable
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
            # No rules pass the filter -- create empty tensors
            Bmax = rules_bodies_idx.shape[1] if rules_bodies_idx.dim() > 1 else 1
            filtered_heads = rules_heads_idx.new_zeros(0, 3)
            filtered_bodies = rules_bodies_idx.new_zeros(0, Bmax, 3)
            filtered_lens = rule_lens.new_zeros(0)

        print(
            f"  LazyGrounder: {n_orig} rules -> {n_filt} rules "
            f"({n_orig - n_filt} filtered, "
            f"{len(reachable)} reachable predicates)"
        )

        # Create inner grounder with filtered rules
        self._inner = ParametrizedBCGrounder(
            facts_idx,
            filtered_heads,
            filtered_bodies,
            filtered_lens,
            constant_no,
            padding_idx,
            device,
            **kwargs,
        )

        # Expose inner grounder properties
        self.effective_total_G: int = self._inner.effective_total_G

    @property
    def max_body_atoms(self) -> int:
        return self._inner.M

    @staticmethod
    def _compute_reachable_predicates(
        rules_heads_idx: Tensor,    # [R, 3]
        rules_bodies_idx: Tensor,   # [R, Bmax, 3]
        rule_lens: Tensor,          # [R]
        query_predicates: Optional[Set[int]] = None,
    ) -> Set[int]:
        """BFS on head->body predicate graph to find reachable predicates.

        Operates on predicate indices from the rule tensors. Builds head->body
        pred adjacency from ``rules_heads_idx`` and ``rules_bodies_idx``.

        Args:
            rules_heads_idx:  [R, 3] rule heads.
            rules_bodies_idx: [R, Bmax, 3] rule bodies.
            rule_lens:        [R] body lengths.
            query_predicates: Starting predicate index set. If None, uses all
                rule-head predicates (conservative -- no filtering).

        Returns:
            Set of reachable predicate indices.
        """
        R = rules_heads_idx.shape[0]
        if R == 0:
            return set()

        # Build adjacency: head_pred -> {body_preds}
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

        # Start from query predicates or all head predicates
        if query_predicates is None:
            seeds = all_head_preds
        else:
            seeds = query_predicates & all_head_preds

        # BFS: a predicate is reachable if it's a seed or appears in the
        # body of a rule whose head is reachable (backward reachability).
        # head->body means: to prove head, we need body.
        # So we want predicates that are needed to prove seeds.
        # BFS from seeds following head->body edges.
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
        self,
        queries: Tensor,       # [B, 3]
        query_mask: Tensor,    # [B]
    ) -> ForwardResult:
        """Delegates to inner ParametrizedBCGrounder.

        Returns identical output format (ForwardResult).
        """
        return self._inner(queries, query_mask)

    def __repr__(self) -> str:
        return f"LazyGrounder(inner={self._inner!r})"
