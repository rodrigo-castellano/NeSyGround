"""ForwardGrounder — forward chaining as a grounder (closure to fixpoint).

Wraps ``run_forward_chaining`` over a KB's rules+facts, exposing the derived
closure as hashes or triples. Used by ``KB.with_closure`` (soundness) and the
``closure`` factory type. FC operates over the REAL predicate range (max id seen
in facts/rules + 1), NOT the loader's inflated ``predicate_no`` — passing the
inflated value would build thousands of empty per-predicate matrices.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder._build.data.kb import KB
from grounder._build.data.rule_index import compile_rules
from grounder._build.forward.fc import run_forward_chaining


def _real_num_predicates(kb: KB) -> int:
    """Max predicate id across facts + rule heads + (non-pad) rule bodies, +1."""
    pad = kb.padding_idx
    ids = [kb.fact_index.facts_idx[:, 0], kb.rules_heads_idx[:, 0]]
    body_preds = kb.rules_bodies_idx[..., 0].reshape(-1)
    ids.append(body_preds[body_preds != pad])
    allp = torch.cat([t for t in ids if t.numel() > 0])
    return int(allp.max().item()) + 1


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
        self._num_predicates = _real_num_predicates(kb)

    def closure_hashes(self) -> Tuple[Tensor, int]:
        """``(sorted_hashes, n_provable)`` — provable atom hashes
        (``h = pred*E^2 + subj*E + obj``)."""
        return run_forward_chaining(
            self._rules, self.kb.fact_index.facts_idx,
            self._num_entities, self._num_predicates, depth=self.depth,
            device=str(self.kb.device_), method=self.method, join_algo=self.join_algo)

    def closure_facts(self) -> Tensor:
        """Derived closure as ``[N, 3]`` (pred, subj, obj) triples."""
        hashes, n = self.closure_hashes()
        if n == 0:
            return torch.empty(0, 3, dtype=torch.long, device=hashes.device)
        E = self._num_entities
        E2 = E * E
        pred = hashes // E2
        rem = hashes % E2
        return torch.stack([pred, rem // E, rem % E], dim=1)

    def forward(self) -> Tensor:
        """Closure triples (so ``kb.with_closure(fg())`` reads naturally)."""
        return self.closure_facts()


__all__ = ["ForwardGrounder"]
