"""KB — the immutable knowledge base: facts + rules + indices.

Read-only and resolution-agnostic: one KB is shared across grounders (sld/rtf/pbc
each build their own resolver on top). It holds the generic ``SldRuleIndex``
(segment lookup); the pbc resolver builds its own ``PbcRuleIndex`` from the same
rule tensors. ``with_closure`` returns a NEW KB with extra (forward-derived)
facts — the KB itself is never mutated.

    kb = KB(facts, heads, bodies, lens, constant_no=C, predicate_no=P, padding_idx=pad, device=dev)
    g1 = BackwardGrounder(kb, ...);  g2 = BackwardGrounder(kb, resolution="rtf", ...)
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder._build.data import fact_index as fact_index_mod
from grounder._build.data import rule_index as rule_index_mod
from grounder._build.data.encoding import Encoding

FactIndexType = Literal["arg_key", "inverted", "block_sparse"]


class KB(nn.Module):
    """Immutable facts + rules + fact/rule indices."""

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        fact_index_type: FactIndexType = "arg_key",
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
        fact_order: Literal["original", "shuffle"] = "original",
        rule_order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
        pack_base: Optional[int] = None,
    ) -> None:
        super().__init__()
        if facts_idx.numel() == 0:
            raise ValueError("facts_idx is empty — a KB must have at least one fact")
        if rules_heads_idx.shape[0] == 0:
            raise ValueError("rules_heads_idx is empty — a KB must have at least one rule")

        self.constant_no = int(constant_no)
        self.predicate_no = int(predicate_no)
        self.padding_idx = int(padding_idx)
        self.device_ = device
        self.encoding = Encoding.from_constant_no(self.constant_no, self.padding_idx)

        # Index tables are sized predicate_no + 1; padding appears in predicate
        # slots of inactive states, so predicate_no must cover padding_idx.
        index_predicate_no = max(predicate_no, padding_idx)

        facts_idx = facts_idx.to(device=device, dtype=torch.long)
        rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)
        rules_bodies_idx = rules_bodies_idx.to(device=device, dtype=torch.long)
        rule_lens = rule_lens.to(device=device, dtype=torch.long)
        self.M = int(rule_lens.max().item())

        # Remember construction config so with_closure rebuilds identically.
        self._index_cfg = dict(
            fact_index_type=fact_index_type, max_facts_per_query=max_facts_per_query,
            max_memory_mb=max_memory_mb, fact_order=fact_order, rule_order=rule_order,
            order_seed=order_seed, pack_base=pack_base)
        self.register_buffer("rules_heads_idx", rules_heads_idx)
        self.register_buffer("rules_bodies_idx", rules_bodies_idx)
        self.register_buffer("rule_lens", rule_lens)

        self.fact_index = fact_index_mod.create(
            facts_idx, type=fact_index_type,
            constant_no=constant_no, predicate_no=index_predicate_no,
            padding_idx=padding_idx, device=device, pack_base=pack_base,
            max_facts_per_query=max_facts_per_query, max_memory_mb=max_memory_mb,
            order=fact_order, order_seed=order_seed)
        self.rule_index = rule_index_mod.create(
            rules_heads_idx, rules_bodies_idx, rule_lens, resolution="sld",
            predicate_no=index_predicate_no, padding_idx=padding_idx, device=device,
            order=rule_order, order_seed=order_seed)

        self.num_rules = rules_heads_idx.shape[0]
        self.K_r = self.rule_index.max_rule_pairs
        self.K_f = self.fact_index.max_fact_pairs

    @property
    def num_facts(self) -> int:
        return self.fact_index.num_facts

    @property
    def facts_idx(self) -> Tensor:
        """The (sorted) fact triples, via the fact index."""
        return self.fact_index.facts_idx

    def with_closure(self, closure_facts: Tensor) -> "KB":
        """Return a NEW KB whose facts are this KB's ∪ ``closure_facts``.

        Used by forward-chaining soundness (the closure set), NOT on the
        byte-identity path. The KB itself is never mutated.
        """
        extra = closure_facts.to(device=self.device_, dtype=torch.long)
        merged = torch.unique(torch.cat([self.fact_index.facts_idx, extra], 0), dim=0)
        return KB(merged, self.rules_heads_idx, self.rules_bodies_idx, self.rule_lens,
                  constant_no=self.constant_no, predicate_no=self.predicate_no,
                  padding_idx=self.padding_idx, device=self.device_, **self._index_cfg)

    def __repr__(self) -> str:
        return (f"KB(facts={self.num_facts}, rules={self.num_rules}, "
                f"entities={self.constant_no}, predicates={self.predicate_no})")


__all__ = ["KB"]
