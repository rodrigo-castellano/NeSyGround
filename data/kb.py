"""KB — Knowledge Base: facts + rules + indices.

Immutable after construction. Can be shared across multiple grounders.

    kb = KB(facts, heads, bodies, lens, constant_no=C, predicate_no=P, ...)
    g1 = BCGrounder(kb, depth=1)
    g2 = BCGrounder(kb, depth=4, resolution='rtf')
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.fact_index import FactIndex
from grounder.data.rule_index import RuleIndex


class KB(nn.Module):
    """Knowledge base: facts + rules + indices.

    Args:
        facts_idx:          [F, 3] fact triples (pred, arg0, arg1)
        rules_heads_idx:    [R, 3] rule head atoms
        rules_bodies_idx:   [R, Bmax, 3] rule body atoms (padded)
        rule_lens:          [R] number of body atoms per rule
        constant_no:        number of entities (exclusive upper bound)
        predicate_no:       number of predicates (exclusive upper bound)
        padding_idx:        padding value
        device:             target device
        fact_index_type:    'arg_key' | 'inverted' | 'block_sparse'
        max_facts_per_query: K_f for inverted/block_sparse indices
        max_memory_mb:      memory budget for block_sparse dense blocks
        fact_order:         'original' | 'shuffle'
        rule_order:         'original' | 'shuffle'
        order_seed:         random seed for shuffle reproducibility
        pack_base:          multiplier for hash packing (auto if None)
    """

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
        fact_index_type: Literal["arg_key", "inverted", "block_sparse"] = "arg_key",
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

        # Index tables are sized predicate_no + 1. Padding values appear in
        # predicate slots of inactive proof states, so tables must cover
        # padding_idx → predicate_no must be >= padding_idx.
        predicate_no = max(predicate_no, padding_idx)

        # Move tensors to device
        facts_idx = facts_idx.to(device=device, dtype=torch.long)
        rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)
        rules_bodies_idx = rules_bodies_idx.to(device=device, dtype=torch.long)
        rule_lens = rule_lens.to(device=device, dtype=torch.long)

        self.M = int(rule_lens.max().item())

        # --- Build indices ---
        self.fact_index = FactIndex.create(
            facts_idx, type=fact_index_type,
            constant_no=constant_no, predicate_no=predicate_no,
            padding_idx=padding_idx, device=device,
            pack_base=pack_base,
            max_facts_per_query=max_facts_per_query,
            max_memory_mb=max_memory_mb,
            order=fact_order, order_seed=order_seed,
        )
        self.rule_index = RuleIndex(
            rules_heads_idx, rules_bodies_idx, rule_lens,
            predicate_no=predicate_no, padding_idx=padding_idx,
            device=device,
            order=rule_order, order_seed=order_seed,
        )

        self.num_rules = rules_heads_idx.shape[0]
        self.K_r = self.rule_index.max_rule_pairs
        self.K_f = self.fact_index.max_fact_pairs

    @property
    def num_facts(self) -> int:
        return self.fact_index.num_facts

    def __repr__(self) -> str:
        return (f"KB(facts={self.num_facts}, rules={self.num_rules}, "
                f"entities={self.constant_no}, predicates={self.predicate_no})")
