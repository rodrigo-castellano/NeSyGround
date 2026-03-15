"""Grounder — base class owning KB state (facts, rules, indices).

Tensor conventions:
    B = batch, S = states per query, G = max goals, M = max body atoms
    K = max derived children per parent state
    K_f = max fact matches, K_r = max rule matches
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.fact_index import (
    ArgKeyFactIndex,
    BlockSparseFactIndex,
    InvertedFactIndex,
    shuffle_facts_per_predicate,
)
from grounder.rule_index import RuleIndex


class Grounder(nn.Module):
    """Base class owning knowledge-base state: facts, rules, indices.

    Args:
        facts_idx:        [F, 3] fact triples (pred, arg0, arg1)
        rules_heads_idx:  [R, 3] rule head atoms
        rules_bodies_idx: [R, Bmax, 3] rule body atoms (padded)
        rule_lens:        [R] number of body atoms per rule
        constant_no:      highest constant index (variables start at constant_no + 1)
        padding_idx:      padding value
        device:           target device
        predicate_no:     total number of predicates (exclusive upper bound)
        pack_base:        multiplier for hash packing (auto if None)
        fact_index_type:  'arg_key' | 'inverted' | 'block_sparse'
        shuffle_facts:    shuffle facts per predicate before building indices
        shuffle_seed:     random seed for fact shuffling
        num_entities:     total entities (required for inverted/block_sparse)
        max_facts_per_query: K_f for inverted/block_sparse indices
        max_memory_mb:    memory budget for block_sparse dense blocks
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
        predicate_no: Optional[int] = None,
        pack_base: Optional[int] = None,
        fact_index_type: str = "arg_key",
        shuffle_facts: bool = False,
        shuffle_seed: int = 42,
        num_entities: Optional[int] = None,
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
    ) -> None:
        super().__init__()
        self.constant_no = int(constant_no)
        self.padding_idx = int(padding_idx)
        self._device = device

        # Move tensors to device
        facts_idx = facts_idx.to(device=device, dtype=torch.long)
        rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)
        rules_bodies_idx = rules_bodies_idx.to(device=device, dtype=torch.long)
        rule_lens = rule_lens.to(device=device, dtype=torch.long)

        # Max body atoms per rule
        self.M = int(rule_lens.max().item()) if rule_lens.numel() > 0 else 1

        # Pack base for hash computation
        if pack_base is not None:
            self.pack_base = int(pack_base)
        else:
            self.pack_base = max(int(constant_no), int(padding_idx)) + 2

        # --- Build fact index ---
        if fact_index_type == "arg_key":
            self.fact_index = ArgKeyFactIndex(
                facts_idx, constant_no, padding_idx, device,
                pack_base=self.pack_base,
            )
        elif fact_index_type == "inverted":
            assert num_entities is not None and predicate_no is not None
            self.fact_index = InvertedFactIndex(
                facts_idx, constant_no, padding_idx, device,
                num_entities, predicate_no + 1, max_facts_per_query,
            )
        elif fact_index_type == "block_sparse":
            assert num_entities is not None and predicate_no is not None
            self.fact_index = BlockSparseFactIndex(
                facts_idx, constant_no, padding_idx, device,
                num_entities, predicate_no + 1, max_facts_per_query, max_memory_mb,
            )
        else:
            raise ValueError(f"Unknown fact_index_type: {fact_index_type}")

        # Canonical references (from the fact index, which sorts facts)
        self.register_buffer("facts_idx", self.fact_index.facts_idx)
        self.register_buffer("fact_hashes", self.fact_index.fact_hashes)

        # --- Build rule index ---
        # predicate_no must be >= padding_idx so the segment lookup table
        # can handle padding values that appear as predicate indices in
        # inactive proof states at depth >= 2.
        if predicate_no is not None and predicate_no < padding_idx:
            predicate_no = padding_idx
        self.rule_index = RuleIndex(
            rules_heads_idx, rules_bodies_idx, rule_lens, device,
            predicate_no=predicate_no, padding_idx=padding_idx,
        )

        self.num_rules = rules_heads_idx.shape[0]
        self.K_r = self.rule_index.max_rule_pairs

        # K_f: use data-derived max when available
        if hasattr(self.fact_index, 'max_fact_pairs'):
            self.K_f = self.fact_index.max_fact_pairs
        else:
            self.K_f = max_facts_per_query

        # Optional: shuffle facts per predicate
        if shuffle_facts and self.facts_idx.numel() > 0:
            self.facts_idx.copy_(
                shuffle_facts_per_predicate(self.facts_idx, predicate_no, shuffle_seed))

    @property
    def num_facts(self) -> int:
        return self.facts_idx.shape[0]
