"""Rule index package — sorted storage + (for pbc) binding tables.

    create(..., resolution="sld"|"rtf")  -> SldRuleIndex  (segment lookup)
    create(..., resolution="pbc")        -> PbcRuleIndex  (+ binding tables)

The KB builds the resolution-agnostic SldRuleIndex; the pbc resolver builds its
own PbcRuleIndex (binding analysis is pbc-only and forced ``all_anchors``).
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from grounder.data.rule_index.base import RuleIndex, build_csr_offsets
from grounder.data.rule_index.pattern import (
    BINDING_FREE_VAR_OFFSET, BINDING_HEAD_VAR0, BINDING_HEAD_VAR1,
    PatternVariant, RulePattern, compile_rules,
)
from grounder.data.rule_index.pbc import BindingTables, PbcRuleIndex
from grounder.data.rule_index.sld import SldRuleIndex

Resolution = Literal["sld", "rtf", "pbc"]


def create(
    rules_heads_idx: Tensor,
    rules_bodies_idx: Tensor,
    rule_lens: Tensor,
    *,
    resolution: Resolution = "sld",
    constant_no: Optional[int] = None,
    predicate_no: Optional[int] = None,
    num_predicates: Optional[int] = None,
    padding_idx: int = 0,
    device: torch.device,
    all_anchors: bool = True,
    order: Literal["original", "shuffle"] = "original",
    order_seed: int = 42,
) -> RuleIndex:
    """Build the rule index for ``resolution`` (the only public constructor)."""
    if resolution in ("sld", "rtf"):
        return SldRuleIndex(rules_heads_idx, rules_bodies_idx, rule_lens,
                            predicate_no=predicate_no, padding_idx=padding_idx,
                            device=device, order=order, order_seed=order_seed)
    if resolution == "pbc":
        if constant_no is None:
            raise ValueError("pbc rule index requires constant_no (variable threshold)")
        P = num_predicates if num_predicates is not None else (
            (predicate_no + 1) if predicate_no is not None else
            int(rules_heads_idx[:, 0].max().item()) + 2)
        return PbcRuleIndex(rules_heads_idx, rules_bodies_idx, rule_lens,
                            constant_no=constant_no, num_predicates=P,
                            predicate_no=predicate_no, padding_idx=padding_idx,
                            device=device, all_anchors=all_anchors)
    raise ValueError(f"Unknown resolution: {resolution!r}. Choose from sld | rtf | pbc")


__all__ = [
    "create", "Resolution",
    "RuleIndex", "SldRuleIndex", "PbcRuleIndex", "BindingTables",
    "RulePattern", "PatternVariant", "compile_rules", "build_csr_offsets",
    "BINDING_HEAD_VAR0", "BINDING_HEAD_VAR1", "BINDING_FREE_VAR_OFFSET",
]
