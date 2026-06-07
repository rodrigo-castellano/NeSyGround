"""PbcRuleIndex — rule index for Parametrized Backward Chaining (pbc resolution).

Extends the base index with the tensorized binding metadata the pbc resolver
needs to enumerate free variables on the layout (flat/dense) paths. The binding
analysis itself lives ONLY in ``RulePattern`` (pattern.py); this class is a PURE
TENSORIZATION of those results — it derives no new bindings.

``all_anchors`` (forced True for pbc): expand each rule into one variant per body
atom used as the resolution anchor. Anchoring only on the first atom misses
bindings the paper admits; the K_r variants of a logical rule collapse back to
one entry through ``variant_to_orig`` + the ``(orig_rule_idx, head, sorted_body)``
dedup downstream.

BIT-EXACT RECIPES (fingerprint-enforced):
  * variant expansion order: rule-major, then body-atom-index (the anchor)
  * dependency-position remap: ``fv_to_dep`` renumbers free-var references so
    ``arg_source_dep`` / ``fv_enum_bound_src`` reference only earlier dep slots
  * ``arg_source_dep`` / ``body_preds_dep`` use the ORIGINAL (pre-anchor) body
    order so all variants of one rule agree under the terminal dedup
  * clustering: expanded patterns sorted by head pred, CSR segments mapped back
    through the sort permutation to expanded-pattern indices
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder._build.data.rule_index.base import RuleIndex, build_csr_offsets
from grounder._build.data.rule_index.pattern import (
    BINDING_FREE_VAR_OFFSET, PatternVariant, RulePattern,
)


class BindingTables(NamedTuple):
    """The pbc binding bundle the resolver/dedup read (all dep-ordered)."""
    head_preds: Tensor          # [Rp]
    body_preds: Tensor          # [Rp, M] dependency order
    num_body_atoms: Tensor      # [Rp]
    has_free: Tensor            # [Rp]
    arg_source: Tensor          # [Rp, M, 2] dependency order
    fv_enum_pred: Tensor        # [Rp, Fv]
    fv_enum_bound_src: Tensor   # [Rp, Fv] dep-pos remapped
    fv_enum_direction: Tensor   # [Rp, Fv]
    fv_enum_valid: Tensor       # [Rp, Fv]
    arg_source_dep: Tensor      # [Rp, M, 2] ORIGINAL body order, dep-pos fv ids
    body_preds_dep: Tensor      # [Rp, M] ORIGINAL body order
    variant_to_orig: Tensor     # [Rp] -> original (sorted) rule index


class PbcRuleIndex(RuleIndex):
    """Base index + tensorized RulePattern binding metadata for pbc."""

    def __init__(
        self,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        constant_no: int,
        num_predicates: int,
        predicate_no: Optional[int] = None,
        padding_idx: int = 0,
        device: torch.device,
        all_anchors: bool = True,
    ) -> None:
        super().__init__(rules_heads_idx, rules_bodies_idx, rule_lens,
                         predicate_no=predicate_no, padding_idx=padding_idx, device=device)
        R = self.rules_heads_sorted.size(0)
        base_patterns: List[RulePattern] = [
            RulePattern(i, self.rules_heads_sorted[i], self.rules_bodies_sorted[i],
                        int(self.rule_lens_sorted[i].item()), constant_no)
            for i in range(R)]

        if all_anchors:
            self.patterns: List = []
            for p in base_patterns:
                for j in range(p.num_body):
                    bp, bpi, em = p.reorder_with_anchor(j)
                    self.patterns.append(PatternVariant(p, bp, bpi, em))
            self._original_patterns = base_patterns
        else:
            self.patterns = base_patterns

        Rp = len(self.patterns)
        Rt = max(Rp, 1)
        M = max((p.num_body for p in self.patterns), default=1)
        self.max_body = M
        Fv = max(max((p.num_free for p in self.patterns), default=0), 1)
        self.max_free_vars = Fv

        head_preds = torch.zeros(Rt, dtype=torch.long, device=device)
        body_preds = torch.zeros(Rt, M, dtype=torch.long, device=device)
        num_body = torch.zeros(Rt, dtype=torch.long, device=device)
        has_free = torch.zeros(Rt, dtype=torch.bool, device=device)
        enum_pred = torch.zeros(Rt, dtype=torch.long, device=device)
        enum_bound = torch.zeros(Rt, dtype=torch.long, device=device)
        enum_dir = torch.zeros(Rt, dtype=torch.long, device=device)
        arg_source = torch.zeros(Rt, M, 2, dtype=torch.long, device=device)
        fv_enum_pred = torch.zeros(Rt, Fv, dtype=torch.long, device=device)
        fv_enum_bound_src = torch.zeros(Rt, Fv, dtype=torch.long, device=device)
        fv_enum_direction = torch.zeros(Rt, Fv, dtype=torch.long, device=device)
        fv_enum_valid = torch.zeros(Rt, Fv, dtype=torch.bool, device=device)
        arg_source_dep = torch.zeros(Rt, M, 2, dtype=torch.long, device=device)
        body_preds_dep = torch.zeros(Rt, M, dtype=torch.long, device=device)
        variant_to_orig = torch.zeros(Rt, dtype=torch.long, device=device)

        for i, p in enumerate(self.patterns):
            head_preds[i] = p.head_pred_idx
            num_body[i] = p.num_body
            has_free[i] = p.num_free > 0
            variant_to_orig[i] = p.rule_idx
            for j, bp in enumerate(p.body_patterns):
                body_preds[i, j] = bp["pred_idx"]
                arg_source[i, j, 0] = bp["arg0_binding"]
                arg_source[i, j, 1] = bp["arg1_binding"]
            # enum_pred/bound/dir = first free-variable-introducing atom's meta
            for m in p.enum_meta:
                if m["introduces_fv"] >= 0:
                    enum_pred[i] = m["enum_pred"]
                    enum_bound[i] = m["enum_bound_src"]
                    enum_dir[i] = m["enum_direction"]
                    break

            # fv_to_dep: free-var index -> dependency position (enumeration order)
            fv_to_dep, dep_pos = {}, 0
            for m in p.enum_meta:
                if m["introduces_fv"] >= 0:
                    fv_to_dep[m["introduces_fv"]] = dep_pos
                    dep_pos += 1

            dep_pos = 0
            for m in p.enum_meta:
                if m["introduces_fv"] >= 0 and dep_pos < Fv:
                    fv_enum_pred[i, dep_pos] = m["enum_pred"]
                    fv_enum_direction[i, dep_pos] = m["enum_direction"]
                    fv_enum_valid[i, dep_pos] = True
                    bs = m["enum_bound_src"]
                    if bs >= BINDING_FREE_VAR_OFFSET:
                        bs = BINDING_FREE_VAR_OFFSET + fv_to_dep[bs - BINDING_FREE_VAR_OFFSET]
                    fv_enum_bound_src[i, dep_pos] = bs
                    dep_pos += 1

            # *_dep buffers: ORIGINAL body order, fv references remapped to dep-pos
            orig_patterns = getattr(p, "_orig_body_patterns", p.body_patterns)
            orig_preds = getattr(p, "_orig_body_pred_indices", p.body_pred_indices)
            for j in range(min(len(orig_patterns), M)):
                bp = orig_patterns[j]
                body_preds_dep[i, j] = orig_preds[j]
                for a, key in enumerate(("arg0_binding", "arg1_binding")):
                    val = bp[key]
                    if val >= BINDING_FREE_VAR_OFFSET:
                        ofv = val - BINDING_FREE_VAR_OFFSET
                        if ofv in fv_to_dep:
                            val = BINDING_FREE_VAR_OFFSET + fv_to_dep[ofv]
                    arg_source_dep[i, j, a] = val

        for name, t in (
            ("head_preds", head_preds), ("body_preds", body_preds),
            ("num_body_atoms", num_body), ("has_free", has_free),
            ("enum_pred", enum_pred), ("enum_bound", enum_bound), ("enum_dir", enum_dir),
            ("arg_source", arg_source),
            ("fv_enum_pred", fv_enum_pred), ("fv_enum_bound_src", fv_enum_bound_src),
            ("fv_enum_direction", fv_enum_direction), ("fv_enum_valid", fv_enum_valid),
            ("arg_source_dep", arg_source_dep), ("body_preds_dep", body_preds_dep),
            ("variant_to_orig", variant_to_orig),
        ):
            self.register_buffer(name, t)

        self._build_clustering(num_predicates, all_anchors, Rp, R, head_preds, device)

    def _build_clustering(self, P: int, all_anchors: bool, Rp: int, R: int,
                          head_preds: Tensor, device: torch.device) -> None:
        """Predicate→pattern segment lookup over the (possibly expanded) set."""
        if all_anchors:
            expanded = head_preds[:Rp]
            _, cnts = torch.unique(expanded, sorted=True, return_counts=True)
            R_eff = int(cnts.max().item()) if cnts.numel() > 0 else 1
            self._max_rule_pairs = R_eff
            sort_order = torch.argsort(expanded, stable=True)
            uniq_s, cnts_s = torch.unique_consecutive(expanded[sort_order], return_counts=True)
            seg = build_csr_offsets(uniq_s, cnts_s, P, device)
            preds_t = torch.arange(P, device=device).clamp(0, seg.shape[0] - 2)
            starts = seg[preds_t]
            counts = seg[preds_t + 1] - starts
            pos = torch.arange(R_eff, device=device).unsqueeze(0)
            idx = (starts.unsqueeze(1) + pos).clamp(0, max(Rp - 1, 0))
            pred_rule_indices = sort_order[idx]
            pred_rule_mask = pos < counts.unsqueeze(1)
        else:
            R_eff = self._max_rule_pairs
            preds_t = torch.arange(P, device=device).clamp(0, self._seg_offsets.shape[0] - 2)
            starts = self._seg_offsets[preds_t]
            counts = self._seg_offsets[preds_t + 1] - starts
            pos = torch.arange(R_eff, device=device).unsqueeze(0)
            pred_rule_indices = (starts.unsqueeze(1) + pos).clamp(0, max(R - 1, 0))
            pred_rule_mask = pos < counts.unsqueeze(1)
        self.register_buffer("pred_rule_indices", pred_rule_indices)
        self.register_buffer("pred_rule_mask", pred_rule_mask)

    def binding_tables(self) -> BindingTables:
        """The binding bundle (the resolver/dedup never touch RulePattern)."""
        return BindingTables(
            self.head_preds, self.body_preds, self.num_body_atoms, self.has_free,
            self.arg_source, self.fv_enum_pred, self.fv_enum_bound_src,
            self.fv_enum_direction, self.fv_enum_valid,
            self.arg_source_dep, self.body_preds_dep, self.variant_to_orig)


__all__ = ["PbcRuleIndex", "BindingTables"]
