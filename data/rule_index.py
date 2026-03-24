"""Rule indexing — sorted storage, lookup, and binding analysis.

Hierarchy
---------
RuleIndex(nn.Module)          base: sorted rules + segment-based predicate→rule lookup
└── RuleIndexEnum             + per-rule binding analysis + tensorized enum metadata

Factory: ``RuleIndex.create(rules_heads_idx, ..., type='base', ...)``

Supporting classes
------------------
RulePattern       per-rule variable binding pattern (used by RuleIndexEnum and FC)
compile_rules()   standalone: raw tensors → List[RulePattern]
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Binding constants
BINDING_HEAD_VAR0 = 0
BINDING_HEAD_VAR1 = 1
BINDING_FREE_VAR_OFFSET = 2


# ======================================================================
# RuleIndex — base: sorted storage + segment lookup
# ======================================================================

_RULE_INDEX_TYPES = {}  # filled after subclass definitions


class RuleIndex(nn.Module):
    """Sorted rules with predicate→rule segment lookup.

    Sufficient for MGU resolution (SLD, RTF). For enum resolution,
    use ``RuleIndex.create(type='enum', ...)`` or ``RuleIndexEnum`` directly.
    """

    def __init__(
        self,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        predicate_no: Optional[int] = None,
        padding_idx: int = 0,
        device: torch.device,
        order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        R = rules_heads_idx.shape[0]

        if R == 0:
            raise ValueError("rules_heads_idx is empty — cannot build a rule index without rules")

        if order == "shuffle":
            perm = self._shuffle_within_head_pred(
                rules_heads_idx, R, device, order_seed)
            rules_heads_idx = rules_heads_idx[perm]
            rules_bodies_idx = rules_bodies_idx[perm]
            rule_lens = rule_lens[perm]
        sort_perm = torch.argsort(rules_heads_idx[:, 0], stable=True)
        heads = rules_heads_idx.index_select(0, sort_perm).to(device)
        bodies = rules_bodies_idx.index_select(0, sort_perm).to(device)
        idx = sort_perm.to(device)
        lens = rule_lens.index_select(0, sort_perm).to(device)

        preds = heads[:, 0]
        uniq, cnts = torch.unique_consecutive(preds, return_counts=True)

        num_pred = (predicate_no + 1 if predicate_no is not None
                    else int(preds.max().item()) + 2)
        # CSR offsets: offsets[p] = start, offsets[p+1] - offsets[p] = count
        seg_offsets = torch.zeros(num_pred + 1, dtype=torch.long, device=device)
        cum = cnts.cumsum(0)
        mask = uniq < num_pred
        seg_offsets[uniq[mask] + 1] = cum[mask]
        seg_offsets = seg_offsets.cummax(0).values

        self._max_rule_pairs = int(cnts.max().item())

        self.register_buffer("rules_heads_sorted", heads)
        self.register_buffer("rules_bodies_sorted", bodies)
        self.register_buffer("rules_idx_sorted", idx)
        self.register_buffer("rule_lens_sorted", lens)
        self.register_buffer("_seg_offsets", seg_offsets)

    @property
    def num_rules(self) -> int:
        return self.rules_heads_sorted.shape[0]

    @property
    def max_rule_pairs(self) -> int:
        return self._max_rule_pairs

    @property
    def R_eff(self) -> int:
        return self._max_rule_pairs

    @property
    def K_r(self) -> int:
        """Rules per predicate (alias for R_eff)."""
        return self._max_rule_pairs

    @property
    def rules_heads(self) -> Tensor:
        return self.rules_heads_sorted

    @property
    def rules_bodies(self) -> Tensor:
        return self.rules_bodies_sorted

    @property
    def rule_lens(self) -> Tensor:
        return self.rule_lens_sorted

    @torch.no_grad()
    def lookup(
        self, query_preds: Tensor, max_pairs: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predicate→rule segment lookup.

        Returns: (item_idx [B, K], valid_mask [B, K], query_idx [B, K]).
        """
        B = query_preds.shape[0]
        dev = query_preds.device
        if B == 0:
            z = torch.zeros((0, max_pairs), dtype=torch.long, device=dev)
            return z, z.bool(), z

        qp = query_preds.long().clamp(0, self._seg_offsets.shape[0] - 2)
        starts = self._seg_offsets[qp]
        lens = (self._seg_offsets[qp + 1] - starts).clamp(max=max_pairs)
        offsets = torch.arange(max_pairs, device=dev).unsqueeze(0)
        return (
            starts.unsqueeze(1) + offsets,
            offsets < lens.unsqueeze(1),
            torch.arange(B, device=dev).unsqueeze(1).expand(-1, max_pairs),
        )

    lookup_by_segments = lookup

    @staticmethod
    def _shuffle_within_head_pred(
        rules_heads_idx: Tensor, R: int, device: torch.device, seed: int,
    ) -> Tensor:
        """Return a permutation that shuffles rules within each head-predicate group."""
        gen = torch.Generator(device=rules_heads_idx.device).manual_seed(seed)
        head_preds = rules_heads_idx[:, 0]
        sort_order = torch.argsort(head_preds, stable=True)
        sorted_preds = head_preds[sort_order]
        num_preds = int(sorted_preds.max().item()) + 1 if R > 0 else 1
        counts = torch.bincount(sorted_preds.long(), minlength=num_preds)
        starts = torch.zeros(num_preds + 1, dtype=torch.long,
                             device=rules_heads_idx.device)
        starts[1:] = counts.cumsum(0)
        perm = sort_order.clone()
        for p in range(num_preds):
            s, e = starts[p].item(), starts[p + 1].item()
            if e - s > 1:
                perm[s:e] = sort_order[
                    s + torch.randperm(e - s, device=rules_heads_idx.device,
                                       generator=gen)]
        return perm

    @classmethod
    def create(
        cls,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        type: Literal["base", "enum"] = "base",
        constant_no: Optional[int] = None,
        predicate_no: Optional[int] = None,
        padding_idx: int = 0,
        device: torch.device,
        num_predicates: Optional[int] = None,
        order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
    ) -> "RuleIndex":
        """Factory: create a RuleIndex subclass by type name.

        Args:
            type: 'base' (segment lookup only) or 'enum' (+ binding analysis).
            order: 'original' (keep input order) or 'shuffle' (random within
                   each head-predicate group — useful when K_r caps results).
            order_seed: random seed for shuffle reproducibility.
        """
        if type not in _RULE_INDEX_TYPES:
            raise ValueError(
                f"Unknown rule index type: {type!r}. "
                f"Choose from {list(_RULE_INDEX_TYPES)}"
            )
        return _RULE_INDEX_TYPES[type](
            rules_heads_idx, rules_bodies_idx, rule_lens,
            constant_no=constant_no, predicate_no=predicate_no,
            padding_idx=padding_idx, device=device,
            num_predicates=num_predicates,
            order=order, order_seed=order_seed,
        )


# ======================================================================
# RulePattern — per-rule binding analysis
# ======================================================================

class RulePattern:
    """Per-rule variable binding pattern with enumeration ordering.

    Identifies head/free variables, computes binding sources for each
    body atom argument, and reorders body atoms so each has >= 1 known arg.
    """

    def __init__(
        self, rule_idx: int, head: Tensor, body: Tensor,
        body_len: int, constant_no: int,
    ) -> None:
        self.rule_idx = rule_idx
        self.constant_no = constant_no
        self.head_pred_idx: int = head[0].item()
        self.head_var0: int = head[1].item()
        self.head_var1: int = head[2].item()
        self.num_body: int = body_len
        self.body_pred_indices: List[int] = [
            body[j, 0].item() for j in range(body_len)]

        head_vars = {v for v in (self.head_var0, self.head_var1)
                     if v > constant_no}
        body_vars = {body[j, k].item()
                     for j in range(body_len) for k in (1, 2)
                     if body[j, k].item() > constant_no}
        self.free_vars_list = sorted(body_vars - head_vars)
        self.num_free = len(self.free_vars_list)
        self._fv_idx = {v: i for i, v in enumerate(self.free_vars_list)}

        self.body_patterns = [
            {"pred_idx": body[j, 0].item(),
             "arg0_binding": self._binding(body[j, 1].item()),
             "arg1_binding": self._binding(body[j, 2].item()),
             "arg0_var": body[j, 1].item(),
             "arg1_var": body[j, 2].item()}
            for j in range(body_len)]

        # Save original order before reordering (needed by _reorder_body_with_anchor)
        self._orig_body_patterns = list(self.body_patterns)
        self._orig_body_pred_indices = list(self.body_pred_indices)

        self._reorder_body()

    def _binding(self, val: int) -> int:
        if val == self.head_var0:
            return BINDING_HEAD_VAR0
        if val == self.head_var1:
            return BINDING_HEAD_VAR1
        if val in self._fv_idx:
            return BINDING_FREE_VAR_OFFSET + self._fv_idx[val]
        return BINDING_HEAD_VAR0

    def _reorder_body(self) -> None:
        known = {BINDING_HEAD_VAR0, BINDING_HEAD_VAR1}
        remaining = list(range(self.num_body))
        order, meta = [], []

        while remaining:
            found = False
            for idx in remaining:
                bp = self.body_patterns[idx]
                b0, b1 = bp["arg0_binding"], bp["arg1_binding"]
                if b0 in known or b1 in known:
                    if b0 in known and b1 in known:
                        m = {"introduces_fv": -1, "enum_bound_src": 0,
                             "enum_direction": 0, "enum_pred": bp["pred_idx"]}
                    elif b0 in known:
                        m = {"introduces_fv": b1 - BINDING_FREE_VAR_OFFSET,
                             "enum_bound_src": b0, "enum_direction": 0,
                             "enum_pred": bp["pred_idx"]}
                        known.add(b1)
                    else:
                        m = {"introduces_fv": b0 - BINDING_FREE_VAR_OFFSET,
                             "enum_bound_src": b1, "enum_direction": 1,
                             "enum_pred": bp["pred_idx"]}
                        known.add(b0)
                    order.append(idx)
                    meta.append(m)
                    remaining.remove(idx)
                    found = True
                    break
            if not found:
                for idx in remaining:
                    order.append(idx)
                    meta.append({"introduces_fv": -1, "enum_bound_src": 0,
                                 "enum_direction": 0,
                                 "enum_pred": self.body_patterns[idx]["pred_idx"]})
                break

        self.enum_meta = meta
        self.body_patterns = [self.body_patterns[i] for i in order]
        self.body_pred_indices = [self.body_pred_indices[i] for i in order]

    def _reorder_body_with_anchor(
        self, forced_first: int,
    ) -> Tuple[List, List, List]:
        """Variant of _reorder_body that forces a specific body atom first.

        Operates on the original (pre-reorder) body patterns.

        Args:
            forced_first: Index into the original body atom list to place first.

        Returns:
            (body_patterns, body_pred_indices, enum_meta) — new lists,
            ``self`` is not mutated.
        """
        known = {BINDING_HEAD_VAR0, BINDING_HEAD_VAR1}
        remaining = list(range(self.num_body))
        order: List[int] = []
        meta: List[dict] = []

        # ── Force the specified atom first ──
        bp = self._orig_body_patterns[forced_first]
        b0, b1 = bp["arg0_binding"], bp["arg1_binding"]
        if b0 in known and b1 in known:
            m = {"introduces_fv": -1, "enum_bound_src": 0,
                 "enum_direction": 0, "enum_pred": bp["pred_idx"]}
        elif b0 in known:
            m = {"introduces_fv": b1 - BINDING_FREE_VAR_OFFSET,
                 "enum_bound_src": b0, "enum_direction": 0,
                 "enum_pred": bp["pred_idx"]}
            known.add(b1)
        elif b1 in known:
            m = {"introduces_fv": b0 - BINDING_FREE_VAR_OFFSET,
                 "enum_bound_src": b1, "enum_direction": 1,
                 "enum_pred": bp["pred_idx"]}
            known.add(b0)
        else:
            m = {"introduces_fv": -1, "enum_bound_src": 0,
                 "enum_direction": 0, "enum_pred": bp["pred_idx"]}
        order.append(forced_first)
        meta.append(m)
        remaining.remove(forced_first)

        # ── Greedy for the rest (same logic as _reorder_body) ──
        while remaining:
            found = False
            for idx in remaining:
                bp = self._orig_body_patterns[idx]
                b0, b1 = bp["arg0_binding"], bp["arg1_binding"]
                if b0 in known or b1 in known:
                    if b0 in known and b1 in known:
                        m = {"introduces_fv": -1, "enum_bound_src": 0,
                             "enum_direction": 0, "enum_pred": bp["pred_idx"]}
                    elif b0 in known:
                        m = {"introduces_fv": b1 - BINDING_FREE_VAR_OFFSET,
                             "enum_bound_src": b0, "enum_direction": 0,
                             "enum_pred": bp["pred_idx"]}
                        known.add(b1)
                    else:
                        m = {"introduces_fv": b0 - BINDING_FREE_VAR_OFFSET,
                             "enum_bound_src": b1, "enum_direction": 1,
                             "enum_pred": bp["pred_idx"]}
                        known.add(b0)
                    order.append(idx)
                    meta.append(m)
                    remaining.remove(idx)
                    found = True
                    break
            if not found:
                for idx in remaining:
                    order.append(idx)
                    meta.append({"introduces_fv": -1, "enum_bound_src": 0,
                                 "enum_direction": 0,
                                 "enum_pred": self._orig_body_patterns[idx]["pred_idx"]})
                break

        return (
            [self._orig_body_patterns[i] for i in order],
            [self._orig_body_pred_indices[i] for i in order],
            meta,
        )


# ======================================================================
# RuleIndexEnum — adds binding analysis + enum metadata tensors
# ======================================================================

class _PatternVariant:
    """Lightweight anchor variant of a RulePattern (duck-typed)."""

    __slots__ = ('rule_idx', 'head_pred_idx', 'num_body', 'num_free',
                 'body_patterns', 'body_pred_indices', 'enum_meta')

    def __init__(
        self, base: RulePattern,
        body_patterns: List, body_pred_indices: List, enum_meta: List,
    ) -> None:
        self.rule_idx = base.rule_idx
        self.head_pred_idx = base.head_pred_idx
        self.num_body = base.num_body
        self.num_free = base.num_free
        self.body_patterns = body_patterns
        self.body_pred_indices = body_pred_indices
        self.enum_meta = enum_meta


class RuleIndexEnum(RuleIndex):
    """Rule index with binding analysis and enum metadata.

    Extends RuleIndex with per-rule binding patterns and tensorized
    metadata for compiled enumeration resolution.
    """

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
        all_anchors: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            rules_heads_idx, rules_bodies_idx, rule_lens,
            predicate_no=predicate_no, padding_idx=padding_idx,
            device=device,
        )
        R = self.rules_heads_sorted.size(0)

        # Per-rule binding analysis
        base_patterns: List[RulePattern] = [
            RulePattern(i, self.rules_heads_sorted[i],
                        self.rules_bodies_sorted[i],
                        int(self.rule_lens_sorted[i].item()), constant_no)
            for i in range(R)]

        # Expand patterns when all_anchors=True: one variant per body atom
        if all_anchors:
            self.patterns = []
            for p in base_patterns:
                for j in range(p.num_body):
                    bp, bpi, em = p._reorder_body_with_anchor(j)
                    self.patterns.append(_PatternVariant(p, bp, bpi, em))
            # Keep originals for FC / fp_global
            self._original_patterns = base_patterns
        else:
            self.patterns = base_patterns

        Rp = len(self.patterns)  # expanded pattern count

        self.max_body: int = max(
            (p.num_body for p in self.patterns), default=1)
        M = self.max_body
        Rt = max(Rp, 1)

        # Tensorize rule metadata
        head_preds = torch.zeros(Rt, dtype=torch.long, device=device)
        body_preds = torch.zeros(Rt, M, dtype=torch.long, device=device)
        num_body = torch.zeros(Rt, dtype=torch.long, device=device)
        has_free = torch.zeros(Rt, dtype=torch.bool, device=device)
        enum_pred = torch.zeros(Rt, dtype=torch.long, device=device)
        enum_bound = torch.zeros(Rt, dtype=torch.long, device=device)
        enum_dir = torch.zeros(Rt, dtype=torch.long, device=device)
        arg_source = torch.zeros(Rt, M, 2, dtype=torch.long, device=device)

        for i, p in enumerate(self.patterns):
            head_preds[i] = p.head_pred_idx
            num_body[i] = p.num_body
            has_free[i] = p.num_free > 0
            for j, bp in enumerate(p.body_patterns):
                body_preds[i, j] = bp["pred_idx"]
                arg_source[i, j, 0] = bp["arg0_binding"]
                arg_source[i, j, 1] = bp["arg1_binding"]
            for m in p.enum_meta:
                if m["introduces_fv"] >= 0:
                    enum_pred[i] = m["enum_pred"]
                    enum_bound[i] = m["enum_bound_src"]
                    enum_dir[i] = m["enum_direction"]
                    break

        # Per-free-variable enumeration metadata
        # enum_meta from _reorder_body() has introduces_fv, enum_bound_src,
        # enum_direction, enum_pred for EACH body atom in dependency order.
        # Extract into [Rt, max_free_vars] tensors for sequential enumeration.
        max_fv = max((p.num_free for p in self.patterns), default=0)
        self.max_free_vars: int = max(max_fv, 1)  # at least 1 for shape consistency
        Fv = self.max_free_vars

        fv_enum_pred = torch.zeros(Rt, Fv, dtype=torch.long, device=device)
        fv_enum_bound_src = torch.zeros(Rt, Fv, dtype=torch.long, device=device)
        fv_enum_direction = torch.zeros(Rt, Fv, dtype=torch.long, device=device)
        fv_enum_valid = torch.zeros(Rt, Fv, dtype=torch.bool, device=device)

        # arg_source_dep / body_preds_dep: ORIGINAL body order, dep-pos-based
        # free var indices. Uses pre-reorder body patterns so body atoms stay
        # in the original position (important for query exclusion). Only fv
        # index references are remapped to dep-pos numbering.
        arg_source_dep = torch.zeros_like(arg_source)
        body_preds_dep = torch.zeros_like(body_preds)

        # Store metadata in DEPENDENCY order: dep_pos 0 = first fv to enumerate,
        # dep_pos 1 = second fv (may depend on dep_pos 0), etc.
        # bound_src is remapped so that references to other free vars use
        # dep_pos-based indices: BINDING_FREE_VAR_OFFSET + dep_pos.
        # This ensures _enumerate_cartesian can iterate 0, 1, ... and each
        # position only references earlier positions.
        for i, p in enumerate(self.patterns):
            dep_pos = 0
            # Build fv_index → dep_pos mapping for bound_src remapping
            fv_to_dep = {}
            for m in p.enum_meta:
                fv_idx = m["introduces_fv"]
                if fv_idx >= 0:
                    fv_to_dep[fv_idx] = dep_pos
                    dep_pos += 1

            dep_pos = 0
            for m in p.enum_meta:
                fv_idx = m["introduces_fv"]
                if fv_idx >= 0 and dep_pos < Fv:
                    fv_enum_pred[i, dep_pos] = m["enum_pred"]
                    fv_enum_direction[i, dep_pos] = m["enum_direction"]
                    fv_enum_valid[i, dep_pos] = True
                    # Remap bound_src: head vars stay as-is (0, 1),
                    # free var references remap to dep_pos-based indices
                    bs = m["enum_bound_src"]
                    if bs >= BINDING_FREE_VAR_OFFSET:
                        orig_fv = bs - BINDING_FREE_VAR_OFFSET
                        bs = BINDING_FREE_VAR_OFFSET + fv_to_dep[orig_fv]
                    fv_enum_bound_src[i, dep_pos] = bs
                    dep_pos += 1

            # Build arg_source_dep/body_preds_dep from ORIGINAL body order
            # (pre-reorder) with fv indices remapped to dep-pos.
            orig_patterns = getattr(p, '_orig_body_patterns', p.body_patterns)
            orig_preds = getattr(p, '_orig_body_pred_indices', p.body_pred_indices)
            for j in range(min(len(orig_patterns), M)):
                bp = orig_patterns[j]
                body_preds_dep[i, j] = orig_preds[j]
                for a, key in enumerate(("arg0_binding", "arg1_binding")):
                    val = bp[key]
                    if val >= BINDING_FREE_VAR_OFFSET:
                        orig_fv = val - BINDING_FREE_VAR_OFFSET
                        if orig_fv in fv_to_dep:
                            val = BINDING_FREE_VAR_OFFSET + fv_to_dep[orig_fv]
                    arg_source_dep[i, j, a] = val

        self.register_buffer("head_preds", head_preds)
        self.register_buffer("body_preds", body_preds)
        self.register_buffer("num_body_atoms", num_body)
        self.register_buffer("has_free", has_free)
        self.register_buffer("enum_pred", enum_pred)
        self.register_buffer("enum_bound", enum_bound)
        self.register_buffer("enum_dir", enum_dir)
        self.register_buffer("arg_source", arg_source)
        self.register_buffer("fv_enum_pred", fv_enum_pred)
        self.register_buffer("fv_enum_bound_src", fv_enum_bound_src)
        self.register_buffer("fv_enum_direction", fv_enum_direction)
        self.register_buffer("fv_enum_valid", fv_enum_valid)
        self.register_buffer("arg_source_dep", arg_source_dep)
        self.register_buffer("body_preds_dep", body_preds_dep)

        # Predicate → rule clustering
        P = num_predicates
        if all_anchors:
            # Rebuild clustering for expanded pattern set
            expanded_head_preds = head_preds[:Rp]
            uniq, cnts = torch.unique(expanded_head_preds, sorted=True,
                                      return_counts=True)
            R_eff = int(cnts.max().item()) if cnts.numel() > 0 else 1
            self._max_rule_pairs = R_eff

            # Sort expanded patterns by head predicate for CSR-style indexing
            sort_order = torch.argsort(expanded_head_preds, stable=True)
            # Build per-predicate start/count
            sorted_hp = expanded_head_preds[sort_order]
            uniq_s, cnts_s = torch.unique_consecutive(
                sorted_hp, return_counts=True)
            seg_offsets_exp = torch.zeros(P + 1, dtype=torch.long,
                                         device=device)
            cum = cnts_s.cumsum(0)
            mask = uniq_s < P
            seg_offsets_exp[uniq_s[mask] + 1] = cum[mask]
            seg_offsets_exp = seg_offsets_exp.cummax(0).values

            all_preds_t = torch.arange(P, device=device).clamp(
                0, seg_offsets_exp.shape[0] - 2)
            seg_starts = seg_offsets_exp[all_preds_t]
            seg_counts = seg_offsets_exp[all_preds_t + 1] - seg_starts
            pos = torch.arange(R_eff, device=device).unsqueeze(0)
            # Map back through sort_order to expanded pattern indices
            pred_rule_indices = (seg_starts.unsqueeze(1) + pos).clamp(
                0, max(Rp - 1, 0))
            pred_rule_indices = sort_order[pred_rule_indices]
            pred_rule_mask = pos < seg_counts.unsqueeze(1)
        else:
            R_eff = self._max_rule_pairs
            all_preds_t = torch.arange(P, device=device).clamp(
                0, self._seg_offsets.shape[0] - 2)
            seg_starts = self._seg_offsets[all_preds_t]          # [P]
            seg_counts = self._seg_offsets[all_preds_t + 1] - seg_starts  # [P]
            pos = torch.arange(R_eff, device=device).unsqueeze(0)  # [1, R_eff]
            pred_rule_indices = (seg_starts.unsqueeze(1) + pos).clamp(
                0, max(R - 1, 0))
            pred_rule_mask = pos < seg_counts.unsqueeze(1)

        self.register_buffer("pred_rule_indices", pred_rule_indices)
        self.register_buffer("pred_rule_mask", pred_rule_mask)


# ======================================================================
# Registry
# ======================================================================

_RULE_INDEX_TYPES.update({
    "base": RuleIndex,
    "enum": RuleIndexEnum,
})


# ======================================================================
# Standalone compile (for FC and nesy callers without a full RuleIndex)
# ======================================================================

def compile_rules(
    rule_heads: Tensor, rule_bodies: Tensor,
    rule_lens: Tensor, constant_no: int,
) -> List[RulePattern]:
    """Raw tensors → List[RulePattern]."""
    return [RulePattern(i, rule_heads[i], rule_bodies[i],
                        int(rule_lens[i].item()), constant_no)
            for i in range(rule_heads.size(0))]
