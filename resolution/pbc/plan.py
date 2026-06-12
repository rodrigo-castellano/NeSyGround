"""PBC compile — build-time setup: binding tables + shape budgets → PbcPlan.

``build_tables`` builds the ``PbcRuleIndex`` (binding analysis), the head-predicate
mask, the per-fv "any rule uses this slot" list, and the K/Y_r/K_v/Y_q budgets; the
shell registers the buffers + stores the scalars. ``build_plan`` then bundles a
grounder's (registered, on-device) buffers + budgets into one frozen ``PbcPlan``
so the resolve pipeline takes ONE object instead of ~25 loose tensors.

``all_anchors`` is forced True for pbc (no direction-B dual anchoring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from grounder.data.rule_index.pbc import PbcRuleIndex


@dataclass(frozen=True)
class PbcTables:
    """The per-rule binding buffers the resolve pipeline reads (references; the
    grounder owns/registers them so device moves stay correct)."""
    pred_rule_indices: Tensor    # [P, K_r] predicate → candidate rule rows
    pred_rule_mask: Tensor       # [P, K_r] which slots are real rules
    has_free: Tensor             # [R] rule has ≥1 free var
    head_pred_mask: Tensor       # [P] predicate is some rule's head
    body_preds: Tensor           # [R, M] body predicates (dep order)
    num_body_atoms: Tensor       # [R]
    enum_pred: Tensor            # [R] single-fv: enumerate predicate
    enum_bound: Tensor           # [R] single-fv: bound-arg source (0=subj,1=obj)
    enum_dir: Tensor             # [R] single-fv: direction
    check_arg_source: Tensor     # [R, M, 2] body arg → source column
    fv_enum_pred: Tensor         # [R, Fv] multi-fv: per-free-var enumerate pred
    fv_enum_bound_src: Tensor    # [R, Fv]
    fv_enum_direction: Tensor    # [R, Fv]
    fv_enum_valid: Tensor        # [R, Fv]
    arg_source_dep: Tensor       # [R, M, 2] dep-order arg sources (cartesian fill)
    body_preds_dep: Tensor       # [R, M] dep-order body preds


@dataclass(frozen=True)
class PbcPlan:
    """Frozen per-run pbc plan: binding tables + derived budgets."""
    tables: PbcTables
    M: int                       # max body atoms
    V: int                       # max free vars
    E: int                       # entity count (cartesian_product space)
    Y_r: int                     # groundings per rule
    K: int                       # children-per-state cap
    Y_q: int                     # collected-groundings budget
    K_v: int                     # candidates per free var
    fv_any_valid: tuple          # per fv-slot: any rule uses it
    cartesian_product: bool      # all-entities ablation


def build_plan(grounder) -> PbcPlan:
    """Bundle a grounder's registered pbc buffers + budgets into a PbcPlan."""
    t = PbcTables(
        grounder.pred_rule_indices, grounder.pred_rule_mask, grounder.has_free,
        grounder.head_pred_mask, grounder.body_preds, grounder.num_body_atoms,
        grounder.enum_pred_a, grounder.enum_bound_binding_a, grounder.enum_direction_a,
        grounder.check_arg_source_a, grounder.fv_enum_pred, grounder.fv_enum_bound_src,
        grounder.fv_enum_direction, grounder.fv_enum_valid,
        grounder.arg_source_dep, grounder.body_preds_dep)
    return PbcPlan(
        tables=t, M=grounder.kb.M, V=grounder.V, E=grounder._E, Y_r=grounder.Y_r,
        K=grounder.K, Y_q=grounder.Y_q, K_v=grounder.K_v,
        fv_any_valid=tuple(grounder._fv_any_valid),
        cartesian_product=grounder._cartesian_product)


def build_tables(grounder, *, max_children: int, max_total_groundings: int,
                 max_groundings_per_rule: Optional[int]) -> None:
    """Build the pbc binding buffers + budgets and WIRE them onto the grounder:
    register the buffers, set ``_enum_ri``/``_P``/``_E``/``K_r``/``K``/``Y_q``/``Y_r``/
    ``V``/``K_v``/``_fv_any_valid``/``max_vars_per_rule``/``_variant_to_orig_t``, then the
    optional S-bump. S is otherwise shell-owned.

    BIT-EXACT: P = max(fact predicates, rule-HEAD predicates); E from the fact index range."""
    kb = grounder.kb
    rule_index, fact_index = kb.rule_index, kb.fact_index
    facts_idx, constant_no, device = fact_index.facts_idx, kb.constant_no, kb.device_
    all_anchors, cartesian_product = grounder._all_anchors, grounder._cartesian_product
    max_groundings_per_query = max_groundings_per_rule       # the per-rule Y_r cap
    P_facts = getattr(fact_index, "_num_predicates", None)
    if P_facts is None:
        P_facts = int(facts_idx[:, 0].max().item()) + 1 if facts_idx.numel() > 0 else 1
    rh = rule_index.rules_heads_sorted
    P_heads = int(rh[:, 0].max().item()) + 1 if rh.numel() > 0 else 0
    P = max(P_facts, P_heads)
    E = getattr(fact_index, "_num_entities", None)
    if E is None:
        E = (int(max(facts_idx[:, 1].max().item(), facts_idx[:, 2].max().item())) + 1
             if facts_idx.numel() > 0 else 1)

    enum_ri = PbcRuleIndex(
        rule_index.rules_heads_sorted, rule_index.rules_bodies_sorted,
        rule_index.rule_lens_sorted, constant_no=constant_no, num_predicates=P,
        predicate_no=P - 1 if P > 0 else None, padding_idx=rule_index.padding_idx,
        device=device, all_anchors=all_anchors)

    head_pred_mask = torch.zeros(P, dtype=torch.bool, device=device)
    for p in enum_ri.patterns:
        head_pred_mask[p.head_pred_idx] = True
    fv_any_valid = [bool(enum_ri.fv_enum_valid[:, fv].any().item())
                    for fv in range(enum_ri.max_free_vars)]

    buffers: Dict[str, Tensor] = {
        "head_pred_mask": head_pred_mask,
        "has_free": enum_ri.has_free,
        "enum_pred_a": enum_ri.enum_pred,
        "enum_bound_binding_a": enum_ri.enum_bound,
        "enum_direction_a": enum_ri.enum_dir,
        "check_arg_source_a": enum_ri.arg_source,
        "head_preds": enum_ri.head_preds,
        "body_preds": enum_ri.body_preds,
        "num_body_atoms": enum_ri.num_body_atoms,
        "pred_rule_indices": enum_ri.pred_rule_indices,
        "pred_rule_mask": enum_ri.pred_rule_mask,
        "fv_enum_pred": enum_ri.fv_enum_pred,
        "fv_enum_bound_src": enum_ri.fv_enum_bound_src,
        "fv_enum_direction": enum_ri.fv_enum_direction,
        "fv_enum_valid": enum_ri.fv_enum_valid,
        "arg_source_dep": enum_ri.arg_source_dep,
        "body_preds_dep": enum_ri.body_preds_dep,
    }

    K_r = enum_ri.K_r
    V = enum_ri.max_free_vars
    K_f = getattr(fact_index, "_max_facts_per_query", max_groundings_per_query)
    if cartesian_product:
        Y_r = E
        K_v = E
    else:
        Y_r = max_groundings_per_query
        K_v = min(K_f, Y_r)
    children_cap = max_children if max_children is not None else max_total_groundings
    K = min(K_r * Y_r, children_cap)           # children per state (capped by max_children)
    Y_q = min(max_total_groundings, K_r * Y_r) # collected-groundings budget; ==K when max_children unset

    for name, tensor in buffers.items():
        grounder.register_buffer(name, tensor)
    grounder._enum_ri = enum_ri
    grounder._P, grounder._E = P, E
    grounder.K_r, grounder.K, grounder.Y_q = K_r, K, Y_q
    grounder.Y_r = Y_r
    grounder.V, grounder.K_v = V, K_v
    grounder._fv_any_valid = fv_any_valid
    grounder.max_vars_per_rule = 3
    # all_anchors dedup/remap needs the variant→original rule mapping on device.
    if all_anchors:
        grounder.register_buffer("_variant_to_orig_t", enum_ri.variant_to_orig.to(
            dtype=torch.long, device=device))
    # Optional S bump: enlarge S toward K_r*K_v when depth>1 (pbc only).
    if grounder._bump_s_to_k and grounder.depth > 1 and grounder.width is not None and grounder.width > 0:
        grounder.G = max(grounder.G, min(K, max(K_r * K_v, 1)))


__all__ = ["PbcTables", "PbcPlan", "build_plan", "build_tables"]
