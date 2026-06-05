"""Compiled enumeration resolution — pre-compiled bindings, no MGU.

Enumerates entity candidates from the fact index using pre-compiled binding
tables (body_arg_sources). At runtime, fills body templates directly from
query args and enumerated candidates. No unification needed.

Public API:
  init_enum():           build all enum metadata buffers and compiled rules
  resolve_enum():        fact-anchored candidates per rule (parametrized)
  resolve_enum_full():   all entities as candidates (width=None/inf)
  resolve_enum_step():   adapter from resolve_enum to common 7-tensor format
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from grounder.data.rule_index import RuleIndexEnum
from grounder.types import FlatResolvedChildren, ResolvedChildren


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def init_enum(
    rule_index,
    fact_index,
    facts_idx: Tensor,
    constant_no: int,
    num_rules: int,
    M: int,
    *,
    width: Optional[int],
    max_groundings_per_query: int,
    max_total_groundings: int,
    max_states: Optional[int],
    device: torch.device,
    cartesian_product: bool = False,
    all_anchors: bool = False,
    flat_intermediate: bool = False,
) -> Dict:
    """Build all enum metadata from a RuleIndex.

    Returns dict with 'compiled' (CompiledRules), 'buffers' (tensors to
    register), and scalar config values.
    """
    # P and E. ``fact_index._num_predicates`` only counts predicates
    # that appear in facts, but rules can reference predicates that
    # never appear as facts (e.g. an inverse predicate that's only
    # ever derived). Take the max across facts AND rule HEADS so the
    # head predicate mask is sized correctly. Body atoms are NOT
    # included: ``rules_bodies_sorted`` may contain ``padding_idx``
    # as a padding marker (rules with fewer body atoms than ``M_max``)
    # which is not a real predicate and would inflate P unnecessarily
    # (on wn18rr this would push P from 12 to 40573).
    P_facts = getattr(fact_index, "_num_predicates", None)
    if P_facts is None:
        P_facts = (int(facts_idx[:, 0].max().item()) + 1
                   if facts_idx.numel() > 0 else 1)
    P_heads = 0
    rh = rule_index.rules_heads_sorted
    if rh.numel() > 0:
        P_heads = int(rh[:, 0].max().item()) + 1
    P = max(P_facts, P_heads)
    E = getattr(fact_index, "_num_entities", None)
    if E is None:
        E = (int(max(facts_idx[:, 1].max().item(),
                     facts_idx[:, 2].max().item())) + 1
             if facts_idx.numel() > 0 else 1)

    # Build enum rule index (binding analysis + metadata tensors)
    enum_ri = RuleIndexEnum(
        rule_index.rules_heads_sorted,
        rule_index.rules_bodies_sorted,
        rule_index.rule_lens_sorted,
        constant_no=constant_no,
        num_predicates=P,
        predicate_no=P - 1 if P > 0 else None,
        padding_idx=rule_index.padding_idx,
        device=device,
        all_anchors=all_anchors,
    )

    # Head predicate mask
    head_pred_mask = torch.zeros(P, dtype=torch.bool, device=device)
    for p in enum_ri.patterns:
        head_pred_mask[p.head_pred_idx] = True

    # Per-fv-slot: does ANY rule use this free var?
    # Used at Python level (not traced) to skip unnecessary Cartesian expansion.
    fv_any_valid = [bool(enum_ri.fv_enum_valid[:, fv].any().item())
                    for fv in range(enum_ri.max_free_vars)]

    # Collect buffers
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
        # Per-free-variable enumeration metadata
        "fv_enum_pred": enum_ri.fv_enum_pred,
        "fv_enum_bound_src": enum_ri.fv_enum_bound_src,
        "fv_enum_direction": enum_ri.fv_enum_direction,
        "fv_enum_valid": enum_ri.fv_enum_valid,
        "arg_source_dep": enum_ri.arg_source_dep,
        "body_preds_dep": enum_ri.body_preds_dep,
    }

    # Dual anchoring (direction B)
    # Skip when all_anchors — every body atom is already tried as anchor.
    any_dual = False
    if width is not None and width > 0 and not all_anchors:
        R = max(num_rules, 1)
        Mb = enum_ri.max_body

        has_dual = torch.zeros(R, dtype=torch.bool, device=device)
        enum_pred_b = torch.zeros(R, dtype=torch.long, device=device)
        enum_bound_b = torch.zeros(R, dtype=torch.long, device=device)
        enum_dir_b = torch.zeros(R, dtype=torch.long, device=device)
        arg_source_b = torch.zeros(R, Mb, 2, dtype=torch.long, device=device)

        for i, p in enumerate(enum_ri.patterns):
            bp0 = p.body_patterns[0] if p.num_body > 0 else None
            bp1 = p.body_patterns[1] if p.num_body > 1 else None
            bp0_has_free = (bp0 is not None
                            and (bp0["arg0_binding"] >= 2
                                 or bp0["arg1_binding"] >= 2))
            bp1_has_free = (bp1 is not None
                            and (bp1["arg0_binding"] >= 2
                                 or bp1["arg1_binding"] >= 2))
            bp1_can_anchor = (bp1_has_free and bp1 is not None
                              and (bp1["arg0_binding"] in (0, 1)
                                   or bp1["arg1_binding"] in (0, 1)))
            if bp0_has_free and bp1_can_anchor:
                has_dual[i] = True
                enum_pred_b[i] = bp1["pred_idx"]
                if bp1["arg0_binding"] >= 2:
                    enum_dir_b[i] = 1
                    enum_bound_b[i] = bp1["arg1_binding"]
                else:
                    enum_dir_b[i] = 0
                    enum_bound_b[i] = bp1["arg0_binding"]
                for j, bp in enumerate(p.body_patterns):
                    for a, key in enumerate(("arg0_binding", "arg1_binding")):
                        b = bp[key]
                        arg_source_b[i, j, a] = (
                            0 if b == 0 else (1 if b == 1 else 2))

        any_dual = bool(has_dual.any().item())
        buffers["has_dual"] = has_dual
        buffers["enum_pred_b"] = enum_pred_b
        buffers["enum_bound_binding_b"] = enum_bound_b
        buffers["enum_direction_b"] = enum_dir_b
        buffers["check_arg_source_b"] = arg_source_b

    # Budgets
    K_r = enum_ri.K_r
    V = enum_ri.max_free_vars
    K_f = getattr(fact_index, '_max_facts_per_query', max_groundings_per_query)

    if cartesian_product:
        G_r = E
        K_v = E
    else:
        # G_r: user-specified grounding budget per rule.
        # K_v: candidates per free variable from fact_index.enumerate.
        # Set to G_r so 1-free-var rules match the old _enumerate_dir behavior
        # (at most G_r candidates per rule). For multi-free-var rules, the
        # Cartesian product is capped to G_r via topk inside _enumerate_cartesian.
        G_r = max_groundings_per_query
        K_v = min(K_f, G_r)

    K = min(K_r * G_r, max_total_groundings)
    S = max_states if max_states is not None else 256
    C = min(max_total_groundings, K_r * G_r)

    print(f"  BCGrounder(enum): {num_rules} rules, K_r={K_r}, "
          f"width={width}, S={S}, K={K}, C={C}, "
          f"K_f={K_f}, K_v={K_v}, V={V}, "
          f"G_r={G_r}"
          + (f", cartesian_product=True, E={E}" if cartesian_product else "")
          + (f", all_anchors=True" if all_anchors else ""))

    return {
        "buffers": buffers,
        "enum_rule_index": enum_ri,
        "P": P, "E": E,
        "K_r": K_r,
        "K": K,
        "S": S,
        "C": C,
        "any_dual": any_dual,
        "G_r": G_r,
        "cartesian_product": cartesian_product,
        "V": V,
        "K_v": K_v,
        "fv_any_valid": fv_any_valid,
        "flat_intermediate": flat_intermediate,
    }


def resolve_enum_step(
    queries: Tensor,           # [B, S, 3]
    remaining: Tensor,         # [B, S, G, 3]
    grounding_body: Tensor,    # [B, S, M, 3]
    state_valid: Tensor,       # [B, S]
    active_mask: Tensor,       # [B, S]
    *,
    fact_index,
    d,                         # int (eager) or 0-dim Tensor (compiled)
    depth: int,
    width: Optional[int],
    is_last=None,              # Optional[Tensor] (compiled path provides
                               #  ``d == depth-1`` as a 0-dim bool)
    M: int,
    padding_idx: int,
    G_r: int,
    K: int,
    any_dual: bool,
    # All the metadata buffers:
    pred_rule_indices: Tensor,
    pred_rule_mask: Tensor,
    has_free: Tensor,
    body_preds: Tensor,
    num_body_atoms: Tensor,
    enum_pred_a: Tensor,
    enum_bound_binding_a: Tensor,
    enum_direction_a: Tensor,
    check_arg_source_a: Tensor,
    head_pred_mask: Optional[Tensor],
    # Optional dual
    has_dual: Optional[Tensor] = None,
    enum_pred_b: Optional[Tensor] = None,
    enum_bound_binding_b: Optional[Tensor] = None,
    enum_direction_b: Optional[Tensor] = None,
    check_arg_source_b: Optional[Tensor] = None,
    collect_evidence: bool = True,
    cartesian_product: bool = False,
    E: int = 0,
    w_last_depth: int = 0,
    # Per-free-var metadata for Cartesian enumeration
    fv_enum_pred: Optional[Tensor] = None,
    fv_enum_bound_src: Optional[Tensor] = None,
    fv_enum_direction: Optional[Tensor] = None,
    fv_enum_valid: Optional[Tensor] = None,
    V: int = 1,
    K_v: int = 64,
    fv_any_valid: Optional[list] = None,
    arg_source_dep: Optional[Tensor] = None,
    body_preds_dep: Optional[Tensor] = None,
    flat_intermediate: bool = False,
    dedup_goals: bool = False,
) -> ResolvedChildren:
    """Adapter: resolve_enum output -> common 9-tensor format used by _pack.

    1. Determine width for this step.
    2. Flatten [B, S, 3] -> [N, 3].
    3. Call resolve_enum with all metadata buffers.
    4. Handle direction B (dual anchoring): pad and concat.
    5. Flatten K_r * G_r -> K_total, cap to K via topk.
    6. Reshape back to [B, S, K, ...].
    7. Build rule_goals: body atoms at 0..M-1, remaining parent goals at M..G-1.
    8. Build rule_gbody: parent's grounding_body expanded to K children.
    9. Return 7-tensor tuple (fact_goals, fact_gbody, fact_success,
       rule_goals, rule_gbody, rule_success, sub_rule_idx) where fact_* are
       empty (K_f=0).

    Args:
        queries: [B, S, 3] selected goal atoms.
        remaining: [B, S, G, 3] remaining proof goals.
        grounding_body: [B, S, M, 3] parent grounding body.
        state_valid: [B, S] validity mask.
        active_mask: [B, S] active state mask.
        fact_index: FactIndex with .enumerate() and .exists().
        d: Current depth step index.
        depth: Total proof depth.
        width: Width bound (None=inf).
        M: Max body atoms.
        padding_idx: Padding index value.
        G_r: Max groundings per query.
        K: Children budget per state.
        any_dual: Whether any rule supports dual anchoring.
        pred_rule_indices..head_pred_mask: Direction A metadata buffers.
        has_dual..check_arg_source_b: Optional direction B metadata buffers.

    Returns:
        9-tuple of tensors:
            fact_goals:    [B, S, 0, G, 3]
            fact_gbody:    [B, S, 0, M, 3]
            fact_success:  [B, S, 0]
            rule_goals:    [B, S, K, G, 3]
            rule_gbody:    [B, S, K, M, 3]
            rule_success:  [B, S, K]
            sub_rule_idx:  [B, S, K]
            fact_subs:     [B, S, 0, 2, 2]
            rule_subs:     [B, S, K, 2, 2]
    """
    B, S, _ = queries.shape
    G = remaining.shape[2]
    pad = padding_idx
    dev = queries.device

    # ``d`` is a Python int when called from the eager step loop, and a
    # 0-dim long tensor when called from the compiled step. Compute
    # ``is_last`` consistently:
    #   - eager: a Python bool (specialised at trace time → static
    #     ``width_d``, static ``head_pred_mask_d``);
    #   - compiled: ``is_last`` is a 0-dim bool tensor passed in by the
    #     caller, so ``width_d`` becomes a 0-dim long tensor and the
    #     head-pred prune is gated by ``is_last`` inside the filter.
    if is_last is None:
        # Eager path — d is a Python int.
        is_last_static = (d == depth - 1)
    else:
        is_last_static = None        # tensor branch handled downstream

    # 1. Width: last step uses w_last_depth (default 0 = all body atoms
    #    must be facts) so proof_goals become empty and collection works.
    if is_last_static is not None:
        width_d = w_last_depth if (is_last_static and width is not None) else width
    else:
        # Compiled: tensor select between ``width`` and ``w_last_depth``.
        if width is None:
            width_d = None
        else:
            width_d = torch.where(
                is_last,
                torch.tensor(w_last_depth, dtype=torch.long, device=dev),
                torch.tensor(width, dtype=torch.long, device=dev),
            )

    # 1b. Head-pred prune is only meaningful when an unknown body atom
    #     could later be proved by some rule — i.e. at intermediate
    #     steps. At the last step (no further depth), admit unknowns
    #     regardless of their predicate, matching keras-ns
    #     prune_incomplete_proofs=False semantics. In compiled mode the
    #     tensor mask is always passed; the filter uses ``is_last`` to
    #     conditionally skip the prune.
    if is_last_static is not None:
        head_pred_mask_d: Optional[Tensor] = (
            None if is_last_static else head_pred_mask
        )
    else:
        head_pred_mask_d = head_pred_mask     # always pass; filter gates

    # ── Flat intermediate path (zero grounding loss) ──
    if flat_intermediate and fv_enum_pred is not None and V >= 1:
        return _resolve_enum_step_flat(
            queries, remaining, grounding_body, state_valid, active_mask,
            fact_index=fact_index, d=d, depth=depth, width=width,
            M=M, padding_idx=padding_idx, G_r=G_r, K=K,
            pred_rule_indices=pred_rule_indices, pred_rule_mask=pred_rule_mask,
            has_free=has_free, body_preds=body_preds, num_body_atoms=num_body_atoms,
            check_arg_source_a=check_arg_source_a, head_pred_mask=head_pred_mask_d,
            fv_enum_pred=fv_enum_pred, fv_enum_bound_src=fv_enum_bound_src,
            fv_enum_direction=fv_enum_direction, fv_enum_valid=fv_enum_valid,
            V=V, fv_any_valid=fv_any_valid,
            arg_source_dep=arg_source_dep, body_preds_dep=body_preds_dep,
            collect_evidence=collect_evidence, w_last_depth=w_last_depth,
            dedup_goals=dedup_goals,
        )

    # 2. Flatten [B, S, 3] -> [N, 3]
    N = B * S
    flat_q = queries.reshape(N, 3)
    flat_valid = (active_mask & state_valid).reshape(N)

    # 3. Call resolve_enum (dense path)
    result = resolve_enum(
        flat_q, flat_valid, fact_index,
        pred_rule_indices=pred_rule_indices,
        pred_rule_mask=pred_rule_mask,
        has_free=has_free,
        body_preds=body_preds,
        num_body_atoms=num_body_atoms,
        enum_pred_a=enum_pred_a,
        enum_bound_binding_a=enum_bound_binding_a,
        enum_direction_a=enum_direction_a,
        check_arg_source_a=check_arg_source_a,
        head_pred_mask=head_pred_mask_d,
        G_r=G_r, M=M, width=width_d,
        is_last=is_last,
        has_dual=has_dual,
        enum_pred_b=enum_pred_b,
        enum_bound_binding_b=enum_bound_binding_b,
        enum_direction_b=enum_direction_b,
        check_arg_source_b=check_arg_source_b,
        any_dual=any_dual,
        cartesian_product=cartesian_product,
        E=E,
        fv_enum_pred=fv_enum_pred,
        fv_enum_bound_src=fv_enum_bound_src,
        fv_enum_direction=fv_enum_direction,
        fv_enum_valid=fv_enum_valid,
        V=V,
        K_v=K_v,
        fv_any_valid=fv_any_valid,
        arg_source_dep=arg_source_dep,
        body_preds_dep=body_preds_dep,
    )

    # 4. Unpack tuple and handle direction B
    (body_a, _exists_a, _body_active, gmask_a, _cmask_a,
     _active_mask_out, aidx,
     body_b, _exists_b, _body_active_b, gmask_b, _cmask_b) = result

    K_r = aidx.size(1)
    G_use_a = body_a.size(2)
    G_use_b = body_b.size(2)

    if G_use_b > 0:
        # Direct assembly into combined tensor (avoids pad + pad + cat)
        G_total = G_use_a + G_use_b
        body_all = torch.full(
            (N, K_r, G_total, M, 3), pad, dtype=torch.long, device=dev)
        body_all[:, :, :G_use_a] = body_a
        body_all[:, :, G_use_a:G_use_a + G_use_b] = body_b[:, :, :G_use_b]
        gmask_all = torch.zeros(
            N, K_r, G_total, dtype=torch.bool, device=dev)
        gmask_all[:, :, :G_use_a] = gmask_a
        gmask_all[:, :, G_use_a:G_use_a + G_use_b] = gmask_b[:, :, :G_use_b]
    else:
        body_all = body_a
        gmask_all = gmask_a

    # 5. Flatten K_r * G_r -> K_total, cap to K via topk
    G_use_total = body_all.size(2)
    K_total = K_r * G_use_total

    body_flat = body_all.reshape(N, K_total, M, 3)
    success_flat = gmask_all.reshape(N, K_total)
    ridx_flat = aidx.unsqueeze(2).expand(
        -1, -1, G_use_total).reshape(N, K_total)

    if K_total > K:
        _, top_idx = success_flat.to(torch.int8).topk(
            K, dim=1, largest=True, sorted=False)
        idx_body = top_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, M, 3)
        body_flat = body_flat.gather(1, idx_body)
        success_flat = success_flat.gather(1, top_idx)
        ridx_flat = ridx_flat.gather(1, top_idx)
    else:
        K = K_total

    # 6. Reshape back to [B, S, K, ...]
    body_flat = body_flat.reshape(B, S, K, M, 3)
    success_flat = success_flat.reshape(B, S, K)
    ridx_flat = ridx_flat.reshape(B, S, K)

    # 7. Build rule_goals: body atoms at positions 0..M-1,
    #    remaining parent goals at M..G-1.
    Bmax = M
    rule_goals = torch.full(
        (B, S, K, G, 3), pad, dtype=torch.long, device=dev)
    rule_goals[:, :, :, :Bmax, :] = body_flat
    # Copy remaining goals from parent (positions 1.. since 0 was selected)
    n_rem = min(G - Bmax, G - 1)
    if n_rem > 0:
        rem = remaining[:, :, 1:1 + n_rem, :]   # [B, S, n_rem, 3]
        rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = \
            rem.unsqueeze(2).expand(-1, -1, K, -1, -1)

    # 8. Build rule_gbody: parent's grounding_body expanded to K children
    G_body = grounding_body.shape[2]  # G_body >= M (accumulated body dim)
    if collect_evidence:
        rule_gbody = grounding_body.unsqueeze(2).expand(
            -1, -1, K, -1, -1)   # [B, S, K, G_body, 3]
    else:
        rule_gbody = torch.zeros(
            B, S, K, G_body, 3, dtype=torch.long, device=dev)

    # 9. Empty fact results (enum resolves everything through rules)
    fact_goals = torch.full(
        (B, S, 0, G, 3), pad, dtype=torch.long, device=dev)
    fact_gbody = torch.zeros(
        B, S, 0, G_body, 3, dtype=torch.long, device=dev)
    fact_success = torch.zeros(
        B, S, 0, dtype=torch.bool, device=dev)

    # 10. Zero subs (enum has no MGU — body atoms are pre-filled)
    fact_subs = torch.full(
        (B, S, 0, 2, 2), pad, dtype=torch.long, device=dev)
    rule_subs = torch.full(
        (B, S, K, 2, 2), pad, dtype=torch.long, device=dev)

    return ResolvedChildren(fact_goals, fact_gbody, fact_success,
                            rule_goals, rule_gbody, success_flat,
                            ridx_flat, fact_subs, rule_subs)


def resolve_enum(
    queries: Tensor,              # [B, 3]
    query_mask: Tensor,           # [B]
    fact_index,                   # FactIndex with .enumerate() and .exists()
    *,
    pred_rule_indices: Tensor,    # [P, K_r]
    pred_rule_mask: Tensor,       # [P, K_r]
    has_free: Tensor,             # [R]
    body_preds: Tensor,           # [R, M]
    num_body_atoms: Tensor,       # [R]
    enum_pred_a: Tensor,          # [R]
    enum_bound_binding_a: Tensor, # [R]
    enum_direction_a: Tensor,     # [R]
    check_arg_source_a: Tensor,   # [R, M, 2]
    head_pred_mask: Optional[Tensor],   # [P] bool, None to skip head-pred prune
    G_r: int,                     # groundings per rule
    M: int,                       # max body atoms
    width: Optional[int],         # width bound (None=∞); int OR 0-dim Tensor
    is_last: Optional[Tensor] = None,   # 0-dim bool tensor (compiled path)
    has_dual: Optional[Tensor] = None,
    enum_pred_b: Optional[Tensor] = None,
    enum_bound_binding_b: Optional[Tensor] = None,
    enum_direction_b: Optional[Tensor] = None,
    check_arg_source_b: Optional[Tensor] = None,
    any_dual: bool = False,
    cartesian_product: bool = False,
    E: int = 0,
    # Per-free-var metadata for Cartesian enumeration
    fv_enum_pred: Optional[Tensor] = None,
    fv_enum_bound_src: Optional[Tensor] = None,
    fv_enum_direction: Optional[Tensor] = None,
    fv_enum_valid: Optional[Tensor] = None,
    V: int = 1,
    K_v: int = 64,
    fv_any_valid: Optional[list] = None,
    arg_source_dep: Optional[Tensor] = None,
    body_preds_dep: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Parametrized compiled enumeration with optional dual anchoring.

    Direction A uses sequential Cartesian enumeration of ALL free variables.
    Direction B (dual anchoring) uses single-anchor enumeration.

    Returns fixed-length 12-tuple (compile-safe, no dicts):
        body_atoms_a    [B, K_r, G_a, M, 3]
        exists_a        [B, K_r, G_a, M]
        body_active     [B, K_r, 1, M]
        grounding_mask_a [B, K_r, G_a]
        cand_mask_a     [B, K_r, G_a]
        active_mask     [B, K_r]
        active_idx      [B, K_r]
        body_atoms_b    [B, K_r, G_b, M, 3]   (zero-sized G_b=0 when no dual)
        exists_b        [B, K_r, G_b, M]
        body_active_b   [B, K_r, 1, M]        (zeros when no dual)
        grounding_mask_b [B, K_r, G_b]
        cand_mask_b     [B, K_r, G_b]
    """
    B = queries.size(0)
    dev = queries.device
    qp, qs, qo = queries[:, 0], queries[:, 1], queries[:, 2]

    # ── Rule clustering ──
    active_idx = pred_rule_indices[qp]             # [B, K_r]
    K_r = active_idx.size(1)
    active_mask = pred_rule_mask[qp] & query_mask.unsqueeze(1)
    has_free_q = has_free[active_idx]               # [B, K_r]
    body_preds_q = body_preds[active_idx]           # [B, K_r, M]
    num_body_q = num_body_atoms[active_idx]         # [B, K_r]

    # ── Direction A: Cartesian enumeration of all free variables ──
    use_cartesian_fv = (
        fv_enum_pred is not None and not cartesian_product
        and V >= 2
    )

    if use_cartesian_fv:
        fv_pred_q = fv_enum_pred[active_idx]        # [B, K_r, Fv]
        fv_bound_q = fv_enum_bound_src[active_idx]  # [B, K_r, Fv]
        fv_dir_q = fv_enum_direction[active_idx]    # [B, K_r, Fv]
        fv_valid_q = fv_enum_valid[active_idx]      # [B, K_r, Fv]

        if cartesian_product:
            G_r = E
        else:
            G_r = min(G_r, fact_index._max_facts_per_query)

        # dep buffers: original body order with dep-pos-based fv indices.
        # Source columns are [qs, qo, dep0_cand, dep1_cand, ...].
        dep_src = (arg_source_dep[active_idx]
                   if arg_source_dep is not None
                   else check_arg_source_a[active_idx])
        dep_bpreds = (body_preds_dep[active_idx]
                      if body_preds_dep is not None
                      else body_preds_q)

        source_a, cmask_a, G_r = _enumerate_cartesian(
            B, K_r, qs, qo,
            fv_pred_q, fv_bound_q, fv_dir_q, fv_valid_q,
            has_free_q, active_mask, fact_index,
            K_v=K_v, V=V,
            G_cap=G_r,
            fv_any_valid=fv_any_valid,
            check_arg_source_q=dep_src,
            body_preds_q=dep_bpreds,
            num_body_q=num_body_q,
            M=M,
        )
        body_a = _fill_body_extended(
            source_a, dep_src, dep_bpreds,
        )
    else:
        # Legacy path for V <= 1 or cartesian_product flag
        if cartesian_product:
            G_r = E
        else:
            G_r = min(G_r, fact_index._max_facts_per_query)
        cands_a, cmask_a = _enumerate_dir(
            B, K_r, G_r, qs, qo,
            enum_pred_a[active_idx],
            enum_bound_binding_a[active_idx],
            enum_direction_a[active_idx],
            fact_index,
            cartesian_product=cartesian_product,
            E=E,
        )
        G_r = cands_a.size(2)
        cmask_a = cmask_a & has_free_q.unsqueeze(2)
        cmask_a[:, :, 0] = cmask_a[:, :, 0] | (~has_free_q & active_mask)
        body_a = _fill_body(
            B, K_r, G_r, M, qs, qo, cands_a,
            check_arg_source_a[active_idx], body_preds_q,
        )

    exists_a = fact_index.exists(
        body_a.reshape(-1, 3)).view(B, K_r, G_r, M)

    atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
    body_active = atom_idx < num_body_q.view(B, K_r, 1, 1)

    # Ensure inactive body atoms have padding predicates so downstream
    # prune_ground_facts recognizes them as padding (not spurious atoms).
    body_a = body_a.masked_fill(~body_active.unsqueeze(-1), fact_index._padding_idx)

    mask_a = _apply_enum_filters(
        body_a, exists_a, body_active, active_mask, cmask_a,
        queries, G_r, width, head_pred_mask, is_last=is_last,
    )

    # ── Direction B (dual anchoring) ──
    if any_dual and not cartesian_product:
        has_dual_q = has_dual[active_idx]
        G_b = G_r // 2
        G_use_b = min(G_b, fact_index._max_facts_per_query)

        cands_b, cmask_b = _enumerate_dir(
            B, K_r, G_use_b, qs, qo,
            enum_pred_b[active_idx],
            enum_bound_binding_b[active_idx],
            enum_direction_b[active_idx],
            fact_index,
        )
        G_use_b = cands_b.size(2)
        cmask_b = cmask_b & has_dual_q.unsqueeze(2)

        body_b = _fill_body(
            B, K_r, G_use_b, M, qs, qo, cands_b,
            check_arg_source_b[active_idx], body_preds_q,
        )
        exists_b = fact_index.exists(
            body_b.reshape(-1, 3)).view(B, K_r, G_use_b, M)
        body_active_b = atom_idx < num_body_q.view(B, K_r, 1, 1)

        mask_b = _apply_enum_filters(
            body_b, exists_b, body_active_b, active_mask, cmask_b,
            queries, G_use_b, width, head_pred_mask, is_last=is_last,
        )

        # Exclude fully proven from B (avoid A duplicates)
        all_proven_b = (
            exists_b | ~body_active_b.expand(-1, -1, G_use_b, -1)
        ).all(dim=-1)
        mask_b = mask_b & ~all_proven_b
    else:
        # Empty B direction (zero-sized G_b=0)
        body_b = torch.zeros(B, K_r, 0, M, 3, dtype=torch.long, device=dev)
        exists_b = torch.zeros(B, K_r, 0, M, dtype=torch.bool, device=dev)
        body_active_b = torch.zeros(B, K_r, 1, M, dtype=torch.bool, device=dev)
        mask_b = torch.zeros(B, K_r, 0, dtype=torch.bool, device=dev)
        cmask_b = torch.zeros(B, K_r, 0, dtype=torch.bool, device=dev)

    return (body_a, exists_a, body_active, mask_a, cmask_a,
            active_mask, active_idx,
            body_b, exists_b, body_active_b, mask_b, cmask_b)


def resolve_enum_full(
    queries: Tensor,              # [B, 3]
    query_mask: Tensor,           # [B]
    fact_index,                   # FactIndex with .exists()
    all_entities: Tensor,         # [E]
    slot0_mask: Tensor,           # [E] bool (True at index 0 only)
    *,
    pred_rule_indices: Tensor,
    pred_rule_mask: Tensor,
    has_free: Tensor,
    body_preds: Tensor,
    num_body_atoms: Tensor,
    check_arg_source_a: Tensor,
    M: int,
) -> Dict[str, Tensor]:
    """Full entity enumeration — all entities as candidates (width=None/∞).

    Used when width >= max_body_atoms: every entity is a candidate, no
    fact-anchored enumeration needed.

    Returns same dict structure as resolve_enum() but without direction B.
    """
    B = queries.size(0)
    dev = queries.device
    E = all_entities.size(0)
    qp, qs, qo = queries[:, 0], queries[:, 1], queries[:, 2]

    # Rule clustering
    active_idx = pred_rule_indices[qp]
    K_r = active_idx.size(1)
    active_mask = pred_rule_mask[qp] & query_mask.unsqueeze(1)
    has_free_q = has_free[active_idx]
    body_preds_q = body_preds[active_idx]
    num_body_q = num_body_atoms[active_idx]

    # All entities as candidates
    candidates = all_entities.view(1, 1, E).expand(B, K_r, -1)
    free_mask = has_free_q.unsqueeze(2) & active_mask.unsqueeze(2)
    bound_mask = (~has_free_q).unsqueeze(2) & active_mask.unsqueeze(2)
    cand_mask = free_mask.expand(-1, -1, E) | (
        bound_mask & slot0_mask.view(1, 1, E))

    body_atoms = _fill_body(
        B, K_r, E, M, qs, qo, candidates,
        check_arg_source_a[active_idx], body_preds_q,
    )

    # Query exclusion only (no width filtering for full enum)
    atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
    body_active = atom_idx < num_body_q.view(B, K_r, 1, 1)
    query_exp = queries.view(B, 1, 1, 1, 3).expand(-1, K_r, E, M, -1)
    is_query = (body_atoms == query_exp).all(dim=-1)
    has_query_atom = (
        is_query & body_active.expand(-1, -1, E, -1)).any(dim=-1)
    grounding_mask = cand_mask & ~has_query_atom

    return {
        'body_atoms_a': body_atoms,       # [B, K_r, E, M, 3]
        'grounding_mask_a': grounding_mask,  # [B, K_r, E]
        'active_mask': active_mask,        # [B, K_r]
        'active_idx': active_idx,          # [B, K_r]
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _cartesian_expand_one_fv(
    B: int, K_r: int, query_subjs: Tensor, query_objs: Tensor, fact_index,
    ep: Tensor, eb: Tensor, ed: Tensor, ev: Tensor,
    all_cands: list, all_masks: list, G_current: int,
    k_cap: Optional[int],
) -> Tuple[list, list, int]:
    """Enumerate one free variable's candidates and Cartesian-expand the
    running ``(all_cands, all_masks)`` by it (interleaved layout so a later
    topk picks diverse fv0 candidates before repeats).

    Shared by the dense (``_enumerate_cartesian``) and flat
    (``_enumerate_cartesian_flat``) paths; ``k_cap`` caps ``K_use`` to ``K_v``
    on the dense path and is ``None`` (use the full ``K_f``) on the flat path.
    Returns the grown ``(all_cands, all_masks, G_new)``.
    """
    # Build source for the bound-value lookup: [qs, qo, cand_0, cand_1, ...]
    qs_exp = query_subjs.view(B, 1, 1).expand(B, K_r, G_current)
    qo_exp = query_objs.view(B, 1, 1).expand(B, K_r, G_current)
    src = torch.stack([qs_exp, qo_exp] + all_cands, dim=3)  # [B,K_r,G_current,2+]
    W_cur = src.size(3)
    eb_idx = eb.clamp(max=W_cur - 1).view(B, K_r, 1, 1).expand(
        B, K_r, G_current, 1)
    bound_vals = src.gather(3, eb_idx).squeeze(3)           # [B, K_r, G_current]

    flat_pred = ep.view(B, K_r, 1).expand(B, K_r, G_current).reshape(-1)
    flat_bound = bound_vals.reshape(-1)
    flat_dir = ed.view(B, K_r, 1).expand(B, K_r, G_current).reshape(-1)

    new_cands, new_mask = fact_index.enumerate(flat_pred, flat_bound, flat_dir)
    K_fi = new_cands.size(1)
    K_use = K_fi if k_cap is None else min(K_fi, k_cap)

    new_cands = new_cands[:, :K_use].reshape(B, K_r, G_current, K_use)
    new_mask = new_mask[:, :K_use].reshape(B, K_r, G_current, K_use)
    # Mask rules that don't have this free variable
    new_mask = new_mask & ev.view(B, K_r, 1, 1)

    G_new = G_current * K_use
    # Previous candidates: repeat each K_use times, interleaved.
    expanded_cands = [
        prev.unsqueeze(3).expand(B, K_r, G_current, K_use
                                 ).transpose(2, 3).reshape(B, K_r, G_new)
        for prev in all_cands]
    expanded_masks = [
        prev_m.unsqueeze(3).expand(B, K_r, G_current, K_use
                                   ).transpose(2, 3).reshape(B, K_r, G_new)
        for prev_m in all_masks]
    # New candidates: each repeated G_current times (one per prev slot).
    expanded_cands.append(new_cands.transpose(2, 3).reshape(B, K_r, G_new))
    expanded_masks.append(new_mask.transpose(2, 3).reshape(B, K_r, G_new))
    return expanded_cands, expanded_masks, G_new


def _enumerate_cartesian(
    B: int, K_r: int,
    query_subjs: Tensor,     # [B]
    query_objs: Tensor,      # [B]
    fv_pred_q: Tensor,       # [B, K_r, Fv]
    fv_bound_q: Tensor,      # [B, K_r, Fv]
    fv_dir_q: Tensor,        # [B, K_r, Fv]
    fv_valid_q: Tensor,      # [B, K_r, Fv]
    has_free_q: Tensor,      # [B, K_r]
    active_mask: Tensor,     # [B, K_r]
    fact_index,
    K_v: int,
    V: int,
    G_cap: int = 0,
    fv_any_valid: Optional[list] = None,
    # Body metadata for exists-informed pruning
    check_arg_source_q: Optional[Tensor] = None,  # [B, K_r, M, 2]
    body_preds_q: Optional[Tensor] = None,        # [B, K_r, M]
    num_body_q: Optional[Tensor] = None,           # [B, K_r]
    M: int = 0,
) -> Tuple[Tensor, Tensor, int]:
    """Sequential Cartesian enumeration of all free variables.

    For each free variable in dependency order:
    1. Get bound values from previously-resolved variables
    2. Enumerate candidates from fact_index
    3. Expand the grounding dimension with Cartesian product
    4. If G_current exceeds G_cap, topk-cap to G_cap using
       exists-informed scoring (valid body atoms get priority)

    Returns:
        source: [B, K_r, G_final, 2 + V]
            Column layout: [qs, qo, fv0_cand, fv1_cand, ...]
        combined_mask: [B, K_r, G_final]
            True where all relevant free vars have valid candidates.
        G_final: int, the grounding dimension.
    """
    dev = query_subjs.device
    if G_cap <= 0:
        G_cap = K_v  # default: same as single free var

    # Per-free-var candidate lists (grow via Cartesian expansion)
    all_cands: list = []   # [B, K_r, G_current] per free var
    all_masks: list = []   # [B, K_r, G_current] validity per free var
    G_current = 1          # starts at 1 (no enumeration yet)

    for fv_idx in range(V):
        # Skip fv slots where NO rule has this free variable (Python-level,
        # not traced). Avoids Cartesian expansion creating K_use duplicates
        # for free vars that only exist in some rules.
        if fv_any_valid is not None and not fv_any_valid[fv_idx]:
            # Still need to add a placeholder for this fv slot
            all_cands.append(torch.zeros(B, K_r, G_current, dtype=torch.long, device=dev))
            all_masks.append(torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev))
            continue

        ep = fv_pred_q[:, :, fv_idx]    # [B, K_r]
        eb = fv_bound_q[:, :, fv_idx]   # [B, K_r]
        ed = fv_dir_q[:, :, fv_idx]     # [B, K_r]
        ev = fv_valid_q[:, :, fv_idx]   # [B, K_r]

        # Enumerate this fv + Cartesian-expand (dense: cap K_use to K_v).
        all_cands, all_masks, G_current = _cartesian_expand_one_fv(
            B, K_r, query_subjs, query_objs, fact_index,
            ep, eb, ed, ev, all_cands, all_masks, G_current, K_v)

        # Cap G_current to G_cap via topk (keeps memory bounded).
        # For 1-free-var rules: G_current = K_v ≤ G_cap (no cap needed).
        # For 2+-free-var rules: G_current = K_v^2 > G_cap → cap here.
        if G_current > G_cap:
            # Score: slot is "valid" if all RELEVANT free vars are valid.
            # This avoids bias toward rules with more free vars.
            combined = torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev)
            for fi in range(len(all_masks)):
                fv_rel = fv_valid_q[:, :, fi].unsqueeze(2)
                combined = combined & (all_masks[fi] | ~fv_rel)
            _, top_idx = combined.to(torch.int8).topk(
                G_cap, dim=2, largest=True, sorted=False)
            all_cands = [c.gather(2, top_idx) for c in all_cands]
            all_masks = [m.to(torch.long).gather(2, top_idx).bool()
                         for m in all_masks]
            G_current = G_cap

    G_final = G_current

    # Combined mask: valid if all RELEVANT free vars have valid candidates.
    # For rules with fewer free vars, irrelevant slots pass by default.
    combined_mask = torch.ones(B, K_r, G_final, dtype=torch.bool, device=dev)
    for fv_idx in range(V):
        fv_relevant = fv_valid_q[:, :, fv_idx].unsqueeze(2)  # [B, K_r, 1]
        combined_mask = combined_mask & (all_masks[fv_idx] | ~fv_relevant)

    # Bound rules (no free vars): only slot 0 is valid
    combined_mask = combined_mask & has_free_q.unsqueeze(2)
    combined_mask[:, :, 0] = combined_mask[:, :, 0] | (
        ~has_free_q & active_mask)

    # Build extended source: [B, K_r, G_final, 2 + V]
    qs_final = query_subjs.view(B, 1, 1).expand(B, K_r, G_final)
    qo_final = query_objs.view(B, 1, 1).expand(B, K_r, G_final)
    source = torch.stack([qs_final, qo_final] + all_cands, dim=3)

    return source, combined_mask, G_final


def _fill_body_atoms(source_m: Tensor, check_arg_m: Tensor,
                     body_preds_m: Tensor) -> Tensor:
    """Rank-agnostic body-atom fill shared by the dense and flat paths.

    ``source_m`` [..., M, W]; ``check_arg_m`` [..., M, 2] indexes the trailing
    W axis (clamped to W-1 — indices >= W are unresolved free vars, filtered
    out downstream by the exists-check); ``body_preds_m`` [..., M]. Returns
    [..., M, 3] = (pred, arg0, arg1).
    """
    W = source_m.size(-1)
    idx_0 = check_arg_m[..., 0].clamp(max=W - 1).unsqueeze(-1)
    arg0 = source_m.gather(-1, idx_0).squeeze(-1)
    idx_1 = check_arg_m[..., 1].clamp(max=W - 1).unsqueeze(-1)
    arg1 = source_m.gather(-1, idx_1).squeeze(-1)
    return torch.stack([body_preds_m, arg0, arg1], dim=-1)


def _fill_body_extended(
    source: Tensor,                # [B, K_r, G_r, W]
    check_arg_source_q: Tensor,    # [B, K_r, M, 2]
    body_preds_q: Tensor,          # [B, K_r, M]
) -> Tensor:
    """Fill body atoms from an extended dense source.

    check_arg_source_q values index into source's last dimension:
      0 = query subject, 1 = query object, 2 = fv0 candidate, 3 = fv1, ...

    Returns: [B, K_r, G_r, M, 3] body atoms.
    """
    G_r = source.size(2)
    M = body_preds_q.size(2)
    source_m = source.unsqueeze(3).expand(-1, -1, -1, M, -1)        # [B,K_r,G_r,M,W]
    check_m = check_arg_source_q.unsqueeze(2).expand(-1, -1, G_r, -1, -1)  # [B,K_r,G_r,M,2]
    preds_m = body_preds_q.unsqueeze(2).expand(-1, -1, G_r, -1)     # [B,K_r,G_r,M]
    return _fill_body_atoms(source_m, check_m, preds_m)


def _enumerate_dir(
    B: int, K_r: int, G_r: int,
    query_subjs: Tensor,     # [B]
    query_objs: Tensor,      # [B]
    enum_pred_q: Tensor,     # [B, K_r]
    enum_bound_q: Tensor,    # [B, K_r]
    enum_dir_q: Tensor,      # [B, K_r]
    fact_index,
    cartesian_product: bool = False,
    E: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Enumerate candidates for one direction across clustered rules.

    When ``cartesian_product=True``, returns all entity indices ``[0..E-1]``
    as candidates for every (query, rule) pair instead of fact-anchored lookup.

    Returns:
        candidates: [B, K_r, G_actual]
        cand_mask:  [B, K_r, G_actual]
    """
    if cartesian_product:
        dev = query_subjs.device
        # All entities as candidates for every (query, rule) pair
        candidates = torch.arange(E, device=dev).unsqueeze(0).unsqueeze(0).expand(
            B * K_r, 1, -1).reshape(B, K_r, E)
        cand_mask = torch.ones(B, K_r, E, dtype=torch.bool, device=dev)
        return candidates, cand_mask

    source = torch.stack([query_subjs, query_objs], dim=1)  # [B, 2]
    enum_bound_vals = source.gather(1, enum_bound_q)         # [B, K_r]

    candidates, cand_mask = fact_index.enumerate(
        enum_pred_q.reshape(-1),
        enum_bound_vals.reshape(-1),
        enum_dir_q.reshape(-1),
    )
    G_fi = candidates.size(1)
    G_actual = min(G_r, G_fi)
    candidates = candidates[:, :G_actual].reshape(B, K_r, G_actual)
    cand_mask = cand_mask[:, :G_actual].reshape(B, K_r, G_actual)
    return candidates, cand_mask


def _fill_body(
    B: int, K_r: int, G_r: int, M: int,
    query_subjs: Tensor,          # [B]
    query_objs: Tensor,           # [B]
    candidates: Tensor,           # [B, K_r, G_r]
    check_arg_source_q: Tensor,   # [B, K_r, M, 2]
    body_preds_q: Tensor,         # [B, K_r, M]
) -> Tensor:
    """Fill body atoms from (subj, obj, candidate) binding table.

    check_arg_source_q[..., 0/1] maps each body atom argument to its source:
      0 = query subject, 1 = query object, 2 = enumerated candidate.

    Returns: [B, K_r, G_r, M, 3] body atoms.
    """
    q_s = query_subjs.view(B, 1, 1).expand(-1, K_r, G_r)
    q_o = query_objs.view(B, 1, 1).expand(-1, K_r, G_r)
    source = torch.stack([q_s, q_o, candidates], dim=3)       # [B, K_r, G_r, 3]
    # W=3 here, so _fill_body_extended's clamp(max=W-1) == the old clamp(max=2):
    # indices >= 3 are unresolved free variables (filtered out by exists-check).
    return _fill_body_extended(source, check_arg_source_q, body_preds_q)



def _all_body_active_ok(ok: Tensor, body_active: Tensor) -> Tensor:
    """Reduce a per-atom OK mask over body atoms: an entry survives when every
    ACTIVE body atom is OK. ``ok`` / ``body_active`` are [..., M] → returns
    [...]. Rank-agnostic (dense [B,K_r,G_r,M] and flat [T,M] both reduce the
    trailing M axis)."""
    return (ok | ~body_active).all(dim=-1)


def _head_pred_all_ok(body_pred_vals: Tensor, exists: Tensor,
                      body_active: Tensor, head_pred_mask: Tensor) -> Tensor:
    """Head-predicate prune mask: every active body atom is either a fact
    (``exists``) or carries a head predicate (provable at a later step).
    Shared by the dense and flat filters. Padded body predicates are clamped
    into range — the ``~body_active`` term admits padded slots regardless, so
    the clamped value is never read."""
    P = head_pred_mask.size(0)
    head_pred_ok = head_pred_mask[body_pred_vals.clamp(max=P - 1)]
    return _all_body_active_ok(exists | head_pred_ok, body_active)


def _apply_enum_filters(
    body_atoms: Tensor,       # [B, K_r, G_r, M, 3]
    exists: Tensor,           # [B, K_r, G_r, M]
    body_active: Tensor,      # [B, K_r, 1, M]
    active_mask: Tensor,      # [B, K_r]
    cand_mask: Tensor,        # [B, K_r, G_r]
    queries: Tensor,          # [B, 3]
    G_r: int,
    width,                    # int (eager) or 0-dim Tensor (compiled), or None
    head_pred_mask: Optional[Tensor],   # [P] bool, None to skip head-pred prune
    *,
    is_last: Optional[Tensor] = None,   # 0-dim bool tensor (compiled path)
                                        # — when True, skip head-pred prune
                                        # and force the all-exist branch
                                        # (matches eager ``head_pred_mask=None``
                                        # + ``width==0`` semantics).
) -> Tensor:
    """Apply width filtering, query exclusion, and head predicate pruning.

    Returns: [B, K_r, G_r] grounding mask.

    Pass ``head_pred_mask=None`` to disable the unprovable-unknown filter
    (matches keras-ns ``prune_incomplete_proofs=False``). Used at the last
    proof step, where there is no further depth to prove unknowns at.

    In the compiled path the caller passes ``head_pred_mask`` as a tensor
    and ``is_last`` as a 0-dim bool — the head-pred prune is gated via
    ``torch.where(is_last, ...)`` instead of a Python ``if``.
    """
    B, K_r = active_mask.shape
    M = body_atoms.shape[3]
    body_active_exp = body_active.expand(-1, -1, G_r, -1)

    # Width filtering
    if width is None:
        mask = active_mask.unsqueeze(2) & cand_mask
    else:
        num_unknown = (body_active_exp & ~exists).sum(dim=-1)
        mask = (num_unknown <= width) & active_mask.unsqueeze(2) & cand_mask

    # Query exclusion: no body atom equals the query
    query_exp = queries.view(B, 1, 1, 1, 3).expand(-1, K_r, G_r, M, -1)
    is_query = (body_atoms == query_exp).all(dim=-1)
    has_query_atom = (is_query & body_active_exp).any(dim=-1)
    mask = mask & ~has_query_atom

    # Head predicate pruning (only when width is bounded AND a mask is given)
    if width is not None:
        if head_pred_mask is not None:
            all_ok = _head_pred_all_ok(
                body_atoms[..., 0], exists, body_active_exp, head_pred_mask)
            if is_last is not None:
                # Skip the prune at the last step (matches the eager
                # ``head_pred_mask=None`` branch).
                all_ok = torch.where(
                    is_last, torch.ones_like(all_ok), all_ok)
            mask = mask & all_ok

        # ``width == 0`` triggers the strict all-exist branch (every
        # body atom must be a fact). When ``width`` is a tensor, gate
        # the branch on ``is_last`` so it fires only at the last step
        # (which is where ``width_d == w_last_depth == 0`` for u=0 in
        # the compiled path).
        if isinstance(width, torch.Tensor):
            if is_last is not None:
                all_exist = _all_body_active_ok(exists, body_active_exp)
                # Apply strict branch only when is_last (mirrors the
                # eager ``width_d == 0 and last step``).
                strict_mask = mask & all_exist
                mask = torch.where(is_last, strict_mask, mask)
        elif width == 0:
            all_exist = _all_body_active_ok(exists, body_active_exp)
            mask = mask & all_exist

    return mask


# ═══════════════════════════════════════════════════════════════════════
# Flat intermediate: complete step (enumerate → fill → exists → filter → dense)
# ═══════════════════════════════════════════════════════════════════════


# Cache for constant ``torch.arange(n)`` tensors keyed by (n, device).
# These index/compare-only constants (atom_idx, K_r, B aranges) recur on
# every flat step but never depend on data, so caching them once removes a
# fresh ``arange`` launch (kernel + the host-side dispatch) per use. Returned
# tensors MUST be treated read-only — every call site only indexes, compares,
# or broadcasts against them, never mutates in place. Only the eager FLAT
# path calls this (compile gates to the dense path), but the ``is_compiling``
# guard keeps a cached tensor from ever aliasing inside a CUDA graph.
_ARANGE_CACHE: Dict[Tuple[int, str], Tensor] = {}


def _arange_cached(n: int, device) -> Tensor:
    """Return a cached ``torch.arange(n, device=device, dtype=long)`` view.

    Read-only: callers must not mutate the result. ``n`` is a static
    (rule-derived) dimension so the cache stays tiny and bounded. Materialised
    fresh under ``torch.compile`` — a persistent cached tensor is not
    CUDA-graph-safe (it aliases across graph replays).
    """
    if torch.compiler.is_compiling():
        return torch.arange(n, device=device)
    key = (int(n), str(device))
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device)
        _ARANGE_CACHE[key] = t
    return t


def _cumcount_flat(keys: Tensor) -> Tensor:
    """Compute 0-based position within each group of equal keys.

    For keys [A, A, B, A, B, C] returns [0, 1, 0, 2, 1, 0].
    Uses sort + cummax trick — no data-dependent control flow.
    """
    T = keys.size(0)
    if T == 0:
        return keys.new_empty(0, dtype=torch.long)
    dev = keys.device
    sort_perm = torch.argsort(keys, stable=True)
    sorted_keys = keys[sort_perm]
    running_idx = torch.arange(T, device=dev)
    # ``group_change[0]=True``, then ``sorted[i] != sorted[i-1]``. Built via
    # cat (alloc + concat) instead of a full ones-alloc + indexed
    # scatter-write. ``group_starts = running_idx`` where a group begins else
    # 0 == ``running_idx * group_change`` (running_idx ≥ 0, change ∈ {0,1}),
    # dropping the ``where`` + ``zeros_like`` alloc.
    ne = sorted_keys[1:] != sorted_keys[:-1]
    group_change = torch.cat(
        [torch.ones(1, dtype=torch.bool, device=dev), ne], dim=0)
    group_starts = running_idx * group_change
    group_starts, _ = group_starts.cummax(0)
    group_pos = running_idx - group_starts
    result = torch.zeros(T, dtype=torch.long, device=dev)
    result[sort_perm] = group_pos
    return result


def _resolve_enum_step_flat(
    queries: Tensor, remaining: Tensor, grounding_body: Tensor,
    state_valid: Tensor, active_mask: Tensor, *,
    fact_index, d: int, depth: int, width: Optional[int], M: int,
    padding_idx: int, G_r: int, K: int,
    pred_rule_indices: Tensor, pred_rule_mask: Tensor,
    has_free: Tensor, body_preds: Tensor, num_body_atoms: Tensor,
    check_arg_source_a: Tensor, head_pred_mask: Optional[Tensor],
    fv_enum_pred: Tensor, fv_enum_bound_src: Tensor,
    fv_enum_direction: Tensor, fv_enum_valid: Tensor,
    V: int, fv_any_valid: Optional[list],
    arg_source_dep: Optional[Tensor], body_preds_dep: Optional[Tensor],
    collect_evidence: bool, w_last_depth: int,
    dedup_goals: bool = False,
) -> ResolvedChildren:
    """Flat intermediate path for resolve_enum_step.

    Enumerates ALL valid candidates (no G_r cap), fills body atoms,
    checks exists, filters — all on flat tensors. Then converts the
    surviving entries into dense [B, S, K_cap, G, 3] for pack_states.

    ``dedup_goals`` (task #48 Phase-2, resolve-SKIP): when True, the
    expensive survivor pipeline (enumerate → fill → exists → filter) runs
    over the DISTINCT selected goals only, then each distinct goal's
    survivors are scattered back to every active state sharing that goal.
    The survivor set is a pure function of ``(goal_triple, is_last)`` +
    fixed rule/fact buffers, so this is byte-identical (the per-state child
    is reconstructed as survivors ⊕ that state's own remaining goals +
    grounding_body). Skips the ~Nx redundancy when many states share a
    goal (674x at d>=1 on countries_s3 BC13).
    """
    B, S, _ = queries.shape
    G = remaining.shape[2]
    pad = padding_idx
    dev = queries.device
    N = B * S

    width_d = width
    if d == depth - 1 and width is not None:
        width_d = w_last_depth
    # Last step: skip head-pred prune (no further depth to prove unknowns).
    head_pred_mask_d: Optional[Tensor] = (
        None if d == depth - 1 else head_pred_mask
    )

    # 1. Flatten queries — full [N, 3] view first.
    flat_q_full = queries.reshape(N, 3)
    flat_valid_full = (active_mask & state_valid).reshape(N)

    # 1b. Compact to active rows only (Option C: dynamic-shape compaction).
    # The flat path is always eager (the compiled path is gated to dense),
    # so dropping invalid rows is a pure win — every downstream tensor
    # sized [N, K_r, ...] becomes [N_eff, K_r, ...] with N_eff = active
    # state count. When all queries are active (typical at depth 0), the
    # nonzero-then-index pair reduces to a no-op aside from one O(N)
    # boolean scan; the saving at depth>=1 (where state_valid sparsity is
    # high) more than pays for it.
    active_pos = torch.nonzero(
        flat_valid_full, as_tuple=False).squeeze(1)        # [N_eff]
    N_eff = active_pos.size(0)
    if N_eff == 0:
        G_body = grounding_body.shape[2]
        return _empty_flat_resolved(B, S, G, G_body, pad, dev)
    flat_q_active = flat_q_full[active_pos]                 # [N_eff, 3]

    # 1c. Resolve-SKIP (task #48 Phase-2): the survivor set is a pure
    # function of the goal triple (+ is_last via width_d/head_pred_mask_d).
    # Dedup the active goals so the expensive enumerate/fill/exists/filter
    # pipeline runs ONCE per distinct goal; survivors are scattered back to
    # every active state sharing that goal in step 8b. ``state_to_goal``
    # maps each active row -> its distinct-goal index (used for expansion).
    if dedup_goals:
        distinct_goals, state_to_goal = torch.unique(
            flat_q_active, dim=0, return_inverse=True)      # [D,3], [N_eff]
        flat_q = distinct_goals                             # pipeline operates on goals
    else:
        state_to_goal = None
        flat_q = flat_q_active
    qp, qs, qo = flat_q[:, 0], flat_q[:, 1], flat_q[:, 2]
    N_goals = flat_q.size(0)   # == D (distinct) when dedup, else N_eff

    # 2. Rule clustering (same as dense path) — over compacted rows.
    active_idx = pred_rule_indices[qp]                     # [N_goals, K_r]
    K_r = active_idx.size(1)
    # All compacted rows are valid by construction, so the per-(state,K_r)
    # mask reduces to ``pred_rule_mask`` (which K_r slots correspond to
    # a real rule for the query's predicate).
    amask = pred_rule_mask[qp]                             # [N_eff, K_r]
    has_free_q = has_free[active_idx]
    # (per-(state,K_r) ``num_body_atoms`` is not needed here — the flat path
    # gathers ``nbody_flat`` per surviving candidate from ``rule_global_idx``
    # below; the dense [N_eff, K_r] gather was dead.)

    # Gather per-free-var metadata
    fv_pred_q = fv_enum_pred[active_idx]
    fv_bound_q = fv_enum_bound_src[active_idx]
    fv_dir_q = fv_enum_direction[active_idx]
    fv_valid_q = fv_enum_valid[active_idx]

    # 3. Flat Cartesian enumeration (zero waste) over compacted rows.
    # Enumerate returns the row index (``n_eff`` per active state, or the
    # distinct-goal index when ``dedup_goals``) and the K_r-position
    # directly — no encode/decode round-trip needed here.
    flat_source, flat_n_idx_eff, flat_r_idx = _enumerate_cartesian_flat(
        N_goals, K_r, qs, qo,
        fv_pred_q, fv_bound_q, fv_dir_q, fv_valid_q,
        has_free_q, amask, fact_index,
        V=V,
        fv_any_valid=fv_any_valid,
    )
    T = flat_source.size(0)  # total valid candidates

    if T == 0:
        # No valid candidates — return empty ResolvedChildren
        G_body = grounding_body.shape[2]
        return _empty_flat_resolved(B, S, G, G_body, pad, dev)

    # 4. Gather per-entry metadata for fill_body
    dep_src = (arg_source_dep if arg_source_dep is not None
               else check_arg_source_a)
    dep_bpreds = (body_preds_dep if body_preds_dep is not None
                  else body_preds)
    # ``flat_n_idx_eff`` indexes the compacted rows; it is converted back to
    # the full [B*S] index via ``active_pos`` only at the boundary where
    # b_idx / s_idx for the dense state layout are needed.

    # Gather per-entry arg_source and body_preds
    # dep_src: [R_total, M, 2], active_idx: [N_eff, K_r] → per (n,r): active_idx[n,r]
    rule_global_idx = active_idx[flat_n_idx_eff, flat_r_idx]  # [T] — global rule index
    check_flat = dep_src[rule_global_idx]                  # [T, M, 2]
    bpreds_flat = dep_bpreds[rule_global_idx]              # [T, M]
    nbody_flat = num_body_atoms[rule_global_idx]           # [T]

    # 5. Fill body atoms
    flat_body = _fill_body_flat(flat_source, check_flat, bpreds_flat)  # [T, M, 3]

    # 6. Exists check
    flat_exists = fact_index.exists(flat_body.reshape(-1, 3)).reshape(T, M)

    # Mask inactive body atoms with padding. ``body_active`` is reused
    # by ``_apply_filters_flat`` below (identical ``atom_idx < nbody``),
    # so it's computed once here instead of twice.
    atom_idx = _arange_cached(M, dev)  # [M], cached constant
    body_active = atom_idx.unsqueeze(0) < nbody_flat.unsqueeze(1)
    flat_body = flat_body.masked_fill(~body_active.unsqueeze(-1), pad)

    # 7. Filters
    # Need queries [N_eff, 3] indexed by flat_n_idx_eff (compacted index).
    fmask = _apply_filters_flat(
        flat_body, flat_exists, flat_n_idx_eff, nbody_flat,
        flat_q, width_d, head_pred_mask_d, M, body_active=body_active)

    # 8. Extract surviving entries
    surv_idx = torch.nonzero(fmask, as_tuple=False).squeeze(1)  # [T_surv]
    T_surv = surv_idx.size(0)

    if T_surv == 0:
        G_body = grounding_body.shape[2]
        return _empty_flat_resolved(B, S, G, G_body, pad, dev)

    surv_body = flat_body[surv_idx]                    # [T_surv, M, 3]
    surv_g_idx = flat_n_idx_eff[surv_idx]              # [T_surv] — row index
    surv_rule_idx = rule_global_idx[surv_idx]          # [T_surv]

    # 8b. Resolve-SKIP expansion: when ``dedup_goals``, each survivor was
    # computed for a DISTINCT goal; replicate it to every active state
    # sharing that goal. The per-state child = these (goal-determined)
    # survivors ⊕ that state's own remaining goals + grounding_body
    # (attached in steps 9+). Build the join by sorting survivors and
    # states by goal and gathering via offsets (vectorised, no python loop).
    if dedup_goals:
        # states grouped by goal: st_cnt[g] = #active states with goal g.
        # survivors grouped by goal: sv_cnt[g] = #survivors for goal g.
        D = N_goals
        # active-state compacted indices grouped by their goal:
        st_order = torch.argsort(state_to_goal, stable=True)     # [N_eff]
        st_cnt = torch.bincount(state_to_goal, minlength=D)      # [D]
        st_off = torch.zeros(D + 1, dtype=torch.long, device=dev)
        st_off[1:] = torch.cumsum(st_cnt, 0)
        # survivors grouped by goal:
        sv_order = torch.argsort(surv_g_idx, stable=True)        # [T_surv]
        sv_cnt = torch.bincount(surv_g_idx, minlength=D)         # [D]
        sv_off = torch.zeros(D + 1, dtype=torch.long, device=dev)
        sv_off[1:] = torch.cumsum(sv_cnt, 0)
        # For each (survivor j of goal g, state k of goal g) emit one row.
        # Total expanded rows = sum_g sv_cnt[g]*st_cnt[g]. Build index lists.
        per_goal_total = sv_cnt * st_cnt                         # [D]
        E_tot = int(per_goal_total.sum().item())
        if E_tot == 0:
            G_body = grounding_body.shape[2]
            return _empty_flat_resolved(B, S, G, G_body, pad, dev)
        # position within the flattened expansion -> (goal, sv_local, st_local)
        e_goal = torch.repeat_interleave(
            torch.arange(D, device=dev), per_goal_total)         # [E_tot]
        e_within = (torch.arange(E_tot, device=dev)
                    - torch.repeat_interleave(
                        torch.cumsum(per_goal_total, 0) - per_goal_total,
                        per_goal_total))                          # [E_tot] 0..(sv*st-1)
        g_st_cnt = st_cnt[e_goal]                                 # [E_tot]
        sv_local = e_within // g_st_cnt                           # which survivor
        st_local = e_within % g_st_cnt                            # which state
        # gather the survivor row (in sv_order space) and state (in st_order):
        sv_pos = sv_off[e_goal] + sv_local                        # idx into sv_order
        st_pos = st_off[e_goal] + st_local                        # idx into st_order
        sv_rows = sv_order[sv_pos]                                # [E_tot] -> survivor row
        st_rows_eff = st_order[st_pos]                            # [E_tot] -> active-state compacted idx
        surv_body = surv_body[sv_rows]
        surv_rule_idx = surv_rule_idx[sv_rows]
        surv_n_idx_eff = st_rows_eff
    else:
        surv_n_idx_eff = surv_g_idx
    # After (optional) expansion the row count is E_tot; rebind T_surv so
    # steps 9+ allocate the right shapes on either path.
    T_surv = surv_body.size(0)
    # Map compacted index back to full [B*S] index for the dense
    # b/s decomposition expected by the downstream pack.
    surv_n_idx = active_pos[surv_n_idx_eff]            # [T_surv (or E_tot)]

    # Body atoms are already in canonical order via ``arg_source_dep`` /
    # ``body_preds_dep`` (which the ``_PatternVariant`` _orig_body_patterns
    # propagation makes invariant across anchor variants). The previous
    # per-step argsort to canonicalise body atom order was redundant
    # given that fix, so it's removed to save Python-dispatch cycles
    # on small workloads (~5 tensor ops per step).

    # 9. Build flat goals [T_surv, G, 3] — body atoms + remaining parent goals
    surv_b_idx = surv_n_idx // S       # [T_surv] — batch index (full N)
    surv_s_idx = surv_n_idx % S        # [T_surv] — state index (full N)

    # Build flat_goals: [T_surv, G, 3]
    flat_goals = torch.full(
        (T_surv, G, 3), pad, dtype=torch.long, device=dev)
    flat_goals[:, :M, :] = surv_body   # body atoms at positions 0..M-1 (canonical)

    # Copy remaining goals from parent at positions M..G-1
    n_rem = min(G - M, G - 1)
    if n_rem > 0:
        rem = remaining[surv_b_idx, surv_s_idx, 1:1 + n_rem, :]  # [T_surv, n_rem, 3]
        flat_goals[:, M:M + n_rem, :] = rem

    # Grounding body field (shape [T_surv, G_body, 3]). The downstream
    # ``pack_states_flat`` rebuilds out_gbody from ``flat_goals[:M_rule]`` and
    # never reads this field, and no other consumer touches it — so the
    # expensive ``grounding_body[surv_b_idx, surv_s_idx]`` parent gather it
    # used to hold (under collect_evidence) was dead. A same-shape zeros
    # placeholder keeps the FlatResolvedChildren contract byte-identically
    # for the grounding output while dropping that gather from the hot path.
    G_body = grounding_body.shape[2]
    flat_gbody = torch.zeros(T_surv, G_body, 3, dtype=torch.long, device=dev)

    flat_subs = torch.full(
        (T_surv, 2, 2), pad, dtype=torch.long, device=dev)

    return FlatResolvedChildren(
        flat_goals=flat_goals,
        flat_gbody=flat_gbody,
        flat_rule_idx=surv_rule_idx,
        flat_b_idx=surv_b_idx,
        flat_s_idx=surv_s_idx,
        flat_subs=flat_subs,
        B=B, S=S,
    )


def _empty_flat_resolved(B, S, G, G_body, pad, dev):
    """Return empty FlatResolvedChildren when no candidates survive."""
    return FlatResolvedChildren(
        flat_goals=torch.full((0, G, 3), pad, dtype=torch.long, device=dev),
        flat_gbody=torch.zeros(0, G_body, 3, dtype=torch.long, device=dev),
        flat_rule_idx=torch.zeros(0, dtype=torch.long, device=dev),
        flat_b_idx=torch.zeros(0, dtype=torch.long, device=dev),
        flat_s_idx=torch.zeros(0, dtype=torch.long, device=dev),
        flat_subs=torch.full((0, 2, 2), pad, dtype=torch.long, device=dev),
        B=B, S=S,
    )


# ═══════════════════════════════════════════════════════════════════════
# Flat intermediate helpers (zero grounding loss)
# ═══════════════════════════════════════════════════════════════════════


def _enumerate_cartesian_flat(
    B: int, K_r: int,
    query_subjs: Tensor,     # [B]
    query_objs: Tensor,      # [B]
    fv_pred_q: Tensor,       # [B, K_r, Fv]
    fv_bound_q: Tensor,      # [B, K_r, Fv]
    fv_dir_q: Tensor,        # [B, K_r, Fv]
    fv_valid_q: Tensor,      # [B, K_r, Fv]
    has_free_q: Tensor,      # [B, K_r]
    active_mask: Tensor,     # [B, K_r]
    fact_index,
    V: int,
    fv_any_valid: Optional[list] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Cartesian enumeration returning flat+offsets (zero padding waste).

    Same algorithm as _enumerate_cartesian but instead of pre-allocating
    a padded [B, K_r, G_r, ...] tensor, produces only valid entries:

    Returns:
        flat_source: [total_valid, 2 + V]
            Each row: [qs, qo, fv0_cand, fv1_cand, ...]
        b_idx: [total_valid]
            Compacted query index (0..B-1) each entry belongs to.
        r_idx: [total_valid]
            Which K_r-position (rule) each entry belongs to (into active_idx).
    """
    dev = query_subjs.device
    K_f = fact_index._max_facts_per_query

    # Per-free-var candidate lists — start dense [B, K_r, G_current]
    # then extract valid entries at the end.
    all_cands: list = []
    all_masks: list = []
    G_current = 1

    for fv_idx in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv_idx]:
            all_cands.append(torch.zeros(B, K_r, G_current, dtype=torch.long, device=dev))
            all_masks.append(torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev))
            continue

        ep = fv_pred_q[:, :, fv_idx]
        eb = fv_bound_q[:, :, fv_idx]
        ed = fv_dir_q[:, :, fv_idx]
        ev = fv_valid_q[:, :, fv_idx]

        # Enumerate this fv + Cartesian-expand (flat: no cap, use full K_f).
        # No dedup in the flat path — keep all entries including duplicates;
        # the flat→dense conversion handles dedup via the K_cap topk.
        all_cands, all_masks, G_current = _cartesian_expand_one_fv(
            B, K_r, query_subjs, query_objs, fact_index,
            ep, eb, ed, ev, all_cands, all_masks, G_current, None)

    # Combined mask: valid if all RELEVANT free vars have valid candidates.
    # Seed from the first free-var term instead of a ``torch.ones`` alloc +
    # redundant leading ``&`` (``ones & X == X``); fold the remaining terms in.
    combined_mask: Optional[Tensor] = None
    for fv_idx in range(V):
        fv_relevant = fv_valid_q[:, :, fv_idx].unsqueeze(2)
        term = all_masks[fv_idx] | ~fv_relevant
        combined_mask = term if combined_mask is None else combined_mask & term
    if combined_mask is None:
        combined_mask = torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev)

    combined_mask = combined_mask & has_free_q.unsqueeze(2)
    combined_mask[:, :, 0] = combined_mask[:, :, 0] | (~has_free_q & active_mask)
    # Apply per-(state, K_r) active_mask: ``pred_rule_mask[qp]`` marks
    # which K_r positions actually correspond to a rule with head=qp's
    # predicate. Padded K_r slots (mask=False) should produce no
    # candidates regardless of has_free / fv_valid status. Without
    # this AND the candidates from the padded rule (rule_idx 0) leak
    # through and get recorded as spurious apps with wrong heads.
    combined_mask = combined_mask & active_mask.unsqueeze(2)

    # ── Extract valid entries into flat representation ──
    # combined_mask: [B, K_r, G_current] → nonzero gives [total_valid, 3]
    valid_idx = torch.nonzero(combined_mask, as_tuple=False)  # [total_valid, 3]
    total_valid = valid_idx.size(0)

    b_idx = valid_idx[:, 0]   # [total_valid] — batch index
    r_idx = valid_idx[:, 1]   # [total_valid] — rule index
    g_idx = valid_idx[:, 2]   # [total_valid] — grounding index

    # Build flat_source: [total_valid, 2 + V]
    flat_parts = [
        query_subjs[b_idx],  # qs
        query_objs[b_idx],   # qo
    ]
    for c in all_cands:
        flat_parts.append(c[b_idx, r_idx, g_idx])
    flat_source = torch.stack(flat_parts, dim=1)  # [total_valid, 2+Fv]

    # Return the compacted-query index (``b_idx``) and the K_r-position
    # (``r_idx``) directly. The caller previously recombined them into
    # ``b_idx * K_r + r_idx`` only to split them back out with ``// K_r`` /
    # ``% K_r`` — returning both avoids the encode + two decodes per step.
    return flat_source, b_idx, r_idx


def _fill_body_flat(
    flat_source: Tensor,               # [total_valid, W]
    check_arg_source_flat: Tensor,     # [total_valid, M, 2]
    body_preds_flat: Tensor,           # [total_valid, M]
) -> Tensor:
    """Fill body atoms from flat source tensor (no padding).

    Same logic as _fill_body_extended but operates on flat [total_valid, ...]
    instead of [B, K_r, G_r, ...].

    Returns: [total_valid, M, 3]
    """
    M = body_preds_flat.size(1)
    source_m = flat_source.unsqueeze(1).expand(-1, M, -1)  # [T, M, W]
    return _fill_body_atoms(source_m, check_arg_source_flat, body_preds_flat)


def _apply_filters_flat(
    flat_body: Tensor,        # [total_valid, M, 3]
    flat_exists: Tensor,      # [total_valid, M] bool
    flat_b_idx: Tensor,       # [total_valid] batch index
    flat_num_body: Tensor,    # [total_valid] num active body atoms per entry
    queries: Tensor,          # [B, 3] original queries
    width: Optional[int],
    head_pred_mask: Optional[Tensor],   # [P] bool, None to skip head-pred prune
    M: int,
    body_active: Optional[Tensor] = None,  # [total_valid, M] precomputed
) -> Tensor:
    """Apply width filtering, query exclusion, head pred pruning on flat body.

    Returns: [total_valid] bool mask of surviving entries.

    ``body_active`` (``atom_idx < flat_num_body``) is recomputed here only
    when the caller did not already build it. The flat resolve path builds
    the identical mask to pad inactive body atoms and passes it through,
    eliminating a redundant arange + broadcast-compare per step.
    """
    T = flat_body.size(0)
    dev = flat_body.device

    if body_active is None:
        atom_idx = _arange_cached(M, dev).unsqueeze(0)  # [1, M]
        body_active = atom_idx < flat_num_body.unsqueeze(1)   # [T, M]

    # Width filtering
    if width is None:
        mask = torch.ones(T, dtype=torch.bool, device=dev)
    else:
        num_unknown = (body_active & ~flat_exists).sum(dim=-1)  # [T]
        mask = num_unknown <= width

    # Query exclusion: no body atom equals the query
    query_exp = queries[flat_b_idx]  # [T, 3]
    is_query = (flat_body == query_exp.unsqueeze(1)).all(dim=-1)  # [T, M]
    has_query_atom = (is_query & body_active).any(dim=-1)  # [T]
    mask = mask & ~has_query_atom

    # Head predicate pruning (when width is bounded AND a mask is given)
    if width is not None:
        if head_pred_mask is not None:
            all_ok = _head_pred_all_ok(
                flat_body[..., 0], flat_exists, body_active, head_pred_mask)
            mask = mask & all_ok

        if width == 0:
            all_exist = _all_body_active_ok(flat_exists, body_active)
            mask = mask & all_exist

    return mask


# ═══════════════════════════════════════════════════════════════════════
# Public aliases (re-exported by resolution/__init__.py)
# ═══════════════════════════════════════════════════════════════════════

enumerate_candidates = _enumerate_dir
fill_body_templates = _fill_body
