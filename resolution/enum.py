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
from grounder.types import ResolvedChildren


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
) -> Dict:
    """Build all enum metadata from a RuleIndex.

    Returns dict with 'compiled' (CompiledRules), 'buffers' (tensors to
    register), and scalar config values.
    """
    # P and E
    P = getattr(fact_index, "_num_predicates", None)
    if P is None:
        P = (int(facts_idx[:, 0].max().item()) + 1
             if facts_idx.numel() > 0 else 1)
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
    R_eff = enum_ri.R_eff
    G_use = E if cartesian_product else max_groundings_per_query
    K_enum = min(R_eff * G_use, max_total_groundings)
    S = max_states if max_states is not None else max_total_groundings
    effective_total_G = min(max_total_groundings, R_eff * G_use)

    print(f"  BCGrounder(enum): {num_rules} rules, R_eff={R_eff}, "
          f"width={width}, S={S}, K_enum={K_enum}, tG={effective_total_G}"
          + (f", cartesian_product=True, E={E}" if cartesian_product else "")
          + (f", all_anchors=True" if all_anchors else ""))

    return {
        "buffers": buffers,
        "enum_rule_index": enum_ri,
        "P": P, "E": E,
        "R_eff": R_eff,
        "K_enum": K_enum,
        "S": S,
        "effective_total_G": effective_total_G,
        "any_dual": any_dual,
        "enum_G": G_use,
        "cartesian_product": cartesian_product,
    }


def resolve_enum_step(
    queries: Tensor,           # [B, S, 3]
    remaining: Tensor,         # [B, S, G, 3]
    grounding_body: Tensor,    # [B, S, M, 3]
    state_valid: Tensor,       # [B, S]
    active_mask: Tensor,       # [B, S]
    *,
    fact_index,
    d: int,
    depth: int,
    width: Optional[int],
    M: int,
    padding_idx: int,
    enum_G: int,
    K_enum: int,
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
    head_pred_mask: Tensor,
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
) -> ResolvedChildren:
    """Adapter: resolve_enum output -> common 9-tensor format used by _pack.

    1. Determine width for this step.
    2. Flatten [B, S, 3] -> [N, 3].
    3. Call resolve_enum with all metadata buffers.
    4. Handle direction B (dual anchoring): pad and concat.
    5. Flatten Re * G_use -> K_total, cap to K_enum via topk.
    6. Reshape back to [B, S, K_enum, ...].
    7. Build rule_goals: body atoms at 0..M-1, remaining parent goals at M..G-1.
    8. Build rule_gbody: parent's grounding_body expanded to K_enum children.
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
        enum_G: Max groundings per query.
        K_enum: Children budget per state.
        any_dual: Whether any rule supports dual anchoring.
        pred_rule_indices..head_pred_mask: Direction A metadata buffers.
        has_dual..check_arg_source_b: Optional direction B metadata buffers.

    Returns:
        9-tuple of tensors:
            fact_goals:    [B, S, 0, G, 3]
            fact_gbody:    [B, S, 0, M, 3]
            fact_success:  [B, S, 0]
            rule_goals:    [B, S, K_enum, G, 3]
            rule_gbody:    [B, S, K_enum, M, 3]
            rule_success:  [B, S, K_enum]
            sub_rule_idx:  [B, S, K_enum]
            fact_subs:     [B, S, 0, 2, 2]
            rule_subs:     [B, S, K_enum, 2, 2]
    """
    B, S, _ = queries.shape
    G = remaining.shape[2]
    pad = padding_idx
    dev = queries.device

    # 1. Width: last step uses w_last_depth (default 0 = all body atoms
    #    must be facts) so proof_goals become empty and collection works.
    width_d = width
    if d == depth - 1 and width is not None:
        width_d = w_last_depth

    # 2. Flatten [B, S, 3] -> [N, 3]
    N = B * S
    flat_q = queries.reshape(N, 3)
    flat_valid = (active_mask & state_valid).reshape(N)

    # 3. Call resolve_enum
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
        head_pred_mask=head_pred_mask,
        G=enum_G, M=M, width=width_d,
        has_dual=has_dual,
        enum_pred_b=enum_pred_b,
        enum_bound_binding_b=enum_bound_binding_b,
        enum_direction_b=enum_direction_b,
        check_arg_source_b=check_arg_source_b,
        any_dual=any_dual,
        cartesian_product=cartesian_product,
        E=E,
    )

    # 4. Unpack tuple and handle direction B
    (body_a, _exists_a, _body_active, gmask_a, _cmask_a,
     _active_mask_out, aidx,
     body_b, _exists_b, _body_active_b, gmask_b, _cmask_b) = result

    Re = aidx.size(1)
    G_use_a = body_a.size(2)
    G_use_b = body_b.size(2)

    if G_use_b > 0:
        # Direct assembly into combined tensor (avoids pad + pad + cat)
        G_total = G_use_a + G_use_b
        body_all = torch.full(
            (N, Re, G_total, M, 3), pad, dtype=torch.long, device=dev)
        body_all[:, :, :G_use_a] = body_a
        body_all[:, :, G_use_a:G_use_a + G_use_b] = body_b[:, :, :G_use_b]
        gmask_all = torch.zeros(
            N, Re, G_total, dtype=torch.bool, device=dev)
        gmask_all[:, :, :G_use_a] = gmask_a
        gmask_all[:, :, G_use_a:G_use_a + G_use_b] = gmask_b[:, :, :G_use_b]
    else:
        body_all = body_a
        gmask_all = gmask_a

    # 5. Flatten Re * G_use -> K_total, cap to K_enum via topk
    G_use_total = body_all.size(2)
    K_total = Re * G_use_total

    body_flat = body_all.reshape(N, K_total, M, 3)
    success_flat = gmask_all.reshape(N, K_total)
    ridx_flat = aidx.unsqueeze(2).expand(
        -1, -1, G_use_total).reshape(N, K_total)

    if K_total > K_enum:
        _, top_idx = success_flat.to(torch.int8).topk(
            K_enum, dim=1, largest=True, sorted=False)
        idx_body = top_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, M, 3)
        body_flat = body_flat.gather(1, idx_body)
        success_flat = success_flat.gather(1, top_idx)
        ridx_flat = ridx_flat.gather(1, top_idx)
    else:
        K_enum = K_total

    # 6. Reshape back to [B, S, K_enum, ...]
    body_flat = body_flat.reshape(B, S, K_enum, M, 3)
    success_flat = success_flat.reshape(B, S, K_enum)
    ridx_flat = ridx_flat.reshape(B, S, K_enum)

    # 7. Build rule_goals: body atoms at positions 0..M-1,
    #    remaining parent goals at M..G-1.
    Bmax = M
    rule_goals = torch.full(
        (B, S, K_enum, G, 3), pad, dtype=torch.long, device=dev)
    rule_goals[:, :, :, :Bmax, :] = body_flat
    # Copy remaining goals from parent (positions 1.. since 0 was selected)
    n_rem = min(G - Bmax, G - 1)
    if n_rem > 0:
        rem = remaining[:, :, 1:1 + n_rem, :]   # [B, S, n_rem, 3]
        rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = \
            rem.unsqueeze(2).expand(-1, -1, K_enum, -1, -1)

    # 8. Build rule_gbody: parent's grounding_body expanded to K_enum children
    G_body = grounding_body.shape[2]  # G_body >= M (accumulated body dim)
    if collect_evidence:
        rule_gbody = grounding_body.unsqueeze(2).expand(
            -1, -1, K_enum, -1, -1)   # [B, S, K_enum, G_body, 3]
    else:
        rule_gbody = torch.zeros(
            B, S, K_enum, G_body, 3, dtype=torch.long, device=dev)

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
        (B, S, K_enum, 2, 2), pad, dtype=torch.long, device=dev)

    return ResolvedChildren(fact_goals, fact_gbody, fact_success,
                            rule_goals, rule_gbody, success_flat,
                            ridx_flat, fact_subs, rule_subs)


def resolve_enum(
    queries: Tensor,              # [B, 3]
    query_mask: Tensor,           # [B]
    fact_index,                   # FactIndex with .enumerate() and .exists()
    *,
    pred_rule_indices: Tensor,    # [P, R_eff]
    pred_rule_mask: Tensor,       # [P, R_eff]
    has_free: Tensor,             # [R]
    body_preds: Tensor,           # [R, M]
    num_body_atoms: Tensor,       # [R]
    enum_pred_a: Tensor,          # [R]
    enum_bound_binding_a: Tensor, # [R]
    enum_direction_a: Tensor,     # [R]
    check_arg_source_a: Tensor,   # [R, M, 2]
    head_pred_mask: Tensor,       # [P] bool
    G: int,                       # max groundings per rule per query
    M: int,                       # max body atoms
    width: Optional[int],         # width bound (None=∞)
    has_dual: Optional[Tensor] = None,
    enum_pred_b: Optional[Tensor] = None,
    enum_bound_binding_b: Optional[Tensor] = None,
    enum_direction_b: Optional[Tensor] = None,
    check_arg_source_b: Optional[Tensor] = None,
    any_dual: bool = False,
    cartesian_product: bool = False,
    E: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Parametrized compiled enumeration with optional dual anchoring.

    Returns fixed-length 12-tuple (compile-safe, no dicts):
        body_atoms_a    [B, Re, G_a, M, 3]
        exists_a        [B, Re, G_a, M]
        body_active     [B, Re, 1, M]
        grounding_mask_a [B, Re, G_a]
        cand_mask_a     [B, Re, G_a]
        active_mask     [B, Re]
        active_idx      [B, Re]
        body_atoms_b    [B, Re, G_b, M, 3]   (zero-sized G_b=0 when no dual)
        exists_b        [B, Re, G_b, M]
        body_active_b   [B, Re, 1, M]        (zeros when no dual)
        grounding_mask_b [B, Re, G_b]
        cand_mask_b     [B, Re, G_b]
    """
    B = queries.size(0)
    dev = queries.device
    qp, qs, qo = queries[:, 0], queries[:, 1], queries[:, 2]

    # ── Rule clustering ──
    active_idx = pred_rule_indices[qp]             # [B, Re]
    Re = active_idx.size(1)
    active_mask = pred_rule_mask[qp] & query_mask.unsqueeze(1)
    has_free_q = has_free[active_idx]               # [B, Re]
    body_preds_q = body_preds[active_idx]           # [B, Re, M]
    num_body_q = num_body_atoms[active_idx]         # [B, Re]

    # ── Direction A: enumerate + fill + filter ──
    if cartesian_product:
        G_use = E
    else:
        G_use = min(G, fact_index._max_facts_per_query)
    cands_a, cmask_a = _enumerate_dir(
        B, Re, G_use, qs, qo,
        enum_pred_a[active_idx],
        enum_bound_binding_a[active_idx],
        enum_direction_a[active_idx],
        fact_index,
        cartesian_product=cartesian_product,
        E=E,
    )
    G_use = cands_a.size(2)

    # Free rules get enumerated candidates; bound rules use slot 0
    cmask_a = cmask_a & has_free_q.unsqueeze(2)
    cmask_a[:, :, 0] = cmask_a[:, :, 0] | (~has_free_q & active_mask)

    body_a = _fill_body(
        B, Re, G_use, M, qs, qo, cands_a,
        check_arg_source_a[active_idx], body_preds_q,
    )
    # Resolve body atoms with unresolved free variables (binding >= 3)
    # before the exists check. Without this, rules with 2+ free variables
    # (e.g. neighborOf(X,Y), neighborOf(Y,K), locatedInCR(K,Z)) get
    # garbage values for K and fail the exists check at width=0.
    body_a = _resolve_free_vars(
        body_a, check_arg_source_a[active_idx], fact_index, active_idx)
    exists_a = fact_index.exists(
        body_a.reshape(-1, 3)).view(B, Re, G_use, M)

    atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
    body_active = atom_idx < num_body_q.view(B, Re, 1, 1)

    # Ensure inactive body atoms have padding predicates so downstream
    # prune_ground_facts recognizes them as padding (not spurious atoms).
    body_a = body_a.masked_fill(~body_active.unsqueeze(-1), fact_index._padding_idx)

    mask_a = _apply_enum_filters(
        body_a, exists_a, body_active, active_mask, cmask_a,
        queries, G_use, width, head_pred_mask,
    )

    # ── Direction B (dual anchoring) ──
    if any_dual and not cartesian_product:
        has_dual_q = has_dual[active_idx]
        G_b = G // 2
        G_use_b = min(G_b, fact_index._max_facts_per_query)

        cands_b, cmask_b = _enumerate_dir(
            B, Re, G_use_b, qs, qo,
            enum_pred_b[active_idx],
            enum_bound_binding_b[active_idx],
            enum_direction_b[active_idx],
            fact_index,
        )
        G_use_b = cands_b.size(2)
        cmask_b = cmask_b & has_dual_q.unsqueeze(2)

        body_b = _fill_body(
            B, Re, G_use_b, M, qs, qo, cands_b,
            check_arg_source_b[active_idx], body_preds_q,
        )
        body_b = _resolve_free_vars(
            body_b, check_arg_source_b[active_idx], fact_index, active_idx)
        exists_b = fact_index.exists(
            body_b.reshape(-1, 3)).view(B, Re, G_use_b, M)
        body_active_b = atom_idx < num_body_q.view(B, Re, 1, 1)

        mask_b = _apply_enum_filters(
            body_b, exists_b, body_active_b, active_mask, cmask_b,
            queries, G_use_b, width, head_pred_mask,
        )

        # Exclude fully proven from B (avoid A duplicates)
        all_proven_b = (
            exists_b | ~body_active_b.expand(-1, -1, G_use_b, -1)
        ).all(dim=-1)
        mask_b = mask_b & ~all_proven_b
    else:
        # Empty B direction (zero-sized G_b=0)
        body_b = torch.zeros(B, Re, 0, M, 3, dtype=torch.long, device=dev)
        exists_b = torch.zeros(B, Re, 0, M, dtype=torch.bool, device=dev)
        body_active_b = torch.zeros(B, Re, 1, M, dtype=torch.bool, device=dev)
        mask_b = torch.zeros(B, Re, 0, dtype=torch.bool, device=dev)
        cmask_b = torch.zeros(B, Re, 0, dtype=torch.bool, device=dev)

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
    Re = active_idx.size(1)
    active_mask = pred_rule_mask[qp] & query_mask.unsqueeze(1)
    has_free_q = has_free[active_idx]
    body_preds_q = body_preds[active_idx]
    num_body_q = num_body_atoms[active_idx]

    # All entities as candidates
    candidates = all_entities.view(1, 1, E).expand(B, Re, -1)
    free_mask = has_free_q.unsqueeze(2) & active_mask.unsqueeze(2)
    bound_mask = (~has_free_q).unsqueeze(2) & active_mask.unsqueeze(2)
    cand_mask = free_mask.expand(-1, -1, E) | (
        bound_mask & slot0_mask.view(1, 1, E))

    body_atoms = _fill_body(
        B, Re, E, M, qs, qo, candidates,
        check_arg_source_a[active_idx], body_preds_q,
    )
    body_atoms = _resolve_free_vars(
        body_atoms, check_arg_source_a[active_idx], fact_index, active_idx)

    # Query exclusion only (no width filtering for full enum)
    atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
    body_active = atom_idx < num_body_q.view(B, Re, 1, 1)
    query_exp = queries.view(B, 1, 1, 1, 3).expand(-1, Re, E, M, -1)
    is_query = (body_atoms == query_exp).all(dim=-1)
    has_query_atom = (
        is_query & body_active.expand(-1, -1, E, -1)).any(dim=-1)
    grounding_mask = cand_mask & ~has_query_atom

    return {
        'body_atoms_a': body_atoms,       # [B, Re, E, M, 3]
        'grounding_mask_a': grounding_mask,  # [B, Re, E]
        'active_mask': active_mask,        # [B, Re]
        'active_idx': active_idx,          # [B, Re]
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _enumerate_dir(
    B: int, Re: int, G_use: int,
    query_subjs: Tensor,     # [B]
    query_objs: Tensor,      # [B]
    enum_pred_q: Tensor,     # [B, Re]
    enum_bound_q: Tensor,    # [B, Re]
    enum_dir_q: Tensor,      # [B, Re]
    fact_index,
    cartesian_product: bool = False,
    E: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Enumerate candidates for one direction across clustered rules.

    When ``cartesian_product=True``, returns all entity indices ``[0..E-1]``
    as candidates for every (query, rule) pair instead of fact-anchored lookup.

    Returns:
        candidates: [B, Re, G_actual]
        cand_mask:  [B, Re, G_actual]
    """
    if cartesian_product:
        dev = query_subjs.device
        # All entities as candidates for every (query, rule) pair
        candidates = torch.arange(E, device=dev).unsqueeze(0).unsqueeze(0).expand(
            B * Re, 1, -1).reshape(B, Re, E)
        cand_mask = torch.ones(B, Re, E, dtype=torch.bool, device=dev)
        return candidates, cand_mask

    source = torch.stack([query_subjs, query_objs], dim=1)  # [B, 2]
    enum_bound_vals = source.gather(1, enum_bound_q)         # [B, Re]

    candidates, cand_mask = fact_index.enumerate(
        enum_pred_q.reshape(-1),
        enum_bound_vals.reshape(-1),
        enum_dir_q.reshape(-1),
    )
    G_fi = candidates.size(1)
    G_actual = min(G_use, G_fi)
    candidates = candidates[:, :G_actual].reshape(B, Re, G_actual)
    cand_mask = cand_mask[:, :G_actual].reshape(B, Re, G_actual)
    return candidates, cand_mask


def _fill_body(
    B: int, Re: int, G_use: int, M: int,
    query_subjs: Tensor,          # [B]
    query_objs: Tensor,           # [B]
    candidates: Tensor,           # [B, Re, G_use]
    check_arg_source_q: Tensor,   # [B, Re, M, 2]
    body_preds_q: Tensor,         # [B, Re, M]
) -> Tensor:
    """Fill body atoms from (subj, obj, candidate) binding table.

    check_arg_source_q[..., 0/1] maps each body atom argument to its source:
      0 = query subject, 1 = query object, 2 = enumerated candidate.

    Returns: [B, Re, G_use, M, 3] body atoms.
    """
    q_s = query_subjs.view(B, 1, 1).expand(-1, Re, G_use)
    q_o = query_objs.view(B, 1, 1).expand(-1, Re, G_use)
    source = torch.stack([q_s, q_o, candidates], dim=3)       # [B, Re, G_use, 3]
    source_exp = source.unsqueeze(3).expand(-1, -1, -1, M, -1)  # [B, Re, G_use, M, 3]

    # Clamp source indices to [0, 2]: values >= 3 are unresolved free variables
    # which produce invalid body atoms (safely filtered out by exists-check).
    idx_0 = check_arg_source_q[:, :, :, 0].clamp(max=2).view(
        B, Re, 1, M).expand(-1, -1, G_use, -1)
    arg0 = source_exp.gather(4, idx_0.unsqueeze(-1)).squeeze(-1)

    idx_1 = check_arg_source_q[:, :, :, 1].clamp(max=2).view(
        B, Re, 1, M).expand(-1, -1, G_use, -1)
    arg1 = source_exp.gather(4, idx_1.unsqueeze(-1)).squeeze(-1)

    preds_exp = body_preds_q.unsqueeze(2).expand(-1, -1, G_use, -1)
    return torch.stack([preds_exp, arg0, arg1], dim=-1)


def _resolve_free_vars(
    body_atoms: Tensor,            # [B, Re, G_use, M, 3]
    check_arg_source_q: Tensor,    # [B, Re, M, 2]
    fact_index,
    active_idx: Tensor,            # [B, Re]
) -> Tensor:
    """Resolve body atoms with unresolved free variables (binding >= 3).

    _fill_body clamps free-variable indices to source[2], producing garbage
    body atoms.  For each body atom where one argument is an unresolved free
    variable, enumerate ALL valid values from the fact_index and replicate
    the grounding for each, matching keras-ns behavior.

    The G_use (grounding) dimension is expanded by up to K_free (number of
    valid free-var values) for each body position with a free variable.
    The expansion reuses existing G_use slots by interleaving: the original
    G_use groundings are replicated K_free times along the G dimension.

    Fully static-shape: no data-dependent branching, compatible with
    torch.compile(fullgraph=True).

    Args:
        body_atoms: [B, Re, G_use, M, 3] body atoms (some with garbage args)
        check_arg_source_q: [B, Re, M, 2] binding sources per body atom arg
        fact_index: FactIndex with enumerate() method
        active_idx: [B, Re] rule indices (for gathering check_arg_source)

    Returns:
        body_atoms: [B, Re, G_use, M, 3] with free variables resolved.
                    Shape unchanged; multiple free-var values fill different
                    G_use slots via the candidate dimension.
    """
    B, Re, G_use, M, _ = body_atoms.shape
    src0 = check_arg_source_q[:, :, :, 0]  # [B, Re, M]
    src1 = check_arg_source_q[:, :, :, 1]  # [B, Re, M]

    # Identify body positions where arg0 or arg1 is a free variable (>= 3)
    has_free_arg0 = (src0 >= 3)  # [B, Re, M]
    has_free_arg1 = (src1 >= 3)  # [B, Re, M]

    # No early exit — always run to keep fullgraph static control flow.
    body_atoms = body_atoms.clone()

    for m_idx in range(M):
        free0_m = has_free_arg0[:, :, m_idx]  # [B, Re]
        free1_m = has_free_arg1[:, :, m_idx]  # [B, Re]

        preds_m = body_atoms[:, :, :, m_idx, 0]  # [B, Re, G_use]
        arg0_m = body_atoms[:, :, :, m_idx, 1]   # [B, Re, G_use]
        arg1_m = body_atoms[:, :, :, m_idx, 2]   # [B, Re, G_use]

        # Resolve free arg0 (bound arg1 → enumerate arg0, direction=1)
        flat_p = preds_m.reshape(-1)
        flat_b0 = arg1_m.reshape(-1)
        flat_dir1 = torch.ones_like(flat_p)
        cands0, cmask0 = fact_index.enumerate(flat_p, flat_b0, flat_dir1)
        # K_free = cands0.size(1); use ALL valid values, not just first
        K_free = cands0.size(1)
        # For each grounding slot, pick the K_free-th valid value cyclically
        # across the G_use dimension. Slot g gets candidate g % K_free.
        g_indices = torch.arange(G_use, device=body_atoms.device)
        k_idx = (g_indices % K_free).view(1, 1, G_use)
        k_idx_exp = k_idx.expand(B * Re, -1, -1).reshape(-1, G_use)
        # Gather the k_idx-th candidate for each flat position
        # cands0: [B*Re*G_use, K_free] → need per-slot selection
        cands0_per_slot = cands0.reshape(B * Re, G_use, K_free)
        cmask0_per_slot = cmask0.reshape(B * Re, G_use, K_free)
        selected0 = cands0_per_slot.gather(2, k_idx_exp.unsqueeze(-1)).squeeze(-1)
        selected0_valid = cmask0_per_slot.gather(2, k_idx_exp.unsqueeze(-1)).squeeze(-1)
        selected0 = selected0.reshape(B, Re, G_use)
        selected0_valid = selected0_valid.reshape(B, Re, G_use)
        update0 = free0_m.unsqueeze(2).expand(-1, -1, G_use) & selected0_valid
        body_atoms[:, :, :, m_idx, 1] = torch.where(
            update0, selected0, body_atoms[:, :, :, m_idx, 1])

        # Resolve free arg1 (bound arg0 → enumerate arg1, direction=0)
        arg0_m = body_atoms[:, :, :, m_idx, 1]
        flat_b1 = arg0_m.reshape(-1)
        flat_dir0 = torch.zeros_like(flat_p)
        cands1, cmask1 = fact_index.enumerate(flat_p, flat_b1, flat_dir0)
        K_free1 = cands1.size(1)
        k_idx1 = (g_indices % K_free1).view(1, 1, G_use)
        k_idx1_exp = k_idx1.expand(B * Re, -1, -1).reshape(-1, G_use)
        cands1_per_slot = cands1.reshape(B * Re, G_use, K_free1)
        cmask1_per_slot = cmask1.reshape(B * Re, G_use, K_free1)
        selected1 = cands1_per_slot.gather(2, k_idx1_exp.unsqueeze(-1)).squeeze(-1)
        selected1_valid = cmask1_per_slot.gather(2, k_idx1_exp.unsqueeze(-1)).squeeze(-1)
        selected1 = selected1.reshape(B, Re, G_use)
        selected1_valid = selected1_valid.reshape(B, Re, G_use)
        update1 = free1_m.unsqueeze(2).expand(-1, -1, G_use) & selected1_valid
        body_atoms[:, :, :, m_idx, 2] = torch.where(
            update1, selected1, body_atoms[:, :, :, m_idx, 2])

    return body_atoms


def _apply_enum_filters(
    body_atoms: Tensor,       # [B, Re, G_use, M, 3]
    exists: Tensor,           # [B, Re, G_use, M]
    body_active: Tensor,      # [B, Re, 1, M]
    active_mask: Tensor,      # [B, Re]
    cand_mask: Tensor,        # [B, Re, G_use]
    queries: Tensor,          # [B, 3]
    G_use: int,
    width: Optional[int],
    head_pred_mask: Tensor,   # [P] bool
) -> Tensor:
    """Apply width filtering, query exclusion, and head predicate pruning.

    Returns: [B, Re, G_use] grounding mask.
    """
    B, Re = active_mask.shape
    M = body_atoms.shape[3]
    body_active_exp = body_active.expand(-1, -1, G_use, -1)

    # Width filtering
    if width is None:
        mask = active_mask.unsqueeze(2) & cand_mask
    else:
        num_unknown = (body_active_exp & ~exists).sum(dim=-1)
        mask = (num_unknown <= width) & active_mask.unsqueeze(2) & cand_mask

    # Query exclusion: no body atom equals the query
    query_exp = queries.view(B, 1, 1, 1, 3).expand(-1, Re, G_use, M, -1)
    is_query = (body_atoms == query_exp).all(dim=-1)
    has_query_atom = (is_query & body_active_exp).any(dim=-1)
    mask = mask & ~has_query_atom

    # Head predicate pruning (only when width is bounded)
    if width is not None:
        body_pred_vals = body_atoms[..., 0]
        head_pred_ok = head_pred_mask[body_pred_vals]
        unknown_ok = exists | head_pred_ok
        all_ok = (unknown_ok | ~body_active_exp).all(dim=-1)
        mask = mask & all_ok

        if width == 0:
            all_exist = (exists | ~body_active_exp).all(dim=-1)
            mask = mask & all_exist

    return mask


# ═══════════════════════════════════════════════════════════════════════
# Public aliases (re-exported by resolution/__init__.py)
# ═══════════════════════════════════════════════════════════════════════

enumerate_candidates = _enumerate_dir
fill_body_templates = _fill_body
