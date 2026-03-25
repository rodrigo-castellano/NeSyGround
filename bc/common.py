"""BC shared utilities: packing, compaction, grounding collection, deduplication.

Provides scatter-based compaction of derived states into fixed-shape output
tensors and grounding collection.

All functions are pure (no class dependencies) and torch.compile compatible.
"""

from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

import torch
from torch import Tensor

from grounder.types import PackedStates
# Re-export prune_ground_facts from its canonical location for compatibility
from grounder.filters.search.prune_facts import prune_ground_facts


# ---------------------------------------------------------------------------
# Packing and compaction (from packing.py)
# ---------------------------------------------------------------------------


def compact_atoms(
    states: Tensor,             # [..., M, 3]
    padding_idx: int,
) -> Tensor:
    """Left-align atoms by removing gaps after pruning.

    Works for any shape [..., M, 3] (BE uses [B, K, M, 3], TS uses [B, S, G, 3]).
    Non-padding atoms are moved to the front within each (*, M) slice.

    Args:
        states: [..., M, 3] tensor with potential gaps (padding) in the M dimension
        padding_idx: padding value

    Returns:
        [..., M, 3] with atoms left-aligned
    """
    if states.numel() == 0:
        return states

    *leading, M, _ = states.shape
    # Flatten leading dims for uniform processing
    flat = states.reshape(-1, M, 3)
    N = flat.shape[0]
    device = states.device
    pad = padding_idx

    valid_atom = (flat[:, :, 0] != pad)  # [N, M]
    pos = torch.cumsum(valid_atom.long(), dim=1) - 1
    M_t = torch.tensor(M, dtype=pos.dtype, device=device)
    sort_key = torch.where(valid_atom, pos, M_t)

    sorted_indices = torch.argsort(sort_key, dim=1, stable=True)
    sorted_indices_exp = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
    result = torch.gather(flat, 1, sorted_indices_exp)

    return result.reshape(*leading, M, 3)



def pack_states(
    fact_goals: Tensor,        # [B, S, K_f, G, 3]
    fact_gbody: Tensor,        # [B, S, K_f, M_work, 3]
    fact_success: Tensor,      # [B, S, K_f]
    rule_goals: Tensor,        # [B, S, K_r, G, 3]
    rule_gbody: Tensor,        # [B, S, K_r, M_work, 3]
    rule_success: Tensor,      # [B, S, K_r]
    sub_rule_idx: Tensor,      # [B, S, K_r]
    fact_subs: Tensor,         # [B, S, K_f, 2, 2]
    rule_subs: Tensor,         # [B, S, K_r, 2, 2]
    top_ridx: Tensor,          # [B, S]
    grounding_body: Tensor,    # [B, S, M_work, 3]
    body_count: Tensor,        # [B, S, D] (structured) or [B, S] (legacy)
    S_out: int,
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
) -> PackedStates:
    """Pack resolution children into compacted proof states.

    M-sized working buffer version: grounding_body is [B, S, M_work, 3] where
    M_work = kb.M (max body atoms in a single rule). The G_body-sized
    accumulated body is handled in a separate sync step in bc.py.

    For rule children: capture the current rule's body atoms from rule_goals.
    For fact children: set to padding (no new body atoms this depth).

    Also computes parent_map, winning_subs, has_new_body for the sync step.

    Args:
        fact_goals..rule_subs: 9-tensor resolved output from resolution.
        top_ridx:       [B, S] parent rule indices.
        grounding_body: [B, S, M_work, 3] parent working body (M-sized).
        body_count:     [B, S] number of valid body atoms per state.
        S_out:          output state budget.
        padding_idx:    padding value.

    Returns:
        grounding_body: [B, S_out, M_work, 3] — M-sized working buffer
        proof_goals:    [B, S_out, G, 3]
        top_ridx:       [B, S_out]
        state_valid:    [B, S_out]
        body_count:     [B, S_out] — inherited from parent (not yet accumulated)
        parent_map:     [B, S_out] — parent state index for each output state
        winning_subs:   [B, S_out, 2, 2] — subs for each output state
        has_new_body:   [B, S_out] — True for rule children with valid matches
    """
    B, S_in = top_ridx.shape
    K_f = fact_goals.shape[2]
    K_r = rule_goals.shape[2]
    M_work = grounding_body.shape[2]
    pad = padding_idx
    dev = top_ridx.device

    n_f = S_in * K_f
    n_r = S_in * K_r

    G = rule_goals.shape[3]

    # ── Fact children: flatten, inherit parent ridx and body_count ──
    bc_is_3d = body_count.dim() == 3   # [B, S, D] vs [B, S]
    if K_f > 0:
        f_goals = fact_goals.reshape(B, n_f, G, 3)
        f_valid = fact_success.reshape(B, n_f)
        f_ridx = top_ridx.unsqueeze(2).expand(
            B, S_in, K_f).reshape(B, n_f)
        if bc_is_3d:
            D_bc = body_count.shape[2]
            f_bcount = body_count.unsqueeze(2).expand(
                B, S_in, K_f, D_bc).reshape(B, n_f, D_bc)
        else:
            f_bcount = body_count.unsqueeze(2).expand(
                B, S_in, K_f).reshape(B, n_f)
        f_subs = fact_subs.reshape(B, n_f, 2, 2)
        # Parent indices for fact children: child j has parent j // K_f
        f_parents = torch.arange(S_in, device=dev).unsqueeze(1).expand(
            S_in, K_f).reshape(n_f)
        f_parents = f_parents.unsqueeze(0).expand(B, n_f)
        if collect_evidence:
            # Skip facts when grounding_body not yet established
            if bc_is_3d:
                uninit = (body_count.sum(dim=-1) == 0)  # [B, S]
            else:
                uninit = (body_count == 0)
            f_valid = f_valid & ~uninit.unsqueeze(-1).expand(
                B, S_in, K_f).reshape(B, n_f)
        # Fact children: no new body atoms (padding)
        f_gbody = torch.full(
            (B, n_f, M_work, 3), pad, dtype=torch.long, device=dev)
        f_has_new = torch.zeros(B, n_f, dtype=torch.bool, device=dev)
    else:
        f_gbody = torch.full(
            (B, 0, M_work, 3), pad, dtype=torch.long, device=dev)
        f_goals = torch.full((B, 0, G, 3), pad, dtype=torch.long, device=dev)
        f_valid = torch.zeros(B, 0, dtype=torch.bool, device=dev)
        f_ridx = torch.zeros(B, 0, dtype=torch.long, device=dev)
        if bc_is_3d:
            D_bc = body_count.shape[2]
            f_bcount = torch.zeros(B, 0, D_bc, dtype=torch.long, device=dev)
        else:
            f_bcount = torch.zeros(B, 0, dtype=torch.long, device=dev)
        f_subs = torch.full((B, 0, 2, 2), pad, dtype=torch.long, device=dev)
        f_parents = torch.zeros(B, 0, dtype=torch.long, device=dev)
        f_has_new = torch.zeros(B, 0, dtype=torch.bool, device=dev)

    # ── Rule children: flatten, capture body atoms, propagate ridx ──
    first = (top_ridx == -1).unsqueeze(2).expand(
        B, S_in, K_r).reshape(B, n_r)              # [B, n_r] first resolution?

    if collect_evidence:
        # Extract new body atoms from rule_goals (first M_rule slots are body).
        if M_rule <= 0:
            M_rule = M_work
        # Capture body atoms from rule_goals into M-sized working buffer
        new_body_atoms = rule_goals[:, :, :, :M_rule, :].reshape(
            B, n_r, M_rule, 3)                                  # [B, n_r, M_rule, 3]
        # Pad/truncate to M_work if M_rule != M_work
        if M_rule < M_work:
            r_gbody = torch.full(
                (B, n_r, M_work, 3), pad, dtype=torch.long, device=dev)
            r_gbody[:, :, :M_rule, :] = new_body_atoms
        elif M_rule > M_work:
            r_gbody = new_body_atoms[:, :, :M_work, :]
        else:
            r_gbody = new_body_atoms
        r_has_new = rule_success.reshape(B, n_r)  # has new body if rule succeeded
    else:
        r_gbody = torch.full(
            (B, n_r, M_work, 3), pad, dtype=torch.long, device=dev)
        r_has_new = torch.zeros(B, n_r, dtype=torch.bool, device=dev)

    if bc_is_3d:
        r_bcount = body_count.unsqueeze(2).expand(
            B, S_in, K_r, D_bc).reshape(B, n_r, D_bc)
    else:
        r_bcount = body_count.unsqueeze(2).expand(
            B, S_in, K_r).reshape(B, n_r)  # inherited from parent

    r_ridx = torch.where(
        first,
        sub_rule_idx.reshape(B, n_r),              # new rule index
        top_ridx.unsqueeze(2).expand(
            B, S_in, K_r).reshape(B, n_r),          # parent's
    )
    r_goals = rule_goals.reshape(B, n_r, G, 3)
    r_valid = rule_success.reshape(B, n_r)
    r_subs = rule_subs.reshape(B, n_r, 2, 2)
    # Parent indices for rule children: child j has parent j // K_r
    r_parents = torch.arange(S_in, device=dev).unsqueeze(1).expand(
        S_in, K_r).reshape(n_r)
    r_parents = r_parents.unsqueeze(0).expand(B, n_r)

    # ── Current-depth rule index (for per-depth evidence) ──
    # Fact children: -1 (no rule at this depth); Rule children: sub_rule_idx
    f_current_ridx = torch.full(
        (B, n_f), -1, dtype=torch.long, device=dev) if K_f > 0 else (
        torch.zeros(B, 0, dtype=torch.long, device=dev))
    r_current_ridx = sub_rule_idx.reshape(B, n_r)

    # ── Concatenate all children (skip cat when K_f=0) ──
    if K_f == 0:
        all_gbody = r_gbody
        all_goals = r_goals
        all_valid = r_valid
        all_ridx = r_ridx
        all_bcount = r_bcount
        all_subs = r_subs
        all_parents = r_parents
        all_has_new = r_has_new
        all_current_ridx = r_current_ridx
    else:
        all_gbody = torch.cat([f_gbody, r_gbody], dim=1)     # [B, N, M_work, 3]
        all_goals = torch.cat([f_goals, r_goals], dim=1)      # [B, N, G, 3]
        all_valid = torch.cat([f_valid, r_valid], dim=1)      # [B, N]
        all_ridx = torch.cat([f_ridx, r_ridx], dim=1)        # [B, N]
        all_bcount = torch.cat([f_bcount, r_bcount], dim=1)   # [B, N]
        all_subs = torch.cat([f_subs, r_subs], dim=1)        # [B, N, 2, 2]
        all_parents = torch.cat([f_parents, r_parents], dim=1)  # [B, N]
        all_has_new = torch.cat([f_has_new, r_has_new], dim=1)  # [B, N]
        all_current_ridx = torch.cat([f_current_ridx, r_current_ridx], dim=1)

    # ── Scatter-compact to S_out ──
    cumsum = all_valid.long().cumsum(dim=1)
    target = torch.where(
        all_valid, cumsum - 1,
        torch.tensor(S_out, dtype=torch.long, device=dev),
    ).clamp(min=0, max=S_out)

    out_gbody = torch.full(
        (B, S_out + 1, M_work, 3), pad, dtype=torch.long, device=dev)
    out_goals = torch.full(
        (B, S_out + 1, G, 3), pad, dtype=torch.long, device=dev)
    out_ridx = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    if bc_is_3d:
        out_bcount = torch.zeros(B, S_out + 1, D_bc, dtype=torch.long, device=dev)
    else:
        out_bcount = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_subs = torch.full(
        (B, S_out + 1, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out + 1, dtype=torch.bool, device=dev)
    out_cur_ridx = torch.full(
        (B, S_out + 1), -1, dtype=torch.long, device=dev)

    ti = target.unsqueeze(-1).unsqueeze(-1)
    out_gbody.scatter_(1, ti.expand(-1, -1, M_work, 3), all_gbody)
    out_goals.scatter_(1, ti.expand(-1, -1, G, 3), all_goals)
    out_ridx.scatter_(1, target, all_ridx)
    if bc_is_3d:
        out_bcount.scatter_(1, target[:, :, None].expand(-1, -1, D_bc),
                            all_bcount)
    else:
        out_bcount.scatter_(1, target, all_bcount)
    out_subs.scatter_(
        1, target.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2), all_subs)
    out_parents.scatter_(1, target, all_parents)
    out_has_new.scatter_(1, target, all_has_new)
    out_cur_ridx.scatter_(1, target, all_current_ridx)

    counts = all_valid.sum(dim=1).clamp(max=S_out)
    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.unsqueeze(1)

    return PackedStates(out_gbody[:, :S_out], out_goals[:, :S_out],
                        out_ridx[:, :S_out], out_valid, out_bcount[:, :S_out],
                        out_parents[:, :S_out], out_subs[:, :S_out],
                        out_has_new[:, :S_out], out_cur_ridx[:, :S_out])


# ---------------------------------------------------------------------------
# Flat packing: FlatResolvedChildren → dense PackedStates
# ---------------------------------------------------------------------------


def pack_states_flat(
    flat_resolved,           # FlatResolvedChildren
    top_ridx: Tensor,        # [B, S] parent rule indices
    grounding_body: Tensor,  # [B, S, M_work, 3] parent working body
    body_count: Tensor,      # [B, S, D] or [B, S] valid atom count per state
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
    dedup: bool = True,
) -> PackedStates:
    """Pack flat resolve output into dense [B, S_out, ...] state tensors.

    S_out is dynamic — computed from the actual number of unique valid children
    per batch element. No fixed S_max cap.

    Deduplicates children with identical proof goals within each batch element
    (same hash = same state, no need to explore twice).
    """
    B = flat_resolved.B
    pad = padding_idx
    dev = flat_resolved.flat_goals.device

    flat_goals = flat_resolved.flat_goals       # [T, G, 3]
    flat_gbody = flat_resolved.flat_gbody       # [T, A, 3]
    flat_ridx = flat_resolved.flat_rule_idx     # [T]
    flat_b = flat_resolved.flat_b_idx           # [T]
    flat_s = flat_resolved.flat_s_idx           # [T]
    flat_subs = flat_resolved.flat_subs         # [T, 2, 2]
    T = flat_goals.size(0)
    G = flat_goals.size(1)
    M_work = grounding_body.shape[2]

    bc_is_3d = body_count.dim() == 3
    if T == 0:
        S_out = 1
        out_valid = torch.zeros(B, S_out, dtype=torch.bool, device=dev)
        out_goals = torch.full((B, S_out, G, 3), pad, dtype=torch.long, device=dev)
        out_gbody = torch.full((B, S_out, M_work, 3), pad, dtype=torch.long, device=dev)
        out_ridx = torch.zeros(B, S_out, dtype=torch.long, device=dev)
        if bc_is_3d:
            D_bc = body_count.shape[2]
            out_bcount = torch.zeros(B, S_out, D_bc, dtype=torch.long, device=dev)
        else:
            out_bcount = torch.zeros(B, S_out, dtype=torch.long, device=dev)
        out_parents = torch.zeros(B, S_out, dtype=torch.long, device=dev)
        out_subs = torch.full((B, S_out, 2, 2), pad, dtype=torch.long, device=dev)
        out_has_new = torch.zeros(B, S_out, dtype=torch.bool, device=dev)
        out_cur_ridx = torch.full((B, S_out), -1, dtype=torch.long, device=dev)
        return PackedStates(out_gbody, out_goals, out_ridx, out_valid,
                            out_bcount, out_parents, out_subs, out_has_new,
                            out_cur_ridx)

    # ── Dedup: remove children with identical proof goals within each batch ──
    if dedup:
        P1, P2, P3, P4 = 1_000_003, 999_983, 999_979, 999_961
        atom_h = (flat_goals[..., 0].long() * P1
                  + flat_goals[..., 1].long() * P2
                  + flat_goals[..., 2].long() * P3)       # [T, G]
        powers = P4 ** torch.arange(G - 1, -1, -1, device=dev)
        goal_hash = (atom_h * powers).sum(dim=-1)          # [T]
        compound = flat_b.long() * P1 + goal_hash
        sorted_c, sort_idx = compound.sort()
        is_dup = torch.zeros(T, dtype=torch.bool, device=dev)
        is_dup[1:] = sorted_c[1:] == sorted_c[:-1]
        is_dup_orig = is_dup[sort_idx.argsort()]
        keep = ~is_dup_orig

        flat_goals = flat_goals[keep]
        flat_gbody = flat_gbody[keep]
        flat_ridx = flat_ridx[keep]
        flat_b = flat_b[keep]
        flat_s = flat_s[keep]
        flat_subs = flat_subs[keep]
        T = flat_goals.size(0)

    # ── Dynamic S_out: max unique children per batch element ──
    from grounder.resolution.enum import _cumcount_flat
    counts = torch.zeros(B, dtype=torch.long, device=dev)
    if T > 0:
        counts.scatter_add_(0, flat_b, torch.ones(T, dtype=torch.long, device=dev))
    S_out = max(int(counts.max().item()), 1)  # one .item() graph break

    # Per-batch cumcount: assign each child a sequential position
    pos = _cumcount_flat(flat_b)  # [T]

    # ── Build dense output tensors [B, S_out, ...] ──
    out_goals = torch.full((B, S_out, G, 3), pad, dtype=torch.long, device=dev)
    out_gbody = torch.full((B, S_out, M_work, 3), pad, dtype=torch.long, device=dev)
    out_ridx = torch.zeros(B, S_out, dtype=torch.long, device=dev)
    if bc_is_3d:
        D_bc = body_count.shape[2]
        out_bcount = torch.zeros(B, S_out, D_bc, dtype=torch.long, device=dev)
    else:
        out_bcount = torch.zeros(B, S_out, dtype=torch.long, device=dev)
    out_subs = torch.full((B, S_out, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out, dtype=torch.bool, device=dev)
    out_cur_ridx = torch.full((B, S_out), -1, dtype=torch.long, device=dev)

    if T > 0:
        out_goals[flat_b, pos] = flat_goals

        if M_rule <= 0:
            M_rule = M_work
        new_body = flat_goals[:, :M_rule, :]
        if M_rule < M_work:
            new_body = torch.nn.functional.pad(
                new_body, (0, 0, 0, M_work - M_rule), value=pad)
        out_gbody[flat_b, pos] = new_body

        parent_ridx = top_ridx[flat_b, flat_s]
        first = (parent_ridx == -1)
        out_ridx[flat_b, pos] = torch.where(first, flat_ridx, flat_ridx)
        out_bcount[flat_b, pos] = body_count[flat_b, flat_s]
        out_parents[flat_b, pos] = flat_s
        out_subs[flat_b, pos] = flat_subs
        out_has_new[flat_b, pos] = True
        out_cur_ridx[flat_b, pos] = flat_ridx  # current depth's rule index

    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.clamp(max=S_out).unsqueeze(1)

    return PackedStates(out_gbody, out_goals, out_ridx, out_valid,
                        out_bcount, out_parents, out_subs, out_has_new,
                        out_cur_ridx)


# ---------------------------------------------------------------------------
# Post-processing: grounding collection (from postprocessing.py)
# ---------------------------------------------------------------------------


def collect_groundings(
    grounding_body: Tensor,     # [B, S, D, M, 3] structured
    proof_goals: Tensor,        # [B, S, G, 3]
    state_valid: Tensor,        # [B, S]
    ridx_per_depth: Tensor,     # [B, S, D]
    collected_body: Tensor,     # [B, C, D, M, 3]
    collected_mask: Tensor,     # [B, C]
    collected_ridx: Tensor,     # [B, C, D]
    constant_no: int,
    pad_idx: int,
    C: int,
    body_count: Tensor,          # [B, S, D]
    collected_bcount: Tensor,    # [B, C, D]
    collect_mode: str = "terminal",
    deactivate: bool = True,
    head_per_depth: Optional[Tensor] = None,   # [B, S, D, 3]
    collected_head: Optional[Tensor] = None,    # [B, C, D, 3]
) -> Tuple:
    """Collect completed groundings into output buffer.

    Handles structured body [B, S, D, M, 3] and per-depth rule indices [B, S, D].

    Args:
        deactivate: If True (default), collected states are deactivated so they
            are not explored further. Set to False when collecting intermediate
            (grounded) states that should continue to deeper depths.

    Returns:
        out_body:    [B, C, D, M, 3]
        out_mask:    [B, C]
        out_ridx:    [B, C, D]
        state_valid: [B, S] updated
        out_bcount:  [B, C, D]
    """
    B, S, D_dim, M_dim, _ = grounding_body.shape
    dev = grounding_body.device
    E = constant_no + 1
    G_body_flat = D_dim * M_dim

    is_padding = (proof_goals[:, :, :, 0] == pad_idx)  # [B, S, G]

    # Flatten body for ground-check: [B, S, D*M, 3]
    body_flat = grounding_body.reshape(B, S, G_body_flat, 3)
    body_args = body_flat[:, :, :, 1:3]
    body_active = (body_flat[:, :, :, 0] != pad_idx)
    is_ground = ((body_args < E) | ~body_active.unsqueeze(-1)).all(dim=-1).all(dim=-1)

    if collect_mode == "grounded":
        goal_args = proof_goals[:, :, :, 1:3]
        goal_grounded = (goal_args < E).all(dim=-1)
        all_goals_ok = (is_padding | goal_grounded).all(dim=2)
    else:
        all_goals_ok = is_padding.all(dim=2)

    valid_grounding = all_goals_ok & is_ground & state_valid

    has_head = head_per_depth is not None and collected_head is not None

    n_new = S
    n_cat = C + n_new
    # Cat along dim=1 — inner dims D, M, 3 carried through
    cb = torch.cat([collected_body, grounding_body], dim=1)     # [B, C+S, D, M, 3]
    cm = torch.cat([collected_mask, valid_grounding], dim=1)    # [B, C+S]
    cr = torch.cat([collected_ridx, ridx_per_depth], dim=1)     # [B, C+S, D]
    c_bc = torch.cat([collected_bcount, body_count], dim=1)     # [B, C+S, D]
    if has_head:
        c_hd = torch.cat([collected_head, head_per_depth], dim=1)  # [B, C+S, D, 3]

    # Dedup: hash over flat body + all D rule indices
    cb_flat = cb.reshape(B, n_cat, G_body_flat, 3)
    cm = _dedup_groundings(cb_flat, cr, cm, G_body_flat)

    n_k = min(C, n_cat)
    _, ki = cm.to(torch.int8).topk(
        n_k, dim=1, largest=True, sorted=False)

    # Gather with structured dimensions
    ki_body = ki[:, :, None, None, None].expand(-1, -1, D_dim, M_dim, 3)
    ki_ridx = ki[:, :, None].expand(-1, -1, D_dim)
    ki_head = ki[:, :, None, None].expand(-1, -1, D_dim, 3) if has_head else None

    if n_k < C:
        p2 = C - n_k
        out_body = torch.nn.functional.pad(
            cb.gather(1, ki_body), (0, 0, 0, 0, 0, 0, 0, p2))
        out_mask = torch.nn.functional.pad(cm.gather(1, ki), (0, p2))
        out_ridx = torch.nn.functional.pad(cr.gather(1, ki_ridx), (0, 0, 0, p2))
        out_bcount = torch.nn.functional.pad(
            c_bc.gather(1, ki_ridx), (0, 0, 0, p2))
        out_head = (torch.nn.functional.pad(
            c_hd.gather(1, ki_head), (0, 0, 0, 0, 0, p2))
            if has_head else None)
    else:
        out_body = cb.gather(1, ki_body)
        out_mask = cm.gather(1, ki)
        out_ridx = cr.gather(1, ki_ridx)
        out_bcount = c_bc.gather(1, ki_ridx)
        out_head = c_hd.gather(1, ki_head) if has_head else None

    if deactivate:
        state_valid = state_valid & ~valid_grounding

    return out_body, out_mask, out_ridx, state_valid, out_bcount, out_head


def _dedup_groundings(
    body: Tensor,       # [B, N, G_body, 3] (flat view)
    ridx: Tensor,       # [B, N] or [B, N, D]
    mask: Tensor,       # [B, N]
    G_body: int,
) -> Tensor:
    """Remove duplicate groundings based on (ridx, body) hash.

    Args:
        body: [B, N, G_body, 3] grounding body atoms (flat view)
        ridx: [B, N, D] per-depth rule indices, or [B, N] single rule index
        mask: [B, N] validity mask
        G_body: number of body atom slots in flat view

    Returns:
        mask: [B, N] updated mask with duplicates removed
    """
    B, N = mask.shape
    dev = mask.device
    P1, P2, P3, P4 = 1_000_003, 999_983, 999_979, 999_961

    # Body hash: [B, N]
    atom_hashes = (body[..., 0].long() * P1
                   + body[..., 1].long() * P2
                   + body[..., 2].long() * P3)               # [B, N, G_body]
    powers = P4 ** torch.arange(G_body - 1, -1, -1, device=dev)
    body_hash = (atom_hashes * powers).sum(dim=-1)            # [B, N]

    # Rule index hash: include all D dimensions if structured
    if ridx.dim() == 3:
        D = ridx.shape[2]
        r_powers = P4 ** torch.arange(D - 1, -1, -1, device=dev)
        ridx_hash = (ridx.long() * r_powers).sum(dim=-1)     # [B, N]
    else:
        ridx_hash = ridx.long()                                # [B, N]

    g_hash = ridx_hash * P1 + body_hash                       # [B, N]

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    gh = torch.where(mask, g_hash, sentinel.expand(B, N))
    sorted_gh, sort_idx = gh.sort(dim=1)
    prev_gh = torch.nn.functional.pad(
        sorted_gh[:, :-1], (1, 0), value=-2)
    is_dup = (sorted_gh == prev_gh)
    inv_sort = sort_idx.argsort(dim=1)
    is_dup_orig = is_dup.gather(1, inv_sort)
    return mask & ~is_dup_orig


# ---------------------------------------------------------------------------
# rule2groundings pruning + tensor conversion
# ---------------------------------------------------------------------------


def prune_rule_groundings(
    rule2groundings: Dict[int, Set[Tuple]],
    fact_set: Set[Tuple[int, int, int]],
    max_iterations: int = 10,
) -> Dict[int, Set[Tuple]]:
    """Iterative fixed-point pruning of rule groundings (Kleene T_P).

    Equivalent to keras's PruneIncompleteProofs. Keeps only groundings
    whose body atoms are all either facts or heads of other proved groundings.

    Args:
        rule2groundings: rule_idx → set of (head, body) tuples
        fact_set: set of (pred, subj, obj) known facts
        max_iterations: convergence bound

    Returns:
        Pruned dict with same structure.
    """
    # Compute proved heads: start with facts
    proved: Set[Tuple[int, int, int]] = set(fact_set)

    for _ in range(max_iterations):
        prev_size = len(proved)
        # Add heads of groundings whose bodies are all proved
        for r, groundings in rule2groundings.items():
            for head, body in groundings:
                if all(atom in proved for atom in body):
                    proved.add(head)
        if len(proved) == prev_size:
            break  # converged

    # Filter: keep groundings whose bodies are all proved
    pruned: Dict[int, Set[Tuple]] = {}
    for r, groundings in rule2groundings.items():
        kept = set()
        for head, body in groundings:
            if all(atom in proved for atom in body):
                kept.add((head, body))
        if kept:
            pruned[r] = kept
    return pruned


def build_rule_grounding_tensors(
    rule2groundings: Dict[int, Set[Tuple]],
    num_rules: int,
    device: torch.device,
) -> "RuleGroundings":
    """Convert Python rule2groundings to (A_in, A_out) tensors.

    Each entry in rule2groundings[r] is (head_tuple, body_tuple) where:
    - head_tuple = (pred, subj, obj)
    - body_tuple = ((p,s,o), (p,s,o), ...)

    Builds a global atom table and per-rule index tensors.

    Returns:
        RuleGroundings with atom_table [num_atoms, 3] and per-rule A_in/A_out.
    """
    from grounder.types import RuleGroundings

    # 1. Collect all unique atoms
    all_atoms: Dict[Tuple[int, int, int], int] = {}

    def get_idx(atom: Tuple[int, int, int]) -> int:
        if atom not in all_atoms:
            all_atoms[atom] = len(all_atoms)
        return all_atoms[atom]

    # Pre-scan to build atom table
    for r, groundings in rule2groundings.items():
        for head, body in groundings:
            get_idx(head)
            for atom in body:
                get_idx(atom)

    num_atoms = len(all_atoms)
    atom_table = torch.zeros(num_atoms, 3, dtype=torch.long, device=device)
    for atom, idx in all_atoms.items():
        atom_table[idx, 0] = atom[0]
        atom_table[idx, 1] = atom[1]
        atom_table[idx, 2] = atom[2]

    # 2. Build per-rule A_in, A_out
    A_in: Dict[int, Tensor] = {}
    A_out: Dict[int, Tensor] = {}

    for r in range(num_rules):
        groundings = rule2groundings.get(r, set())
        if not groundings:
            A_in[r] = torch.zeros(0, 0, dtype=torch.long, device=device)
            A_out[r] = torch.zeros(0, 1, dtype=torch.long, device=device)
            continue

        g_list = sorted(groundings)  # deterministic order
        G_r = len(g_list)
        M_r = max(len(body) for _, body in g_list)

        a_in = torch.zeros(G_r, M_r, dtype=torch.long, device=device)
        a_out = torch.zeros(G_r, 1, dtype=torch.long, device=device)

        for g, (head, body) in enumerate(g_list):
            a_out[g, 0] = all_atoms[head]
            for m, atom in enumerate(body):
                a_in[g, m] = all_atoms[atom]

        A_in[r] = a_in
        A_out[r] = a_out

    return RuleGroundings(
        atom_table=atom_table,
        A_in=A_in,
        A_out=A_out,
        num_atoms=num_atoms,
        num_rules=num_rules,
    )
