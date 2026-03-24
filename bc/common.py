"""BC shared utilities: packing, compaction, grounding collection, deduplication.

Provides scatter-based compaction of derived states into fixed-shape output
tensors and grounding collection.

All functions are pure (no class dependencies) and torch.compile compatible.
"""

from __future__ import annotations
from typing import Optional, Tuple

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
    body_count: Tensor,        # [B, S]
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
    if K_f > 0:
        f_goals = fact_goals.reshape(B, n_f, G, 3)
        f_valid = fact_success.reshape(B, n_f)
        f_ridx = top_ridx.unsqueeze(2).expand(
            B, S_in, K_f).reshape(B, n_f)
        f_bcount = body_count.unsqueeze(2).expand(
            B, S_in, K_f).reshape(B, n_f)
        f_subs = fact_subs.reshape(B, n_f, 2, 2)
        # Parent indices for fact children: child j has parent j // K_f
        f_parents = torch.arange(S_in, device=dev).unsqueeze(1).expand(
            S_in, K_f).reshape(n_f)
        f_parents = f_parents.unsqueeze(0).expand(B, n_f)
        if collect_evidence:
            # Skip facts when grounding_body not yet established
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
    else:
        all_gbody = torch.cat([f_gbody, r_gbody], dim=1)     # [B, N, M_work, 3]
        all_goals = torch.cat([f_goals, r_goals], dim=1)      # [B, N, G, 3]
        all_valid = torch.cat([f_valid, r_valid], dim=1)      # [B, N]
        all_ridx = torch.cat([f_ridx, r_ridx], dim=1)        # [B, N]
        all_bcount = torch.cat([f_bcount, r_bcount], dim=1)   # [B, N]
        all_subs = torch.cat([f_subs, r_subs], dim=1)        # [B, N, 2, 2]
        all_parents = torch.cat([f_parents, r_parents], dim=1)  # [B, N]
        all_has_new = torch.cat([f_has_new, r_has_new], dim=1)  # [B, N]

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
    out_bcount = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_subs = torch.full(
        (B, S_out + 1, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out + 1, dtype=torch.bool, device=dev)

    ti = target.unsqueeze(-1).unsqueeze(-1)
    out_gbody.scatter_(1, ti.expand(-1, -1, M_work, 3), all_gbody)
    out_goals.scatter_(1, ti.expand(-1, -1, G, 3), all_goals)
    out_ridx.scatter_(1, target, all_ridx)
    out_bcount.scatter_(1, target, all_bcount)
    out_subs.scatter_(
        1, target.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2), all_subs)
    out_parents.scatter_(1, target, all_parents)
    out_has_new.scatter_(1, target, all_has_new)

    counts = all_valid.sum(dim=1).clamp(max=S_out)
    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.unsqueeze(1)

    return PackedStates(out_gbody[:, :S_out], out_goals[:, :S_out],
                        out_ridx[:, :S_out], out_valid, out_bcount[:, :S_out],
                        out_parents[:, :S_out], out_subs[:, :S_out],
                        out_has_new[:, :S_out])


# ---------------------------------------------------------------------------
# Post-processing: grounding collection (from postprocessing.py)
# ---------------------------------------------------------------------------


def collect_groundings(
    grounding_body: Tensor,     # [B, S, G_body, 3]
    proof_goals: Tensor,        # [B, S, G, 3]
    state_valid: Tensor,        # [B, S]
    top_ridx: Tensor,           # [B, S]
    collected_body: Tensor,     # [B, C, G_body, 3]
    collected_mask: Tensor,     # [B, C]
    collected_ridx: Tensor,     # [B, C]
    constant_no: int,
    pad_idx: int,
    C: int,
    body_count: Tensor,          # [B, S]
    collected_bcount: Tensor,    # [B, C]
    collect_mode: str = "terminal",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collect completed groundings into output buffer (TS-specific).

    Args:
        grounding_body: [B, S, G_body, 3] current grounding bodies (accumulated)
        proof_goals: [B, S, G, 3] current proof goals (after pruning + compaction)
        state_valid: [B, S] active state mask
        top_ridx: [B, S] rule index per state
        collected_body: [B, C, G_body, 3] accumulated grounding bodies
        collected_mask: [B, C] accumulated validity mask
        collected_ridx: [B, C] accumulated rule indices
        constant_no: highest constant index
        pad_idx: padding value
        C: max number of collected groundings (C)
        body_count: [B, S] number of valid body atoms per state
        collected_bcount: [B, C] accumulated body counts
        collect_mode: 'terminal' (all goals padding) or 'grounded' (goals
            may contain grounded unknowns — for the u-variant / nesy scoring).

    Returns:
        out_body:    [B, C, G_body, 3] updated collected bodies
        out_mask:    [B, C] updated collected mask
        out_ridx:    [B, C] updated collected rule indices
        state_valid: [B, S] updated (terminal states deactivated)
        out_bcount:  [B, C] updated collected body counts
    """
    B, S, G_body, _ = grounding_body.shape
    dev = grounding_body.device
    E = constant_no + 1
    # C is the collected groundings budget (parameter)

    is_padding = (proof_goals[:, :, :, 0] == pad_idx)  # [B, S, G]

    body_args = grounding_body[:, :, :, 1:3]
    body_active = (grounding_body[:, :, :, 0] != pad_idx)   # [B, S, G_body]
    is_ground = ((body_args < E) | ~body_active.unsqueeze(-1)).all(dim=-1).all(dim=-1)

    if collect_mode == "grounded":
        # Accept states where all goals are either padding or grounded
        # (no variables). Remaining grounded goals are open unknowns
        # for downstream nesy scoring.
        goal_args = proof_goals[:, :, :, 1:3]             # [B, S, G, 2]
        goal_grounded = (goal_args < E).all(dim=-1)        # [B, S, G]
        all_goals_ok = (is_padding | goal_grounded).all(dim=2)
    else:
        # Terminal: all goals must be padding (fully resolved)
        all_goals_ok = is_padding.all(dim=2)

    valid_grounding = all_goals_ok & is_ground & state_valid

    n_new = S
    body_new = grounding_body
    ridx_new = top_ridx

    n_cat = C + n_new
    cb = torch.cat([collected_body, body_new], dim=1)
    cm = torch.cat([collected_mask, valid_grounding], dim=1)
    cr = torch.cat([collected_ridx, ridx_new], dim=1)

    # Thread body_count through
    c_bc = torch.cat([collected_bcount, body_count], dim=1)  # [B, C + S]

    cm = _dedup_groundings(cb, cr, cm, G_body)

    n_k = min(C, n_cat)
    _, ki = cm.to(torch.int8).topk(
        n_k, dim=1, largest=True, sorted=False)

    if n_k < C:
        p2 = C - n_k
        out_body = torch.nn.functional.pad(
            cb.gather(1, ki.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, G_body, 3)), (0, 0, 0, 0, 0, p2))
        out_mask = torch.nn.functional.pad(
            cm.gather(1, ki), (0, p2))
        out_ridx = torch.nn.functional.pad(
            cr.gather(1, ki), (0, p2))
        out_bcount = torch.nn.functional.pad(
            c_bc.gather(1, ki), (0, p2))
    else:
        out_body = cb.gather(
            1, ki.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, G_body, 3))
        out_mask = cm.gather(1, ki)
        out_ridx = cr.gather(1, ki)
        out_bcount = c_bc.gather(1, ki)

    state_valid = state_valid & ~valid_grounding

    return out_body, out_mask, out_ridx, state_valid, out_bcount


def _dedup_groundings(
    body: Tensor,       # [B, N, G_body, 3]
    ridx: Tensor,       # [B, N]
    mask: Tensor,       # [B, N]
    G_body: int,
) -> Tensor:
    """Remove duplicate groundings based on (ridx, body) hash.

    Args:
        body: [B, N, G_body, 3] grounding body atoms
        ridx: [B, N] rule index per grounding
        mask: [B, N] validity mask
        G_body: number of body atom slots (accumulated capacity)

    Returns:
        mask: [B, N] updated mask with duplicates removed
    """
    B, N = mask.shape
    dev = mask.device
    # Vectorized prime-mixing hash (overflow-safe, no E*E needed)
    P1, P2, P3, P4 = 1_000_003, 999_983, 999_979, 999_961
    atom_hashes = (body[..., 0].long() * P1
                   + body[..., 1].long() * P2
                   + body[..., 2].long() * P3)               # [B, N, G_body]
    powers = P4 ** torch.arange(G_body - 1, -1, -1, device=dev)   # [G_body]
    g_hash = ridx.long() * P1 + (atom_hashes * powers).sum(dim=-1)  # [B, N]
    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    gh = torch.where(mask, g_hash, sentinel.expand(B, N))
    sorted_gh, sort_idx = gh.sort(dim=1)
    prev_gh = torch.nn.functional.pad(
        sorted_gh[:, :-1], (1, 0), value=-2)
    is_dup = (sorted_gh == prev_gh)
    inv_sort = sort_idx.argsort(dim=1)
    is_dup_orig = is_dup.gather(1, inv_sort)
    return mask & ~is_dup_orig
