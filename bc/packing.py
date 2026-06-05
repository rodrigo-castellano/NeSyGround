"""BC packing utilities: polynomial-hash power cache, atom compaction,
and fixed-shape state packing (dense ``pack_states`` + flat
``pack_states_flat``).

Pure and torch.compile-compatible. Split out of ``bc/common.py``; import
these names from there (a re-export facade) or from here directly.
"""

from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

import torch
from torch import Tensor

from grounder.types import PackedStates


# Cache for the constant polynomial-hash power vectors ``P ** arange(n,-1,-1)``.
# The hash bases (P4/P5) and the slot count ``n`` (G / M / D / G_body) are all
# static per grounder, so the power vector recurs identically on every pack /
# dedup call. Caching it removes a fresh ``arange`` + ``pow`` (two kernel
# launches) per use on the eager hot path. Read-only — never mutated in place.
#
# Used ONLY in eager mode (see ``_pow_desc``): ``_dedup_groundings`` runs
# inside ``collect_groundings_step`` which is part of the compiled ``step``
# unit, and a persistent cached tensor captured inside a CUDA graph aliases
# across replays ("accessing tensor output of CUDAGraphs that has been
# overwritten"). Under compile the power vector is materialised fresh in-graph.
_POW_CACHE: Dict[Tuple[int, int, str], Tensor] = {}


def _pow_desc(base: int, n: int, device) -> Tensor:
    """Return ``base ** torch.arange(n - 1, -1, -1, device)`` (read-only).

    Eager: served from ``_POW_CACHE`` (one fewer arange+pow per call on the
    hot path). Compiling: materialised fresh — a persistent cached tensor is
    not CUDA-graph-safe (it aliases across graph replays).
    """
    if torch.compiler.is_compiling():
        return base ** torch.arange(n - 1, -1, -1, device=device)
    key = (int(base), int(n), str(device))
    t = _POW_CACHE.get(key)
    if t is None:
        t = base ** torch.arange(n - 1, -1, -1, device=device)
        _POW_CACHE[key] = t
    return t


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
            # Skip facts when grounding_body is uninitialised AND we are
            # not at the initial depth-0 resolution. At depth 0 (top_ridx
            # == -1, no rule applied yet) a direct fact match against the
            # query IS a valid 0-body proof ("the query is itself a fact
            # in the KB" — true for any base fact or, under the fp_global
            # KB augmentation, any closure atom). Previously this case
            # was dropped, which left 2+-hop queries and wide-existential
            # rules with no grounding even when the head was derivable.
            if bc_is_3d:
                uninit = (body_count.sum(dim=-1) == 0)  # [B, S]
            else:
                uninit = (body_count == 0)
            is_initial = (top_ridx == -1)  # [B, S] — pre-first-rule
            skip_fact = uninit & ~is_initial
            f_valid = f_valid & ~skip_fact.unsqueeze(-1).expand(
                B, S_in, K_f).reshape(B, n_f)
        # Fact children: no new body atoms (padding). Substitution
        # propagation across depths is handled by _sync_accumulated
        # writing winning_subs into accumulated_body.
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
    flat_ridx = flat_resolved.flat_rule_idx     # [T]
    flat_b = flat_resolved.flat_b_idx           # [T]
    flat_s = flat_resolved.flat_s_idx           # [T]
    # ``flat_resolved.flat_gbody`` / ``.flat_subs`` are intentionally unused:
    # out_gbody is rebuilt from flat_goals[:M_rule] and out_subs stays all-pad.
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
        # Body atoms at [:M_rule] are stored in anchor-variant order
        # (all_anchors=True permutes them). Sort within the body prefix
        # so anchor variants of the same logical rule application share
        # a hash; leave parent-inherited goals at [M_rule:] in natural
        # order (anchor-invariant). Mirrors the canonical-order pattern
        # in ``BCGrounder._collect_r2g_tensor`` for terminal collection.
        if M_rule > 0 and M_rule < G:
            body_h, _ = atom_h[:, :M_rule].sort(dim=-1)
            atom_h = torch.cat([body_h, atom_h[:, M_rule:]], dim=-1)
        elif M_rule > 0 and G > 0:
            atom_h, _ = atom_h.sort(dim=-1)
        powers = _pow_desc(P4, G, dev)
        goal_hash = (atom_h * powers).sum(dim=-1)          # [T]
        compound = flat_b.long() * P1 + goal_hash
        sorted_c, sort_idx = compound.sort()
        # ``is_dup[i] = sorted_c[i] == sorted_c[i-1]`` with is_dup[0]=False.
        # Built via cat (one alloc + concat) instead of zeros + indexed
        # scatter-write, which dispatched two kernels for the same result.
        eq = sorted_c[1:] == sorted_c[:-1]                 # [T-1]
        is_dup = torch.cat(
            [eq.new_zeros(1), eq], dim=0)                  # [T]
        is_dup_orig = is_dup[sort_idx.argsort()]
        keep = ~is_dup_orig

        # ``flat_gbody`` / ``flat_subs`` are never read after dedup (out_gbody
        # is rebuilt from flat_goals[:M_rule]; out_subs stays all-pad), so
        # their boolean-mask gathers are dropped — pure dead work.
        flat_goals = flat_goals[keep]
        flat_ridx = flat_ridx[keep]
        flat_b = flat_b[keep]
        flat_s = flat_s[keep]
        T = flat_goals.size(0)

    # ── Dynamic S_out: max unique children per batch element ──
    from grounder.resolution.enum import _cumcount_flat
    # ``bincount`` is a single kernel and gives the identical per-batch
    # counts as the previous zeros + ones + scatter_add_ trio.
    if T > 0:
        counts = torch.bincount(flat_b, minlength=B)
    else:
        counts = torch.zeros(B, dtype=torch.long, device=dev)
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

        # ``torch.where(first, flat_ridx, flat_ridx)`` is the identity on
        # ``flat_ridx`` regardless of ``first``, so the parent-ridx gather +
        # eq + where (3 dead kernels) are dropped — the result is unchanged.
        out_ridx[flat_b, pos] = flat_ridx
        out_bcount[flat_b, pos] = body_count[flat_b, flat_s]
        out_parents[flat_b, pos] = flat_s
        # ``out_subs`` stays all-pad: ``flat_subs`` is all-padding in the flat
        # path and ``out_subs`` is already initialised to pad, so the old
        # ``out_subs[flat_b, pos] = flat_subs`` scatter wrote pad over pad — a
        # byte-identical no-op and is dropped.
        out_has_new[flat_b, pos] = True
        out_cur_ridx[flat_b, pos] = flat_ridx  # current depth's rule index

    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.clamp(max=S_out).unsqueeze(1)

    return PackedStates(out_gbody, out_goals, out_ridx, out_valid,
                        out_bcount, out_parents, out_subs, out_has_new,
                        out_cur_ridx)


# ---------------------------------------------------------------------------
# Post-processing: grounding collection (from postprocessing.py)
# ---------------------------------------------------------------------------
