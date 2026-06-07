"""State packing — compact resolution children back to per-state proof tapes.

Ported from OLD ``bc/packing.py``:
  * ``compact_atoms``       — left-align atoms after pruning gaps.
  * ``pack_states``         — dense cumsum-scatter compaction (NO topk).
  * ``pack_states_flat``    — flat polynomial-hash dedup + dynamic S_out.

Adapted to the NEW ``ResolvedChildren`` / ``FlatResolvedChildren`` field names
(leading ``layout``; ``flat_batch_idx`` / ``flat_state_idx`` /
``flat_grounding_body``). Returns an internal ``_Packed`` NamedTuple that carries
exactly the pieces the engine step + sync need — this is a private engine seam, NOT
a glossary-registered output type.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import torch
from torch import Tensor

from grounder._build.resolution.pbc.candidates import cumcount_flat


_POW_CACHE: Dict[Tuple[int, int, str], Tensor] = {}


def _pow_desc(base: int, n: int, device) -> Tensor:
    """``base ** arange(n-1, -1, -1)`` (cached read-only, eager)."""
    if torch.compiler.is_compiling():
        return base ** torch.arange(n - 1, -1, -1, device=device)
    key = (int(base), int(n), str(device))
    t = _POW_CACHE.get(key)
    if t is None:
        t = base ** torch.arange(n - 1, -1, -1, device=device)
        _POW_CACHE[key] = t
    return t


class _Packed(NamedTuple):
    """Internal pack result (private engine seam)."""
    grounding_body: Tensor      # [B, S_out, M_work, 3]
    proof_goals: Tensor         # [B, S_out, G, 3]
    top_ridx: Tensor            # [B, S_out]
    state_valid: Tensor         # [B, S_out]
    body_count: Tensor          # [B, S_out, D]  (inherited from parent)
    parent_map: Tensor          # [B, S_out]
    winning_subs: Tensor        # [B, S_out, 2, 2]
    has_new_body: Tensor        # [B, S_out]
    current_ridx: Tensor        # [B, S_out]


def compact_atoms(states: Tensor, padding_idx: int) -> Tensor:
    """Left-align non-padding atoms within each ``[..., M, 3]`` slice."""
    if states.numel() == 0:
        return states
    *leading, M, _ = states.shape
    flat = states.reshape(-1, M, 3)
    device = states.device
    pad = padding_idx
    valid_atom = (flat[:, :, 0] != pad)
    pos = torch.cumsum(valid_atom.long(), dim=1) - 1
    M_t = torch.tensor(M, dtype=pos.dtype, device=device)
    sort_key = torch.where(valid_atom, pos, M_t)
    sorted_indices = torch.argsort(sort_key, dim=1, stable=True)
    sorted_indices_exp = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
    result = torch.gather(flat, 1, sorted_indices_exp)
    return result.reshape(*leading, M, 3)


def pack_states(
    resolved,                  # ResolvedChildren (leading layout field)
    top_ridx: Tensor,          # [B, S]
    grounding_body: Tensor,    # [B, S, M_work, 3]
    body_count: Tensor,        # [B, S, D]
    S_out: int,
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
) -> _Packed:
    """Dense pack: facts concat FIRST then rules; cumsum-scatter compaction."""
    (_layout, fact_goals, _fact_gbody, fact_success,
     rule_goals, _rule_gbody, rule_success, sub_rule_idx,
     fact_subs, rule_subs) = resolved

    B, S_in = top_ridx.shape
    K_f = fact_goals.shape[2]
    K_r = rule_goals.shape[2]
    M_work = grounding_body.shape[2]
    pad = padding_idx
    dev = top_ridx.device

    n_f = S_in * K_f
    n_r = S_in * K_r
    G = rule_goals.shape[3]

    bc_is_3d = body_count.dim() == 3
    if K_f > 0:
        f_goals = fact_goals.reshape(B, n_f, G, 3)
        f_valid = fact_success.reshape(B, n_f)
        f_ridx = top_ridx.unsqueeze(2).expand(B, S_in, K_f).reshape(B, n_f)
        if bc_is_3d:
            D_bc = body_count.shape[2]
            f_bcount = body_count.unsqueeze(2).expand(
                B, S_in, K_f, D_bc).reshape(B, n_f, D_bc)
        else:
            f_bcount = body_count.unsqueeze(2).expand(B, S_in, K_f).reshape(B, n_f)
        f_subs = fact_subs.reshape(B, n_f, 2, 2)
        f_parents = torch.arange(S_in, device=dev).unsqueeze(1).expand(
            S_in, K_f).reshape(n_f)
        f_parents = f_parents.unsqueeze(0).expand(B, n_f)
        if collect_evidence:
            if bc_is_3d:
                uninit = (body_count.sum(dim=-1) == 0)
            else:
                uninit = (body_count == 0)
            is_initial = (top_ridx == -1)
            skip_fact = uninit & ~is_initial
            f_valid = f_valid & ~skip_fact.unsqueeze(-1).expand(
                B, S_in, K_f).reshape(B, n_f)
        f_gbody = torch.full((B, n_f, M_work, 3), pad, dtype=torch.long, device=dev)
        f_has_new = torch.zeros(B, n_f, dtype=torch.bool, device=dev)
    else:
        f_gbody = torch.full((B, 0, M_work, 3), pad, dtype=torch.long, device=dev)
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

    first = (top_ridx == -1).unsqueeze(2).expand(B, S_in, K_r).reshape(B, n_r)

    if collect_evidence:
        if M_rule <= 0:
            M_rule = M_work
        new_body_atoms = rule_goals[:, :, :, :M_rule, :].reshape(B, n_r, M_rule, 3)
        if M_rule < M_work:
            r_gbody = torch.full((B, n_r, M_work, 3), pad, dtype=torch.long, device=dev)
            r_gbody[:, :, :M_rule, :] = new_body_atoms
        elif M_rule > M_work:
            r_gbody = new_body_atoms[:, :, :M_work, :]
        else:
            r_gbody = new_body_atoms
        r_has_new = rule_success.reshape(B, n_r)
    else:
        r_gbody = torch.full((B, n_r, M_work, 3), pad, dtype=torch.long, device=dev)
        r_has_new = torch.zeros(B, n_r, dtype=torch.bool, device=dev)

    if bc_is_3d:
        r_bcount = body_count.unsqueeze(2).expand(
            B, S_in, K_r, D_bc).reshape(B, n_r, D_bc)
    else:
        r_bcount = body_count.unsqueeze(2).expand(B, S_in, K_r).reshape(B, n_r)

    r_ridx = torch.where(
        first,
        sub_rule_idx.reshape(B, n_r),
        top_ridx.unsqueeze(2).expand(B, S_in, K_r).reshape(B, n_r),
    )
    r_goals = rule_goals.reshape(B, n_r, G, 3)
    r_valid = rule_success.reshape(B, n_r)
    r_subs = rule_subs.reshape(B, n_r, 2, 2)
    r_parents = torch.arange(S_in, device=dev).unsqueeze(1).expand(
        S_in, K_r).reshape(n_r)
    r_parents = r_parents.unsqueeze(0).expand(B, n_r)

    f_current_ridx = torch.full(
        (B, n_f), -1, dtype=torch.long, device=dev) if K_f > 0 else (
        torch.zeros(B, 0, dtype=torch.long, device=dev))
    r_current_ridx = sub_rule_idx.reshape(B, n_r)

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
        all_gbody = torch.cat([f_gbody, r_gbody], dim=1)
        all_goals = torch.cat([f_goals, r_goals], dim=1)
        all_valid = torch.cat([f_valid, r_valid], dim=1)
        all_ridx = torch.cat([f_ridx, r_ridx], dim=1)
        all_bcount = torch.cat([f_bcount, r_bcount], dim=1)
        all_subs = torch.cat([f_subs, r_subs], dim=1)
        all_parents = torch.cat([f_parents, r_parents], dim=1)
        all_has_new = torch.cat([f_has_new, r_has_new], dim=1)
        all_current_ridx = torch.cat([f_current_ridx, r_current_ridx], dim=1)

    cumsum = all_valid.long().cumsum(dim=1)
    target = torch.where(
        all_valid, cumsum - 1,
        torch.tensor(S_out, dtype=torch.long, device=dev),
    ).clamp(min=0, max=S_out)

    out_gbody = torch.full((B, S_out + 1, M_work, 3), pad, dtype=torch.long, device=dev)
    out_goals = torch.full((B, S_out + 1, G, 3), pad, dtype=torch.long, device=dev)
    out_ridx = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    if bc_is_3d:
        out_bcount = torch.zeros(B, S_out + 1, D_bc, dtype=torch.long, device=dev)
    else:
        out_bcount = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_subs = torch.full((B, S_out + 1, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out + 1, dtype=torch.bool, device=dev)
    out_cur_ridx = torch.full((B, S_out + 1), -1, dtype=torch.long, device=dev)

    ti = target.unsqueeze(-1).unsqueeze(-1)
    out_gbody.scatter_(1, ti.expand(-1, -1, M_work, 3), all_gbody)
    out_goals.scatter_(1, ti.expand(-1, -1, G, 3), all_goals)
    out_ridx.scatter_(1, target, all_ridx)
    if bc_is_3d:
        out_bcount.scatter_(1, target[:, :, None].expand(-1, -1, D_bc), all_bcount)
    else:
        out_bcount.scatter_(1, target, all_bcount)
    out_subs.scatter_(
        1, target.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2), all_subs)
    out_parents.scatter_(1, target, all_parents)
    out_has_new.scatter_(1, target, all_has_new)
    out_cur_ridx.scatter_(1, target, all_current_ridx)

    counts = all_valid.sum(dim=1).clamp(max=S_out)
    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.unsqueeze(1)

    return _Packed(out_gbody[:, :S_out], out_goals[:, :S_out],
                   out_ridx[:, :S_out], out_valid, out_bcount[:, :S_out],
                   out_parents[:, :S_out], out_subs[:, :S_out],
                   out_has_new[:, :S_out], out_cur_ridx[:, :S_out])


def pack_states_flat(
    flat_resolved,             # FlatResolvedChildren (NEW field names)
    top_ridx: Tensor,          # [B, S]
    grounding_body: Tensor,    # [B, S, M_work, 3]
    body_count: Tensor,        # [B, S, D] or [B, S]
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
    dedup: bool = True,
    subs_noop: bool = True,
) -> _Packed:
    """Flat pack: polynomial-hash dedup, dynamic S_out, positions via cumcount."""
    B = flat_resolved.B
    pad = padding_idx
    dev = flat_resolved.flat_goals.device

    flat_goals = flat_resolved.flat_goals       # [T, G, 3]
    flat_ridx = flat_resolved.flat_rule_idx     # [T]
    flat_b = flat_resolved.flat_batch_idx       # [T]
    flat_s = flat_resolved.flat_state_idx       # [T]
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
        return _Packed(out_gbody, out_goals, out_ridx, out_valid,
                       out_bcount, out_parents, out_subs, out_has_new,
                       out_cur_ridx)

    if dedup:
        P1, P2, P3, P4 = 1_000_003, 999_983, 999_979, 999_961
        atom_h = (flat_goals[..., 0].long() * P1
                  + flat_goals[..., 1].long() * P2
                  + flat_goals[..., 2].long() * P3)
        if M_rule > 0 and M_rule < G:
            body_h, _ = atom_h[:, :M_rule].sort(dim=-1)
            atom_h = torch.cat([body_h, atom_h[:, M_rule:]], dim=-1)
        elif M_rule > 0 and G > 0:
            atom_h, _ = atom_h.sort(dim=-1)
        powers = _pow_desc(P4, G, dev)
        goal_hash = (atom_h * powers).sum(dim=-1)
        compound = flat_b.long() * P1 + goal_hash
        sorted_c, sort_idx = compound.sort()
        eq = sorted_c[1:] == sorted_c[:-1]
        is_dup = torch.cat([eq.new_zeros(1), eq], dim=0)
        is_dup_orig = is_dup[sort_idx.argsort()]
        keep = ~is_dup_orig
        flat_goals = flat_goals[keep]
        flat_ridx = flat_ridx[keep]
        flat_b = flat_b[keep]
        flat_s = flat_s[keep]
        T = flat_goals.size(0)

    if T > 0:
        counts = torch.bincount(flat_b, minlength=B)
    else:
        counts = torch.zeros(B, dtype=torch.long, device=dev)
    S_out = max(int(counts.max().item()), 1)

    pos = cumcount_flat(flat_b)

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
        out_bcount[flat_b, pos] = body_count[flat_b, flat_s]
        out_parents[flat_b, pos] = flat_s
        if subs_noop:                                   # ENUM — verbatim current writes
            out_gbody[flat_b, pos] = new_body
            out_ridx[flat_b, pos] = flat_ridx
            out_has_new[flat_b, pos] = True
            out_cur_ridx[flat_b, pos] = flat_ridx
        else:                                           # SLD/RTF — dense-equivalent provenance
            is_fact = flat_resolved.flat_is_fact        # [T] bool (dedup OFF -> aligned to pos)
            top = flat_resolved.flat_top_ridx           # [T] long
            first = (top == -1)
            eff_ridx = torch.where(is_fact, top, torch.where(first, flat_ridx, top))
            out_ridx[flat_b, pos] = eff_ridx
            out_has_new[flat_b, pos] = ~is_fact
            out_cur_ridx[flat_b, pos] = torch.where(
                is_fact, torch.full_like(flat_ridx, -1), flat_ridx)
            out_gbody[flat_b, pos] = torch.where(
                is_fact.view(-1, 1, 1), torch.full_like(new_body, pad), new_body)
            out_subs[flat_b, pos] = flat_resolved.flat_subs

    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.clamp(max=S_out).unsqueeze(1)

    return _Packed(out_gbody, out_goals, out_ridx, out_valid,
                   out_bcount, out_parents, out_subs, out_has_new,
                   out_cur_ridx)


__all__ = ["compact_atoms", "pack_states", "pack_states_flat", "_Packed", "_pow_desc"]
