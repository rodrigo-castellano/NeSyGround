"""State packing — compact resolution children back to per-goal proof tapes.

  * ``compact_atoms``       — left-align atoms after pruning gaps.
  * ``pack_states``         — dense cumsum-scatter compaction (NO topk).
  * ``pack_states_flat``    — flat polynomial-hash dedup + dynamic G_out.

Reads the ``ResolvedChildren`` / ``FlatResolvedChildren`` field names (leading
``layout``; ``flat_batch_idx`` / ``flat_state_idx`` / ``flat_grounding_body``).
Returns an internal ``_Packed`` NamedTuple that carries exactly the pieces the
engine step + sync need — this is a private engine seam, NOT a glossary-registered
output type.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from grounder.resolution.pbc.candidates import cumcount_flat


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
    grounding_body: Tensor      # [B, G_out, M_work, 3]
    goal_atoms: Tensor          # [B, G_out, L, 3]
    top_rule_idx: Tensor        # [B, G_out]
    goal_valid: Tensor          # [B, G_out]
    body_count: Tensor          # [B, G_out, D]  (inherited from parent)
    parent_map: Tensor          # [B, G_out]
    winning_subs: Tensor        # [B, G_out, 2, 2]
    has_new_body: Tensor        # [B, G_out]
    current_rule_idx: Tensor    # [B, G_out]


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
    top_rule_idx: Tensor,          # [B, G]
    grounding_body: Tensor,    # [B, G, M_work, 3]
    body_count: Tensor,        # [B, G, D]
    S_out: int,
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
) -> _Packed:
    """Dense pack: facts concat FIRST then rules; cumsum-scatter compaction."""
    (_layout, fact_goals, _fact_grounding_body, fact_success,
     rule_goals, _rule_grounding_body, rule_success, sub_rule_idx,
     fact_subs, rule_subs) = resolved

    B, S_in = top_rule_idx.shape
    K_f = fact_goals.shape[2]
    K_r = rule_goals.shape[2]
    M_work = grounding_body.shape[2]
    pad = padding_idx
    dev = top_rule_idx.device

    n_f = S_in * K_f
    n_r = S_in * K_r
    G = rule_goals.shape[3]
    D_bc = body_count.shape[2]                          # body_count always 3-D

    # ── RULE children (always present); first-step rules adopt sub_rule_idx as top ──
    first = (top_rule_idx == -1).unsqueeze(2).expand(B, S_in, K_r).reshape(B, n_r)
    if collect_evidence:
        r_grounding_body = rule_goals[:, :, :, :M_rule, :].reshape(B, n_r, M_rule, 3)
        r_has_new = rule_success.reshape(B, n_r)
    else:
        r_grounding_body = torch.full((B, n_r, M_work, 3), pad, dtype=torch.long, device=dev)
        r_has_new = torch.zeros(B, n_r, dtype=torch.bool, device=dev)
    r_body_count = body_count.unsqueeze(2).expand(B, S_in, K_r, D_bc).reshape(B, n_r, D_bc)
    r_rule_idx = torch.where(
        first, sub_rule_idx.reshape(B, n_r),
        top_rule_idx.unsqueeze(2).expand(B, S_in, K_r).reshape(B, n_r))
    r_goals = rule_goals.reshape(B, n_r, G, 3)
    r_valid = rule_success.reshape(B, n_r)
    r_subs = rule_subs.reshape(B, n_r, 2, 2)
    r_parents = (torch.arange(S_in, device=dev).unsqueeze(1)
                 .expand(S_in, K_r).reshape(n_r).unsqueeze(0).expand(B, n_r))
    r_current_rule_idx = sub_rule_idx.reshape(B, n_r)

    # ── FACT children prepended (facts FIRST, matching the flat path); else rules-only ──
    if K_f > 0:
        f_goals = fact_goals.reshape(B, n_f, G, 3)
        f_valid = fact_success.reshape(B, n_f)
        f_rule_idx = top_rule_idx.unsqueeze(2).expand(B, S_in, K_f).reshape(B, n_f)
        f_body_count = body_count.unsqueeze(2).expand(B, S_in, K_f, D_bc).reshape(B, n_f, D_bc)
        f_subs = fact_subs.reshape(B, n_f, 2, 2)
        f_parents = (torch.arange(S_in, device=dev).unsqueeze(1)
                     .expand(S_in, K_f).reshape(n_f).unsqueeze(0).expand(B, n_f))
        if collect_evidence:                            # skip facts on a started-but-non-initial state
            skip_fact = (body_count.sum(dim=-1) == 0) & ~(top_rule_idx == -1)
            f_valid = f_valid & ~skip_fact.unsqueeze(-1).expand(B, S_in, K_f).reshape(B, n_f)
        f_grounding_body = torch.full((B, n_f, M_work, 3), pad, dtype=torch.long, device=dev)
        f_has_new = torch.zeros(B, n_f, dtype=torch.bool, device=dev)
        f_current_rule_idx = torch.full((B, n_f), -1, dtype=torch.long, device=dev)
        all_grounding_body = torch.cat([f_grounding_body, r_grounding_body], dim=1)
        all_goals = torch.cat([f_goals, r_goals], dim=1)
        all_valid = torch.cat([f_valid, r_valid], dim=1)
        all_rule_idx = torch.cat([f_rule_idx, r_rule_idx], dim=1)
        all_body_count = torch.cat([f_body_count, r_body_count], dim=1)
        all_subs = torch.cat([f_subs, r_subs], dim=1)
        all_parents = torch.cat([f_parents, r_parents], dim=1)
        all_has_new = torch.cat([f_has_new, r_has_new], dim=1)
        all_current_rule_idx = torch.cat([f_current_rule_idx, r_current_rule_idx], dim=1)
    else:
        all_grounding_body, all_goals, all_valid = r_grounding_body, r_goals, r_valid
        all_rule_idx, all_body_count, all_subs = r_rule_idx, r_body_count, r_subs
        all_parents, all_has_new = r_parents, r_has_new
        all_current_rule_idx = r_current_rule_idx

    cumsum = all_valid.long().cumsum(dim=1)
    target = torch.where(
        all_valid, cumsum - 1,
        torch.tensor(S_out, dtype=torch.long, device=dev),
    ).clamp(min=0, max=S_out)

    out_grounding_body = torch.full((B, S_out + 1, M_work, 3), pad, dtype=torch.long, device=dev)
    out_goals = torch.full((B, S_out + 1, G, 3), pad, dtype=torch.long, device=dev)
    out_rule_idx = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_body_count = torch.zeros(B, S_out + 1, D_bc, dtype=torch.long, device=dev)
    out_subs = torch.full((B, S_out + 1, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out + 1, dtype=torch.bool, device=dev)
    out_cur_rule_idx = torch.full((B, S_out + 1), -1, dtype=torch.long, device=dev)

    ti = target.unsqueeze(-1).unsqueeze(-1)
    out_grounding_body.scatter_(1, ti.expand(-1, -1, M_work, 3), all_grounding_body)
    out_goals.scatter_(1, ti.expand(-1, -1, G, 3), all_goals)
    out_rule_idx.scatter_(1, target, all_rule_idx)
    out_body_count.scatter_(1, target[:, :, None].expand(-1, -1, D_bc), all_body_count)
    out_subs.scatter_(
        1, target.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2), all_subs)
    out_parents.scatter_(1, target, all_parents)
    out_has_new.scatter_(1, target, all_has_new)
    out_cur_rule_idx.scatter_(1, target, all_current_rule_idx)

    counts = all_valid.sum(dim=1).clamp(max=S_out)
    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.unsqueeze(1)

    return _Packed(out_grounding_body[:, :S_out], out_goals[:, :S_out],
                   out_rule_idx[:, :S_out], out_valid, out_body_count[:, :S_out],
                   out_parents[:, :S_out], out_subs[:, :S_out],
                   out_has_new[:, :S_out], out_cur_rule_idx[:, :S_out])


def pack_states_flat(
    flat_resolved,             # FlatResolvedChildren
    top_rule_idx: Tensor,          # [B, G]
    grounding_body: Tensor,    # [B, G, M_work, 3]
    body_count: Tensor,        # [B, G, D] or [B, G]
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
    dedup: bool = True,
    subs_noop: bool = True,
    S_cap: Optional[int] = None,
) -> _Packed:
    """Flat pack: polynomial-hash dedup, dynamic G_out, positions via cumcount.

    ``S_cap`` (sld/rtf flat path) caps G_out at the fixed dense width and truncates
    the per-batch tail in concat order to match dense pack_states' [:G] slice.
    """
    B = flat_resolved.B
    pad = padding_idx
    dev = flat_resolved.flat_child_goals.device

    flat_goals = flat_resolved.flat_child_goals  # [T, L, 3]
    flat_rule_idx = flat_resolved.flat_rule_idx     # [T]
    flat_b = flat_resolved.flat_batch_idx       # [T]
    flat_s = flat_resolved.flat_state_idx       # [T]
    T = flat_goals.size(0)
    G = flat_goals.size(1)
    M_work = grounding_body.shape[2]

    D_bc = body_count.shape[2]                          # body_count always 3-D
    if T == 0:
        S_out = 1
        out_valid = torch.zeros(B, S_out, dtype=torch.bool, device=dev)
        out_goals = torch.full((B, S_out, G, 3), pad, dtype=torch.long, device=dev)
        out_grounding_body = torch.full((B, S_out, M_work, 3), pad, dtype=torch.long, device=dev)
        out_rule_idx = torch.zeros(B, S_out, dtype=torch.long, device=dev)
        out_body_count = torch.zeros(B, S_out, D_bc, dtype=torch.long, device=dev)
        out_parents = torch.zeros(B, S_out, dtype=torch.long, device=dev)
        out_subs = torch.full((B, S_out, 2, 2), pad, dtype=torch.long, device=dev)
        out_has_new = torch.zeros(B, S_out, dtype=torch.bool, device=dev)
        out_cur_rule_idx = torch.full((B, S_out), -1, dtype=torch.long, device=dev)
        return _Packed(out_grounding_body, out_goals, out_rule_idx, out_valid,
                       out_body_count, out_parents, out_subs, out_has_new,
                       out_cur_rule_idx)

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
        flat_rule_idx = flat_rule_idx[keep]
        flat_b = flat_b[keep]
        flat_s = flat_s[keep]
        T = flat_goals.size(0)

    counts = torch.bincount(flat_b, minlength=B)

    pos = cumcount_flat(flat_b)

    # SLD/RTF provenance fields (aligned to flat_goals; dedup is OFF here).
    if not subs_noop:
        flat_is_fact = flat_resolved.flat_is_fact
        flat_top = flat_resolved.flat_top_rule_idx
        flat_subs = flat_resolved.flat_subs
    else:
        flat_is_fact = flat_top = flat_subs = None

    if S_cap is not None:
        S_out = min(max(int(counts.max().item()), 1), int(S_cap))
        keep = pos < S_out                              # truncate per-batch tail in concat order
        flat_goals = flat_goals[keep]
        flat_rule_idx = flat_rule_idx[keep]
        flat_b = flat_b[keep]
        flat_s = flat_s[keep]
        pos = pos[keep]
        if not subs_noop:
            flat_is_fact = flat_is_fact[keep]
            flat_top = flat_top[keep]
            flat_subs = flat_subs[keep]
        T = flat_goals.size(0)
    else:
        S_out = max(int(counts.max().item()), 1)

    out_goals = torch.full((B, S_out, G, 3), pad, dtype=torch.long, device=dev)
    out_grounding_body = torch.full((B, S_out, M_work, 3), pad, dtype=torch.long, device=dev)
    out_rule_idx = torch.zeros(B, S_out, dtype=torch.long, device=dev)
    out_body_count = torch.zeros(B, S_out, D_bc, dtype=torch.long, device=dev)
    out_subs = torch.full((B, S_out, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out, dtype=torch.bool, device=dev)
    out_cur_rule_idx = torch.full((B, S_out), -1, dtype=torch.long, device=dev)

    out_goals[flat_b, pos] = flat_goals
    new_body = flat_goals[:, :M_rule, :]
    out_body_count[flat_b, pos] = body_count[flat_b, flat_s]
    out_parents[flat_b, pos] = flat_s
    if subs_noop:                                   # ENUM — verbatim current writes
        out_grounding_body[flat_b, pos] = new_body
        out_rule_idx[flat_b, pos] = flat_rule_idx
        out_has_new[flat_b, pos] = True
        out_cur_rule_idx[flat_b, pos] = flat_rule_idx
    else:                                           # SLD/RTF — dense-equivalent provenance
        is_fact = flat_is_fact                       # [T] bool (dedup OFF -> aligned to pos)
        top = flat_top                               # [T] long
        first = (top == -1)
        eff_rule_idx = torch.where(is_fact, top, torch.where(first, flat_rule_idx, top))
        out_rule_idx[flat_b, pos] = eff_rule_idx
        out_has_new[flat_b, pos] = ~is_fact
        out_cur_rule_idx[flat_b, pos] = torch.where(
            is_fact, torch.full_like(flat_rule_idx, -1), flat_rule_idx)
        out_grounding_body[flat_b, pos] = torch.where(
            is_fact.view(-1, 1, 1), torch.full_like(new_body, pad), new_body)
        out_subs[flat_b, pos] = flat_subs

    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.clamp(max=S_out).unsqueeze(1)

    return _Packed(out_grounding_body, out_goals, out_rule_idx, out_valid,
                   out_body_count, out_parents, out_subs, out_has_new,
                   out_cur_rule_idx)


__all__ = ["compact_atoms", "pack_states", "pack_states_flat", "_Packed", "_pow_desc"]
