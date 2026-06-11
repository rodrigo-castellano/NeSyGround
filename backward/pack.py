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


def compact_atoms(states: Tensor, padding_idx: int,
                  valid: Optional[Tensor] = None) -> Tensor:
    """Left-align non-padding atoms within each ``[..., M, 3]`` slice.

    Scatter-based: valid atoms write to their cumsum rank (unique by
    construction); padding atoms all write to one discarded trash slot ``M``
    (which write wins is irrelevant — the slot is sliced away, and the kept
    ``[:, :M]`` region is fully deterministic since valid targets are
    unique). Replaces the former stable-argsort+gather formulation, which
    cost ~7 ms/call on family-BC12 eval chunks (segmented argsort over a
    tiny M is a CUDA worst case; one scatter is ~10× cheaper). Kept slots
    differ from the old version only in padding rows' args (junk preserved
    by the gather, clean ``pad`` triples here) — consumers test
    ``atom[0] != pad`` only, and the 40-cell fingerprints are unchanged.

    ``valid`` (optional ``[..., M]`` bool) supplies the keep mask directly —
    the prune_ground_facts fusion: the caller skips materializing the
    hole-punched tensor (one full read+write round) and the mask compare here.
    Atoms with ``valid=False`` go to the trash slot regardless of content.
    """
    if states.numel() == 0:
        return states
    *leading, M, _ = states.shape
    flat = states.reshape(-1, M, 3)
    pad = padding_idx
    valid_atom = (flat[:, :, 0] != pad) if valid is None else valid.reshape(-1, M)
    pos = torch.cumsum(valid_atom, dim=1, dtype=torch.long) - 1
    tgt = torch.where(valid_atom, pos, M)
    out = flat.new_full((flat.shape[0], M + 1, 3), pad)
    out.scatter_(1, tgt.unsqueeze(-1).expand(-1, -1, 3), flat)
    return out[:, :M].reshape(*leading, M, 3)


def pack_states(
    resolved,                  # ResolvedChildren (leading layout field)
    top_rule_idx: Tensor,          # [B, G]
    grounding_body: Tensor,    # [B, G, M_work, 3]
    body_count: Tensor,        # [B, G, D]
    S_out: int,
    padding_idx: int,
    collect_evidence: bool = True,
    M_rule: int = 0,
    parent_goals: Optional[Tensor] = None,
) -> _Packed:
    """Dense pack: facts compact FIRST then rules (same concat-order slot layout as
    before, but each region scatters DIRECTLY into the output — no [B, n_f+n_r, …]
    concat copies) ; per-parent fields (rule_idx, body_count) are reconstructed by
    gathering through ``out_parents`` instead of materializing [B, n_r, …] sources.

    ``parent_goals`` ([B, S_in, G, 3], the pre-pack frontier goals): only read when
    ``rule_goals`` carries the BODY region alone (G_emit < G — the fused pbc-dense
    emit, see ``DenseMaterializer.emit``); the parent remaining-tail
    ``parent_goals[:, :, 1:1+n_rem]`` is then rebuilt by a parents_s gather,
    value-identical to emit's [B,S,K,G,3] tail broadcast + scatter."""
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
    G = fact_goals.shape[3]                             # output goal width (L)
    G_emit = rule_goals.shape[3]                        # == G, or M (fused body-only emit)
    D_bc = body_count.shape[2]                          # body_count always 3-D

    # ── per-region validity + concat-order compaction targets (facts FIRST) ──
    r_valid = rule_success.reshape(B, n_r)
    if K_f > 0:
        f_valid = fact_success.reshape(B, n_f)
        if collect_evidence:                            # skip facts on a started-but-non-initial state
            skip_fact = (body_count.sum(dim=-1) == 0) & ~(top_rule_idx == -1)
            f_valid = f_valid & ~skip_fact.unsqueeze(-1).expand(B, S_in, K_f).reshape(B, n_f)
        cs_f = f_valid.long().cumsum(dim=1)             # [B, n_f]
        n_valid_f = cs_f[:, -1:]                        # [B, 1]
        cs_r = r_valid.long().cumsum(dim=1) + n_valid_f
        target_f = torch.where(
            f_valid, cs_f - 1,
            torch.tensor(S_out, dtype=torch.long, device=dev)).clamp_(min=0, max=S_out)
    else:
        cs_r = r_valid.long().cumsum(dim=1)
    target_r = torch.where(
        r_valid, cs_r - 1,
        torch.tensor(S_out, dtype=torch.long, device=dev)).clamp_(min=0, max=S_out)
    counts = cs_r[:, -1].clamp(max=S_out)

    # ── output buffers (slot S_out is the discard slot for invalid/overflow rows) ──
    out_grounding_body = torch.full((B, S_out + 1, M_work, 3), pad, dtype=torch.long, device=dev)
    out_goals = torch.full((B, S_out + 1, G, 3), pad, dtype=torch.long, device=dev)
    out_subs = torch.full((B, S_out + 1, 2, 2), pad, dtype=torch.long, device=dev)
    out_parents = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)
    out_has_new = torch.zeros(B, S_out + 1, dtype=torch.bool, device=dev)
    out_cur_rule_idx = torch.full((B, S_out + 1), -1, dtype=torch.long, device=dev)

    # ── RULE region (always present); sources are views, never concat copies ──
    ti_r = target_r.unsqueeze(-1).unsqueeze(-1)
    out_goals[:, :, :G_emit].scatter_(
        1, ti_r.expand(-1, -1, G_emit, 3), rule_goals.reshape(B, n_r, G_emit, 3))
    out_subs.scatter_(1, ti_r.expand(-1, -1, 2, 2), rule_subs.reshape(B, n_r, 2, 2))
    r_parents = (torch.arange(S_in, device=dev).unsqueeze(1)
                 .expand(S_in, K_r).reshape(n_r).unsqueeze(0).expand(B, n_r))
    out_parents.scatter_(1, target_r, r_parents)
    out_cur_rule_idx.scatter_(1, target_r, sub_rule_idx.reshape(B, n_r))
    if collect_evidence:
        out_grounding_body.scatter_(
            1, ti_r.expand(-1, -1, M_rule, 3),
            rule_goals[:, :, :, :M_rule, :].reshape(B, n_r, M_rule, 3))
        out_has_new.scatter_(1, target_r, r_valid)
    # else: grounding_body stays pad, has_new stays False (== scattering pad/False)

    # ── FACT region (facts compact first; pad/False/-1 sources need no scatter) ──
    if K_f > 0:
        ti_f = target_f.unsqueeze(-1).unsqueeze(-1)
        out_goals.scatter_(1, ti_f.expand(-1, -1, G, 3), fact_goals.reshape(B, n_f, G, 3))
        out_subs.scatter_(1, ti_f.expand(-1, -1, 2, 2), fact_subs.reshape(B, n_f, 2, 2))
        f_parents = (torch.arange(S_in, device=dev).unsqueeze(1)
                     .expand(S_in, K_f).reshape(n_f).unsqueeze(0).expand(B, n_f))
        out_parents.scatter_(1, target_f, f_parents)
        # facts: cur_rule_idx == -1 (init), grounding_body == pad, has_new == False.
        f_neg1 = torch.full((1, 1), -1, dtype=torch.long, device=dev).expand(B, n_f)
        out_cur_rule_idx.scatter_(1, target_f, f_neg1)

    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.unsqueeze(1)
    parents_s = out_parents[:, :S_out]
    cur_s = out_cur_rule_idx[:, :S_out]

    if G_emit < G:
        # Fused body-only emit (pbc dense; no fact region): rebuild the parent
        # remaining-tail goals[1:1+n_rem] through parents_s. Valid slots get the
        # parent tail (== emit's per-K broadcast); invalid slots stay pad.
        assert parent_goals is not None and K_f == 0
        n_rem = min(G - G_emit, G - 1)
        if n_rem > 0:
            tail = parent_goals[:, :, 1:1 + n_rem, :].gather(
                1, parents_s.unsqueeze(-1).unsqueeze(-1).expand(B, S_out, n_rem, 3))
            pad_t = torch.full((), pad, dtype=torch.long, device=dev)
            out_goals[:, :S_out, G_emit:G_emit + n_rem, :] = torch.where(
                out_valid.unsqueeze(-1).unsqueeze(-1), tail, pad_t)

    # rule_idx per slot: facts inherit top_rule_idx[parent]; rules take
    # sub_rule_idx when the parent is first-step (top == -1), else top — for BOTH
    # cases this equals where(top[parent] == -1, cur_rule_idx, top[parent]).
    gathered_top = top_rule_idx.gather(1, parents_s)
    zero = torch.zeros((), dtype=torch.long, device=dev)
    out_rule_idx = torch.where(
        out_valid, torch.where(gathered_top == -1, cur_s, gathered_top), zero)
    # body_count is inherited from the parent for facts AND rules alike.
    out_body_count = torch.where(
        out_valid.unsqueeze(-1),
        body_count.gather(1, parents_s.unsqueeze(-1).expand(-1, -1, D_bc)), zero)

    return _Packed(out_grounding_body[:, :S_out], out_goals[:, :S_out],
                   out_rule_idx, out_valid, out_body_count,
                   parents_s, out_subs[:, :S_out],
                   out_has_new[:, :S_out], cur_s)


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
        # inverse permutation via scatter (not a second sort via .argsort())
        keep = torch.empty_like(is_dup)
        keep[sort_idx] = ~is_dup
        # one nonzero, then index_select per field (boolean indexing would
        # re-run nonzero for every field)
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        flat_goals = flat_goals[keep_idx]
        flat_rule_idx = flat_rule_idx[keep_idx]
        flat_b = flat_b[keep_idx]
        flat_s = flat_s[keep_idx]
        T = flat_goals.size(0)

    counts = torch.bincount(flat_b, minlength=B)

    # NOTE: an assume_sorted shortcut here (skipping cumcount's argsort when
    # subs_noop) was tried 2026-06-11 and REVERTED: the pbc flat emit is
    # (b, r, g)-sorted on the plain path, but the dedup_goals expansion
    # branch reorders flat_b — join_ab caught 48/250 spurious survivors on
    # countries_s3 while the fingerprint cells (which don't run that knob
    # combo) stayed green. The sortedness precondition is not carried
    # through the seams; the ~0.4 ms/step win does not justify it.
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
        # truncate per-batch tail in concat order; single nonzero + index_select
        keep_idx = torch.nonzero(pos < S_out, as_tuple=False).squeeze(1)
        flat_goals = flat_goals[keep_idx]
        flat_rule_idx = flat_rule_idx[keep_idx]
        flat_b = flat_b[keep_idx]
        flat_s = flat_s[keep_idx]
        pos = pos[keep_idx]
        if not subs_noop:
            flat_is_fact = flat_is_fact[keep_idx]
            flat_top = flat_top[keep_idx]
            flat_subs = flat_subs[keep_idx]
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

    # Single-kernel scatters on the flattened (b, slot) axis instead of
    # ``out[flat_b, pos] = …`` indexed writes: under torch deterministic mode
    # the eager int64 indexed write routes to the serialized
    # ``indexing_backward_kernel`` (~2×2.1 ms per train step measured); a
    # ``scatter_`` does not. Targets ``flat_b * S_out + pos`` are unique by
    # construction (``pos`` is the per-batch cumcount and ``pos < S_out`` on
    # both the capped and uncapped paths), so the result is byte-identical.
    tgt = flat_b * S_out + pos                       # [T] unique flattened slots
    t1 = tgt.view(-1, 1, 1)
    out_goals.view(B * S_out, G, 3).scatter_(0, t1.expand(-1, G, 3), flat_goals)
    new_body = flat_goals[:, :M_rule, :]
    out_body_count.view(B * S_out, D_bc).scatter_(
        0, tgt.unsqueeze(-1).expand(-1, D_bc), body_count[flat_b, flat_s])
    out_parents.view(-1).scatter_(0, tgt, flat_s)
    if subs_noop:                                   # ENUM — verbatim current writes
        out_grounding_body.view(B * S_out, M_work, 3).scatter_(
            0, t1.expand(-1, M_rule, 3), new_body)
        out_rule_idx.view(-1).scatter_(0, tgt, flat_rule_idx)
        out_has_new.view(-1).scatter_(0, tgt, True)
        out_cur_rule_idx.view(-1).scatter_(0, tgt, flat_rule_idx)
    else:                                           # SLD/RTF — dense-equivalent provenance
        is_fact = flat_is_fact                       # [T] bool (dedup OFF -> aligned to pos)
        top = flat_top                               # [T] long
        first = (top == -1)
        # truth-table fold of where(is_fact, top, where(first, rule, top))
        eff_rule_idx = torch.where(~is_fact & first, flat_rule_idx, top)
        out_rule_idx.view(-1).scatter_(0, tgt, eff_rule_idx)
        out_has_new.view(-1).scatter_(0, tgt, ~is_fact)
        out_cur_rule_idx.view(-1).scatter_(
            0, tgt, torch.where(is_fact, -1, flat_rule_idx))
        out_grounding_body.view(B * S_out, M_work, 3).scatter_(
            0, t1.expand(-1, M_rule, 3),
            torch.where(is_fact.view(-1, 1, 1), pad, new_body))
        out_subs.view(B * S_out, 2, 2).scatter_(0, t1.expand(-1, 2, 2), flat_subs)

    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.clamp(max=S_out).unsqueeze(1)

    return _Packed(out_grounding_body, out_goals, out_rule_idx, out_valid,
                   out_body_count, out_parents, out_subs, out_has_new,
                   out_cur_rule_idx)


__all__ = ["compact_atoms", "pack_states", "pack_states_flat", "_Packed", "_pow_desc"]
