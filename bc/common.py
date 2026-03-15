"""BC shared utilities: packing, compaction, grounding collection, deduplication.

Provides scatter-based compaction of derived states into fixed-shape output
tensors, ground-fact pruning, and grounding collection.

All functions are pure (no class dependencies) and torch.compile compatible.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.fact_index import fact_contains


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
    fact_gbody: Tensor,        # [B, S, K_f, M, 3]
    fact_success: Tensor,      # [B, S, K_f]
    rule_goals: Tensor,        # [B, S, K_r, G, 3]
    rule_gbody: Tensor,        # [B, S, K_r, M, 3]
    rule_success: Tensor,      # [B, S, K_r]
    sub_rule_idx: Tensor,      # [B, S, K_r]
    top_ridx: Tensor,          # [B, S]
    grounding_body: Tensor,    # [B, S, M, 3]
    S_out: int,
    padding_idx: int,
    track_grounding_body: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack resolution children into compacted proof states.

    Single function: flatten → propagate gbody/ridx → compact to S_out.

    Grounding body propagation:
      - First resolution (top_ridx == -1): child gbody = rule body atoms.
      - Later resolutions: child gbody = parent's gbody (preserved).

    Args:
        fact_goals..sub_rule_idx: 7-tensor resolved output from resolution.
        top_ridx:       [B, S] parent rule indices.
        grounding_body: [B, S, M, 3] parent grounding bodies.
        S_out:          output state budget.
        padding_idx:    padding value.

    Returns:
        grounding_body: [B, S_out, M, 3]
        proof_goals:    [B, S_out, G, 3]
        top_ridx:       [B, S_out]
        state_valid:    [B, S_out]
    """
    B, S_in = top_ridx.shape
    K_f = fact_goals.shape[2]
    K_r = rule_goals.shape[2]
    M = grounding_body.shape[2]
    G = rule_goals.shape[3]
    pad = padding_idx
    dev = top_ridx.device

    n_f = S_in * K_f
    n_r = S_in * K_r

    # ── Fact children: flatten, inherit parent ridx ──
    if K_f > 0:
        f_gbody = fact_gbody.reshape(B, n_f, M, 3)
        f_goals = fact_goals.reshape(B, n_f, G, 3)
        f_valid = fact_success.reshape(B, n_f)
        f_ridx = top_ridx.unsqueeze(2).expand(
            B, S_in, K_f).reshape(B, n_f)
        if track_grounding_body:
            # Skip facts when grounding_body not yet established
            uninit = (grounding_body[:, :, 0, 0] == pad)
            f_valid = f_valid & ~uninit.unsqueeze(-1).expand(
                B, S_in, K_f).reshape(B, n_f)
    else:
        f_gbody = torch.zeros(B, 0, M, 3, dtype=torch.long, device=dev)
        f_goals = torch.full((B, 0, G, 3), pad, dtype=torch.long, device=dev)
        f_valid = torch.zeros(B, 0, dtype=torch.bool, device=dev)
        f_ridx = torch.zeros(B, 0, dtype=torch.long, device=dev)

    # ── Rule children: flatten, propagate gbody and ridx ──
    first = (top_ridx == -1).unsqueeze(2).expand(
        B, S_in, K_r).reshape(B, n_r)              # [B, n_r] first resolution?

    if track_grounding_body:
        r_gbody = torch.where(
            first.unsqueeze(-1).unsqueeze(-1).expand(B, n_r, M, 3),
            rule_goals[:, :, :, :M, :].reshape(B, n_r, M, 3),  # body atoms
            rule_gbody.reshape(B, n_r, M, 3),                    # parent's
        )
    else:
        r_gbody = torch.zeros(B, n_r, M, 3, dtype=torch.long, device=dev)

    r_ridx = torch.where(
        first,
        sub_rule_idx.reshape(B, n_r),              # new rule index
        top_ridx.unsqueeze(2).expand(
            B, S_in, K_r).reshape(B, n_r),          # parent's
    )
    r_goals = rule_goals.reshape(B, n_r, G, 3)
    r_valid = rule_success.reshape(B, n_r)

    # ── Concatenate all children ──
    all_gbody = torch.cat([f_gbody, r_gbody], dim=1)   # [B, N, M, 3]
    all_goals = torch.cat([f_goals, r_goals], dim=1)    # [B, N, G, 3]
    all_valid = torch.cat([f_valid, r_valid], dim=1)    # [B, N]
    all_ridx = torch.cat([f_ridx, r_ridx], dim=1)      # [B, N]

    # ── Scatter-compact to S_out ──
    cumsum = all_valid.long().cumsum(dim=1)
    target = torch.where(
        all_valid, cumsum - 1,
        torch.tensor(S_out, dtype=torch.long, device=dev),
    ).clamp(min=0, max=S_out)

    out_gbody = torch.zeros(
        B, S_out + 1, M, 3, dtype=torch.long, device=dev)
    out_goals = torch.full(
        (B, S_out + 1, G, 3), pad, dtype=torch.long, device=dev)
    out_ridx = torch.zeros(B, S_out + 1, dtype=torch.long, device=dev)

    ti = target.unsqueeze(-1).unsqueeze(-1)
    out_gbody.scatter_(1, ti.expand(-1, -1, M, 3), all_gbody)
    out_goals.scatter_(1, ti.expand(-1, -1, G, 3), all_goals)
    out_ridx.scatter_(1, target, all_ridx)

    counts = all_valid.sum(dim=1).clamp(max=S_out)
    out_valid = torch.arange(S_out, device=dev).unsqueeze(0) < counts.unsqueeze(1)

    return (out_gbody[:, :S_out], out_goals[:, :S_out],
            out_ridx[:, :S_out], out_valid)


# ---------------------------------------------------------------------------
# Post-processing: ground-fact pruning, grounding collection (from postprocessing.py)
# ---------------------------------------------------------------------------

def prune_ground_facts(
    candidates: Tensor,         # [..., M, 3] — any leading dims (e.g. [B,K,M,3])
    valid_mask: Tensor,         # [...] matching leading dims
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: Optional[int] = None,
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Remove known ground facts from candidates (fixed shape, fully vectorized).

    Works on any 4D state tensor [..., M, 3] (e.g. [B,K,M,3] or [B,S,G,3]).
    Also treats True predicate atoms as "facts" (proof indicators).

    Args:
        candidates: [..., M, 3] candidate derived states
        valid_mask: [...] which candidates are valid
        fact_hashes: [F] sorted int64 fact hashes for membership testing
        pack_base: packing base for hash computation
        constant_no: highest constant index
        padding_idx: padding value
        true_pred_idx: predicate index for True atoms (treated as resolved)
        excluded_queries: [B, 1, 3] atoms NOT to prune (cycle prevention)

    Returns:
        pruned_states: [..., M, 3] with facts removed
        pruned_counts: [...] new atom counts per candidate
        is_proof: [...] whether candidate became empty (proof found)
    """
    B, K, M, _ = candidates.shape
    pad = padding_idx

    # Check which atoms are ground facts
    preds = candidates[:, :, :, 0]               # [B, K, M]
    args = candidates[:, :, :, 1:3]              # [B, K, M, 2]

    valid_atom = (preds != pad)                  # [B, K, M]
    is_ground = (args <= constant_no).all(dim=-1)  # [B, K, M]
    ground_atoms = valid_atom & is_ground        # [B, K, M]

    # Check ALL atoms against fact hashes (fully vectorized)
    flat_atoms = candidates.reshape(-1, 3)        # [B*K*M, 3]
    is_fact_flat = fact_contains(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.reshape(B, K, M)

    # Only mark as fact if it was actually a ground atom
    is_fact = is_fact & ground_atoms

    # Handle Exclusion: Keep excluded atoms (don't prune them)
    if excluded_queries is not None:
        excl_first = excluded_queries[:, 0, :]  # [B, 3]
        excl_exp = excl_first.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 3]
        is_excluded_atom = (candidates == excl_exp).all(dim=-1) & ground_atoms
        is_fact = is_fact & ~is_excluded_atom

    # Also treat True predicate atoms as "facts" (proof indicators)
    if true_pred_idx is not None:
        is_true_pred = (preds == true_pred_idx)  # [B, K, M]
        is_fact = is_fact | is_true_pred

    # Atoms to keep: valid AND NOT a known fact
    keep_atom = valid_atom & ~is_fact  # [B, K, M]

    # Compute new counts
    pruned_counts = keep_atom.sum(dim=-1)  # [B, K]

    # Detect proofs: candidate with zero remaining atoms
    is_proof = (pruned_counts == 0) & valid_mask

    # Mask out removed atoms (gaps remain, downstream handles via compact_atoms)
    pad_t = torch.tensor(pad, dtype=candidates.dtype, device=candidates.device)
    pruned_states = torch.where(
        keep_atom.unsqueeze(-1),
        candidates,
        pad_t,
    )

    return pruned_states, pruned_counts, is_proof


def collect_groundings(
    grounding_body: Tensor,     # [B, S, M, 3]
    proof_goals: Tensor,        # [B, S, G, 3]
    state_valid: Tensor,        # [B, S]
    top_ridx: Tensor,           # [B, S]
    collected_body: Tensor,     # [B, tG, M, 3]
    collected_mask: Tensor,     # [B, tG]
    collected_ridx: Tensor,     # [B, tG]
    constant_no: int,
    pad_idx: int,
    effective_total_G: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collect completed groundings into output buffer (TS-specific).

    A state is terminal when all proof goals are padding (resolved).
    Ground facts are already removed by prune_ground_facts_exists,
    and atoms are left-aligned by compact_atoms.

    Args:
        grounding_body: [B, S, M, 3] current grounding bodies
        proof_goals: [B, S, G, 3] current proof goals (after pruning + compaction)
        state_valid: [B, S] active state mask
        top_ridx: [B, S] rule index per state
        collected_body: [B, tG, M, 3] accumulated grounding bodies
        collected_mask: [B, tG] accumulated validity mask
        collected_ridx: [B, tG] accumulated rule indices
        constant_no: highest constant index
        pad_idx: padding value
        effective_total_G: max number of collected groundings (tG)

    Returns:
        out_body:    [B, tG, M, 3] updated collected bodies
        out_mask:    [B, tG] updated collected mask
        out_ridx:    [B, tG] updated collected rule indices
        state_valid: [B, S] updated (terminal states deactivated)
    """
    B, S, M, _ = grounding_body.shape
    dev = grounding_body.device
    E = constant_no + 1
    tG = effective_total_G

    is_padding = (proof_goals[:, :, :, 0] == pad_idx)  # [B, S, G]

    # Terminal = all goals are padding (all resolved)
    all_goals_done = is_padding.all(dim=2)

    body_args = grounding_body[:, :, :, 1:3]
    is_ground = (body_args < E).all(dim=-1).all(dim=-1)

    valid_grounding = all_goals_done & is_ground & state_valid

    n_new = S
    body_new = grounding_body
    ridx_new = top_ridx

    n_cat = tG + n_new
    cb = torch.cat([collected_body, body_new], dim=1)
    cm = torch.cat([collected_mask, valid_grounding], dim=1)
    cr = torch.cat([collected_ridx, ridx_new], dim=1)

    cm = _dedup_groundings(cb, cr, cm, M)

    n_k = min(tG, n_cat)
    _, ki = cm.to(torch.int8).topk(
        n_k, dim=1, largest=True, sorted=False)

    if n_k < tG:
        p2 = tG - n_k
        out_body = torch.nn.functional.pad(
            cb.gather(1, ki.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, M, 3)), (0, 0, 0, 0, 0, p2))
        out_mask = torch.nn.functional.pad(
            cm.gather(1, ki), (0, p2))
        out_ridx = torch.nn.functional.pad(
            cr.gather(1, ki), (0, p2))
    else:
        out_body = cb.gather(
            1, ki.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
        out_mask = cm.gather(1, ki)
        out_ridx = cr.gather(1, ki)

    state_valid = state_valid & ~valid_grounding

    return out_body, out_mask, out_ridx, state_valid


def _dedup_groundings(
    body: Tensor,       # [B, N, M, 3]
    ridx: Tensor,       # [B, N]
    mask: Tensor,       # [B, N]
    M: int,
) -> Tensor:
    """Remove duplicate groundings based on (ridx, body) hash.

    Args:
        body: [B, N, M, 3] grounding body atoms
        ridx: [B, N] rule index per grounding
        mask: [B, N] validity mask
        M: number of body atoms

    Returns:
        mask: [B, N] updated mask with duplicates removed
    """
    B, N = mask.shape
    dev = mask.device
    # Vectorized prime-mixing hash (overflow-safe, no E*E needed)
    P1, P2, P3, P4 = 1_000_003, 999_983, 999_979, 999_961
    atom_hashes = (body[..., 0].long() * P1
                   + body[..., 1].long() * P2
                   + body[..., 2].long() * P3)               # [B, N, M]
    powers = P4 ** torch.arange(M - 1, -1, -1, device=dev)   # [M]
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
