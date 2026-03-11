"""Post-processing: ground-fact pruning, grounding collection, deduplication.

Provides both BE-style (hash-based) and TS-style (index-based) pruning,
plus TS-specific grounding collection and deduplication.

Depends on ``fact_index.fact_contains`` for hash-based membership tests
and ``packing.compact_atoms`` for atom compaction.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from grounder.fact_index import fact_contains


# ---------------------------------------------------------------------------
# BE-style prune_ground_facts (hash-based, from BE _postprocessing.py)
# ---------------------------------------------------------------------------

def prune_ground_facts(
    candidates: Tensor,         # [B, K, M, 3]
    valid_mask: Tensor,         # [B, K]
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: Optional[int] = None,
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Remove known ground facts from candidates (fixed shape, fully vectorized).

    Also treats True predicate atoms as "facts" (proof indicators).

    Args:
        candidates: [B, K, M, 3] candidate derived states
        valid_mask: [B, K] which candidates are valid
        fact_hashes: [F] sorted int64 fact hashes for membership testing
        pack_base: packing base for hash computation
        constant_no: highest constant index
        padding_idx: padding value
        true_pred_idx: predicate index for True atoms (treated as resolved)
        excluded_queries: [B, 1, 3] atoms NOT to prune (cycle prevention)

    Returns:
        pruned_states: [B, K, M, 3] with facts removed
        pruned_counts: [B, K] new atom counts per candidate
        is_proof: [B, K] whether candidate became empty (proof found)
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
    flat_atoms = candidates.view(-1, 3)          # [B*K*M, 3]
    is_fact_flat = fact_contains(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.view(B, K, M)

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


def prune_ground_facts_3d(
    proof_goals: Tensor,        # [B, S, G, 3]
    state_valid: Tensor,        # [B, S]
    fact_hashes: Tensor,        # [F] sorted fact hashes
    pack_base: int,
    constant_no: int,
    padding_idx: int,
) -> Tensor:
    """Remove known ground facts from proof goals (hash-based, 3D variant).

    Unlike ``prune_ground_facts`` which returns counts/is_proof for [B, K, M, 3],
    this simpler variant works on [B, S, G, 3] and just replaces known facts
    with padding.  Call compact_atoms afterwards to left-align.

    Args:
        proof_goals: [B, S, G, 3] proof-goal states
        state_valid: [B, S] validity mask (unused but kept for API symmetry)
        fact_hashes: [F] sorted int64 fact hashes
        pack_base:   multiplier for pack_triples_64
        constant_no: highest constant index
        padding_idx: padding index

    Returns:
        [B, S, G, 3] with known ground facts replaced by padding
    """
    B, S, G, _ = proof_goals.shape
    pad = padding_idx

    preds = proof_goals[:, :, :, 0]
    valid_atom = (preds != pad)
    is_ground = (
        (proof_goals[:, :, :, 1] <= constant_no)
        & (proof_goals[:, :, :, 2] <= constant_no)
    )
    ground_atoms = valid_atom & is_ground

    flat_atoms = proof_goals.reshape(-1, 3)
    is_fact_flat = fact_contains(flat_atoms, fact_hashes, pack_base)
    is_fact = is_fact_flat.view(B, S, G) & ground_atoms

    keep = valid_atom & ~is_fact
    return torch.where(
        keep.unsqueeze(-1).expand_as(proof_goals),
        proof_goals,
        torch.tensor(pad, dtype=torch.long, device=proof_goals.device),
    )


# ---------------------------------------------------------------------------
# TS-style prune_ground_facts (index-based, from TS _prune_ground_facts)
# ---------------------------------------------------------------------------

def prune_ground_facts_exists(
    proof_goals: Tensor,        # [B, S, G, 3]
    state_valid: Tensor,        # [B, S]
    exists_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    constant_no: int,
    pad_idx: int,
) -> Tensor:
    """Remove known ground facts from proof_goals using an exists_fn callable.

    TS variant that uses fact_index.exists(pred, subj, obj) -> bool instead of
    hash-based membership.  Call compact_atoms afterwards to left-align.

    Args:
        proof_goals: [B, S, G, 3] proof goal atoms
        state_valid: [B, S] which states are active
        exists_fn: callable(pred [N], subj [N], obj [N]) -> [N] bool
        constant_no: highest constant index
        pad_idx: padding value

    Returns:
        proof_goals: [B, S, G, 3] pruned (not yet left-aligned)
    """
    B, S, G, _ = proof_goals.shape
    E = constant_no + 1

    preds = proof_goals[:, :, :, 0]                      # [B, S, G]
    valid_atom = (preds != pad_idx)
    is_ground = (
        (proof_goals[:, :, :, 1] < E)
        & (proof_goals[:, :, :, 2] < E)
    )
    ground_atoms = valid_atom & is_ground                 # [B, S, G]

    # Vectorized existence check via the provided callable
    # Clamp to safe ranges for the index lookup
    num_preds = preds.max().item() + 1 if preds.numel() > 0 else 1
    safe_pred = preds.clamp(0, max(num_preds - 1, 0))
    safe_subj = proof_goals[:, :, :, 1].clamp(0, E - 1)
    safe_obj = proof_goals[:, :, :, 2].clamp(0, E - 1)
    is_fact = exists_fn(
        safe_pred.reshape(-1),
        safe_subj.reshape(-1),
        safe_obj.reshape(-1),
    ).view(B, S, G)

    is_known_fact = ground_atoms & is_fact

    # Replace known facts with padding
    keep = valid_atom & ~is_known_fact
    return torch.where(
        keep.unsqueeze(-1).expand_as(proof_goals),
        proof_goals,
        torch.tensor(pad_idx, dtype=torch.long, device=proof_goals.device),
    )


# ---------------------------------------------------------------------------
# TS grounding collection (from TS BCPrologStatic._collect_groundings)
# ---------------------------------------------------------------------------

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

    cm = dedup_groundings(cb, cr, cm, M, E)

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


# ---------------------------------------------------------------------------
# TS grounding deduplication (from TS BCPrologStatic._dedup_groundings)
# ---------------------------------------------------------------------------

def dedup_groundings(
    body: Tensor,       # [B, N, M, 3]
    ridx: Tensor,       # [B, N]
    mask: Tensor,       # [B, N]
    M: int,
    E: int,
) -> Tensor:
    """Remove duplicate groundings based on (ridx, body) hash.

    Args:
        body: [B, N, M, 3] grounding body atoms
        ridx: [B, N] rule index per grounding
        mask: [B, N] validity mask
        M: number of body atoms
        E: entity count (constant_no + 1), used as hash base

    Returns:
        mask: [B, N] updated mask with duplicates removed
    """
    B, N = mask.shape
    dev = mask.device
    PRIME = 1_000_003
    g_hash = ridx.long()
    for j in range(M):
        atom_h = (body[:, :, j, 0].long() * (E * E)
                  + body[:, :, j, 1].long() * E
                  + body[:, :, j, 2].long())
        g_hash = g_hash * PRIME + atom_h
    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    gh = torch.where(mask, g_hash, sentinel.expand(B, N))
    sorted_gh, sort_idx = gh.sort(dim=1)
    prev_gh = torch.nn.functional.pad(
        sorted_gh[:, :-1], (1, 0), value=-2)
    is_dup = (sorted_gh == prev_gh)
    inv_sort = sort_idx.argsort(dim=1)
    is_dup_orig = is_dup.gather(1, inv_sort)
    return mask & ~is_dup_orig
