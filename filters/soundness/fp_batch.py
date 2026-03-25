"""Cross-query Kleene T_P fixed-point filter for transitive provability.

Iteratively marks groundings as proved when all their body atoms are either
base facts or heads of already-proved groundings *across the entire batch*.

Algorithm:
  1. Hash body atoms and head atoms for all groundings.
  2. Check which body atoms are base facts.
  3. Seed: groundings where all active body atoms are facts.
  4. Fixed-point (bounded by depth+1 iterations):
     a. Collect proved head hashes from ALL batch elements into a global pool.
     b. Sort the pool for binary search.
     c. For each body atom, check membership in the global pool.
     d. Mark groundings as proved if all body atoms are facts or in pool.
  5. Return proved mask.

Soundness guarantee:
  - Iteration 0: only atoms with all-fact proofs are marked.
  - Iteration k: only atoms whose dependencies proved at k-1.
  - Circular deps (p<-q, q<-p) never bootstrap → both stay False.
  - Converges to the minimal Herbrand model restricted to collected groundings.
"""

from __future__ import annotations

import torch
from torch import Tensor


def apply_fp_batch(
    body: Tensor,           # [B, N, M, 3]
    mask: Tensor,           # [B, N] bool
    queries: Tensor,        # [B, 3]
    fact_index,             # FactIndex with .exists()
    pack_base: int,
    padding_idx: int,
    depth: int,
    grounding_heads: Tensor = None,  # [B, N, D, 3] per-grounding heads
) -> Tensor:
    """Cross-query Kleene T_P fixed-point on collected groundings.

    A grounding is proved if all its active body atoms are base facts or
    heads of other proved groundings *anywhere in the batch*.

    Args:
        body:              [B, N, M, 3] collected grounding body atoms.
        mask:              [B, N] validity mask.
        queries:           [B, 3] original query atoms (head of each grounding).
        fact_index:        FactIndex with .exists() method.
        pack_base:         hash packing base.
        padding_idx:       padding value.
        depth:             number of fixed-point iterations.
        grounding_heads:   [B, N, D, 3] per-depth head atoms (optional).
            When provided, each grounding contributes D heads (one per depth)
            to the proved pool, enabling cross-proof transitive closure for
            intermediate groundings collected in 'grounded' mode.

    Returns:
        [B, N] bool — which groundings are transitively proved.
    """
    B, N, M, _ = body.shape
    pb = pack_base
    dev = body.device

    # --- Hashes ---
    body_hashes = (body[..., 0].long() * (pb * pb)
                   + body[..., 1].long() * pb
                   + body[..., 2].long())             # [B, N, M]

    if grounding_heads is not None:
        # Per-grounding, per-depth heads: [B, N, D, 3] → [B, N*D] hashes
        D_h = grounding_heads.shape[2]
        gh = grounding_heads.long()
        all_head_hashes = (gh[..., 0] * (pb * pb) + gh[..., 1] * pb
                           + gh[..., 2])              # [B, N, D]
        # Valid heads: not padding
        head_valid = grounding_heads[..., 0] != padding_idx  # [B, N, D]
        # Flatten: each grounding contributes D head entries
        head_hashes_flat = all_head_hashes.reshape(B, N * D_h)  # [B, N*D]
        head_valid_flat = head_valid.reshape(B, N * D_h)        # [B, N*D]
        # Expand mask: grounding must be valid for its heads to count
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D_h).reshape(B, N * D_h)
        head_valid_flat = head_valid_flat & mask_expanded
        N_pool = N * D_h
    else:
        q_hash = (queries[:, 0].long() * (pb * pb)
                  + queries[:, 1].long() * pb
                  + queries[:, 2].long())              # [B]
        head_hashes_flat = q_hash.unsqueeze(1).expand(-1, N)  # [B, N]
        head_valid_flat = mask  # [B, N]
        N_pool = N

    # --- Per-depth virtual grounding mode ---
    if grounding_heads is not None:
        # Treat each (grounding, depth) as a separate virtual grounding.
        # Virtual grounding v=(c,d): body=body_structured[c,d,:M,:],
        # head=grounding_heads[c,d,:]. This matches keras's per-rule-application
        # representation.
        #
        # body is [B, N, D*M, 3] flat — reshape to [B, N, D, M_per, 3]
        M_per = M // D_h  # M atoms per depth (D*M_per = M)
        body_structured = body.reshape(B, N, D_h, M_per, 3)

        # Flatten to virtual groundings: [B, N*D, M_per, 3]
        V = N * D_h
        vbody = body_structured.reshape(B, V, M_per, 3)
        vhead = grounding_heads.reshape(B, V, 3)  # [B, V, 3]

        # Virtual mask: grounding valid AND depth active
        vmask_base = mask.unsqueeze(-1).expand(-1, -1, D_h)
        vhead_active = grounding_heads[..., 0] != padding_idx
        vmask = (vmask_base & vhead_active).reshape(B, V)  # [B, V]

        # Hashes
        vbody_h = (vbody[..., 0].long() * (pb * pb)
                   + vbody[..., 1].long() * pb
                   + vbody[..., 2].long())  # [B, V, M_per]
        vhead_h = (vhead[..., 0].long() * (pb * pb)
                   + vhead[..., 1].long() * pb
                   + vhead[..., 2].long())  # [B, V]

        # Fact check per virtual grounding
        vis_fact = fact_index.exists(
            vbody.reshape(-1, 3)).view(B, V, M_per)
        vbody_active = vbody[..., 0] != padding_idx

        # Seed
        vproved = (vis_fact | ~vbody_active).all(dim=-1) & vmask

        sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
        for _ in range(depth + 1):
            pool = torch.where(vproved, vhead_h, sentinel.expand(B, V))
            pool_sorted, _ = pool.reshape(B * V).sort()
            flat_vb = vbody_h.reshape(B * V * M_per)
            pos = torch.searchsorted(pool_sorted, flat_vb)
            pos = pos.clamp(max=B * V - 1)
            found = pool_sorted[pos] == flat_vb
            in_pool = found.view(B, V, M_per)
            vproved = (vis_fact | in_pool | ~vbody_active).all(dim=-1) & vmask

        # A grounding is proved if ALL its active depths are proved
        vproved_per_depth = vproved.reshape(B, N, D_h)
        depth_inactive = ~vhead_active  # inactive depths don't block
        proved = (vproved_per_depth | depth_inactive).all(dim=-1) & mask
        return proved

    # --- Standard mode (flat body, query = head) ---
    is_fact = fact_index.exists(
        body.reshape(-1, 3)).view(B, N, M)
    body_active = body[..., 0] != padding_idx

    proved = (is_fact | ~body_active).all(dim=-1) & mask

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    for _ in range(depth + 1):
        all_heads = head_hashes_flat.reshape(B * N_pool)
        all_valid = proved.reshape(B * N_pool)
        proved_pool = torch.where(
            all_valid, all_heads, sentinel.expand(B * N_pool))
        proved_pool_sorted, _ = proved_pool.sort()

        flat_body = body_hashes.reshape(B * N * M)
        pos = torch.searchsorted(proved_pool_sorted, flat_body)
        pos = pos.clamp(max=B * N_pool - 1)
        found = proved_pool_sorted[pos] == flat_body
        in_proved = found.view(B, N, M)

        atom_ok = is_fact | in_proved | ~body_active
        proved = atom_ok.all(dim=-1) & mask

    return proved
