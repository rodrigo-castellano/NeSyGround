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
) -> Tensor:
    """Cross-query Kleene T_P fixed-point on collected groundings.

    A grounding is proved if all its active body atoms are base facts or
    heads of other proved groundings *anywhere in the batch*.

    Args:
        body:        [B, N, M, 3] collected grounding body atoms.
        mask:        [B, N] validity mask.
        queries:     [B, 3] original query atoms (head of each grounding).
        fact_index:  FactIndex with .exists() method.
        pack_base:   hash packing base.
        padding_idx: padding value.
        depth:       number of fixed-point iterations.

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
    q_hash = (queries[:, 0].long() * (pb * pb)
              + queries[:, 1].long() * pb
              + queries[:, 2].long())                  # [B]
    head_hashes = q_hash.unsqueeze(1).expand(-1, N)   # [B, N]

    # --- Fact check ---
    is_fact = fact_index.exists(
        body.reshape(-1, 3)).view(B, N, M)            # [B, N, M]
    body_active = body[..., 0] != padding_idx         # [B, N, M]

    # --- Seed: all active body atoms are facts ---
    proved = (is_fact | ~body_active).all(dim=-1) & mask  # [B, N]

    # --- Fixed-point iterations ---
    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)

    for _ in range(depth + 1):
        # Collect proved heads across ENTIRE batch into 1D pool
        all_heads = head_hashes.reshape(B * N)           # [B*N]
        all_proved = proved.reshape(B * N)                # [B*N]
        proved_pool = torch.where(all_proved, all_heads, sentinel.expand_as(all_heads))
        proved_pool_sorted, _ = proved_pool.sort()        # [B*N]

        # Check each body atom against the global pool
        flat_body = body_hashes.reshape(B * N * M)        # [B*N*M]
        pos = torch.searchsorted(proved_pool_sorted, flat_body)
        pos = pos.clamp(max=B * N - 1)
        found = proved_pool_sorted[pos] == flat_body
        in_proved = found.view(B, N, M)                   # [B, N, M]

        atom_ok = is_fact | in_proved | ~body_active
        new_proved = atom_ok.all(dim=-1) & mask

        # Early exit if converged
        if (new_proved == proved).all():
            break
        proved = new_proved

    return proved
