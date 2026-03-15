"""PruneIncompleteProofs — fixed-point filter for transitive provability.

Iteratively marks groundings as proved when all their body atoms are either
base facts or heads of already-proved groundings.

Algorithm:
  1. Initially proved: all active body atoms are base facts.
  2. Fixed-point: head hashes of proved groundings → sort →
     binary-search each body hash → mark newly proved.
  3. Repeat (bounded by depth+1 iterations).
"""

from __future__ import annotations

import torch
from torch import Tensor


def apply_prune(
    body: Tensor,           # [B, N, M, 3]
    mask: Tensor,           # [B, N] bool
    queries: Tensor,        # [B, 3]
    fact_index,             # FactIndex with .exists()
    pack_base: int,
    padding_idx: int,
    depth: int,
) -> Tensor:
    """PruneIncompleteProofs on collected groundings.

    Fixed-point: a grounding is proved if all its active body atoms are
    base facts or heads of other proved groundings in the batch.

    Args:
        body:        [B, N, M, 3] collected grounding body atoms.
        mask:        [B, N] validity mask.
        queries:     [B, 3] original query atoms (head hash source).
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

    # Compute hashes and fact existence
    body_hashes = (body[..., 0].long() * (pb * pb)
                   + body[..., 1].long() * pb
                   + body[..., 2].long())             # [B, N, M]
    is_fact = fact_index.exists(
        body.reshape(-1, 3)).view(B, N, M)            # [B, N, M]
    body_active = body[..., 0] != padding_idx         # [B, N, M]

    q_hash = (queries[:, 0].long() * (pb * pb)
              + queries[:, 1].long() * pb
              + queries[:, 2].long())                  # [B]
    head_hash = q_hash.unsqueeze(1).expand(-1, N)     # [B, N]

    # Initially proved: ALL active body atoms are base facts
    proved = (is_fact | ~body_active).all(dim=-1) & mask  # [B, N]

    # Fixed-point iterations
    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    body_hashes_flat = body_hashes.reshape(B, N * M)

    for _ in range(depth + 1):
        proved_h = torch.where(proved, head_hash, sentinel.expand_as(head_hash))
        proved_sorted, _ = proved_h.sort(dim=1)

        pos = torch.searchsorted(proved_sorted, body_hashes_flat)
        pos = pos.clamp(max=N - 1)
        found = proved_sorted.gather(1, pos) == body_hashes_flat
        body_in_proved = found.view(B, N, M)

        atom_ok = is_fact | body_in_proved | ~body_active
        proved = atom_ok.all(dim=-1) & mask

    return proved
