"""GPU-based deduplication suite for BFS frontier management.

Pure tensor operations — no engine dependency. Ported from
kge_experiments/engines/generate_depths.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Position-weighted primes for hashing multi-atom states
_HASH_PRIMES = [
    2654435761, 2246822519, 3266489917, 668265263, 374761393,
    2147483587, 1073741789, 536870909, 268435399, 134217689,
    67108859, 33554393, 16777213, 8388593, 4194301,
    2097143, 1048573, 524287, 262139, 131071,
]


def hash_states(
    states: Tensor,         # [N, A, 3]
    owner: Tensor,          # [N]
    pack_base: int,
    padding_idx: int,
) -> Tensor:
    """Compute per-state hashes combining owner ID and atom content.

    Atoms are packed to int64, sorted per state for canonical ordering,
    then combined with position-weighted primes. Owner ID is mixed into
    upper bits so different queries' identical states get distinct hashes.

    Returns: [N] int64 hash keys.
    """
    N, A, _ = states.shape
    device = states.device

    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Pack each atom: (pred * base + arg0) * base + arg1
    p = states[:, :, 0].long()
    a = states[:, :, 1].long()
    b = states[:, :, 2].long()
    atom_keys = ((p * pack_base) + a) * pack_base + b

    # Padding atoms -> large sentinel so they sort to the end
    is_pad = (states[:, :, 0] == padding_idx)
    atom_keys = torch.where(
        is_pad,
        torch.tensor(2**62, dtype=torch.int64, device=device),
        atom_keys,
    )

    # Sort per state for canonical ordering
    atom_keys, _ = atom_keys.sort(dim=1)

    # Position-weighted hash
    n_primes = min(A, len(_HASH_PRIMES))
    primes = torch.tensor(_HASH_PRIMES[:n_primes], dtype=torch.int64, device=device)
    state_hash = (atom_keys[:, :n_primes] * primes.unsqueeze(0)).sum(dim=1)

    # Mix in remaining atoms if A > n_primes
    if A > n_primes:
        state_hash = state_hash + atom_keys[:, n_primes:].sum(dim=1) * 997

    # Combine with owner: shift owner into upper bits
    combined = (owner.long() << 44) | (state_hash & 0x0FFF_FFFF_FFFF)
    return combined


def dedup_within_depth(
    states: Tensor,         # [N, A, 3]
    owner: Tensor,          # [N]
    next_vars: Tensor,      # [N]
    pack_base: int,
    padding_idx: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Remove duplicate states within a single depth level.

    Returns: (unique_states, unique_owner, unique_next_vars, unique_hashes)
    """
    if states.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.int64, device=states.device)
        return states, owner, next_vars, empty

    hashes = hash_states(states, owner, pack_base, padding_idx)
    unique_hashes, inv = torch.unique(hashes, return_inverse=True)

    # Keep first occurrence via scatter
    N = states.shape[0]
    first_idx = torch.full((unique_hashes.shape[0],), N, dtype=torch.long, device=states.device)
    src_idx = torch.arange(N, device=states.device).flip(0)
    first_idx.scatter_(0, inv.flip(0), src_idx)

    keep = first_idx[first_idx < N]
    keep, _ = keep.sort()

    return states[keep], owner[keep], next_vars[keep], hashes[keep]


def dedup_cross_depth(
    states: Tensor,         # [N, A, 3]
    owner: Tensor,          # [N]
    next_vars: Tensor,      # [N]
    hashes: Tensor,         # [N] from within-depth dedup
    visited: Tensor,        # [V] sorted visited hashes
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Remove states already visited in prior depths.

    Returns: (filtered_states, filtered_owner, filtered_next_vars, filtered_hashes)
    """
    if states.shape[0] == 0 or visited.shape[0] == 0:
        return states, owner, next_vars, hashes

    V = visited.shape[0]
    idx = torch.searchsorted(visited, hashes)
    found = (idx < V) & (visited[idx.clamp(max=V - 1)] == hashes)
    keep = ~found

    if keep.all():
        return states, owner, next_vars, hashes

    return states[keep], owner[keep], next_vars[keep], hashes[keep]


def cap_frontier_per_query(
    states: Tensor,         # [N, A, 3]
    owner: Tensor,          # [N]
    next_vars: Tensor,      # [N]
    max_per_query: int,
    num_queries: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Cap the number of frontier states per query.

    Keeps the first max_per_query states per owner (in current order).
    Returns: (capped_states, capped_owner, capped_next_vars)
    """
    N = states.shape[0]
    device = states.device

    if N == 0 or max_per_query <= 0:
        return states, owner, next_vars

    # Sort by owner to group
    sorted_idx = torch.argsort(owner, stable=True)
    sorted_owner = owner[sorted_idx]

    # Within-group rank via cumsum grouped by owner change points
    ones = torch.ones(N, dtype=torch.long, device=device)
    group_start = torch.zeros(N, dtype=torch.bool, device=device)
    group_start[0] = True
    group_start[1:] = sorted_owner[1:] != sorted_owner[:-1]

    cumsum = ones.cumsum(0)
    group_offsets = torch.zeros(N, dtype=torch.long, device=device)
    group_offsets[group_start] = cumsum[group_start] - 1
    group_offsets = group_offsets.cummax(0)[0]
    rank = cumsum - 1 - group_offsets

    keep_sorted = rank < max_per_query
    keep_original_idx = sorted_idx[keep_sorted]

    return states[keep_original_idx], owner[keep_original_idx], next_vars[keep_original_idx]
