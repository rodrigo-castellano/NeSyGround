"""FC provable set filter — check body atoms against forward-chaining provable set.

Body atoms are accepted if they are base facts OR in the provable set I_D
computed at init via forward chaining.
"""

from __future__ import annotations

from torch import Tensor

from grounder.filters.soundness import check_in_provable


def apply_fp_global(
    body: Tensor,               # [B, N, M, 3]
    mask: Tensor,               # [B, N] bool
    fact_index,                 # FactIndex with .exists()
    pack_base: int,
    padding_idx: int,
    provable_hashes: Tensor,    # [I_max] sorted provable atom hashes
) -> Tensor:
    """Filter groundings: keep those whose body atoms are all facts or provable.

    Args:
        body:            [B, N, M, 3] collected grounding body atoms.
        mask:            [B, N] validity mask.
        fact_index:      FactIndex with .exists() method.
        pack_base:       hash packing base.
        padding_idx:     padding value.
        provable_hashes: [I_max] sorted hashes of provable atoms.

    Returns:
        [B, N] bool — which groundings survive the filter.
    """
    B, N, M, _ = body.shape
    pb = pack_base

    body_hashes = (body[..., 0].long() * (pb * pb)
                   + body[..., 1].long() * pb
                   + body[..., 2].long())
    is_fact = fact_index.exists(body.reshape(-1, 3)).view(B, N, M)
    in_provable = check_in_provable(body_hashes, provable_hashes)
    body_active = body[..., 0] != padding_idx

    all_proved = ((is_fact | in_provable) | ~body_active).all(dim=-1)
    return mask & all_proved
