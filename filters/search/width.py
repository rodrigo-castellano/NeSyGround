"""Width filter — reject states with too many unknown ground body atoms.

For SLD/RTF resolution (between RESOLVE and PACK).
Enum resolution handles width internally in _apply_enum_filters.
"""

from __future__ import annotations

from torch import Tensor


def filter_width(
    rule_goals: Tensor,      # [B, S, K_r, G, 3]
    rule_success: Tensor,    # [B, S, K_r]
    *,
    fact_index,              # FactIndex with .exists()
    constant_no: int,
    padding_idx: int,
    M: int,
    width: int,
) -> Tensor:
    """Kill rule-derived states with > `width` ground non-fact body atoms.

    Args:
        rule_goals:   [B, S, K_r, G, 3] resolved rule-derived goals.
        rule_success: [B, S, K_r] validity mask for rule children.
        fact_index:   FactIndex with .exists() method.
        constant_no:  max constant index (constants are <= constant_no).
        padding_idx:  padding value.
        M:            number of body atom slots.
        width:        max allowed ground non-fact body atoms.

    Returns:
        [B, S, K_r] filtered rule_success mask.
    """
    body = rule_goals[:, :, :, :M, :]                  # [B, S, K_r, M, 3]
    is_active = body[..., 0] != padding_idx             # [B, S, K_r, M]
    is_ground = (body[..., 1:] <= constant_no).all(dim=-1) & is_active  # [B, S, K_r, M]

    flat = body.reshape(-1, 3)                          # [B*S*K_r*M, 3]
    is_fact_flat = fact_index.exists(flat)               # [B*S*K_r*M]
    is_fact = is_fact_flat.view(body.shape[:-1]) & is_ground  # [B, S, K_r, M]

    ground_not_fact = is_ground & ~is_fact              # [B, S, K_r, M]
    count = ground_not_fact.sum(dim=-1)                 # [B, S, K_r]
    return rule_success & (count <= width)
