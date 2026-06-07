"""Internal utilities for nesy hooks."""

from torch import Tensor


def _topk_select(
    body: Tensor,       # [B, tG_in, M, 3]
    mask: Tensor,       # [B, tG_in]
    rule_idx: Tensor,   # [B, tG_in]
    scores: Tensor,     # [B, tG_in]
    output_tG: int,
) -> tuple:
    """Select top-k groundings by score. Returns (body, mask, rule_idx)."""
    B, tG_in, M, _ = body.shape
    _, top_idx = scores.topk(output_tG, dim=1, largest=True, sorted=False)
    idx_body = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)
    return (
        body.gather(1, idx_body),
        mask.gather(1, top_idx),
        rule_idx.gather(1, top_idx),
    )
