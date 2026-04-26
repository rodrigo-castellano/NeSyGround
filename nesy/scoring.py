"""Re-exports from kge_kernels.scoring with one model-aware convenience.

``precompute_partial_scores`` is the only function here that diverges from a
plain re-export: tkk's :func:`kge_kernels.scoring.precompute_partial_scores`
accepts a :class:`KGEBackend`; consumers in this repo (and downstream in
DpRL) pass a raw ``nn.Module``. The wrapper builds the backend with sigmoid
normalization (``build_backend``) and forwards.
"""

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from kge_kernels.scoring import (
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
    score_partial_atoms,
)
from kge_kernels.scoring import precompute_partial_scores as _precompute_partial_scores


def precompute_partial_scores(
    model: nn.Module,
    pred_remap: Tensor,
    const_remap: Tensor,
    batch_chunk: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Model-aware wrapper around tkk's backend-typed precompute."""
    return _precompute_partial_scores(
        build_backend(model), pred_remap, const_remap, batch_chunk=batch_chunk,
    )


__all__ = [
    "build_backend",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "score_partial_atoms",
]
