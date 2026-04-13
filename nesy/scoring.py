"""Re-exports from kge_kernels.scoring."""

from kge_kernels.scoring import (
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
    precompute_partial_scores_from_model as precompute_partial_scores,
    score_partial_atoms,
)

__all__ = [
    "build_backend",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "score_partial_atoms",
]
