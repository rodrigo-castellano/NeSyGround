"""Backward-compatible re-exports from kge_kernels.adapter."""

from kge_kernels.adapter import (
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
    precompute_partial_scores,
)
from kge_kernels.partial import score_partial_atoms

__all__ = [
    "build_backend",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "score_partial_atoms",
]
