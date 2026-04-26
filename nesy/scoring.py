"""Re-exports from :mod:`kge_kernels.scoring`.

After tkk's scoring-folder consolidation there is no per-mode scoring
wrapper anymore — KGE scoring is just ``model.score(h, r, t)`` (with
optional ``torch.sigmoid`` if you want probabilities). This module
re-exports the partial-atom helpers that DO live in tkk.
"""

from kge_kernels.scoring import (
    precompute_partial_scores,
    score_partial_atoms,
)

__all__ = [
    "precompute_partial_scores",
    "score_partial_atoms",
]
