"""Re-exports from :mod:`kge_kernels.scoring`.

After tkk's scoring-folder consolidation there is no per-mode scoring
wrapper anymore — KGE scoring is just ``model.score(h, r, t)`` (with
optional ``torch.sigmoid`` if you want probabilities). This module
re-exports the partial-atom scorer classes that DO live in tkk.
"""

from kge_kernels.scoring import LazyPartialScorer, PartialScorer

__all__ = [
    "LazyPartialScorer",
    "PartialScorer",
]
