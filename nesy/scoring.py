"""Re-exports from :mod:`kge_kernels.scoring`.

This module is the historical entry point for grounder consumers (DpRL,
ns) to access KGE scoring helpers. After tkk's scoring-folder
consolidation there is just one entry point — :func:`kge_score` — plus
the partial-atom helpers; the per-mode wrappers (``kge_score_triples``
etc.) and the ``KGEBackend`` adapter are gone.
"""

from kge_kernels.scoring import (
    kge_score,
    precompute_partial_scores,
    score_partial_atoms,
)

__all__ = [
    "kge_score",
    "precompute_partial_scores",
    "score_partial_atoms",
]
