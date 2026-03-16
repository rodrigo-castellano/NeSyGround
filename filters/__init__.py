"""Grounding filters — taxonomy:

Soundness filters (terminal):
  soundness/fp_batch.py  — cross-query Kleene T_P fixed-point
  soundness/fp_global.py — FC provable set check (precomputed)

Search filters (per-step):
  search/width.py      — reject states with too many unknowns
  search/prune_dead.py — kill provably dead states

Shared utilities:
  _hash.py             — hash_atoms
  check_in_provable    — binary search membership in sorted hash tensor
"""

from grounder.filters.soundness import check_in_provable
from grounder.filters.search import filter_width, filter_prune_dead

__all__ = [
    "check_in_provable",
    "filter_width",
    "filter_prune_dead",
]
