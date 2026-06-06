"""GroundingCache — the opt-in grounding-memoization state for a grounder.

A single state container for the three OPTIONAL, default-OFF memoization
layers; the standard depth-loop path never consults it:

* per-query **tabling** (#47) — :mod:`grounder.bc.tabling`
* per-subgoal **memo** (#48) — :mod:`grounder.bc.subgoal`
* within-batch **resolve-skip** (#48 Phase-2) — the flat enum resolve

The cache LOGIC lives in those modules; this is just the state they read
and write. :class:`grounder.bc.bc.BCGrounder` composes one of these
(``self._cache``) and exposes thin ``_*`` properties that delegate here, so
the cache modules + external callers (e.g. the torch-ns env toggles) keep
using ``grounder._tabling_enabled`` etc. unchanged.

Grounding is weight-independent (a pure function of the KB), so these caches
are byte-identical to the uncached path — verified by ``ground_fingerprint``
with each layer toggled on vs off.
"""
from __future__ import annotations

from typing import Any, Dict


class GroundingCache:
    """Default-OFF memoization state for one :class:`BCGrounder`.

    ``kb`` is used only to stamp KB-immutability (the caches are valid only
    while facts + rules are unchanged for the grounder's lifetime).
    """

    def __init__(self, kb) -> None:
        # ── Per-query tabling (#47) ──
        self.tabling_enabled = False
        # atom_hash -> (rule_idx[F], head[F,3], body[F,M,3], query_triple[3])
        self.grounding_cache: Dict[int, Any] = {}
        # Query-index offset reconciling the considered accumulator's
        # batch-local b_idx with the write-back miss_positions map
        # (reset per forward / chunk; 0 for the single-batch path).
        self.chunk_query_offset = 0
        # KB-immutability stamp — asserted on every consult.
        self.tabling_kb_stamp = (id(kb.fact_index), int(kb.num_rules))
        # Cap on distinct cached keys; when full, serve hits + compute misses
        # fresh (passthrough) instead of inserting.
        self.tabling_max_entries = 2_000_000
        self.tabling_full_warned = False
        # Per-forward hit/miss counters (diagnostic).
        self.tabling_last_hits = 0
        self.tabling_last_misses = 0

        # ── Per-subgoal memo (#48) ──
        # (selected_goal_hash, is_last) -> stored considered rows.
        self.subgoal_enabled = False
        self.subgoal_memo: Dict[Any, Any] = {}
        self.subgoal_max_entries = 2_000_000
        self.subgoal_full_warned = False
        self.subgoal_last_hits = 0
        self.subgoal_last_misses = 0

        # ── Resolve-skip (#48 Phase-2) ──
        # Within-batch goal dedup in the flat enum resolve (run the pipeline
        # once per distinct selected goal, scatter survivors).
        self.resolve_skip_enabled = False


__all__ = ["GroundingCache"]
