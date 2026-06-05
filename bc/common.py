"""BC shared utilities — re-export facade.

The implementations were split by responsibility into sibling modules:

* ``bc/packing.py``         — ``compact_atoms``, ``pack_states``,
                              ``pack_states_flat`` (+ ``_pow_desc`` / ``_POW_CACHE``)
* ``bc/postprocessing.py``  — ``collect_groundings``, ``_dedup_groundings``
* ``bc/grounding_convert.py`` — ``prune_rule_groundings``,
                              ``build_rule_grounding_tensors``

This module re-exports them so existing ``from grounder.bc.common import X``
imports keep working unchanged.
"""

from __future__ import annotations

from grounder.bc.packing import (
    _POW_CACHE,
    _pow_desc,
    compact_atoms,
    pack_states,
    pack_states_flat,
)
from grounder.bc.postprocessing import (
    _dedup_groundings,
    collect_groundings,
)
from grounder.bc.grounding_convert import (
    build_rule_grounding_tensors,
    prune_rule_groundings,
)
# Re-export prune_ground_facts from its canonical location for compatibility.
from grounder.filters.search.prune_facts import prune_ground_facts

__all__ = [
    "_POW_CACHE",
    "_pow_desc",
    "compact_atoms",
    "pack_states",
    "pack_states_flat",
    "collect_groundings",
    "_dedup_groundings",
    "prune_rule_groundings",
    "build_rule_grounding_tensors",
    "prune_ground_facts",
]
