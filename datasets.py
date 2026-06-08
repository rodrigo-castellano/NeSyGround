"""Per-dataset grounder EXECUTION defaults — the single source of truth so callers
never re-guess the optimal batch/chunk/layout for a dataset.

Keyed by dataset name; values are GROUNDER ctor kwargs (chunk_size, layout, ...).
These are grounding-OUTPUT-INVARIANT (chunking + layout never change the firing
set — the fingerprint gates this), so they are pure memory/speed tuning:

  chunk_size : None = auto memory-budget (default); 0 = one pass; N = queries/chunk.
  layout     : 'auto' (default) | 'flat' | 'dense'.

Consumers (torch-ns adapter via ``create_grounder(dataset=...)``, the grounder
scale tests) read this so tuning lives in ONE place. The paper datasets run safe
+ non-regressing under auto chunking with the consumer's per-cell test batch
size, so they keep auto; the large datasets pin an explicit chunk to bound the
V≥2 Cartesian enumeration peak (the static budget under-counts it) for
depth-3/2 all-groundings runs.
"""
from __future__ import annotations

from typing import Any, Dict

DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # paper datasets — auto chunking + the consumer's per-cell test batch size is
    # verified safe (no OOM) and non-regressing vs the old grounder.
    "family":       {},
    "wn18rr":       {},
    "countries_s2": {},
    "countries_s3": {},
    "ablation_d2":  {},
    "ablation_d3":  {},
    "nations":      {},
    "umls":         {},
    # large / high-fanout — resolution-agnostic chunk only (materialization is a
    # PBC-only knob and would ConfigError under SLD, so it is NOT pinned here; the
    # scale experiment chooses resolution + materialization). Measured GPU 24GB,
    # w=1, keep-all (chunk bounds the per-chunk enumeration peak):
    #   fb15k237 d3: SLD chunk=16 -> 7.7GB, 57534 firings/200q, 9ms/q (~3min full)
    #     — the recommended scale path; PBC cartesian/join enum OOMs (~3GB/query,
    #     a few very-high-fanout queries exceed 24GB).
    #   yago310 d2: SLD chunk=16 -> 11.2GB (1450/50q); PBC join chunk=16 -> 11.7GB.
    "fb15k237":     {"chunk_size": 16},
    "yago310":      {"chunk_size": 16},
}


def grounder_defaults(dataset: str | None) -> Dict[str, Any]:
    """Grounder ctor-kwarg defaults for ``dataset`` ({} if unknown/None)."""
    if not dataset:
        return {}
    return dict(DATASET_DEFAULTS.get(dataset, {}))


__all__ = ["DATASET_DEFAULTS", "grounder_defaults"]
