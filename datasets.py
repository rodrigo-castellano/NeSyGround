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
    # family/wn18rr: one grounder-chunk (chunk_size=0). The consumer (eval_scores)
    # already bounds each run_bc batch (candidate pool capped to N_full atoms), so
    # the grounder's auto memory-budget needlessly RE-chunks it — that double
    # chunking is the measured ~1.2x exhaustive-eval regression vs old. One pass
    # matches old. Output-invariant (fingerprint batch-invariance) -> MRR-safe.
    "family":       {"chunk_size": 0},
    "wn18rr":       {"chunk_size": 0},
    # smaller / high-fanout datasets keep auto (countries_s3 d3 needs the budget
    # chunker to bound its Cartesian peak; one pass would OOM).
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
