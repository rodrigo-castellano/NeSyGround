"""PBC (Parametrized Backward Chaining) candidate generation.

  plan       — build_tables (build binding tables + budgets) → PbcPlan via build_plan
  enumerate  — cluster + enumerate (dense_single|dense_cartesian|flat_cartesian|flat_pruned) + fill
  guided     — GuidedBeam / GuidedStats (KGE prior over the flat-pruned expansion)
  width      — the BC_{w,d,u} width filter + depth_gate
  resolve    — resolve_step driver + Dense/Flat materializers + PbcResolver

The flat path prunes by pushing the width predicate INTO the enumeration as a
per-fv branch pruner (``FlatMaterializer(prune=True)``) — a
set-identical optimization of the one-shot cartesian path (gated by join_ab.py),
not byte-identity. ``guided_topk`` layers a KGE beam on the same pruned path.
"""
from __future__ import annotations

from grounder.resolution.pbc.plan import PbcPlan, PbcTables, build_plan, build_tables
from grounder.resolution.pbc.resolve import (
    DenseMaterializer, FlatMaterializer, PbcResolver, StepInputs, resolve_step,
)

__all__ = [
    "build_tables", "build_plan", "PbcPlan", "PbcTables",
    "resolve_step", "StepInputs",
    "DenseMaterializer", "FlatMaterializer", "PbcResolver",
]
