"""BackwardGrounder — query-directed proof search (sld / rtf / pbc).

The backward shell: reads its typed ``Backward`` config, wires the per-resolution
setup (``init_mgu`` for sld/rtf, ``init_enum`` for pbc), fixes the shared static
layout (G, A, S, Y_q), and drives the depth loop via ``backward.loop.run_backward``.
The single runtime verb is ``ground(request)``; ``request.output_spec`` selects
which tiers are produced (and what the engine collects).

  pbc      → BC_{w,d,u}: all_anchors forced, filter fp_batch when u=0.
  sld/rtf  → plain backward chaining, filter none.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from grounder.api.config import Backward, PBC, RTF, SLD
from grounder.core import BackwardResult, GroundRequest, OutputSpec, Tier
from grounder.data.kb import KB
from grounder.backward.loop import run_backward
from grounder.base.errors import ConfigError
from grounder.base.types import Layout
from grounder.execution.capability import (
    COMPILED_DYNAMIC, COMPILED_STEP, EAGER, CapabilityRow, Cell,
)

# str knob surface → internal layout / compile spec.
_LAYOUT = {"auto": None, "dense": Layout.DENSE, "flat": Layout.FLAT}
_COMPILE = {"off": EAGER, "graph": COMPILED_STEP, "dynamic": COMPILED_DYNAMIC}

# The (layout, compile) cells a backward grounder declares it supports.
_CELLS = frozenset({
    Cell(Layout.DENSE, EAGER), Cell(Layout.DENSE, COMPILED_STEP),
    Cell(Layout.DENSE, COMPILED_DYNAMIC), Cell(Layout.FLAT, EAGER),
})


def _validate(config: Backward, layout: str, compile: str) -> None:
    """All ctor contract checks, raised before any state is built."""
    res = config.resolution
    if getattr(res, "materialization", "cartesian") == "join" and not isinstance(res, PBC):
        raise ConfigError("materialization='join' requires resolution='pbc'")
    if getattr(res, "guided_topk", None) is not None and config.guided_scorer is None:
        raise ConfigError(
            "guided_topk requires Backward.guided_scorer (nesy.hooks.GuidedScorer)")
    if layout not in _LAYOUT:
        raise ConfigError(f"layout must be auto|dense|flat, got {layout!r}")
    if compile not in _COMPILE:
        raise ConfigError(f"compile must be off|graph|dynamic, got {compile!r}")


class BackwardGrounder(nn.Module):
    """Unified backward-chaining grounder (sld / rtf / pbc), built from a ``Backward`` config."""

    def __init__(self, kb: KB, config: Backward, *, layout: str = "auto",
                 compile: str = "off", chunk_size: Optional[int] = None) -> None:
        super().__init__()
        _validate(config, layout, compile)
        # Build inputs snapshotted for rebound() (re-snapshot over a rewritten KB).
        self._build = dict(config=config, layout=layout, compile=compile, chunk_size=chunk_size)
        self.kb = kb
        self.num_rules = kb.num_rules

        # ── resolution family (sld / rtf / pbc) ──
        res = config.resolution
        pbc = res if isinstance(res, PBC) else None
        self.resolution = "pbc" if pbc else "rtf" if isinstance(res, RTF) else "sld"

        # ── PBC-only knobs (inert defaults for sld/rtf) ──
        self.width = pbc.width if pbc else 1
        self.w_last_depth = pbc.u if pbc else 0
        self._cartesian_product = pbc.cartesian_product if pbc else False
        self._all_anchors = pbc is not None                  # forced for pbc
        flat_intermediate = pbc.flat_intermediate if pbc else False
        materialization = pbc.materialization if pbc else "cartesian"

        # ── materialization routing: "join" → L3 JoinResolver (pbc tables, flat-eager);
        #    guided beam (KGE prior) needs the join's incremental per-fv expansion ──
        if pbc is not None and pbc.guided_topk is not None:
            materialization = "join"
        self._dispatch_resolution = "join" if materialization == "join" else self.resolution
        if materialization == "join":
            flat_intermediate, layout = True, "flat"

        # ── shared search knobs (mirrored onto self for RunPlan.snapshot) ──
        self.depth = config.depth
        self.prune_facts = config.prune_facts
        self._pack_dedup = config.pack_dedup
        self._init_state_shape = config.init_state_shape
        self._bump_s_to_k = config.bump_s_to_k

        # ── nesy hooks (resolution + step/grounding injection points) ──
        self.hooks = list(config.hooks) if config.hooks else []
        self.fact_hook, self.rule_hook = config.fact_hook, config.rule_hook

        # ── guided beam (KGE prior; join path only) ──
        self.guided_topk = pbc.guided_topk if pbc else None
        self.guided_tnorm = pbc.guided_tnorm if pbc else "min"
        self.guided_scorer = config.guided_scorer
        self.guided_stats = None      # GuidedStats census counters (attach post-ctor)

        # ── collection / output (which tiers + soundness filter) ──
        self._collect_mode = "terminal"
        # collect_evidence / _collect_rule_groundings come from the request's output_spec (set in ground()).
        self.collect_evidence = self._collect_rule_groundings = False
        # Default spec always present so plan.snapshot reads an attribute (no frozenset under compile).
        self.output_spec = OutputSpec(frozenset({Tier.PROOF_STATE}))
        # Filter default: pbc with u=0 → fp_batch; else none.
        self.filter_mode = config.filter or ("fp_batch" if (pbc and self.w_last_depth == 0) else "none")

        # ── standardization: lazily build the terminal var-renaming fn only when requested ──
        self._standardize_fn = None
        if config.standardization is not None:
            from grounder.resolution.standardize import build_standardize_fn
            self._standardize_fn = build_standardize_fn(config.standardization)

        # ── exec-surface knobs ──
        self._layout_knob, self._compile_knob, self._chunk_size = layout, compile, chunk_size
        self._knobs_set = layout != "auto" or compile != "off" or chunk_size is not None
        flat_requested = {"dense": False, "flat": True}.get(layout, flat_intermediate)

        # ── shared static layout: G, A, S computed ONCE ──
        M, D = kb.M, self.depth
        mg = config.max_goals if config.max_goals is not None else M + (M - 1) * D
        self.max_goals = max(mg, M)                                          # G
        self.A = D * M                                                       # A
        self.S = config.max_states if config.max_states is not None else 256  # S

        # Universal children-per-state cap: None → family default
        # (sld/rtf 550; pbc the grounding budget, since a pbc child IS a grounding).
        children_cap = (config.max_children if config.max_children is not None
                        else (config.max_groundings_per_query if pbc else 550))
        self._init_resolution(
            max_children=children_cap,
            max_total_groundings=config.max_groundings_per_query,
            max_groundings_per_rule=pbc.max_groundings_per_rule if pbc else None,
            flat_intermediate=flat_requested)

        # ── pre-resolved exec cell (read by plan.snapshot) ──
        lay = _LAYOUT[layout]
        self._exec_layout = lay if lay is not None else (
            Layout.FLAT if self._flat_intermediate else Layout.DENSE)
        # FLAT is eager-only: an auto-resolved flat layout downgrades a compile pref to
        # EAGER; an explicit flat+compile is illegal, left for capability.validate.
        self._exec_compile = (EAGER if (self._exec_layout is Layout.FLAT and layout == "auto")
                              else _COMPILE[compile])

    def _init_resolution(self, *, max_children: int,
                         max_total_groundings: int, max_groundings_per_rule: Optional[int],
                         flat_intermediate: bool) -> None:
        """Per-resolution build: shape budgets (+ pbc binding buffers). S is already set."""
        if self.resolution in ("sld", "rtf"):
            from grounder.resolution.mgu import init_mgu
            cfg = init_mgu(
                resolution=self.resolution, K_f=self.kb.K_f, K_r=self.kb.K_r,
                rule_index=self.kb.rule_index,
                max_total_groundings=max_total_groundings, max_children=max_children,
                max_groundings_per_rule=max_groundings_per_rule)
            self.K = cfg["K"]
            self.kb.K_f = cfg["K_f"]
            self.max_vars_per_rule = cfg["max_vars_per_rule"]
            self.Y_q = cfg["Y_q"]
            self._max_fact_pairs_body = cfg["max_fact_pairs_body"]
            self._flat_intermediate = False
            return
        if self.resolution != "pbc":
            raise ValueError(f"Unknown resolution: {self.resolution}")

        from grounder.resolution.pbc import init_enum
        meta = init_enum(
            rule_index=self.kb.rule_index, fact_index=self.kb.fact_index,
            facts_idx=self.kb.fact_index.facts_idx, constant_no=self.kb.constant_no,
            num_rules=self.kb.num_rules, M=self.kb.M, width=self.width,
            max_total_groundings=max_total_groundings, max_children=max_children,
            max_groundings_per_query=max_groundings_per_rule,   # the per-rule Y_r cap
            device=self.kb.device_,
            cartesian_product=self._cartesian_product, all_anchors=self._all_anchors,
            flat_intermediate=flat_intermediate)
        for name, tensor in meta["buffers"].items():
            self.register_buffer(name, tensor)
        self._enum_ri = meta["enum_rule_index"]
        self._P, self._E = meta["P"], meta["E"]
        self.K_r, self.K, self.Y_q = meta["K_r"], meta["K"], meta["Y_q"]
        self.Y_r = meta["Y_r"]
        self.V, self.K_v = meta.get("V", 1), meta.get("K_v", 64)
        self._fv_any_valid = meta.get("fv_any_valid", None)
        self._flat_intermediate = meta.get("flat_intermediate", False)
        self.max_vars_per_rule = 3
        # all_anchors dedup/remap needs the variant→original rule mapping on device.
        if self._all_anchors:
            self.register_buffer("_variant_to_orig_t", self._enum_ri.variant_to_orig.to(
                dtype=torch.long, device=self.kb.device_))
        # Optional S bump: enlarge S toward K_r*K_v when depth>1 (pbc only).
        if self._bump_s_to_k and self.depth > 1 and self.width is not None and self.width > 0:
            K_v = meta.get("K_v", 0) or 0
            self.S = max(self.S, min(self.K, max(self.K_r * K_v, 1)))

    # ── Grounder API ──
    @torch.no_grad()
    def ground(self, request: GroundRequest) -> BackwardResult:
        """The single runtime verb — proof search over ``request.queries`` for the tiers
        in ``request.output_spec`` (which also drives what the engine collects)."""
        spec = request.output_spec
        self.output_spec = spec
        self.collect_evidence = spec.trees             # Tier.TREES requested
        self._collect_rule_groundings = spec.firings   # Tier.FIRINGS requested
        result = run_backward(self, request.queries, request.query_mask,
                              excluded_queries=request.excluded_queries)
        if spec.firings:
            # Pool-iter reasoners gather pool[query_pool_idx] regardless of
            # provability, so every query atom needs a slot + query_pool_idx set.
            from dataclasses import replace as _replace
            from grounder.backward.considered import populate_query_pool_idx
            from grounder.base.types import RuleGroundings
            rg = result.rule_groundings
            if rg is None:
                rg = RuleGroundings.empty(
                    num_rules=self.kb.num_rules, M=self.kb.M,
                    device=request.queries.device)
            rg = populate_query_pool_idx(rg, request.queries, self.kb.padding_idx)
            result = _replace(result, rule_groundings=rg)
        return result

    def producible_tiers(self) -> "frozenset[Tier]":
        """BC can fill all three backward tiers."""
        return frozenset({Tier.PROOF_STATE, Tier.FIRINGS, Tier.TREES})

    def capability_row(self) -> CapabilityRow:
        """Declared (layout, compile) cells: dense × {eager, graph, dynamic} + flat-eager."""
        return CapabilityRow(_CELLS)

    def rebound(self, kb: KB) -> "BackwardGrounder":
        """Re-snapshot over a rewritten KB (transforms) with the identical config."""
        return BackwardGrounder(kb, **self._build)

    def __repr__(self) -> str:
        return (f"BackwardGrounder(resolution={self.resolution!r}, filter={self.filter_mode!r}, "
                f"depth={self.depth}, width={self.width}, num_rules={self.kb.num_rules}, "
                f"S={self.S}, Y_q={self.Y_q})")


__all__ = ["BackwardGrounder"]
