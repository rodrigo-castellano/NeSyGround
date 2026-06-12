"""BackwardGrounder — query-directed proof search (sld / rtf / pbc).

The backward shell: reads its typed ``Backward`` config, wires the per-resolution
setup (``init_mgu`` for sld/rtf, ``build_tables`` for pbc), fixes the shared static
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
                 compile: str = "off", chunk_size: Optional[int] = None,
                 init_state_shape: str = "minimal") -> None:
        super().__init__()
        _validate(config, layout, compile)
        # Build inputs snapshotted for rebound() (re-snapshot over a rewritten KB).
        self._build = dict(config=config, layout=layout, compile=compile,
                           chunk_size=chunk_size, init_state_shape=init_state_shape)
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
        # Flat path prunes by default — push width into the enumeration as a per-fv branch
        # pruner (set-identical to one-shot, join_ab-gated). A KGE beam (guided_topk) implies
        # pruning + forces a flat-eager layout (the incremental expansion the beam rides on).
        self._flat_prune = bool(pbc.flat_prune or pbc.guided_topk is not None) if pbc else False
        if pbc is not None and pbc.guided_topk is not None:
            layout = "flat"

        # ── shared search knobs (mirrored onto self for RunPlan.snapshot) ──
        self.depth = config.depth
        self.prune_facts = config.prune_facts
        self._pack_dedup = config.pack_dedup
        self._init_state_shape = init_state_shape   # exec-surface knob (only buffers.py reads it)
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
        # output_spec is the SINGLE source for which tiers to collect; RunPlan derives
        # collect_evidence (TREES) / collect_rule_groundings (FIRINGS) from it. Default spec
        # always present so plan.snapshot reads an attribute (no frozenset under compile).
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

        # ── shared static layout: L, A, G computed ONCE ──
        M, D = kb.M, self.depth
        mg = config.max_atoms if config.max_atoms is not None else M + (M - 1) * D
        self.max_atoms = max(mg, M)                                          # L
        self.A = D * M                                                       # A
        self.G = config.max_goals if config.max_goals is not None else 256  # G

        # Universal children-per-state cap: None → family default
        # (sld/rtf 550; pbc the grounding budget, since a pbc child IS a grounding).
        children_cap = (config.max_children if config.max_children is not None
                        else (config.max_groundings_per_query if pbc else 550))
        self._init_resolution(
            max_children=children_cap,
            max_total_groundings=config.max_groundings_per_query,
            max_groundings_per_rule=pbc.max_groundings_per_rule if pbc else None)

        # ── pre-resolved exec cell (read by plan.snapshot) ──
        lay = _LAYOUT[layout]
        # FLAT is eager-only, so under layout=auto a compile request resolves to DENSE
        # (the compilable layout); without compile, pbc prefers flat, sld/rtf dense.
        auto_flat = pbc is not None and compile == "off"
        self._exec_layout = lay if lay is not None else (
            Layout.FLAT if auto_flat else Layout.DENSE)
        # An auto-resolved flat layout downgrades a compile pref to EAGER; an explicit
        # flat+compile is illegal, left for capability.validate.
        self._exec_compile = (EAGER if (self._exec_layout is Layout.FLAT and layout == "auto")
                              else _COMPILE[compile])

    def _init_resolution(self, *, max_children: int,
                         max_total_groundings: int, max_groundings_per_rule: Optional[int]) -> None:
        """Dispatch to the resolution layer's grounder-setup — ``init_mgu`` (sld/rtf) /
        ``build_tables`` (pbc) own the budget computation AND the wiring (buffers +
        scalars + S-bump) onto self. S is already set; flat-vs-dense is exec-resolved."""
        from grounder.resolution.mgu import init_mgu
        from grounder.resolution.pbc import build_tables
        kw = dict(max_children=max_children, max_total_groundings=max_total_groundings,
                  max_groundings_per_rule=max_groundings_per_rule)
        if self.resolution in ("sld", "rtf"):
            init_mgu(self, **kw)
        elif self.resolution == "pbc":
            build_tables(self, **kw)
        else:
            raise ValueError(f"Unknown resolution: {self.resolution}")

    # ── Grounder API ──
    @torch.no_grad()
    def ground(self, request: GroundRequest) -> BackwardResult:
        """The single runtime verb — proof search over ``request.queries`` for the tiers
        in ``request.output_spec`` (which also drives what the engine collects)."""
        spec = request.output_spec
        self.output_spec = spec     # RunPlan derives collect_evidence/collect_rule_groundings from this
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
                f"G={self.G}, Y_q={self.Y_q})")


__all__ = ["BackwardGrounder"]
