"""BackwardGrounder — query-directed proof search (sld / rtf / pbc).

The backward shell: wires the per-resolution config (init_mgu for sld/rtf, init_enum
for pbc), sets the shared layout (G, A, S, C), and drives the depth loop via
``backward.loop.run_backward``. Ported from OLD ``bc/bc.py`` + ``bc/init_resolution.py``,
trimmed to the eager/CPU fingerprint path (no compile, no chunking, no caches).

  pbc  → BC_{w,d,u}: all_anchors forced True, filter fp_batch (u=0), G_r cap.
  sld/rtf → plain backward chaining, filter none.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder._build.core.request import GroundRequest, OutputSpec, Tier
from grounder._build.core.result import BackwardResult
from grounder._build.data.kb import KB
from grounder._build.backward.loop import run_backward
from grounder._build.errors import ConfigError
from grounder._build.execution.capability import (
    COMPILED_DYNAMIC, COMPILED_STEP, EAGER, CapabilityRow, Cell,
)
from grounder._build.types import Layout, RuleGroundings

# Knob normalization maps (str surface -> internal layout/compile spec).
_LAYOUT_KNOB = {"auto": None, "dense": Layout.DENSE, "flat": Layout.FLAT}
_COMPILE_KNOB = {"off": EAGER, "graph": COMPILED_STEP, "dynamic": COMPILED_DYNAMIC}
# Old torch.compile-mode strings (consumers' `compile_mode=`) -> new `compile` knob.
# None is handled by the ctor guard (None -> "off"); new values pass through.
_COMPILE_MODE_ALIAS = {
    "none": "off", "off": "off", "graph": "graph", "dynamic": "dynamic",
    "default": "graph", "reduce-overhead": "graph", "max-autotune": "graph",
}


class BackwardGrounder(nn.Module):
    """Unified backward-chaining grounder BC_{w,d}."""

    def __init__(
        self,
        kb: KB,
        *,
        resolution: str = "pbc",
        materialization: str = "cartesian",   # pbc layout: "cartesian" | "join" (L3)
        depth: int = 2,
        width: Optional[int] = 1,
        w: Optional[int] = None,
        u: int = 0,
        w_last_depth: Optional[int] = None,
        filter: Optional[str] = None,
        max_total_groundings: int = 64,
        max_groundings_per_query: int = 32,
        max_groundings_per_rule: Optional[int] = None,
        max_goals: Optional[int] = None,
        max_states: Optional[int] = None,
        K_MAX: int = 550,
        max_derived_per_state: Optional[int] = None,
        collect_evidence: bool = True,
        collect_rule_groundings: bool = False,
        cartesian_product: bool = False,
        all_anchors: bool = False,
        flat_intermediate: bool = False,
        layout: str = "auto",            # "auto" | "dense" | "flat"
        compile: str = "off",            # "off" | "graph" | "dynamic"
        compile_mode: Optional[str] = None,  # consumer one-window alias for `compile`
        chunk_size: Optional[int] = None,
        pack_dedup: bool = True,
        prune_facts: bool = False,
        bump_s_to_k: bool = True,
        init_state_shape: str = "minimal",
        fc_method: str = "spmm",
        fc_depth: int = 10,
        hooks: Optional[List] = None,
        fact_hook=None,
        rule_hook=None,
        standardization=None,
    ) -> None:
        super().__init__()
        # `compile_mode` is the consumers' one-window alias for `compile` (old
        # torch.compile mode strings). Only honored when `compile` is at default.
        if compile_mode is not None and compile == "off":
            compile = _COMPILE_MODE_ALIAS.get(compile_mode, compile_mode)
        # Capture the raw ctor inputs (sans kb) for rebound() — re-snapshot over a
        # rewritten KB with the IDENTICAL knobs (transform support).
        self._ctor_kwargs = dict(
            resolution=resolution, materialization=materialization,
            depth=depth, width=width, w=w, u=u,
            w_last_depth=w_last_depth, filter=filter,
            max_total_groundings=max_total_groundings,
            max_groundings_per_query=max_groundings_per_query,
            max_groundings_per_rule=max_groundings_per_rule, max_goals=max_goals,
            max_states=max_states, K_MAX=K_MAX,
            max_derived_per_state=max_derived_per_state,
            collect_evidence=collect_evidence,
            collect_rule_groundings=collect_rule_groundings,
            cartesian_product=cartesian_product, all_anchors=all_anchors,
            flat_intermediate=flat_intermediate, layout=layout, compile=compile,
            chunk_size=chunk_size, pack_dedup=pack_dedup, prune_facts=prune_facts,
            bump_s_to_k=bump_s_to_k, init_state_shape=init_state_shape,
            fc_method=fc_method, fc_depth=fc_depth, hooks=hooks,
            fact_hook=fact_hook, rule_hook=rule_hook, standardization=standardization)
        self.kb = kb
        self.num_rules = kb.num_rules

        # ``w`` is the PBC alias for ``width``; ``u`` aliases ``w_last_depth``.
        if w is not None:
            width = w
        if w_last_depth is None:
            w_last_depth = u

        if materialization not in ("cartesian", "join"):
            raise ConfigError(f"materialization must be cartesian|join, got {materialization!r}")
        if materialization == "join" and resolution != "pbc":
            raise ConfigError("materialization='join' requires resolution='pbc'")
        self.resolution = resolution
        # Dispatch key: "join" routes to the L3 JoinResolver but keeps the pbc init
        # path (same binding tables); flat layout forced (join is flat-eager only).
        self._materialization = materialization
        self._dispatch_resolution = "join" if materialization == "join" else resolution
        if materialization == "join":
            flat_intermediate = True
            layout = "flat"
        self.depth = depth
        self.width = width
        self.collect_evidence = collect_evidence
        self._collect_rule_groundings = collect_rule_groundings
        self.prune_facts = prune_facts
        self._pack_dedup = pack_dedup
        self._init_state_shape = init_state_shape
        self._collect_mode = "terminal"
        self.hooks = list(hooks) if hooks else []
        self.fact_hook = fact_hook
        self.rule_hook = rule_hook
        self._standardize_fn = None

        # ── exec-surface knobs: validate + capture (auto/off/None = legacy path) ──
        if layout not in _LAYOUT_KNOB:
            raise ConfigError(f"layout must be auto|dense|flat, got {layout!r}")
        if compile not in _COMPILE_KNOB:
            raise ConfigError(f"compile must be off|graph|dynamic, got {compile!r}")
        self._layout_knob = layout
        self._compile_knob = compile
        self._chunk_size = chunk_size
        self._knobs_set = (layout != "auto") or (compile != "off") or (chunk_size is not None)

        # ── filter default (pbc with u=0 → fp_batch; sld/rtf → none) ──
        if filter is None:
            if resolution == "pbc":
                filter = "fp_batch" if w_last_depth == 0 else "none"
            else:
                filter = "none"
        self.filter_mode = filter

        # ── enum invariants forced for pbc ──
        if resolution == "pbc":
            all_anchors = True
        self._all_anchors = all_anchors
        self._cartesian_product = cartesian_product
        # layout knob overrides the legacy flat_intermediate flag; auto keeps it.
        if layout == "dense":
            self._flat_intermediate_flag = False
        elif layout == "flat":
            self._flat_intermediate_flag = True
        else:
            self._flat_intermediate_flag = flat_intermediate
        self._w_last_depth = w_last_depth
        self._bump_s_to_k = bump_s_to_k

        # ── shared layout: G, A, S each computed ONCE here ──
        M = kb.M
        D = depth
        mg = max_goals if max_goals is not None else M + (M - 1) * D
        self.max_goals = max(mg, M)   # G: goals per state
        self.A = D * M                # A: accumulated-body capacity
        self.S = max_states if max_states is not None else 256  # S: single owner

        self._init_resolution(
            K_MAX=K_MAX,
            max_derived_per_state=max_derived_per_state,
            max_total_groundings=max_total_groundings,
            max_groundings_per_rule=max_groundings_per_rule,
            max_groundings_per_query=max_groundings_per_query,
            fc_method=fc_method, fc_depth=fc_depth)

        # ── pre-resolved exec cell axes (read by plan.snapshot; no map leak) ──
        self._exec_compile = _COMPILE_KNOB[self._compile_knob]
        _lay = _LAYOUT_KNOB[self._layout_knob]
        self._exec_layout = (_lay if _lay is not None
                             else (Layout.FLAT if self._flat_intermediate else Layout.DENSE))

    # ------------------------------------------------------------------
    # Per-resolution init
    # ------------------------------------------------------------------
    def _init_resolution(self, **kwargs) -> None:
        if self.resolution in ("sld", "rtf"):
            from grounder._build.resolution.mgu import init_mgu
            # K, K_f, max_vars_per_rule, C computed here — S is already self.S.
            cfg = init_mgu(
                resolution=self.resolution, K_f=self.kb.K_f, K_r=self.kb.K_r,
                rule_index=self.kb.rule_index,
                max_total_groundings=kwargs["max_total_groundings"],
                K_MAX=kwargs["K_MAX"], max_derived_per_state=kwargs["max_derived_per_state"],
                max_groundings_per_rule=kwargs["max_groundings_per_rule"])
            self.K = cfg["K"]
            self.kb.K_f = cfg["K_f"]
            self.max_vars_per_rule = cfg["max_vars_per_rule"]
            self.C = cfg["C"]
            self._max_fact_pairs_body = cfg["max_fact_pairs_body"]
            self._flat_intermediate = False
            self._enum_cartesian = False

        elif self.resolution == "pbc":
            from grounder._build.resolution.pbc import init_enum
            # K, K_r, K_v, G_r, C, V, E, P, buffers computed here — S is already self.S.
            meta = init_enum(
                rule_index=self.kb.rule_index, fact_index=self.kb.fact_index,
                facts_idx=self.kb.fact_index.facts_idx, constant_no=self.kb.constant_no,
                num_rules=self.kb.num_rules, M=self.kb.M, width=self.width,
                max_groundings_per_query=kwargs["max_groundings_per_query"],
                max_total_groundings=kwargs["max_total_groundings"],
                device=self.kb.device_,
                cartesian_product=self._cartesian_product, all_anchors=self._all_anchors,
                flat_intermediate=self._flat_intermediate_flag)
            for name, tensor in meta["buffers"].items():
                self.register_buffer(name, tensor)
            self._enum_ri = meta["enum_rule_index"]
            self.max_body_atoms = self._enum_ri.max_body
            self._P, self._E = meta["P"], meta["E"]
            self.K_r = meta["K_r"]
            self.K = meta["K"]
            # Optional S bump: enlarge S to at least K_r*K_v when depth>1 (pbc only).
            if (self._bump_s_to_k and self.depth > 1
                    and self.width is not None and self.width > 0):
                K_v = meta.get("K_v", 0) or 0
                K_actual = max(self.K_r * K_v, 1)
                self.S = max(self.S, min(self.K, K_actual))
            self.C = meta["C"]
            self.G_r = meta["G_r"]
            self._enum_cartesian = meta.get("cartesian_product", False)
            self.V = meta.get("V", 1)
            self.K_v = meta.get("K_v", 64)
            self._fv_any_valid = meta.get("fv_any_valid", None)
            self._flat_intermediate = meta.get("flat_intermediate", False)
            # Variant→original rule mapping (for all_anchors dedup/remap).
            if self._all_anchors:
                self.register_buffer(
                    "_variant_to_orig_t",
                    self._enum_ri.variant_to_orig.to(
                        dtype=torch.long, device=self.kb.device_))
            self.max_vars_per_rule = 3
        else:
            raise ValueError(f"Unknown resolution: {self.resolution}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _default_output_spec(self) -> OutputSpec:
        """The grounder's standing tier set (from its collect flags) — the BC
        request default when ``forward()`` is called without an explicit spec."""
        tiers = {Tier.PROOF_STATE}
        if self._collect_rule_groundings:
            tiers.add(Tier.FIRINGS)
        if self.collect_evidence:
            tiers.add(Tier.TREES)
        return OutputSpec(frozenset(tiers))

    @torch.no_grad()
    def ground(self, request: GroundRequest) -> BackwardResult:
        """The unified runtime verb — proof search over ``request.queries`` with
        the requested tiers (``request.output_spec``)."""
        self.output_spec = request.output_spec
        return run_backward(self, request.queries, request.query_mask,
                            excluded_queries=request.excluded_queries)

    @torch.no_grad()
    def forward(self, queries: Tensor, query_mask: Tensor, **init_kwargs) -> BackwardResult:
        """Compat shim — build a GroundRequest from the standing collect flags and
        delegate to ``ground()`` (byte-identical to the legacy forward path)."""
        excluded = init_kwargs.pop("excluded_queries", None)
        req = GroundRequest(queries=queries, query_mask=query_mask,
                            output_spec=self._default_output_spec(),
                            excluded_queries=excluded)
        if not init_kwargs:
            return self.ground(req)
        # Rare extra init kwargs (initial_goals/next_var) bypass GroundRequest.
        self.output_spec = req.output_spec
        return run_backward(self, queries, query_mask,
                            excluded_queries=excluded, **init_kwargs)

    __call__ = nn.Module.__call__

    def producible_tiers(self) -> "frozenset[Tier]":
        """BC can fill all three backward tiers."""
        return frozenset({Tier.PROOF_STATE, Tier.FIRINGS, Tier.TREES})

    def rebound(self, kb: KB) -> "BackwardGrounder":
        """Re-snapshot over a rewritten KB (transforms) with identical knobs."""
        return BackwardGrounder(kb, **self._ctor_kwargs)

    def capability_row(self) -> CapabilityRow:
        """Declared (layout, compile) cells: dense-eager/-graph/-dynamic + flat-eager (pbc and sld/rtf)."""
        if self.resolution == "pbc":
            cells = {Cell(Layout.DENSE, EAGER), Cell(Layout.DENSE, COMPILED_STEP),
                     Cell(Layout.DENSE, COMPILED_DYNAMIC), Cell(Layout.FLAT, EAGER)}
        else:  # sld / rtf
            cells = {Cell(Layout.DENSE, EAGER), Cell(Layout.DENSE, COMPILED_STEP),
                     Cell(Layout.DENSE, COMPILED_DYNAMIC), Cell(Layout.FLAT, EAGER)}
        return CapabilityRow(frozenset(cells))

    @torch.no_grad()
    def run_bc(self, queries: Tensor, query_mask: Tensor, **init_kwargs) -> RuleGroundings:
        """Rule-evidence entry point — forces collect_rule_groundings, returns rg."""
        prev = self._collect_rule_groundings
        self._collect_rule_groundings = True
        try:
            excluded = init_kwargs.pop("excluded_queries", None)
            spec = self._default_output_spec()  # forces FIRINGS (flag is True here)
            self.output_spec = spec
            out = run_backward(self, queries, query_mask,
                               excluded_queries=excluded, **init_kwargs)
        finally:
            self._collect_rule_groundings = prev
        rg = out.rule_groundings
        if rg is None:
            rg = RuleGroundings.empty(
                num_rules=int(getattr(self.kb, "num_rules", 0) or 0),
                M=self.kb.M, device=queries.device)
        return rg

    def __repr__(self) -> str:
        return (f"BackwardGrounder(resolution={self.resolution!r}, "
                f"filter={self.filter_mode!r}, depth={self.depth}, "
                f"width={self.width}, num_rules={self.kb.num_rules}, "
                f"S={self.S}, C={self.C})")


__all__ = ["BackwardGrounder"]
