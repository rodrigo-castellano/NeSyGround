"""BCGrounder — unified backward chaining with configurable resolution.

Configuration replaces classes:
  resolution: 'sld' | 'rtf' | 'enum'
  filter:     'fp_batch' | 'fp_global' | 'none'
  depth, width, hooks
  standardization: None | StandardizationConfig

Canonical loop (same code path for all resolutions):
  states = init_states(queries, query_mask)
  for d in range(D):
      states = step(states, d)   # SELECT → RESOLVE → PACK → POSTPROCESS
  return GrounderOutput(state, evidence)

Resolution is the only pluggable phase — _select, _pack, _postprocess are shared.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from grounder.bc.cache import GroundingCache
from grounder.config import GrounderConfig
from grounder.data.kb import KB
from grounder.resolution.standardization import StandardizationConfig
from grounder.types import GrounderOutput, ResolvedChildren

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook


def _cache_prop(name: str) -> property:
    """Delegate ``BCGrounder._<…>`` reads/writes to ``self._cache.<name>``
    (the :class:`grounder.bc.cache.GroundingCache`)."""
    return property(lambda self: getattr(self._cache, name),
                    lambda self, v: setattr(self._cache, name, v))


class BCGrounder(nn.Module):
    """Unified backward-chaining grounder BC_{w,d}.

    Configurable with orthogonal choices:
      depth (d):    number of proof steps
      width (w):    max unknown body atoms per grounding (enum only; None=∞)
      resolution:   'sld' | 'rtf' | 'enum'
      filter:       'fp_batch' | 'fp_global' | 'none'
      hooks:        GroundingHook list (post-grounding scoring/filtering)
      fact_hook:    ResolutionFactHook (filters fact candidates during resolution)
      rule_hook:    ResolutionRuleHook (filters rule candidates during resolution)
    """

    def __init__(
        self,
        kb: KB,
        *,
        depth: int = 2,
        width: Optional[int] = 1,
        resolution: str = "enum",
        filter: Optional[str] = None,
        max_total_groundings: int = 64,
        compile_mode: Optional[str] = None,
        hooks: Optional[List] = None,
        fact_hook: Optional[ResolutionFactHook] = None,
        rule_hook: Optional[ResolutionRuleHook] = None,
        # MGU params
        max_goals: Optional[int] = None,
        max_states: Optional[int] = None,
        K_MAX: int = 550,
        max_derived_per_state: Optional[int] = None,
        collect_evidence: bool = True,
        step_prune_dead: bool = False,
        max_groundings_per_rule: Optional[int] = None,
        # Enum params
        max_groundings_per_query: int = 32,
        fc_method: str = "spmm",
        fc_depth: int = 10,
        # Testing/validation enum params (not compile-compatible)
        cartesian_product: bool = False,
        all_anchors: bool = False,
        flat_intermediate: bool = False,
        pack_dedup: bool = True,
        collect_rule_groundings: bool = False,
        w_last_depth: Optional[int] = None,
        collect_mode: str = "terminal",
        # Output variable standardization (for consumers of ungrounded states)
        standardization: Optional[StandardizationConfig] = None,
        # Per-step ground-fact pruning: remove known facts from proof goals
        # between resolution steps. Disabled by default (standard SLD semantics
        # where every resolution step costs 1 depth). Enable for "compressed"
        # depth semantics where ground-fact goals are free.
        prune_facts: bool = False,
        # Bump ``S = max(S, K)`` for ``depth>1, width>0`` to keep every
        # intermediate child. Disabled defaults the per-step allocation
        # to ``[B*S_max, K_r, G_r, M, 3]`` — fits high-fan-out KBs at
        # the cost of truncating at packing. Default True (no-loss).
        bump_s_to_k: bool = True,
        # ``init_state_shape``:
        #   "minimal" (default) — depth-0 state has shape ``[B, 1, ...]``;
        #     only one slot per query is allocated. DpRL-friendly: lets
        #     callers stop after step 0 with the smallest possible
        #     buffer.
        #   "full" — depth-0 state has shape ``[B, S, ...]`` from the
        #     start, with only slot 0 valid. NS-friendly: every step
        #     sees the same shape, so a single compiled graph covers
        #     all depths (no d=0 specialization). Recommended when
        #     ``compile_mode`` is set.
        init_state_shape: str = "minimal",
        # Internal: a pre-built, already-validated GrounderConfig. ``from_config``
        # passes this to avoid re-deriving; public callers pass the kwargs above.
        _config: Optional["GrounderConfig"] = None,
    ) -> None:
        super().__init__()
        self.kb = kb
        # Expose num_rules on the grounder (consumers read it for buffer
        # sizing; the torch-ns adapter previously fell back to 0 via
        # ``getattr(grounder, 'num_rules', 0)`` because it was never set).
        self.num_rules = kb.num_rules

        # GrounderConfig is the single validated construction surface: its
        # ``__post_init__`` owns every kb-independent derivation (filter
        # default, forced all_anchors for enum, w_last_depth, step_width,
        # step_prune_dead, standardization_mode) and all combo validation.
        # BCGrounder just consumes the resolved config + does kb-dependent
        # wiring below.
        self.config = _config if _config is not None else GrounderConfig(
            resolution=resolution, filter=filter, depth=depth, width=width,
            max_total_groundings=max_total_groundings,
            max_groundings_per_query=max_groundings_per_query,
            max_groundings_per_rule=max_groundings_per_rule,
            max_goals=max_goals, max_states=max_states, K_MAX=K_MAX,
            max_derived_per_state=max_derived_per_state,
            w_last_depth=w_last_depth, compile_mode=compile_mode,
            init_state_shape=init_state_shape, bump_s_to_k=bump_s_to_k,
            collect_evidence=collect_evidence, collect_mode=collect_mode,
            collect_rule_groundings=collect_rule_groundings,
            pack_dedup=pack_dedup, hooks=hooks, fact_hook=fact_hook,
            rule_hook=rule_hook, fc_method=fc_method, fc_depth=fc_depth,
            cartesian_product=cartesian_product, all_anchors=all_anchors,
            flat_intermediate=flat_intermediate, standardization=standardization,
            prune_facts=prune_facts, step_prune_dead=step_prune_dead,
        )
        cfg = self.config

        # Optional (default-OFF) grounding-memoization state.
        self._init_grounding_caches()

        # ── Resolved configuration (all derivation done in GrounderConfig) ──
        self.depth = cfg.depth
        self.width = cfg.width
        self.resolution = cfg.resolution
        self.filter_mode = cfg.filter
        # Compile mode is opt-in (default None = eager). The dense enum path
        # pairs with 'reduce-overhead' (static shapes); the flat path stays
        # eager (its torch.nonzero is dynamic-shaped). Don't enable compile
        # globally in sweeps — CUDA-graph weakref bookkeeping accumulates.
        self.compile_mode = cfg.compile_mode
        self.hooks = list(cfg.hooks) if cfg.hooks else []
        self.fact_hook = cfg.fact_hook
        self.rule_hook = cfg.rule_hook
        self.step_hook = None  # Optional StepHook (nn.Module), set externally
        self.collect_evidence = cfg.collect_evidence
        self.prune_facts = cfg.prune_facts
        self._cartesian_product = cfg.cartesian_product
        self._all_anchors = cfg.all_anchors          # forced True for enum
        self._flat_intermediate_flag = cfg.flat_intermediate
        self._pack_dedup = cfg.pack_dedup
        self._collect_rule_groundings = cfg.collect_rule_groundings
        self._bump_s_to_k = cfg.bump_s_to_k
        self._init_state_shape = cfg.init_state_shape
        self._w_last_depth = cfg.w_last_depth        # defaulted to 0 (u=0)
        self._collect_mode = cfg.collect_mode
        self._step_width = cfg.step_width            # SLD/RTF only
        self._step_prune_dead = cfg.step_prune_dead_effective
        self.standardization_mode = cfg.standardization_mode

        # ── Shared layout: G, A, S, C (kb-dependent) ──
        # Standard symbols (see grounder/CLAUDE.md "Naming Convention").
        M = self.kb.M
        D = cfg.depth

        # G (max goals per state): M + (M-1)*D.
        max_goals = cfg.max_goals
        if max_goals is None:
            max_goals = M + (M - 1) * D
        self.max_goals = max(max_goals, M)

        # A (accumulated body capacity: D * M).
        self.A = D * M

        # S (max states per depth step).  Default 256.
        max_states = cfg.max_states if cfg.max_states is not None else 256
        self.S = max_states

        # C (collected groundings budget).
        self.C = cfg.max_total_groundings

        # Init resolution-specific params + compilation
        from grounder.bc.init_resolution import init_resolution
        init_resolution(
            self,
            max_states=max_states, K_MAX=cfg.K_MAX,
            max_derived_per_state=cfg.max_derived_per_state,
            max_total_groundings=cfg.max_total_groundings,
            max_groundings_per_rule=cfg.max_groundings_per_rule,
            max_groundings_per_query=cfg.max_groundings_per_query,
            fc_method=cfg.fc_method, fc_depth=cfg.fc_depth,
        )

        # Output variable standardization
        self._standardize_fn: Optional[Callable] = None
        if cfg.standardization is not None:
            from grounder.resolution.standardization import build_standardize_fn
            self._standardize_fn = build_standardize_fn(
                cfg.standardization, self.kb.device_)

    def _init_grounding_caches(self) -> None:
        """Compose the optional, default-OFF grounding-memoization state.

        The state now lives in its own :class:`grounder.bc.cache.GroundingCache`
        module rather than as ~15 attributes on the grounder; the standard
        depth-loop path never touches it (all layers default OFF). The ``_*``
        names the cache modules + external callers use are kept as thin
        delegating properties (see below), so nothing else changes.
        """
        self._cache = GroundingCache(self.kb)

    # ── Back-compat property surface over the GroundingCache module ──
    # The cache modules (bc/tabling.py, bc/subgoal.py) and external callers
    # (e.g. torch-ns's GROUNDER_RESOLVE_SKIP / GROUNDER_SUBGOAL_MEMO toggles)
    # still read/write ``grounder._tabling_enabled`` etc.; these delegate to
    # the single ``self._cache`` object so the state has one owner.
    _tabling_enabled = _cache_prop("tabling_enabled")
    _grounding_cache = _cache_prop("grounding_cache")
    _chunk_query_offset = _cache_prop("chunk_query_offset")
    _tabling_kb_stamp = _cache_prop("tabling_kb_stamp")
    _tabling_max_entries = _cache_prop("tabling_max_entries")
    _tabling_full_warned = _cache_prop("tabling_full_warned")
    _tabling_last_hits = _cache_prop("tabling_last_hits")
    _tabling_last_misses = _cache_prop("tabling_last_misses")
    _subgoal_enabled = _cache_prop("subgoal_enabled")
    _subgoal_memo = _cache_prop("subgoal_memo")
    _subgoal_max_entries = _cache_prop("subgoal_max_entries")
    _subgoal_full_warned = _cache_prop("subgoal_full_warned")
    _subgoal_last_hits = _cache_prop("subgoal_last_hits")
    _subgoal_last_misses = _cache_prop("subgoal_last_misses")
    _resolve_skip_enabled = _cache_prop("resolve_skip_enabled")

    @classmethod
    def from_config(cls, kb: KB, config: GrounderConfig) -> "BCGrounder":
        """Construct from a :class:`grounder.config.GrounderConfig`.

        The clean construction entry point. The config is already validated +
        derived (in its ``__post_init__``), so it's passed straight through —
        BCGrounder consumes it without re-deriving.
        """
        return cls(kb, _config=config)

    @torch.no_grad()
    def forward(
        self, queries: Tensor, query_mask: Tensor,
        *, batch_size: Optional[int] = None,
        **init_kwargs,
    ) -> GrounderOutput:
        """Ground all ``queries`` and return a ``GrounderOutput``.

        Dispatches to chunked or single-batch path depending on
        ``batch_size`` and compile/resolution combo. See
        :func:`grounder.bc.forward.forward` for the full dispatch rules.
        """
        from grounder.bc.forward import forward as _forward
        return _forward(
            self, queries, query_mask, batch_size=batch_size,
            **init_kwargs,
        )

    def _forward_one_batch_inner(
        self, queries: Tensor, query_mask: Tensor, **init_kwargs,
    ) -> GrounderOutput:
        """Bound-method wrapper for :func:`grounder.bc.forward.forward_one_batch_inner`.

        ``forward_chunked`` torch.compile-wraps this method (rather than
        a closure) so dynamo's per-instance specialization caches by
        ``self`` identity rather than re-tracing per closure object.
        """
        from grounder.bc.forward import forward_one_batch_inner
        return forward_one_batch_inner(self, queries, query_mask, **init_kwargs)

    @torch.no_grad()
    def run_bc(
        self, queries: Tensor, query_mask: Tensor,
        *, batch_size: Optional[int] = None,
        **init_kwargs,
    ):
        """Rule-evidence entry point — for SBR/DCR/R2N pool-iter consumers.

        See :func:`grounder.bc.forward.run_bc` for details.
        """
        from grounder.bc.forward import run_bc as _run_bc
        return _run_bc(
            self, queries, query_mask, batch_size=batch_size,
            **init_kwargs,
        )

    def _apply_hooks(
        self,
        resolved: ResolvedChildren,
        states: Dict[str, Tensor],
    ) -> ResolvedChildren:
        """Apply resolution hooks. Subclasses override for RL filtering."""
        return resolved


    # ==================================================================
    # Helpers
    # ==================================================================

    def __repr__(self) -> str:
        return (
            f"BCGrounder(resolution={self.resolution!r}, "
            f"filter={self.filter_mode!r}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_rules={self.kb.num_rules}, "
            f"S={self.S}, C={self.C})")
