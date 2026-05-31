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

from grounder.data.kb import KB
from grounder.resolution.standardization import StandardizationConfig
from grounder.types import GrounderOutput, ResolvedChildren

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook


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
    ) -> None:
        super().__init__()
        self.kb = kb
        # Expose num_rules on the grounder (consumers read it for buffer
        # sizing; the torch-ns adapter previously fell back to 0 via
        # ``getattr(grounder, 'num_rules', 0)`` because it was never set).
        self.num_rules = kb.num_rules

        # ── Per-query grounding tabling cache (task #47, Design #2) ──
        # Grounding is weight-independent (facts + rules are fixed for the
        # grounder's lifetime), so each query atom's RAW (pre-prune)
        # "considered" firing set is byte-identical across calls. When
        # ``_tabling_enabled`` is True the forward chain serves recurring
        # queries from ``_grounding_cache`` instead of re-running the depth
        # loop. Default OFF — the existing path is untouched unless a caller
        # flips ``grounder._tabling_enabled = True``.
        self._tabling_enabled = False
        self._grounding_cache: Dict[int, Any] = {}
        # Query-index offset for the considered accumulator's bidx column.
        # Reset per public ``forward`` (and per chunk); reconciles the
        # accumulator's batch-local ``b_idx`` with the write-back's
        # ``miss_positions`` map. 0 for the single-batch path.
        self._chunk_query_offset = 0
        # KB-immutability stamp — the cache is only valid while facts + rules
        # are unchanged. Asserted on every consult.
        self._tabling_kb_stamp = (id(self.kb.fact_index), int(self.kb.num_rules))
        # Cap on distinct cached query keys. When full, stop inserting new
        # keys but keep serving hits + computing misses fresh (passthrough).
        self._tabling_max_entries = 2_000_000
        self._tabling_full_warned = False

        # ── Per-subgoal grounding memo (task #48, Design #2) ──
        # Memoizes each SELECTED SUBGOAL's considered firing set keyed by
        # ``(selected_goal_atom_hash, is_last)`` — finer granularity than the
        # per-query cache, so subgoals shared ACROSS queries (and across
        # epochs) reuse one stored set. Default OFF; built on top of #47.
        # See :mod:`grounder.bc.subgoal`.
        self._subgoal_enabled = False
        self._subgoal_memo: Dict[Any, Any] = {}
        self._subgoal_max_entries = 2_000_000
        self._subgoal_full_warned = False

        # ── Resolve-SKIP (task #48 Phase-2) ──
        # Within-batch goal dedup in the flat enum resolve: run the
        # expensive enumerate/fill/exists/filter pipeline ONCE per distinct
        # selected goal, then scatter survivors to every state sharing it.
        # Byte-identical (survivor set is goal-determined; per-state child =
        # survivors + that state's own remaining goals + grounding_body).
        # Default OFF. See ``resolution/enum.py:_resolve_enum_step_flat``.
        self._resolve_skip_enabled = False

        self.depth = depth
        self.width = width
        self.resolution = resolution
        # Default filter is the paper BC_{w,d,u=0} convention for enum:
        # 'fp_batch' (keras ``prune_incomplete_proofs=True`` equivalent).
        # Callers that want u>0 semantics (admit unknown leaves) should
        # pass ``filter='none'`` explicitly along with ``w_last_depth>0``;
        # ``make_bcwd`` does this via its ``u`` parameter.
        # SLD/RTF have no parity story; default 'none'.
        if filter is None:
            if resolution == "closure":
                filter = "none"
            else:
                filter = "fp_batch" if resolution == "enum" else "none"
        self.filter_mode = filter
        # Compile mode is opt-in (default None = eager). The dense
        # ``enum`` path is the right pairing for ``'reduce-overhead'``
        # (static shapes, CUDA-graph capture); the flat path stays
        # eager because its ``torch.nonzero`` produces dynamic shapes
        # incompatible with reduce-overhead. Multi-grounder sweeps
        # should NOT enable compile globally — torch's CUDA-graph
        # weakref bookkeeping accumulates across grounders.
        self.compile_mode = compile_mode
        self.hooks = hooks or []
        self.fact_hook = fact_hook
        self.rule_hook = rule_hook
        self.step_hook = None  # Optional StepHook (nn.Module), set externally
        self.collect_evidence = collect_evidence
        self.prune_facts = prune_facts
        # Enum defaults:
        #   all_anchors=True              — try every body atom as anchor
        #                                    (matches keras's per-i loop;
        #                                    forced for correctness because
        #                                    anchoring only on the first body
        #                                    atom misses bindings keras finds)
        #   cartesian_product=False       — fact-anchored enumeration: candidates
        #                                    come from ``fact_index.enumerate``
        #                                    (the partial atom lookup), not from
        #                                    the full entity domain. This
        #                                    matches keras's
        #                                    ``fact_index._index.get(partial_atom)``
        #                                    and keeps the d=1 body tensor
        #                                    bounded by ``G_r`` (typically
        #                                    ``min(K_f, max_groundings_per_query)``)
        #                                    instead of ``E`` (entity count).
        # Callers may opt back into ``cartesian_product=True`` for full-domain
        # exploration; the count is identical (after filtering) because both
        # admit the same set of valid groundings.
        #
        # ``collect_rule_groundings`` is NOT forced for enum: speed-only
        # callers (test_speed.py) pass ``False`` so the per-step
        # ``_step_compiled`` accumulators stay empty. Forcing True caused
        # the chunked path to leak ~100 MB / chunk on wn18rr (step_body /
        # step_head / step_ridx clones from every chunk's depth=1 step
        # piling up across all 293 chunks until ~24 GB OOM).
        if resolution == "closure":
            pass  # closure never touches enum-specific anchor variants
        elif resolution == "enum":
            if not all_anchors:
                all_anchors = True
        self._cartesian_product = cartesian_product
        self._all_anchors = all_anchors
        self._flat_intermediate_flag = flat_intermediate
        self._pack_dedup = pack_dedup
        self._collect_rule_groundings = collect_rule_groundings
        self._bump_s_to_k = bump_s_to_k
        if init_state_shape not in ("minimal", "full"):
            raise ValueError(
                f"init_state_shape must be 'minimal' or 'full', "
                f"got {init_state_shape!r}")
        self._init_state_shape = init_state_shape
        # Paper BC_{w,d,u} convention: u (= w_last_depth) defaults to 0.
        # All body atoms at the last (= terminal) step must be facts;
        # any rule application with leftover unknown leaves is dropped
        # by terminal collection. Callers that want u>0 (admit unknown
        # leaves; e.g. depth=1 with width>0 to surface single-rule
        # applications as in keras-ns ``prune_incomplete_proofs=False``
        # tests) pass ``w_last_depth=u`` explicitly. ``make_bcwd``
        # exposes this as the ``u`` parameter.
        if w_last_depth is None:
            w_last_depth = 0
        self._w_last_depth = w_last_depth
        self._collect_mode = collect_mode

        # Per-step search filters
        self._step_width = width if resolution in ("sld", "rtf") and width is not None else None

        # prune_dead: only for SLD/RTF
        if step_prune_dead and resolution == "enum":
            import warnings
            warnings.warn(
                "step_prune_dead has no effect with enum resolution "
                "(all body atoms are ground). Ignoring.",
                stacklevel=2,
            )
        self._step_prune_dead = step_prune_dead and resolution in ("sld", "rtf")

        self.standardization_mode = standardization.mode if standardization else None

        # ── Shared layout: G, A, S, C ──
        # Standard symbols (see grounder/CLAUDE.md "Naming Convention").
        M = self.kb.M
        D = depth

        # G (max goals per state): M + (M-1)*D.
        if max_goals is None:
            max_goals = M + (M - 1) * D
        self.max_goals = max(max_goals, M)

        # A (accumulated body capacity: D * M).
        self.A = D * M

        # S (max states per depth step).  Default 256.
        if max_states is None:
            max_states = 256
        self.S = max_states

        # C (collected groundings budget).
        self.C = max_total_groundings

        # Init resolution-specific params + compilation
        from grounder.bc.init_resolution import init_resolution
        init_resolution(
            self,
            max_states=max_states, K_MAX=K_MAX,
            max_derived_per_state=max_derived_per_state,
            max_total_groundings=max_total_groundings,
            max_groundings_per_rule=max_groundings_per_rule,
            max_groundings_per_query=max_groundings_per_query,
            fc_method=fc_method, fc_depth=fc_depth,
        )

        # Output variable standardization
        self._standardize_fn: Optional[Callable] = None
        if standardization is not None:
            from grounder.resolution.standardization import build_standardize_fn
            self._standardize_fn = build_standardize_fn(standardization, self.kb.device_)

    @classmethod
    def from_config(cls, kb: KB, config) -> "BCGrounder":
        """Construct from a :class:`grounder.config.GrounderConfig`.

        The clean construction entry point (Phase 2a). Behavior-neutral:
        expands the config to the exact legacy constructor kwargs.
        """
        return cls(kb, **config.as_kwargs())

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
