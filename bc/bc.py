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

from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data.kb import KB
from grounder.resolution.primitives import apply_substitutions
from grounder.resolution.standardization import StandardizationConfig

if TYPE_CHECKING:
    from grounder.nesy.hooks import ResolutionFactHook, ResolutionRuleHook
from grounder.bc.common import (
    compact_atoms,
    collect_groundings,
    pack_states,
    prune_ground_facts,
)
from grounder.types import (
    GrounderOutput, ProofEvidence, ProofState, ResolvedChildren, SyncParams,
)
from grounder.filters import check_in_fp_global
from grounder.filters.search import filter_width, filter_prune_dead
from grounder.resolution.sld import resolve_sld
from grounder.resolution.rtf import resolve_rtf
from grounder.resolution.enum import resolve_enum_step


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
        filter: str = "fp_batch",
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
        track_grounding_body: bool = True,
        step_prune_dead: bool = False,
        max_groundings_per_rule: Optional[int] = None,
        # Enum params
        max_groundings_per_query: int = 32,
        fc_method: str = "join",
        fc_depth: int = 10,
        # Output variable standardization (for consumers of ungrounded states)
        standardization: Optional[StandardizationConfig] = None,
        # Per-step ground-fact pruning: remove known facts from proof goals
        # between resolution steps. Disabled by default (standard SLD semantics
        # where every resolution step costs 1 depth). Enable for "compressed"
        # depth semantics where ground-fact goals are free.
        prune_facts: bool = False,
    ) -> None:
        super().__init__()
        self.kb = kb

        self.depth = depth
        self.width = width
        self.resolution = resolution
        self.filter_mode = filter
        self.compile_mode = compile_mode
        self.hooks = hooks or []
        self.fact_hook = fact_hook
        self.rule_hook = rule_hook
        self.step_hook = None  # Optional StepHook (nn.Module), set externally
        self.track_grounding_body = track_grounding_body
        self.prune_facts = prune_facts

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

        # Max goals: shared for all resolutions.
        # Must accommodate M body atoms for enum resolution.
        if max_goals is None:
            max_goals = 1 + depth * max(self.kb.M - 1, 1)
        self.max_goals = max(max_goals, self.kb.M)
        # Body capacity: depth * M (each depth adds up to M body atoms)
        self.max_body_capacity = depth * self.kb.M

        # Init resolution + compilation
        self._init_resolution(
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

    # ==================================================================
    # Resolution init
    # ==================================================================

    def _init_resolution(self, **kwargs) -> None:
        """Call resolution module's init, apply results, set up compilation."""
        if self.resolution in ("sld", "rtf"):
            from grounder.resolution.mgu import init_mgu
            cfg = init_mgu(
                resolution=self.resolution,
                K_f=self.kb.K_f, K_r=self.kb.K_r,
                rule_index=self.kb.rule_index,
                max_total_groundings=kwargs["max_total_groundings"],
                K_MAX=kwargs["K_MAX"],
                max_derived_per_state=kwargs["max_derived_per_state"],
                max_states=kwargs["max_states"],
                max_groundings_per_rule=kwargs["max_groundings_per_rule"],
            )
            self.K = cfg["K"]
            self.S = cfg["S"]
            self.kb.K_f = cfg["K_f"]
            self.max_vars_per_rule = cfg["max_vars_per_rule"]
            self.effective_total_G = cfg["effective_total_G"]
            self._max_fact_pairs_body = cfg["max_fact_pairs_body"]

        elif self.resolution == "enum":
            from grounder.resolution.enum import init_enum
            meta = init_enum(
                rule_index=self.kb.rule_index,
                fact_index=self.kb.fact_index,
                facts_idx=self.kb.fact_index.facts_idx,
                constant_no=self.kb.constant_no,
                num_rules=self.kb.num_rules, M=self.kb.M,
                width=self.width,
                max_groundings_per_query=kwargs["max_groundings_per_query"],
                max_total_groundings=kwargs["max_total_groundings"],
                max_states=kwargs["max_states"],
                device=self.kb.device_,
            )
            for name, tensor in meta["buffers"].items():
                self.register_buffer(name, tensor)
            self._enum_ri = meta["enum_rule_index"]
            self.max_body_atoms = self._enum_ri.max_body
            self._P, self._E = meta["P"], meta["E"]
            self.R_eff = meta["R_eff"]
            self._K_enum = meta["K_enum"]
            self.S = meta["S"]
            self.effective_total_G = meta["effective_total_G"]
            self.any_dual = meta["any_dual"]
            self._enum_G = meta["enum_G"]
            self.fc_method = kwargs["fc_method"]
            self.fc_depth = kwargs["fc_depth"]
            self.max_vars_per_rule = 3  # unused for enum, but keeps state uniform

        else:
            raise ValueError(f"Unknown resolution: {self.resolution}")

        # Compilation (all resolutions — shapes are static)
        self._compiled = False
        self._clone_between_steps = False
        if (self.compile_mode
                and self.depth > 1
                and self.kb.device_.type == "cuda"):
            self._fn_step = torch.compile(
                self._step_impl, fullgraph=True,
                mode=self.compile_mode)
            self._clone_between_steps = (
                self.compile_mode == "reduce-overhead")
            self._compiled = True
            self._multi_step = True

        # Per-step search filter buffers
        if self._step_prune_dead:
            P = self.kb.predicate_no + 1
            head_pred_mask = torch.zeros(P, dtype=torch.bool, device=self.kb.device_)
            head_preds = self.kb.rule_index.rules_heads_sorted[:, 0]
            head_pred_mask.scatter_(0, head_preds, True)
            self.register_buffer("_step_head_pred_mask", head_pred_mask)

            fi = self.kb.fact_index
            if hasattr(fi, '_a0_offsets'):
                a0_lens = fi._a0_offsets[1:] - fi._a0_offsets[:-1]
                a1_lens = fi._a1_offsets[1:] - fi._a1_offsets[:-1]
                self.register_buffer("_step_a0_lens", a0_lens)
                self.register_buffer("_step_a1_lens", a1_lens)
                self._step_key_scale = fi._key_scale
                self._step_has_csr = True
            else:
                self._step_has_csr = False
            if hasattr(fi, '_p_offsets'):
                p_lens = fi._p_offsets[1:] - fi._p_offsets[:-1]
                self.register_buffer("_step_p_lens", p_lens)

        # fp_global set I_D (for fp_global filter)
        self._has_fp_global = False
        if self.filter_mode == "fp_global":
            if self.resolution == "enum":
                self._build_fp_global_set(self.kb.device_)
            else:
                # SLD/RTF: build a temporary RuleIndexEnum for FC patterns
                from grounder.data.rule_index import RuleIndexEnum
                P = self.kb.predicate_no + 1
                E = self.kb.constant_no + 1
                enum_ri = RuleIndexEnum(
                    self.kb.rule_index.rules_heads_sorted,
                    self.kb.rule_index.rules_bodies_sorted,
                    self.kb.rule_index.rule_lens_sorted,
                    constant_no=self.kb.constant_no,
                    num_predicates=P,
                    padding_idx=self.kb.padding_idx,
                    device=self.kb.device_,
                )
                self._build_fp_global_set(
                    self.kb.device_, compiled_rules=enum_ri.patterns,
                    P=P, E=E,
                )

    def _build_fp_global_set(
        self, device: torch.device,
        compiled_rules=None, P: int = 0, E: int = 0,
    ) -> None:
        from grounder.fc.fc import run_forward_chaining
        if compiled_rules is None:
            compiled_rules = self._enum_ri.patterns
        if P == 0:
            P = self._P
        if E == 0:
            E = self._E
        method = getattr(self, 'fc_method', 'dynamic')
        if method in ("join", "spmm"):
            method = "dynamic"
        fc_depth = getattr(self, 'fc_depth', 10)
        fp_global_tensor, n_fp_global = run_forward_chaining(
            compiled_rules=compiled_rules,
            facts_idx=self.kb.fact_index.facts_idx,
            num_entities=E,
            num_predicates=P,
            depth=fc_depth,
            device=str(device),
        )
        self.register_buffer("fp_global_hashes", fp_global_tensor)
        self.register_buffer(
            "num_fp_global",
            torch.tensor(n_fp_global, dtype=torch.long, device=device))
        self._has_fp_global = n_fp_global > 0
        self._P_fp_global = P
        self._E_fp_global = E

    # ==================================================================
    # Canonical loop
    # ==================================================================

    @torch.no_grad()
    def forward(
        self, queries: Tensor, query_mask: Tensor, **init_kwargs,
    ) -> GrounderOutput:
        states = self.init_states(queries, query_mask, **init_kwargs)
        for d in range(self.depth):
            states = self.step(states, d)
            if self.step_hook is not None:
                cb, cm, cr = self.step_hook.on_step(
                    states["collected_body"], states["collected_mask"],
                    states["collected_ridx"], d)
                states["collected_body"] = cb
                states["collected_mask"] = cm
                states["collected_ridx"] = cr
        evidence = self.filter_terminal(states)
        # filter='none' returns raw states dict — wrap in ProofEvidence
        # outside the compiled region (dataclass init breaks fullgraph).
        if isinstance(evidence, dict):
            evidence = ProofEvidence(
                body=evidence["collected_body"],
                mask=evidence["collected_mask"],
                count=evidence["collected_mask"].sum(dim=1),
                rule_idx=evidence["collected_ridx"],
                body_count=evidence["collected_bcount"],
            )
        for hook in self.hooks:
            body, mask, ridx = hook.apply(evidence.body, evidence.mask, evidence.rule_idx)
            evidence = ProofEvidence(
                body=body, mask=mask, count=mask.sum(dim=1), rule_idx=ridx,
                body_count=evidence.body_count)
        state = ProofState(
            proof_goals=states["proof_goals"],
            state_valid=states["state_valid"],
            top_ridx=states["top_ridx"],
        )
        return GrounderOutput(state=state, evidence=evidence)


    def init_states(
        self, queries: Tensor, query_mask: Tensor,
        *,
        initial_goals: Optional[Tensor] = None,
        next_var_indices: Optional[Tensor] = None,
        excluded_queries: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Build initial states dict for the proof loop.

        Args:
            queries: [B, 3] query atoms.
            query_mask: [B] validity mask.
            initial_goals: [B, M_in, 3] multi-atom goal list to use instead
                of the single query atom (for RL mid-proof entry).
            next_var_indices: [B] pre-allocated variable counters. Defaults
                to ``constant_no + 1`` (fresh).
            excluded_queries: optional tensor for cycle prevention.
        """
        B = queries.size(0)
        dev = queries.device
        pad = self.kb.padding_idx
        G = self.max_goals
        tG = self.effective_total_G
        M = self.kb.M  # max body atoms in any single rule
        # G_body: accumulated body capacity across all depths.
        # Each depth step adds up to M body atoms (one rule application).
        # Upper bound = depth * M (not G, which bounds open goals).
        G_body = 1 if not self.track_grounding_body else self.depth * M
        # M_work: working buffer for the current depth's body atoms.
        M_work = 1 if not self.track_grounding_body else M

        proof_goals = torch.full(
            (B, 1, G, 3), pad, dtype=torch.long, device=dev)
        if initial_goals is not None:
            M_in = initial_goals.shape[1]
            proof_goals[:, 0, :M_in, :] = initial_goals
        else:
            proof_goals[:, 0, 0, :] = queries
        # M-sized working buffer (current depth's rule body atoms)
        grounding_body = torch.full(
            (B, 1, M_work, 3), pad, dtype=torch.long, device=dev)
        # G_body-sized accumulator (all depths' body atoms)
        accumulated_body = torch.full(
            (B, 1, G_body, 3), pad, dtype=torch.long, device=dev)
        body_count = torch.zeros(B, 1, dtype=torch.long, device=dev)
        top_ridx = torch.full((B, 1), -1, dtype=torch.long, device=dev)
        state_valid = query_mask.unsqueeze(1)

        if next_var_indices is None:
            E = self.kb.constant_no + 1
            next_var_indices = torch.full(
                (B,), E, dtype=torch.long, device=dev)

        states = {
            "queries": queries,
            "query_mask": query_mask,
            "proof_goals": proof_goals,
            "grounding_body": grounding_body,
            "accumulated_body": accumulated_body,
            "body_count": body_count,
            "top_ridx": top_ridx,
            "state_valid": state_valid,
            "next_var_indices": next_var_indices,
            "initial_next_var": next_var_indices,
            "collected_body": queries.new_zeros(B, tG, G_body, 3),
            "collected_mask": torch.zeros(B, tG, dtype=torch.bool, device=dev),
            "collected_ridx": queries.new_zeros(B, tG),
            "collected_bcount": torch.zeros(B, tG, dtype=torch.long, device=dev),
        }
        if initial_goals is not None:
            states["initial_goals"] = initial_goals
        if excluded_queries is not None:
            states["excluded_queries"] = excluded_queries
        return states

    def step(self, states: Dict[str, Tensor], d: int) -> Dict[str, Tensor]:
        """One proof step: SELECT → RESOLVE → PACK → POSTPROCESS."""
        if self.kb.num_rules == 0:
            return states

        # Compiled fast path (all depths; skip last enum step which needs width=0)
        if self._compiled:
            last_enum_step = (
                self.resolution == "enum"
                and d == self.depth - 1
                and self.width is not None
            )
            if not last_enum_step:
                return self._step_compiled(states)

        # ── SELECT ──
        goal_queries, remaining, active_mask = self._select(states)

        # ── RESOLVE ──
        resolved = self._resolve(
            goal_queries, remaining,
            states["grounding_body"], states["state_valid"],
            active_mask, states, d,
        )

        # ── SEARCH FILTERS (between RESOLVE and PACK) ──
        resolved = self._apply_search_filters(resolved)

        # ── HOOKS (between RESOLVE and PACK) ──
        resolved = self._apply_hooks(resolved, states)

        # ── PACK → returns (states, sync) — no dict pollution ──
        states, sync = self._pack(resolved, states)

        # ── POSTPROCESS ──
        states = self._postprocess(states, sync)

        return states

    def filter_terminal(self, states: Dict[str, Tensor]):
        """Apply soundness filter on collected groundings -> ProofEvidence.

        When ``filter='none'``, returns the raw states dict (no collection).
        """
        if self.filter_mode == "none":
            return states

        B = states["collected_body"].size(0)
        tG = self.effective_total_G
        G_body = states["collected_body"].shape[2]  # G (accumulated body dim)
        dev = states["collected_body"].device

        body = states["collected_body"]
        mask = states["collected_mask"]
        ridx = states["collected_ridx"]

        if self.kb.num_rules == 0:
            return self._empty_result(B, tG, G_body, dev)

        if self.filter_mode == "fp_batch":
            from grounder.filters.soundness.fp_batch import apply_fp_batch
            mask = apply_fp_batch(
                body, mask, states["queries"], self.kb.fact_index,
                self.kb.fact_index.pack_base, self.kb.padding_idx, self.depth)

        elif self.filter_mode == "fp_global":
            from grounder.filters.soundness.fp_global import apply_fp_global
            mask = apply_fp_global(
                body, mask, self.kb.fact_index,
                self.kb.fact_index.pack_base, self.kb.padding_idx,
                self.fp_global_hashes)

        count = mask.sum(dim=1)
        bcount = states["collected_bcount"]
        return ProofEvidence(
            body=body, mask=mask,
            count=count, rule_idx=ridx,
            body_count=bcount,
        )

    # ==================================================================
    # Phase 1: SELECT (shared)
    # ==================================================================

    def _select(
        self, states: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract first goal from each proof state."""
        proof_goals = states["proof_goals"]
        active_mask = proof_goals[:, :, 0, 0] != self.kb.padding_idx
        queries = proof_goals[:, :, 0, :]
        queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
        remaining = proof_goals.clone()
        remaining[:, :, 0, :] = self.kb.padding_idx
        return queries, remaining, active_mask

    # ==================================================================
    # Phase 2: RESOLVE (dispatches to resolution module)
    # ==================================================================

    def _resolve(
        self,
        queries: Tensor,           # [B, S, 3]
        remaining: Tensor,         # [B, S, G, 3]
        grounding_body: Tensor,    # [B, S, M, 3]
        state_valid: Tensor,       # [B, S]
        active_mask: Tensor,       # [B, S]
        states: Dict[str, Tensor],
        d: int,
        use_hooks: bool = True,
    ) -> ResolvedChildren:
        """Dispatch to resolution strategy. Returns ResolvedChildren."""
        fh = self.fact_hook if use_hooks else None
        rh = self.rule_hook if use_hooks else None

        if self.resolution == "sld":
            return resolve_sld(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices=states["next_var_indices"],
                fact_index=self.kb.fact_index, facts_idx=self.kb.fact_index.facts_idx,
                rule_index=self.kb.rule_index,
                constant_no=self.kb.constant_no, padding_idx=self.kb.padding_idx,
                K_f=self.kb.K_f, K_r=self.kb.K_r,
                max_vars_per_rule=self.max_vars_per_rule,
                num_rules=self.kb.num_rules,
                track_grounding_body=self.track_grounding_body,
                excluded_queries=states.get("excluded_queries"),
                fact_hook=fh, rule_hook=rh,
            )
        elif self.resolution == "rtf":
            return resolve_rtf(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices=states["next_var_indices"],
                fact_index=self.kb.fact_index, facts_idx=self.kb.fact_index.facts_idx,
                rule_index=self.kb.rule_index,
                constant_no=self.kb.constant_no, padding_idx=self.kb.padding_idx,
                K_f=self.kb.K_f, K_r=self.kb.K_r, K=self.K,
                max_vars_per_rule=self.max_vars_per_rule,
                num_rules=self.kb.num_rules,
                max_fact_pairs_body=self._max_fact_pairs_body,
                track_grounding_body=self.track_grounding_body,
                fact_hook=fh, rule_hook=rh,
            )
        else:
            return resolve_enum_step(
                queries, remaining, grounding_body, state_valid, active_mask,
                fact_index=self.kb.fact_index,
                d=d, depth=self.depth, width=self.width,
                M=self.kb.M, padding_idx=self.kb.padding_idx,
                enum_G=self._enum_G, K_enum=self._K_enum,
                any_dual=self.any_dual,
                pred_rule_indices=self.pred_rule_indices,
                pred_rule_mask=self.pred_rule_mask,
                has_free=self.has_free,
                body_preds=self.body_preds,
                num_body_atoms=self.num_body_atoms,
                enum_pred_a=self.enum_pred_a,
                enum_bound_binding_a=self.enum_bound_binding_a,
                enum_direction_a=self.enum_direction_a,
                check_arg_source_a=self.check_arg_source_a,
                head_pred_mask=self.head_pred_mask,
                has_dual=getattr(self, "has_dual", None),
                enum_pred_b=getattr(self, "enum_pred_b", None),
                enum_bound_binding_b=getattr(self, "enum_bound_binding_b", None),
                enum_direction_b=getattr(self, "enum_direction_b", None),
                check_arg_source_b=getattr(self, "check_arg_source_b", None),
                track_grounding_body=self.track_grounding_body,
            )

    # ==================================================================
    # Hooks (between RESOLVE and PACK)
    # ==================================================================

    def _apply_hooks(
        self,
        resolved: ResolvedChildren,
        states: Dict[str, Tensor],
    ) -> ResolvedChildren:
        """Apply resolution hooks. Subclasses override for RL filtering."""
        return resolved

    # ==================================================================
    # Per-step search filters (between RESOLVE and PACK)
    # ==================================================================

    def _apply_search_filters(
        self,
        resolved: ResolvedChildren,
    ) -> ResolvedChildren:
        """Per-step search filters. No gradients, zero overhead when disabled."""
        if not self._step_prune_dead and self._step_width is None:
            return resolved

        (fg, fgb, fs, rule_goals, rgb, rule_success, sri,
         f_subs, r_subs) = resolved

        if self._step_prune_dead:
            rule_success = filter_prune_dead(
                rule_goals, rule_success,
                head_pred_mask=self._step_head_pred_mask,
                fact_index=self.kb.fact_index,
                constant_no=self.kb.constant_no,
                padding_idx=self.kb.padding_idx,
                M=self.kb.M,
                a0_lens=self._step_a0_lens if self._step_has_csr else None,
                a1_lens=self._step_a1_lens if self._step_has_csr else None,
                p_lens=getattr(self, '_step_p_lens', None),
                key_scale=self._step_key_scale if self._step_has_csr else 0,
            )

        if self._step_width is not None:
            rule_success = filter_width(
                rule_goals, rule_success,
                fact_index=self.kb.fact_index,
                constant_no=self.kb.constant_no,
                padding_idx=self.kb.padding_idx,
                M=self.kb.M,
                width=self._step_width,
            )

        return ResolvedChildren(fg, fgb, fs, rule_goals, rgb, rule_success,
                                sri, f_subs, r_subs)

    # ==================================================================
    # Phase 3: PACK (shared)
    # ==================================================================

    def _pack(
        self,
        resolved: ResolvedChildren,
        states: Dict[str, Tensor],
    ) -> Tuple[Dict, SyncParams]:
        """Flatten S*K children, propagate grounding body, compact to S.

        Returns (states, sync) — no dict pollution with underscore keys.
        """
        S_in = states["top_ridx"].shape[1]

        packed = pack_states(
            *resolved,
            states["top_ridx"], states["grounding_body"],
            states["body_count"],
            self.S, self.kb.padding_idx,
            track_grounding_body=self.track_grounding_body,
            M_rule=self.kb.M,
        )

        states["grounding_body"] = packed.grounding_body
        states["proof_goals"] = packed.proof_goals
        states["top_ridx"] = packed.top_ridx
        states["state_valid"] = packed.state_valid

        sync = SyncParams(
            parent_map=packed.parent_map,
            winning_subs=packed.winning_subs,
            has_new_body=packed.has_new_body,
            parent_bcount=packed.body_count,
        )

        states["next_var_indices"] = (
            states["next_var_indices"] + S_in * self.max_vars_per_rule)
        return states, sync

    def _sync_accumulated(
        self,
        states: Dict[str, Tensor],
        sync: SyncParams,
    ) -> Dict[str, Tensor]:
        """Propagate accumulated_body: gather from parents, apply subs, append new atoms.

        All operations are static-shape and torch.compile(fullgraph=True) compatible.

        Args:
            states: Current states dict with accumulated_body and grounding_body.
            sync: SyncParams with parent_map, winning_subs, has_new_body, parent_bcount.
        """
        parent_map = sync.parent_map
        winning_subs = sync.winning_subs
        has_new_body = sync.has_new_body
        parent_bcount = sync.parent_bcount

        if not self.track_grounding_body:
            states["body_count"] = parent_bcount
            return states

        B, S_out = parent_map.shape
        G_body = states["accumulated_body"].shape[2]
        M_work = states["grounding_body"].shape[2]
        pad = self.kb.padding_idx
        dev = parent_map.device

        # a. Gather accumulated_body from parents
        pi = parent_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, G_body, 3)
        acc = states["accumulated_body"].gather(1, pi)  # [B, S_out, G_body, 3]

        # b. Gather body_count from parents (already passed as parent_bcount)
        bc = parent_bcount  # [B, S_out]

        # c. Apply winning substitutions to accumulated_body
        acc_flat = acc.reshape(B * S_out, G_body, 3)
        subs_flat = winning_subs.reshape(B * S_out, 2, 2)
        acc_flat = apply_substitutions(acc_flat, subs_flat, pad)
        acc = acc_flat.reshape(B, S_out, G_body, 3)

        # d. Append new body atoms from grounding_body where has_new_body
        new_atoms = states["grounding_body"]  # [B, S_out, M_work, 3]
        new_active = (new_atoms[:, :, :, 0] != pad)  # [B, S_out, M_work]
        new_lens = new_active.long().sum(dim=-1)  # [B, S_out]

        local_idx = torch.arange(M_work, device=dev).view(1, 1, M_work)
        raw_write_pos = bc.unsqueeze(-1) + local_idx  # [B, S_out, M_work]
        write_pos = raw_write_pos.clamp(max=G_body - 1)  # [B, S_out, M_work]
        write_mask = (
            has_new_body.unsqueeze(-1)
            & new_active
            & (raw_write_pos < G_body)
        )  # [B, S_out, M_work]

        wi = write_pos.unsqueeze(-1).expand(-1, -1, -1, 3)
        existing = acc.gather(2, wi)
        to_write = torch.where(write_mask.unsqueeze(-1), new_atoms, existing)
        acc.scatter_(2, wi, to_write)

        # e. Update body_count
        bc = bc + torch.where(has_new_body, new_lens, torch.zeros_like(new_lens))
        bc = bc.clamp(max=G_body)

        states["accumulated_body"] = acc
        states["body_count"] = bc
        return states

    # ==================================================================
    # Phase 4: POSTPROCESS (shared)
    # ==================================================================

    def _postprocess_goals(self, states: Dict) -> Dict[str, Tensor]:
        """Optionally prune ground facts, then compact atoms.

        When ``prune_facts=True``, known ground facts are removed from
        proof_goals between steps (compressed depth semantics).
        When ``prune_facts=False`` (default), only compaction is applied
        (standard SLD semantics where every resolution costs 1 depth).

        Safe for torch.compile — ``self.prune_facts`` is a static Python bool.
        """
        if self.prune_facts:
            proof_goals, _, _ = prune_ground_facts(
                states["proof_goals"], states["state_valid"],
                self.kb.fact_index.fact_hashes, self.kb.fact_index.pack_base,
                self.kb.constant_no, self.kb.padding_idx,
                excluded_queries=states.get("excluded_queries"),
            )
            states["proof_goals"] = compact_atoms(proof_goals, self.kb.padding_idx)
        else:
            states["proof_goals"] = compact_atoms(
                states["proof_goals"], self.kb.padding_idx)
        return states

    def _collect_groundings(self, states: Dict) -> Dict[str, Tensor]:
        """Collect completed groundings into output buffer.

        Uses accumulated_body (G_body-sized). Called outside the compiled
        step to keep G_body tensors out of the CUDA graph.
        """
        cb, cm, cr, sv, c_bc = collect_groundings(
            states["accumulated_body"], states["proof_goals"],
            states["state_valid"], states["top_ridx"],
            states["collected_body"], states["collected_mask"],
            states["collected_ridx"],
            self.kb.constant_no, self.kb.padding_idx, self.effective_total_G,
            body_count=states["body_count"],
            collected_bcount=states["collected_bcount"],
        )

        states["collected_body"] = cb
        states["collected_mask"] = cm
        states["collected_ridx"] = cr
        states["state_valid"] = sv
        states["collected_bcount"] = c_bc
        return states

    def _postprocess(self, states: Dict[str, Tensor], sync: SyncParams) -> Dict[str, Tensor]:
        """Full postprocess: prune goals + sync accumulated + collect groundings.

        Used by the non-compiled path. The compiled path splits this into
        _postprocess_goals (inside compiled) + _sync_and_collect (outside).

        Args:
            states: Current proof states dict.
            sync: SyncParams from _pack (parent_map, winning_subs, etc.).
        """
        states = self._postprocess_goals(states)
        states = self._sync_accumulated(states, sync)
        return self._collect_groundings(states)

    # ==================================================================
    # Output variable standardization (optional)
    # ==================================================================

    def standardize_output(
        self,
        states: Tensor,
        counts: Tensor,
        next_var_indices: Tensor,
        input_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Standardize runtime variables in output states.

        Renumbers variables in ``states`` using the configured mode
        (``standardization_mode='offset'`` or ``'canonical'``).

        Does nothing if ``standardization_mode=None`` (default).

        Args:
            states: [B, K, M, 3] derived states to standardize.
            counts: [B] valid state count per batch element.
            next_var_indices: [B] current free variable index.
            input_states: [B, ?, 3] parent states (needed for offset mode).

        Returns:
            std_states: [B, K, M, 3] standardized states.
            new_next_var: [B] updated free variable indices.
        """
        if self._standardize_fn is None:
            return states, next_var_indices
        return self._standardize_fn(states, counts, next_var_indices,
                                    input_states if input_states is not None
                                    else states.new_zeros(0))

    # ==================================================================
    # Compiled step (optimization — same semantics as clean path)
    # ==================================================================

    def _step_compiled(self, states: Dict) -> Dict[str, Tensor]:
        """Compiled step: dict <-> raw tensors."""
        if self._clone_between_steps:
            states = {k: v.clone() if isinstance(v, Tensor) else v
                      for k, v in states.items()}

        (states["grounding_body"], states["accumulated_body"],
         states["body_count"],
         states["proof_goals"],
         states["top_ridx"], states["state_valid"],
         states["next_var_indices"],
         states["collected_body"], states["collected_mask"],
         states["collected_ridx"],
         states["collected_bcount"]) = self._fn_step(
            states["grounding_body"], states["accumulated_body"],
            states["body_count"],
            states["proof_goals"],
            states["top_ridx"], states["state_valid"],
            states["next_var_indices"],
            states["collected_body"], states["collected_mask"],
            states["collected_ridx"],
            states["collected_bcount"],
        )
        return states

    def _step_impl(
        self,
        grounding_body: Tensor,
        accumulated_body: Tensor,
        body_count: Tensor,
        proof_goals: Tensor,
        top_ridx: Tensor,
        state_valid: Tensor,
        next_var_indices: Tensor,
        collected_body: Tensor,
        collected_mask: Tensor,
        collected_ridx: Tensor,
        collected_bcount: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor]:
        """Raw tensor step for torch.compile -- same phases as clean path."""
        states = {
            "grounding_body": grounding_body,
            "accumulated_body": accumulated_body,
            "body_count": body_count,
            "proof_goals": proof_goals,
            "top_ridx": top_ridx,
            "state_valid": state_valid,
            "next_var_indices": next_var_indices,
            "collected_body": collected_body,
            "collected_mask": collected_mask,
            "collected_ridx": collected_ridx,
            "collected_bcount": collected_bcount,
        }

        # SELECT -> RESOLVE -> SEARCH FILTERS -> HOOKS -> PACK -> POSTPROCESS
        queries, remaining, active_mask = self._select(states)
        resolved = self._resolve(
            queries, remaining, grounding_body, state_valid,
            active_mask, states, d=1, use_hooks=False,
        )
        resolved = self._apply_search_filters(resolved)
        resolved = self._apply_hooks(resolved, states)
        states, sync = self._pack(resolved, states)
        states = self._postprocess(states, sync)

        return (states["grounding_body"], states["accumulated_body"],
                states["body_count"],
                states["proof_goals"],
                states["top_ridx"], states["state_valid"],
                states["next_var_indices"],
                states["collected_body"], states["collected_mask"],
                states["collected_ridx"],
                states["collected_bcount"])

    # ==================================================================
    # Provability
    # ==================================================================

    def check_known(self, atoms: Tensor) -> Tensor:
        """Check if atoms are known facts or in fp_global set (I_D)."""
        is_fact = self.kb.fact_index.exists(atoms)
        if hasattr(self, "_has_fp_global") and self._has_fp_global:
            E = getattr(self, '_E_fp_global', getattr(self, '_E', self.kb.constant_no + 1))
            h = atoms[..., 0] * (E * E) + atoms[..., 1] * E + atoms[..., 2]
            in_fp_global = check_in_fp_global(h, self.fp_global_hashes)
            return is_fact | in_fp_global
        return is_fact

    def is_provable(self, atoms: Tensor) -> Tensor:
        return self.check_known(atoms)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _empty_result(
        self, B: int, tG: int, G_body: int, dev: torch.device,
    ) -> ProofEvidence:
        return ProofEvidence(
            body=torch.zeros(
                B, tG, G_body, 3, dtype=torch.long, device=dev),
            mask=torch.zeros(
                B, tG, dtype=torch.bool, device=dev),
            count=torch.zeros(
                B, dtype=torch.long, device=dev),
            rule_idx=torch.zeros(
                B, tG, dtype=torch.long, device=dev),
            body_count=torch.zeros(
                B, tG, dtype=torch.long, device=dev),
        )

    def __repr__(self) -> str:
        return (
            f"BCGrounder(resolution={self.resolution!r}, "
            f"filter={self.filter_mode!r}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_rules={self.kb.num_rules}, "
            f"S={self.S}, tG={self.effective_total_G})")
