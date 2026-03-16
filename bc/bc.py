"""BCGrounder — unified backward chaining with configurable resolution.

Configuration replaces classes:
  resolution: 'sld' | 'rtf' | 'enum'
  filter:     'prune' | 'provset' | 'none'
  depth, width, hooks
  standardization: None | StandardizationConfig

Canonical loop (same code path for all resolutions):
  states = init_states(queries, query_mask)
  for d in range(D):
      states = step(states, d)   # SELECT → RESOLVE → PACK → POSTPROCESS
  return filter_terminal(states)

Resolution is the only pluggable phase — _select, _pack, _postprocess are shared.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple  # Callable used by _standardize_fn

import torch
import torch.nn as nn
from torch import Tensor

from grounder.base import Grounder
from grounder.resolution.standardization import StandardizationConfig
from grounder.bc.common import (
    compact_atoms,
    collect_groundings,
    pack_states,
    prune_ground_facts,
)
from grounder.types import GroundingResult
from grounder.filters import check_in_provable
from grounder.resolution.sld import resolve_sld
from grounder.resolution.rtf import resolve_rtf
from grounder.resolution.enum import resolve_enum_step


class BCGrounder(Grounder):
    """Unified backward-chaining grounder BC_{w,d}.

    Configurable with orthogonal choices:
      depth (d):    number of proof steps
      width (w):    max unknown body atoms per grounding (enum only; None=∞)
      resolution:   'sld' | 'rtf' | 'enum'
      filter:       'prune' | 'provset' | 'none'
      hooks:        GroundingHook list (post-grounding scoring/filtering)
      fact_hook:    ResolutionFactHook (filters fact candidates during resolution)
      rule_hook:    ResolutionRuleHook (filters rule candidates during resolution)
    """

    def __init__(
        self,
        *args,
        depth: int = 2,
        width: Optional[int] = 1,
        resolution: str = "enum",
        filter: str = "prune",
        max_total_groundings: int = 64,
        compile_mode: Optional[str] = None,
        hooks: Optional[List] = None,
        fact_hook=None,
        rule_hook=None,
        # MGU params
        max_goals: Optional[int] = None,
        max_states: Optional[int] = None,
        K_MAX: int = 550,
        max_derived_per_state: Optional[int] = None,
        track_grounding_body: bool = True,
        max_groundings_per_rule: Optional[int] = None,
        # Enum params
        max_groundings_per_query: int = 32,
        fc_method: str = "join",
        fc_depth: int = 10,
        # Output variable standardization (for consumers of ungrounded states)
        standardization: Optional[StandardizationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.depth = depth
        self.width = width
        self.resolution = resolution
        self.filter_mode = filter
        self.compile_mode = compile_mode
        self.hooks = hooks or []
        self.fact_hook = fact_hook
        self.rule_hook = rule_hook
        self.track_grounding_body = track_grounding_body
        self.standardization_mode = standardization.mode if standardization else None

        # Max goals: shared for all resolutions.
        # Must accommodate M body atoms for enum resolution.
        if max_goals is None:
            max_goals = 1 + depth * max(self.M - 1, 1)
        self.max_goals = max(max_goals, self.M)

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
            self._standardize_fn = build_standardize_fn(standardization, self._device)

    # ==================================================================
    # Resolution init
    # ==================================================================

    def _init_resolution(self, **kwargs) -> None:
        """Call resolution module's init, apply results, set up compilation."""
        if self.resolution in ("sld", "rtf"):
            from grounder.resolution.mgu import init_mgu
            cfg = init_mgu(
                resolution=self.resolution,
                K_f=self.K_f, K_r=self.K_r,
                rule_index=self.rule_index,
                max_total_groundings=kwargs["max_total_groundings"],
                K_MAX=kwargs["K_MAX"],
                max_derived_per_state=kwargs["max_derived_per_state"],
                max_states=kwargs["max_states"],
                max_groundings_per_rule=kwargs["max_groundings_per_rule"],
            )
            self.K = cfg["K"]
            self.S = cfg["S"]
            self.K_f = cfg["K_f"]
            self.max_vars_per_rule = cfg["max_vars_per_rule"]
            self.effective_total_G = cfg["effective_total_G"]
            self._max_fact_pairs_body = cfg["max_fact_pairs_body"]

        elif self.resolution == "enum":
            from grounder.resolution.enum import init_enum
            meta = init_enum(
                rule_index=self.rule_index,
                fact_index=self.fact_index,
                facts_idx=self.facts_idx,
                constant_no=self.constant_no,
                num_rules=self.num_rules, M=self.M,
                width=self.width,
                max_groundings_per_query=kwargs["max_groundings_per_query"],
                max_total_groundings=kwargs["max_total_groundings"],
                max_states=kwargs["max_states"],
                device=self._device,
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
                and self._device.type == "cuda"):
            self._fn_step = torch.compile(
                self._step_impl, fullgraph=True,
                mode=self.compile_mode)
            self._clone_between_steps = (
                self.compile_mode == "reduce-overhead")
            self._compiled = True
            self._multi_step = True

        # Provable set (for provset filter)
        self._has_provable_set = False
        if self.filter_mode == "provset" and self.resolution == "enum":
            self._build_provable_set(self._device)

    def _build_provable_set(self, device: torch.device) -> None:
        from grounder.fc.fc import run_forward_chaining
        method = self.fc_method
        if method in ("join", "spmm"):
            method = "dynamic"
        provable_tensor, n_provable = run_forward_chaining(
            compiled_rules=self._enum_ri.patterns,
            facts_idx=self.facts_idx,
            num_entities=self._E,
            num_predicates=self._P,
            depth=self.fc_depth,
            device=str(device),
        )
        self.register_buffer("provable_hashes", provable_tensor)
        self.register_buffer(
            "num_provable",
            torch.tensor(n_provable, dtype=torch.long, device=device))
        self._has_provable_set = n_provable > 0

    # ==================================================================
    # Canonical loop
    # ==================================================================

    @torch.no_grad()
    def forward(
        self, queries: Tensor, query_mask: Tensor, **init_kwargs,
    ) -> GroundingResult:
        states = self.init_states(queries, query_mask, **init_kwargs)
        for d in range(self.depth):
            states = self.step(states, d)
        result = self.filter_terminal(states)
        # filter='none' returns raw states dict — wrap in GroundingResult
        # outside the compiled region (dataclass init breaks fullgraph).
        if isinstance(result, dict):
            result = GroundingResult(
                body=result["collected_body"],
                mask=result["collected_mask"],
                count=result["collected_mask"].sum(dim=1),
                rule_idx=result["collected_ridx"],
            )
        for hook in self.hooks:
            body, mask, ridx = hook.apply(result.body, result.mask, result.rule_idx)
            result = GroundingResult(
                body=body, mask=mask, count=mask.sum(dim=1), rule_idx=ridx)
        return result


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
        pad = self.padding_idx
        G = self.max_goals
        M = self.M
        tG = self.effective_total_G
        M_body = 1 if not self.track_grounding_body else M

        proof_goals = torch.full(
            (B, 1, G, 3), pad, dtype=torch.long, device=dev)
        if initial_goals is not None:
            M_in = initial_goals.shape[1]
            proof_goals[:, 0, :M_in, :] = initial_goals
        else:
            proof_goals[:, 0, 0, :] = queries
        grounding_body = torch.full(
            (B, 1, M_body, 3), pad, dtype=torch.long, device=dev)
        top_ridx = torch.full((B, 1), -1, dtype=torch.long, device=dev)
        state_valid = query_mask.unsqueeze(1)

        if next_var_indices is None:
            E = self.constant_no + 1
            next_var_indices = torch.full(
                (B,), E, dtype=torch.long, device=dev)

        states = {
            "queries": queries,
            "query_mask": query_mask,
            "proof_goals": proof_goals,
            "grounding_body": grounding_body,
            "top_ridx": top_ridx,
            "state_valid": state_valid,
            "next_var_indices": next_var_indices,
            "initial_next_var": next_var_indices,
            "collected_body": queries.new_zeros(B, tG, M, 3),
            "collected_mask": torch.zeros(B, tG, dtype=torch.bool, device=dev),
            "collected_ridx": queries.new_zeros(B, tG),
        }
        if initial_goals is not None:
            states["initial_goals"] = initial_goals
        if excluded_queries is not None:
            states["excluded_queries"] = excluded_queries
        return states

    def step(self, states: Dict, d: int) -> Dict:
        """One proof step: SELECT → RESOLVE → PACK → POSTPROCESS."""
        if self.num_rules == 0:
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

        # ── HOOKS (between RESOLVE and PACK) ──
        resolved = self._apply_hooks(resolved, states)

        # ── PACK ──
        states = self._pack(resolved, states)

        # ── POSTPROCESS ──
        states = self._postprocess(states)

        return states

    def filter_terminal(self, states: Dict):
        """Apply soundness filter on collected groundings → GroundingResult.

        When ``filter='none'``, returns the raw states dict (no collection).
        """
        if self.filter_mode == "none":
            return states

        B = states["collected_body"].size(0)
        tG = self.effective_total_G
        M = self.M
        dev = states["collected_body"].device

        body = states["collected_body"]
        mask = states["collected_mask"]
        ridx = states["collected_ridx"]

        if self.num_rules == 0:
            return self._empty_result(B, tG, M, dev)

        if self.filter_mode == "prune":
            from grounder.filters.prune import apply_prune
            mask = apply_prune(
                body, mask, states["queries"], self.fact_index,
                self.pack_base, self.padding_idx, self.depth)

        elif self.filter_mode == "provset":
            from grounder.filters.provset import apply_provset
            mask = apply_provset(
                body, mask, self.fact_index,
                self.pack_base, self.padding_idx,
                self.provable_hashes)

        count = mask.sum(dim=1)
        return GroundingResult(
            body=body, mask=mask,
            count=count, rule_idx=ridx,
        )

    # ==================================================================
    # Phase 1: SELECT (shared)
    # ==================================================================

    def _select(
        self, states: Dict,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract first goal from each proof state."""
        proof_goals = states["proof_goals"]
        active_mask = proof_goals[:, :, 0, 0] != self.padding_idx
        queries = proof_goals[:, :, 0, :]
        queries = queries * active_mask.unsqueeze(-1).to(queries.dtype)
        remaining = proof_goals.clone()
        remaining[:, :, 0, :] = self.padding_idx
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
        states: Dict,
        d: int,
        use_hooks: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Dispatch to resolution strategy. Returns common 7-tensor format."""
        fh = self.fact_hook if use_hooks else None
        rh = self.rule_hook if use_hooks else None

        if self.resolution == "sld":
            return resolve_sld(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices=states["next_var_indices"],
                fact_index=self.fact_index, facts_idx=self.facts_idx,
                rule_index=self.rule_index,
                constant_no=self.constant_no, padding_idx=self.padding_idx,
                K_f=self.K_f, K_r=self.K_r,
                max_vars_per_rule=self.max_vars_per_rule,
                num_rules=self.num_rules,
                track_grounding_body=self.track_grounding_body,
                excluded_queries=states.get("excluded_queries"),
                fact_hook=fh, rule_hook=rh,
            )
        elif self.resolution == "rtf":
            return resolve_rtf(
                queries, remaining, grounding_body, state_valid, active_mask,
                next_var_indices=states["next_var_indices"],
                fact_index=self.fact_index, facts_idx=self.facts_idx,
                rule_index=self.rule_index,
                constant_no=self.constant_no, padding_idx=self.padding_idx,
                K_f=self.K_f, K_r=self.K_r, K=self.K,
                max_vars_per_rule=self.max_vars_per_rule,
                num_rules=self.num_rules,
                max_fact_pairs_body=self._max_fact_pairs_body,
                track_grounding_body=self.track_grounding_body,
                fact_hook=fh, rule_hook=rh,
            )
        else:
            return resolve_enum_step(
                queries, remaining, grounding_body, state_valid, active_mask,
                fact_index=self.fact_index,
                d=d, depth=self.depth, width=self.width,
                M=self.M, padding_idx=self.padding_idx,
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
        resolved: Tuple[Tensor, ...],
        states: Dict,
    ) -> Tuple[Tensor, ...]:
        """Apply resolution hooks. Subclasses override for RL filtering."""
        return resolved

    # ==================================================================
    # Phase 3: PACK (shared)
    # ==================================================================

    def _pack(
        self,
        resolved: Tuple[Tensor, ...],
        states: Dict,
    ) -> Dict:
        """Flatten S×K children, propagate grounding body, compact to S."""
        S_in = states["top_ridx"].shape[1]

        new_gbody, new_goals, new_ridx, new_valid = pack_states(
            *resolved,
            states["top_ridx"], states["grounding_body"],
            self.S, self.padding_idx,
            track_grounding_body=self.track_grounding_body,
        )

        states["grounding_body"] = new_gbody
        states["proof_goals"] = new_goals
        states["top_ridx"] = new_ridx
        states["state_valid"] = new_valid
        states["next_var_indices"] = (
            states["next_var_indices"] + S_in * self.max_vars_per_rule)
        return states

    # ==================================================================
    # Phase 4: POSTPROCESS (shared)
    # ==================================================================

    def _postprocess(self, states: Dict) -> Dict:
        """Prune ground facts, compact atoms, collect completed groundings."""
        proof_goals, _, _ = prune_ground_facts(
            states["proof_goals"], states["state_valid"],
            self.fact_hashes, self.pack_base,
            self.constant_no, self.padding_idx,
            excluded_queries=states.get("excluded_queries"),
        )
        proof_goals = compact_atoms(proof_goals, self.padding_idx)

        cb, cm, cr, sv = collect_groundings(
            states["grounding_body"], proof_goals,
            states["state_valid"], states["top_ridx"],
            states["collected_body"], states["collected_mask"],
            states["collected_ridx"],
            self.constant_no, self.padding_idx, self.effective_total_G,
        )

        states["proof_goals"] = proof_goals
        states["collected_body"] = cb
        states["collected_mask"] = cm
        states["collected_ridx"] = cr
        states["state_valid"] = sv
        return states

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

    def _step_compiled(self, states: Dict) -> Dict:
        """Compiled step: dict ↔ raw tensors."""
        if self._clone_between_steps:
            states = {k: v.clone() if isinstance(v, Tensor) else v
                      for k, v in states.items()}

        (states["grounding_body"], states["proof_goals"],
         states["top_ridx"], states["state_valid"],
         states["next_var_indices"],
         states["collected_body"], states["collected_mask"],
         states["collected_ridx"]) = self._fn_step(
            states["grounding_body"], states["proof_goals"],
            states["top_ridx"], states["state_valid"],
            states["next_var_indices"],
            states["collected_body"], states["collected_mask"],
            states["collected_ridx"],
        )
        return states

    def _step_impl(
        self,
        grounding_body: Tensor,
        proof_goals: Tensor,
        top_ridx: Tensor,
        state_valid: Tensor,
        next_var_indices: Tensor,
        collected_body: Tensor,
        collected_mask: Tensor,
        collected_ridx: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor]:
        """Raw tensor step for torch.compile — same 4 phases as clean path."""
        states = {
            "grounding_body": grounding_body,
            "proof_goals": proof_goals,
            "top_ridx": top_ridx,
            "state_valid": state_valid,
            "next_var_indices": next_var_indices,
            "collected_body": collected_body,
            "collected_mask": collected_mask,
            "collected_ridx": collected_ridx,
        }

        # SELECT → RESOLVE → HOOKS → PACK → POSTPROCESS
        queries, remaining, active_mask = self._select(states)
        resolved = self._resolve(
            queries, remaining, grounding_body, state_valid,
            active_mask, states, d=1, use_hooks=False,
        )
        resolved = self._apply_hooks(resolved, states)
        states = self._pack(resolved, states)
        states = self._postprocess(states)

        return (states["grounding_body"], states["proof_goals"],
                states["top_ridx"], states["state_valid"],
                states["next_var_indices"],
                states["collected_body"], states["collected_mask"],
                states["collected_ridx"])

    # ==================================================================
    # Provability
    # ==================================================================

    def check_known(self, atoms: Tensor) -> Tensor:
        """Check if atoms are known facts or in provable set."""
        is_fact = self.fact_index.exists(atoms)
        if hasattr(self, "_has_provable_set") and self._has_provable_set:
            E = self._E
            h = atoms[..., 0] * (E * E) + atoms[..., 1] * E + atoms[..., 2]
            in_provable = check_in_provable(h, self.provable_hashes)
            return is_fact | in_provable
        return is_fact

    def is_provable(self, atoms: Tensor) -> Tensor:
        return self.check_known(atoms)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _empty_result(
        self, B: int, tG: int, M: int, dev: torch.device,
    ) -> GroundingResult:
        return GroundingResult(
            body=torch.zeros(
                B, tG, M, 3, dtype=torch.long, device=dev),
            mask=torch.zeros(
                B, tG, dtype=torch.bool, device=dev),
            count=torch.zeros(
                B, dtype=torch.long, device=dev),
            rule_idx=torch.zeros(
                B, tG, dtype=torch.long, device=dev),
        )

    def __repr__(self) -> str:
        return (
            f"BCGrounder(resolution={self.resolution!r}, "
            f"filter={self.filter_mode!r}, "
            f"depth={self.depth}, width={self.width}, "
            f"num_rules={self.num_rules}, "
            f"S={self.S}, tG={self.effective_total_G})")
