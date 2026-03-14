"""ParametrizedBCGrounder -- approximate backward chaining with width/depth.

Port of ns_lib/grounding/parametrized_grounder.py to the grounder/ package.
Works with raw tensors instead of ns_lib domain objects (Rule, KnowledgeBase,
TensorFactIndex).

Inherits from ``grounder.grounders.base.Grounder`` (NOT ``BCGrounder``) so it
owns the fact index and rule index built from raw tensors.  Rule compilation,
vectorized metadata and rule clustering are delegated to
``grounder.compilation``.  The provable set for PruneIncompleteProofs is
computed via ``grounder.forward_chaining.run_forward_chaining``.

Returns :class:`grounder.types.ForwardResult`.

Naming: ``backward_W_D`` (parametrized BC grounder).
- ``backward_0_1``: proven-only (all body atoms must be facts)
- ``backward_1_2``: allows 1 unproven body atom, depth-2 provability check
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.grounders.base import Grounder
from grounder.types import ForwardResult
from grounder.compilation import (
    CompiledRule,
    compile_rules,
    build_vectorized_metadata,
    build_rule_clustering,
    tensorize_rules,
    check_in_provable,
    BINDING_HEAD_VAR0,
    BINDING_HEAD_VAR1,
    BINDING_FREE_VAR_OFFSET,
)
from grounder.forward_chaining import run_forward_chaining


class ParametrizedBCGrounder(Grounder):
    """Fullgraph-compatible approximate backward chaining grounder.

    Matches ns_lib ParametrizedBCGrounder behaviour with configurable
    ``width`` and ``depth``.

    Algorithm:
        1. Rule clustering: match query pred -> relevant rules (R_eff).
        2. Direction A: enumerate candidates from fact index.
        3. Resolve body atoms using check_arg_source.
        4. Width filtering + query exclusion + head predicate pruning.
        5. Direction B (dual anchoring) for dual-capable rules.
        6. PruneIncompleteProofs via precomputed FC provable set.
        7. Flatten + topk selection.

    Args:
        facts_idx:              [F, 3]   fact triples (pred, arg0, arg1).
        rules_heads_idx:        [R, 3]   rule head atoms.
        rules_bodies_idx:       [R, M, 3] rule body atoms (padded).
        rule_lens:              [R]      number of body atoms per rule.
        constant_no:            highest constant index.
        padding_idx:            padding value.
        device:                 target device.
        depth:                  backward chaining depth.
        width:                  max unproven body atoms per grounding.
                                None means w=inf.  When w >= max_body_atoms
                                (or None), delegates to FullBCGrounder.
        max_groundings_per_query: G budget per rule per query.
        max_total_groundings:   static cap on total groundings.
        prune_incomplete_proofs: whether to prune groundings with
                                unprovable atoms.
        fc_method:              forward-chaining method for provable set.
        predicate_no:           total number of predicates (exclusive).
        num_entities:           total number of entities.
        max_facts_per_query:    K_f for inverted/block_sparse index.
        fact_index_type:        'block_sparse' (default) or 'inverted'.
    """

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        constant_no: int,
        padding_idx: int,
        device: torch.device,
        *,
        depth: int = 2,
        width: Optional[int] = 1,
        max_groundings_per_query: int = 32,
        max_total_groundings: int = 64,
        prune_incomplete_proofs: bool = True,
        fc_method: str = "join",
        fc_depth: int = 10,
        predicate_no: Optional[int] = None,
        num_entities: Optional[int] = None,
        max_facts_per_query: int = 64,
        fact_index_type: str = "block_sparse",
        **kwargs,
    ) -> None:
        # Grounder base builds fact_index + rule_index from raw tensors
        super().__init__(
            facts_idx,
            rules_heads_idx,
            rules_bodies_idx,
            rule_lens,
            constant_no,
            padding_idx,
            device,
            predicate_no=predicate_no,
            fact_index_type=fact_index_type,
            num_entities=num_entities,
            max_facts_per_query=max_facts_per_query,
            **kwargs,
        )

        self.depth = depth
        self.width = width
        self.max_groundings_per_query = max_groundings_per_query
        self.max_total_groundings = max_total_groundings
        self.prune = prune_incomplete_proofs
        self.fc_method = fc_method
        self.fc_depth = fc_depth

        # Compile rules from raw tensors
        self.compiled_rules = compile_rules(
            rules_heads_idx, rules_bodies_idx, rule_lens, constant_no,
        )

        # Max body atoms across all rules
        self.max_body_atoms = max(
            (cr.num_body for cr in self.compiled_rules), default=1
        )

        # Resolve predicate_no / num_entities for metadata builders
        P = predicate_no if predicate_no is not None else (
            int(facts_idx[:, 0].max().item()) + 1 if facts_idx.numel() > 0 else 1
        )
        E = num_entities if num_entities is not None else (
            int(max(facts_idx[:, 1].max().item(),
                    facts_idx[:, 2].max().item())) + 1
            if facts_idx.numel() > 0 else 1
        )
        self._P = P
        self._E = E

        # Head predicate mask for pruning unknown body atoms
        dev = device
        head_pred_mask = torch.zeros(P, dtype=torch.bool, device=dev)
        for cr in self.compiled_rules:
            head_pred_mask[cr.head_pred_idx] = True
        self.register_buffer("head_pred_mask", head_pred_mask)

        # Tensorize rule patterns (registered buffers for static rule data)
        head_preds, body_preds, num_body_atoms = tensorize_rules(
            self.compiled_rules, self.max_body_atoms, dev,
        )
        self.register_buffer("head_preds", head_preds)
        self.register_buffer("body_preds", body_preds)
        self.register_buffer("num_body_atoms", num_body_atoms)

        # Build vectorized metadata for single-pass grounding
        self._build_metadata(dev)

        # Build rule clustering: per-predicate rule mapping
        pred_rule_indices, pred_rule_mask, R_eff = build_rule_clustering(
            self.compiled_rules, P, dev,
        )
        self.R_eff = R_eff
        self.register_buffer("pred_rule_indices", pred_rule_indices)
        self.register_buffer("pred_rule_mask", pred_rule_mask)

        print(
            f"  Rule clustering: {self.num_rules} rules -> R_eff={R_eff} "
            f"({P} predicates, max {R_eff} rules/pred)"
        )

        # Precompute provable set for PruneIncompleteProofs
        self._build_provable_set(dev)

        # Check if full enumeration mode should be activated:
        # When w >= max_body_atoms, all body atoms can be unknown for every
        # rule, so fact-anchoring adds no value -- delegate to FullBCGrounder.
        if self.width is None or self.width >= self.max_body_atoms:
            from grounder.grounders.full import FullBCGrounder  # lazy import

            self._full_enum = True
            self._full_grounder = FullBCGrounder(
                facts_idx=facts_idx,
                rules_heads_idx=rules_heads_idx,
                rules_bodies_idx=rules_bodies_idx,
                rule_lens=rule_lens,
                constant_no=constant_no,
                padding_idx=padding_idx,
                device=device,
                max_total_groundings=max_total_groundings,
                predicate_no=predicate_no,
                num_entities=num_entities,
                max_facts_per_query=max_facts_per_query,
                fact_index_type=fact_index_type,
            )
            self.effective_total_G = self._full_grounder.effective_total_G
            w_str = "inf" if self.width is None else str(self.width)
            print(
                f"  Full enumeration mode: w={w_str}"
                f" >= max_body={self.max_body_atoms}"
                f" -> delegating to FullBCGrounder"
            )
        else:
            self._full_enum = False
            self.effective_total_G = min(
                max_total_groundings, self.R_eff * max_groundings_per_query,
            )

    # ==================================================================
    # Vectorized metadata (directions A + B, dual anchoring)
    # ==================================================================

    def _build_metadata(self, device: torch.device) -> None:
        """Build per-rule metadata tensors for vectorized grounding.

        For dual anchoring (width > 0), two sets of metadata are created:
        - Direction A: anchor on the first body atom with free var.
        - Direction B: anchor on the second body atom with free var.
        Each direction gets G // 2 slots.
        """
        R = max(self.num_rules, 1)
        M = self.max_body_atoms

        # Per-rule flags
        has_free = torch.zeros(R, dtype=torch.bool, device=device)
        has_dual = torch.zeros(R, dtype=torch.bool, device=device)

        # Direction A metadata
        enum_pred_a = torch.zeros(R, dtype=torch.long, device=device)
        enum_bound_binding_a = torch.zeros(R, dtype=torch.long, device=device)
        enum_direction_a = torch.zeros(R, dtype=torch.long, device=device)

        # Direction B metadata
        enum_pred_b = torch.zeros(R, dtype=torch.long, device=device)
        enum_bound_binding_b = torch.zeros(R, dtype=torch.long, device=device)
        enum_direction_b = torch.zeros(R, dtype=torch.long, device=device)

        # Arg source tables: 0=subj, 1=obj, 2=candidate
        check_arg_source_a = torch.zeros(R, M, 2, dtype=torch.long, device=device)
        check_arg_source_b = torch.zeros(R, M, 2, dtype=torch.long, device=device)

        # Body atom indices for enumeration vs checking
        enum_body_idx_a = torch.zeros(R, dtype=torch.long, device=device)
        enum_body_idx_b = torch.zeros(R, dtype=torch.long, device=device)
        check_body_idx_a = torch.zeros(R, dtype=torch.long, device=device)
        check_body_idx_b = torch.zeros(R, dtype=torch.long, device=device)

        # Check predicate for existence check (non-enum body atom)
        check_pred_a = torch.zeros(R, dtype=torch.long, device=device)
        check_pred_b = torch.zeros(R, dtype=torch.long, device=device)

        for i, cr in enumerate(self.compiled_rules):
            bp0 = cr.body_patterns[0] if cr.num_body > 0 else None
            bp1 = cr.body_patterns[1] if cr.num_body > 1 else None

            bp0_has_free = (
                bp0 is not None
                and (bp0["arg0_binding"] >= 2 or bp0["arg1_binding"] >= 2)
            )
            bp1_has_free = (
                bp1 is not None
                and (bp1["arg0_binding"] >= 2 or bp1["arg1_binding"] >= 2)
            )

            if bp0_has_free or bp1_has_free:
                has_free[i] = True

            # Dual anchoring requires bp1 to have one arg bound to query
            # (0 or 1) and one free arg.  Rules like bp1(Y,K) where both
            # args are free cannot serve as dual anchor.
            bp1_can_anchor = bp1_has_free and bp1 is not None and (
                bp1["arg0_binding"] in (0, 1) or bp1["arg1_binding"] in (0, 1)
            )
            if bp0_has_free and bp1_can_anchor and (
                self.width is None or self.width > 0
            ):
                has_dual[i] = True

            # -- Direction A: primary enumeration --
            if bp0_has_free:
                enum_body_idx_a[i] = 0
                check_body_idx_a[i] = 1 if cr.num_body > 1 else 0
                self._fill_enum_meta(
                    bp0, enum_pred_a, enum_bound_binding_a,
                    enum_direction_a, i,
                )
                if cr.num_body > 1:
                    check_pred_a[i] = bp1["pred_idx"]
                    self._fill_arg_source(cr, check_arg_source_a, i)
                else:
                    self._fill_arg_source(cr, check_arg_source_a, i)
            elif bp1_has_free:
                enum_body_idx_a[i] = 1
                check_body_idx_a[i] = 0
                self._fill_enum_meta(
                    bp1, enum_pred_a, enum_bound_binding_a,
                    enum_direction_a, i,
                )
                check_pred_a[i] = bp0["pred_idx"]
                self._fill_arg_source(cr, check_arg_source_a, i)
            else:
                # Fully bound -- use direction A with dummy enum
                self._fill_arg_source(cr, check_arg_source_a, i)

            # -- Direction B: secondary (swap enum/check) --
            if has_dual[i]:
                enum_body_idx_b[i] = 1
                check_body_idx_b[i] = 0
                self._fill_enum_meta(
                    bp1, enum_pred_b, enum_bound_binding_b,
                    enum_direction_b, i,
                )
                check_pred_b[i] = bp0["pred_idx"]
                self._fill_arg_source(cr, check_arg_source_b, i)

        self.register_buffer("has_free", has_free)
        self.register_buffer("has_dual", has_dual)
        self.register_buffer("enum_pred_a", enum_pred_a)
        self.register_buffer("enum_bound_binding_a", enum_bound_binding_a)
        self.register_buffer("enum_direction_a", enum_direction_a)
        self.register_buffer("enum_pred_b", enum_pred_b)
        self.register_buffer("enum_bound_binding_b", enum_bound_binding_b)
        self.register_buffer("enum_direction_b", enum_direction_b)
        self.register_buffer("check_arg_source_a", check_arg_source_a)
        self.register_buffer("check_arg_source_b", check_arg_source_b)
        self.register_buffer("enum_body_idx_a", enum_body_idx_a)
        self.register_buffer("enum_body_idx_b", enum_body_idx_b)
        self.register_buffer("check_body_idx_a", check_body_idx_a)
        self.register_buffer("check_body_idx_b", check_body_idx_b)
        self.register_buffer("check_pred_a", check_pred_a)
        self.register_buffer("check_pred_b", check_pred_b)

        # Pre-compute Python bool for compile-time branching
        self.any_dual = bool(has_dual.any().item())

    # ==================================================================
    # Provable set precomputation
    # ==================================================================

    def _build_provable_set(self, device: torch.device) -> None:
        """Precompute provable atoms for PruneIncompleteProofs."""
        if self.prune and self.width is not None and self.width > 0:
            method = self.fc_method
            if method in ("join", "spmm"):
                method = "dynamic"
            provable_tensor, n_provable = run_forward_chaining(
                compiled_rules=self.compiled_rules,
                facts_idx=self.facts_idx,
                num_entities=self._E,
                num_predicates=self._P,
                depth=self.fc_depth,
                device=str(device),
            )
            self.register_buffer("provable_hashes", provable_tensor)
            self.register_buffer(
                "num_provable",
                torch.tensor(n_provable, dtype=torch.long, device=device),
            )
            self._has_provable_set = n_provable > 0
        else:
            self.register_buffer(
                "provable_hashes",
                torch.zeros(1, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "num_provable",
                torch.tensor(0, dtype=torch.long, device=device),
            )
            self._has_provable_set = False

    def _check_provable_precomputed(
        self,
        preds: Tensor,
        subjs: Tensor,
        objs: Tensor,
    ) -> Tensor:
        """Check if atoms are in the precomputed provable set.

        Uses binary search on sorted hash tensor -- O(log K) per atom.
        Fully compatible with torch.compile(fullgraph=True).

        Args:
            preds: (N,) predicate indices.
            subjs: (N,) subject indices.
            objs:  (N,) object indices.

        Returns:
            provable: (N,) boolean tensor.
        """
        E = self._E
        query_hashes = preds * (E * E) + subjs * E + objs

        positions = torch.searchsorted(self.provable_hashes, query_hashes)
        valid = positions < self.num_provable
        max_idx = torch.clamp(self.num_provable - 1, min=0)
        clamped = torch.minimum(positions, max_idx.expand_as(positions))
        clamped = torch.clamp(clamped, min=0)
        match = self.provable_hashes[clamped] == query_hashes

        return valid & match

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _fill_enum_meta(
        bp: dict,
        enum_pred: Tensor,
        enum_bound_binding: Tensor,
        enum_direction: Tensor,
        i: int,
    ) -> None:
        """Fill enumeration metadata for a body pattern."""
        enum_pred[i] = bp["pred_idx"]
        if bp["arg0_binding"] >= 2:
            enum_direction[i] = 1  # enumerate subjects (arg0 is free)
            enum_bound_binding[i] = bp["arg1_binding"]
        else:
            enum_direction[i] = 0  # enumerate objects (arg0 is bound)
            enum_bound_binding[i] = bp["arg0_binding"]

    @staticmethod
    def _fill_arg_source(
        cr: CompiledRule,
        check_arg_source: Tensor,
        i: int,
    ) -> None:
        """Fill arg source table for all body atoms of a rule."""
        for j, bp in enumerate(cr.body_patterns):
            for a, key in enumerate(["arg0_binding", "arg1_binding"]):
                b = bp[key]
                if b == 0:
                    check_arg_source[i, j, a] = 0
                elif b == 1:
                    check_arg_source[i, j, a] = 1
                else:
                    check_arg_source[i, j, a] = 2

    def _vectorized_enumerate_direction(
        self,
        B: int,
        Re: int,
        G_use: int,
        dev: torch.device,
        query_subjs: Tensor,
        query_objs: Tensor,
        enum_pred_q: Tensor,
        enum_bound_q: Tensor,
        enum_dir_q: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Enumerate candidates for one direction across clustered rules.

        Args:
            enum_pred_q:  (B, Re) per-query enumeration predicates.
            enum_bound_q: (B, Re) per-query bound argument bindings.
            enum_dir_q:   (B, Re) per-query enumeration directions.

        Returns:
            candidates: (B, Re, G_use)
            cand_mask:  (B, Re, G_use)
        """
        source = torch.stack([query_subjs, query_objs], dim=1)  # (B, 2)
        enum_bound_vals = source.gather(1, enum_bound_q)  # (B, Re)

        enum_preds_flat = enum_pred_q.reshape(-1)  # (B*Re,)
        enum_bound_flat = enum_bound_vals.reshape(-1)  # (B*Re,)
        enum_dir_flat = enum_dir_q.reshape(-1)  # (B*Re,)

        candidates, cand_mask = self.fact_index.enumerate(
            enum_preds_flat, enum_bound_flat, enum_dir_flat,
        )

        G_fi = candidates.size(1)
        G_use_actual = min(G_use, G_fi)
        candidates = candidates[:, :G_use_actual].reshape(B, Re, G_use_actual)
        cand_mask = cand_mask[:, :G_use_actual].reshape(B, Re, G_use_actual)

        return candidates, cand_mask

    def _vectorized_resolve_body(
        self,
        B: int,
        Re: int,
        G_use: int,
        M: int,
        dev: torch.device,
        query_subjs: Tensor,
        query_objs: Tensor,
        candidates: Tensor,
        check_arg_source_q: Tensor,
        body_preds_q: Tensor,
    ) -> Tensor:
        """Resolve body atom arguments from (subj, obj, candidate) source.

        Args:
            check_arg_source_q: (B, Re, M, 2) per-query arg source table.
            body_preds_q:       (B, Re, M) per-query body predicates.

        Returns:
            body_atoms: (B, Re, G_use, M, 3) = [pred, arg0, arg1]
        """
        q_subjs_exp = query_subjs.view(B, 1, 1).expand(-1, Re, G_use)
        q_objs_exp = query_objs.view(B, 1, 1).expand(-1, Re, G_use)
        source_3 = torch.stack(
            [q_subjs_exp, q_objs_exp, candidates], dim=3,
        )  # (B, Re, G_use, 3)

        # Unrolled body resolution: pre-expand source once, gather both args
        source_exp = source_3.unsqueeze(3).expand(
            -1, -1, -1, M, -1,
        )  # (B, Re, G_use, M, 3)

        idx_0 = check_arg_source_q[:, :, :, 0].view(B, Re, 1, M).expand(
            -1, -1, G_use, -1,
        )
        arg0 = source_exp.gather(4, idx_0.unsqueeze(-1)).squeeze(
            -1,
        )  # (B, Re, G_use, M)

        idx_1 = check_arg_source_q[:, :, :, 1].view(B, Re, 1, M).expand(
            -1, -1, G_use, -1,
        )
        arg1 = source_exp.gather(4, idx_1.unsqueeze(-1)).squeeze(
            -1,
        )  # (B, Re, G_use, M)

        body_preds_exp = body_preds_q.unsqueeze(2).expand(-1, -1, G_use, -1)
        body_atoms = torch.stack([body_preds_exp, arg0, arg1], dim=-1)
        return body_atoms

    # ==================================================================
    # Forward pass (grounding)
    # ==================================================================

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,
        query_mask: Tensor,
    ) -> ForwardResult:
        """Ground rules with approximate backward chaining -- rule-clustered.

        Uses per-predicate rule clustering: each query only processes the
        R_eff rules matching its head predicate.

        Args:
            queries:    (B, 3) -- [pred_idx, subj_idx, obj_idx].
            query_mask: (B,) -- valid queries.

        Returns:
            ForwardResult with collected_body, collected_mask,
            collected_count, collected_ridx.
        """
        # Full enumeration mode: delegate to FullBCGrounder
        if self._full_enum:
            return self._full_grounder(queries, query_mask)

        B = queries.size(0)
        Re = self.R_eff
        G = self.max_groundings_per_query
        M = self.max_body_atoms
        tG = self.effective_total_G
        dev = queries.device

        if self.num_rules == 0:
            return ForwardResult(
                collected_body=queries.new_zeros(B, tG, M, 3),
                collected_mask=torch.zeros(B, tG, dtype=torch.bool, device=dev),
                collected_count=queries.new_zeros(B),
                collected_ridx=queries.new_zeros(B, tG),
            )

        query_preds = queries[:, 0]
        query_subjs = queries[:, 1]
        query_objs = queries[:, 2]

        # == 1. RULE CLUSTERING -- gather only matching rules per query ==
        active_idx = self.pred_rule_indices[query_preds]  # (B, Re)
        active_mask = (
            self.pred_rule_mask[query_preds] & query_mask.unsqueeze(1)
        )  # (B, Re)

        # Gather per-rule metadata for active rules
        has_free_q = self.has_free[active_idx]  # (B, Re)
        has_dual_q = self.has_dual[active_idx]  # (B, Re)
        body_preds_q = self.body_preds[active_idx]  # (B, Re, M)
        num_body_q = self.num_body_atoms[active_idx]  # (B, Re)

        # Direction A metadata
        enum_pred_a_q = self.enum_pred_a[active_idx]  # (B, Re)
        enum_bound_a_q = self.enum_bound_binding_a[active_idx]  # (B, Re)
        enum_dir_a_q = self.enum_direction_a[active_idx]  # (B, Re)
        check_src_a_q = self.check_arg_source_a[active_idx]  # (B, Re, M, 2)

        # == 2. DIRECTION A: Primary enumeration ==
        G_use = min(G, self.fact_index._max_facts_per_query)

        candidates_a, cand_mask_a = self._vectorized_enumerate_direction(
            B, Re, G_use, dev, query_subjs, query_objs,
            enum_pred_a_q, enum_bound_a_q, enum_dir_a_q,
        )
        G_use = candidates_a.size(2)

        # Mask out bound-only rules' candidates
        cand_mask_a = cand_mask_a & has_free_q.unsqueeze(2)
        # For bound-only rules, enable slot 0
        bound_mask = ~has_free_q  # (B, Re)
        cand_mask_a[:, :, 0] = cand_mask_a[:, :, 0] | (bound_mask & active_mask)

        # Resolve body atoms for direction A
        body_atoms_a = self._vectorized_resolve_body(
            B, Re, G_use, M, dev, query_subjs, query_objs, candidates_a,
            check_src_a_q, body_preds_q,
        )

        # Check existence
        body_flat_a = body_atoms_a.reshape(-1, 3)
        exists_flat_a = self.fact_index.exists(body_flat_a)
        exists_a = exists_flat_a.view(B, Re, G_use, M)

        atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
        body_active = atom_idx < num_body_q.view(B, Re, 1, 1)

        # Width filtering: count unknown active body atoms
        num_unknown_a = (body_active & ~exists_a).sum(dim=-1)  # (B, Re, G_use)

        if self.width is None:
            grounding_mask_a = active_mask.unsqueeze(2) & cand_mask_a
        elif self.width == 0:
            within_width_a = num_unknown_a == 0
            grounding_mask_a = (
                within_width_a & active_mask.unsqueeze(2) & cand_mask_a
            )
        else:
            within_width_a = num_unknown_a <= self.width
            grounding_mask_a = (
                within_width_a & active_mask.unsqueeze(2) & cand_mask_a
            )

        # Query exclusion: no body atom equals the query
        query_exp = queries.view(B, 1, 1, 1, 3).expand(-1, Re, G_use, M, -1)
        is_query_a = (body_atoms_a == query_exp).all(dim=-1)
        has_query_atom_a = (
            is_query_a & body_active.expand(-1, -1, G_use, -1)
        ).any(dim=-1)
        grounding_mask_a = grounding_mask_a & ~has_query_atom_a

        if self.width is not None:
            # Head predicate pruning: unknown atoms must have head pred
            body_preds_a_vals = body_atoms_a[..., 0]  # (B, Re, G_use, M)
            head_pred_ok_a = self.head_pred_mask[body_preds_a_vals]
            unknown_ok_a = exists_a | head_pred_ok_a
            all_ok_a = (
                unknown_ok_a | ~body_active.expand(-1, -1, G_use, -1)
            ).all(dim=-1)

            if self.width > 0:
                grounding_mask_a = grounding_mask_a & all_ok_a

            if self.width == 0:
                all_exist_a = (
                    exists_a | ~body_active.expand(-1, -1, G_use, -1)
                ).all(dim=-1)
                grounding_mask_a = grounding_mask_a & all_exist_a

        # == 3. DIRECTION B: Dual anchoring ==
        if self.any_dual:
            # Direction B metadata
            enum_pred_b_q = self.enum_pred_b[active_idx]  # (B, Re)
            enum_bound_b_q = self.enum_bound_binding_b[active_idx]  # (B, Re)
            enum_dir_b_q = self.enum_direction_b[active_idx]  # (B, Re)
            check_src_b_q = self.check_arg_source_b[active_idx]  # (B, Re, M, 2)

            G_b = G // 2
            G_use_b = min(G_b, self.fact_index._max_facts_per_query)

            candidates_b, cand_mask_b = self._vectorized_enumerate_direction(
                B, Re, G_use_b, dev, query_subjs, query_objs,
                enum_pred_b_q, enum_bound_b_q, enum_dir_b_q,
            )
            G_use_b = candidates_b.size(2)

            # Only dual-anchor rules use direction B
            cand_mask_b = cand_mask_b & has_dual_q.unsqueeze(2)

            body_atoms_b = self._vectorized_resolve_body(
                B, Re, G_use_b, M, dev, query_subjs, query_objs, candidates_b,
                check_src_b_q, body_preds_q,
            )

            body_flat_b = body_atoms_b.reshape(-1, 3)
            exists_flat_b = self.fact_index.exists(body_flat_b)
            exists_b = exists_flat_b.view(B, Re, G_use_b, M)

            body_active_b = atom_idx < num_body_q.view(B, Re, 1, 1)
            num_unknown_b = (body_active_b & ~exists_b).sum(dim=-1)

            if self.width is None:
                grounding_mask_b = active_mask.unsqueeze(2) & cand_mask_b
            elif self.width == 0:
                within_width_b = num_unknown_b == 0
                grounding_mask_b = (
                    within_width_b & active_mask.unsqueeze(2) & cand_mask_b
                )
            else:
                within_width_b = num_unknown_b <= self.width
                grounding_mask_b = (
                    within_width_b & active_mask.unsqueeze(2) & cand_mask_b
                )

            # Query exclusion
            query_exp_b = queries.view(B, 1, 1, 1, 3).expand(
                -1, Re, G_use_b, M, -1,
            )
            is_query_b = (body_atoms_b == query_exp_b).all(dim=-1)
            has_query_atom_b = (
                is_query_b & body_active_b.expand(-1, -1, G_use_b, -1)
            ).any(dim=-1)
            grounding_mask_b = grounding_mask_b & ~has_query_atom_b

            if self.width is not None:
                # Head predicate pruning
                body_preds_b_vals = body_atoms_b[..., 0]
                head_pred_ok_b = self.head_pred_mask[body_preds_b_vals]
                unknown_ok_b = exists_b | head_pred_ok_b
                all_ok_b = (
                    unknown_ok_b | ~body_active_b.expand(-1, -1, G_use_b, -1)
                ).all(dim=-1)
                grounding_mask_b = grounding_mask_b & all_ok_b

                if self.width == 0:
                    all_exist_b = (
                        exists_b | ~body_active_b.expand(-1, -1, G_use_b, -1)
                    ).all(dim=-1)
                    grounding_mask_b = grounding_mask_b & all_exist_b

            # Exclude proven groundings from direction B (avoid duplicates)
            all_proven_b = (
                exists_b | ~body_active_b.expand(-1, -1, G_use_b, -1)
            ).all(dim=-1)
            grounding_mask_b = grounding_mask_b & ~all_proven_b

            # == PER-BATCH PruneIncompleteProofs (dual anchoring) ==
            if self._has_provable_set:
                base_mask_a = grounding_mask_a
                base_mask_b = grounding_mask_b

                body_active_exp_a = body_active.expand(-1, -1, G_use, -1)
                body_active_exp_b = body_active_b.expand(-1, -1, G_use_b, -1)

                # Initial query_proved: queries with fully-proven groundings
                fully_proven_a = (
                    (exists_a | ~body_active_exp_a).all(dim=-1)
                    & active_mask.unsqueeze(2)
                    & cand_mask_a
                )
                query_proved = fully_proven_a.any(dim=2).any(dim=1)  # (B,)
                fully_proven_b = (
                    (exists_b | ~body_active_exp_b).all(dim=-1)
                    & active_mask.unsqueeze(2)
                    & cand_mask_b
                )
                query_proved = query_proved | fully_proven_b.any(dim=2).any(
                    dim=1,
                )

                # Precompute hashes (constant across iterations)
                E = self._E
                E2 = E * E
                q_hashes = query_preds * E2 + query_subjs * E + query_objs
                sentinel = torch.tensor(-1, dtype=torch.long, device=dev)

                bh_a = (
                    body_atoms_a[..., 0] * E2
                    + body_atoms_a[..., 1] * E
                    + body_atoms_a[..., 2]
                )
                bh_a_flat = bh_a.reshape(-1)

                bh_b = (
                    body_atoms_b[..., 0] * E2
                    + body_atoms_b[..., 1] * E
                    + body_atoms_b[..., 2]
                )
                bh_b_flat = bh_b.reshape(-1)

                if self.depth >= 2:
                    provable_a = self._check_provable_precomputed(
                        body_atoms_a[..., 0].reshape(-1),
                        body_atoms_a[..., 1].reshape(-1),
                        body_atoms_a[..., 2].reshape(-1),
                    ).view(B, Re, G_use, M)
                    provable_b = self._check_provable_precomputed(
                        body_atoms_b[..., 0].reshape(-1),
                        body_atoms_b[..., 1].reshape(-1),
                        body_atoms_b[..., 2].reshape(-1),
                    ).view(B, Re, G_use_b, M)

                _n_prune = self.depth
                for _prune_iter in range(_n_prune):
                    proved_h = torch.where(
                        query_proved, q_hashes,
                        sentinel.expand_as(q_hashes),
                    )
                    proved_sorted, _ = proved_h.sort()

                    # Direction A
                    pos_a = torch.searchsorted(proved_sorted, bh_a_flat)
                    clamped_a = torch.clamp(pos_a, 0, B - 1)
                    in_proved_a = (pos_a < B) & (
                        proved_sorted[clamped_a] == bh_a_flat
                    )
                    in_proved_a = in_proved_a.view(B, Re, G_use, M)
                    if self.depth >= 2:
                        ok_a = (
                            exists_a | in_proved_a | provable_a
                            | ~body_active_exp_a
                        ).all(dim=-1)
                    else:
                        ok_a = (
                            exists_a | in_proved_a | ~body_active_exp_a
                        ).all(dim=-1)

                    # Direction B
                    pos_b = torch.searchsorted(proved_sorted, bh_b_flat)
                    clamped_b = torch.clamp(pos_b, 0, B - 1)
                    in_proved_b = (pos_b < B) & (
                        proved_sorted[clamped_b] == bh_b_flat
                    )
                    in_proved_b = in_proved_b.view(B, Re, G_use_b, M)
                    if self.depth >= 2:
                        ok_b = (
                            exists_b | in_proved_b | provable_b
                            | ~body_active_exp_b
                        ).all(dim=-1)
                    else:
                        ok_b = (
                            exists_b | in_proved_b | ~body_active_exp_b
                        ).all(dim=-1)

                    valid_a = base_mask_a & ok_a
                    valid_b = base_mask_b & ok_b
                    query_proved = (
                        query_proved
                        | valid_a.any(dim=2).any(dim=1)
                        | valid_b.any(dim=2).any(dim=1)
                    )

                grounding_mask_a = base_mask_a & ok_a
                grounding_mask_b = base_mask_b & ok_b

            # == 4. CONCATENATE directions A and B ==
            rule_indices_a = active_idx.unsqueeze(2).expand(
                -1, -1, G_use,
            )  # (B, Re, G_use)
            rule_indices_b = active_idx.unsqueeze(2).expand(
                -1, -1, G_use,
            )  # padded below

            if G_use < G_use_b:
                pad_a = G_use_b - G_use
                body_atoms_a = torch.nn.functional.pad(
                    body_atoms_a, (0, 0, 0, 0, 0, pad_a),
                )
                grounding_mask_a = torch.nn.functional.pad(
                    grounding_mask_a, (0, pad_a),
                )
                rule_indices_a = torch.nn.functional.pad(
                    rule_indices_a, (0, pad_a),
                )
                G_use = G_use_b
                rule_indices_b = active_idx.unsqueeze(2).expand(
                    -1, -1, G_use_b,
                )
            elif G_use_b < G_use:
                pad_b = G_use - G_use_b
                body_atoms_b = torch.nn.functional.pad(
                    body_atoms_b, (0, 0, 0, 0, 0, pad_b),
                )
                grounding_mask_b = torch.nn.functional.pad(
                    grounding_mask_b, (0, pad_b),
                )
                rule_indices_b = torch.nn.functional.pad(
                    active_idx.unsqueeze(2).expand(-1, -1, G_use_b),
                    (0, pad_b),
                )
            else:
                rule_indices_b = active_idx.unsqueeze(2).expand(
                    -1, -1, G_use_b,
                )

            body_atoms_all = torch.cat(
                [body_atoms_a, body_atoms_b], dim=2,
            )
            grounding_mask_all = torch.cat(
                [grounding_mask_a, grounding_mask_b], dim=2,
            )
            rule_indices_all = torch.cat(
                [rule_indices_a, rule_indices_b], dim=2,
            )
            G_total = body_atoms_all.size(2)
        else:
            # Per-batch PruneIncompleteProofs (no dual anchoring)
            if self._has_provable_set:
                base_mask_a = grounding_mask_a
                body_active_exp_a = body_active.expand(-1, -1, G_use, -1)

                fully_proven_a = (
                    (exists_a | ~body_active_exp_a).all(dim=-1)
                    & active_mask.unsqueeze(2)
                    & cand_mask_a
                )
                query_proved = fully_proven_a.any(dim=2).any(dim=1)

                E = self._E
                E2 = E * E
                q_hashes = query_preds * E2 + query_subjs * E + query_objs
                sentinel = torch.tensor(-1, dtype=torch.long, device=dev)

                bh_a = (
                    body_atoms_a[..., 0] * E2
                    + body_atoms_a[..., 1] * E
                    + body_atoms_a[..., 2]
                )
                bh_a_flat = bh_a.reshape(-1)

                if self.depth >= 2:
                    provable_a = self._check_provable_precomputed(
                        body_atoms_a[..., 0].reshape(-1),
                        body_atoms_a[..., 1].reshape(-1),
                        body_atoms_a[..., 2].reshape(-1),
                    ).view(B, Re, G_use, M)

                _n_prune = self.depth
                for _prune_iter in range(_n_prune):
                    proved_h = torch.where(
                        query_proved, q_hashes,
                        sentinel.expand_as(q_hashes),
                    )
                    proved_sorted, _ = proved_h.sort()

                    pos_a = torch.searchsorted(proved_sorted, bh_a_flat)
                    clamped_a = torch.clamp(pos_a, 0, B - 1)
                    in_proved_a = (pos_a < B) & (
                        proved_sorted[clamped_a] == bh_a_flat
                    )
                    in_proved_a = in_proved_a.view(B, Re, G_use, M)
                    if self.depth >= 2:
                        ok_a = (
                            exists_a | in_proved_a | provable_a
                            | ~body_active_exp_a
                        ).all(dim=-1)
                    else:
                        ok_a = (
                            exists_a | in_proved_a | ~body_active_exp_a
                        ).all(dim=-1)

                    valid_a = base_mask_a & ok_a
                    query_proved = (
                        query_proved | valid_a.any(dim=2).any(dim=1)
                    )

                grounding_mask_a = base_mask_a & ok_a

            body_atoms_all = body_atoms_a
            grounding_mask_all = grounding_mask_a
            rule_indices_all = active_idx.unsqueeze(2).expand(-1, -1, G_use)
            G_total = G_use

        # == 5. FLATTEN + TOPK ==
        total_slots = Re * G_total
        flat_body = body_atoms_all.reshape(B, total_slots, M, 3)
        flat_mask = grounding_mask_all.reshape(B, total_slots)
        flat_ridx = rule_indices_all.reshape(B, total_slots)

        if tG >= total_slots:
            pad_n = tG - total_slots
            if pad_n > 0:
                flat_body = torch.nn.functional.pad(
                    flat_body, (0, 0, 0, 0, 0, pad_n),
                )
                flat_mask = torch.nn.functional.pad(flat_mask, (0, pad_n))
                flat_ridx = torch.nn.functional.pad(flat_ridx, (0, pad_n))
            out_count = flat_mask.sum(dim=1)
            return ForwardResult(
                collected_body=flat_body,
                collected_mask=flat_mask,
                collected_count=out_count,
                collected_ridx=flat_ridx,
            )

        _, top_idx = flat_mask.to(torch.int8).topk(
            tG, dim=1, largest=True, sorted=False,
        )
        idx_body = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)
        out_body = flat_body.gather(1, idx_body)
        out_mask = flat_mask.gather(1, top_idx)
        out_ridx = flat_ridx.gather(1, top_idx)
        out_count = out_mask.sum(dim=1)

        return ForwardResult(
            collected_body=out_body,
            collected_mask=out_mask,
            collected_count=out_count,
            collected_ridx=out_ridx,
        )

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        return (
            f"ParametrizedBCGrounder(num_rules={self.num_rules}, "
            f"R_eff={self.R_eff}, "
            f"depth={self.depth}, width={self.width}, "
            f"max_groundings={self.max_groundings_per_query}, "
            f"max_total_groundings={self.max_total_groundings})"
        )
