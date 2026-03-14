"""FullBCGrounder — full backward chaining grounder BC_{∞,1} for raw tensors.

Port of ns_lib/grounding/full_grounder.py to the grounder/ package.

Implements BC_{∞,1} from the IJCAI paper: the Full Grounder with w=∞, d=1.
Enumerates ALL E constants for free variables and accepts all groundings
(even if body atoms are not in the fact base). This is the exact grounding
used by LTN, SBR, and MLNs.

Unlike BCGrounder which fact-anchors (enumerates from matching facts) and
filters by width/depth, this grounder shows the raw grounding explosion:
candidates grow with E, not D (KG degree).

Key properties:
- O(B * R_eff * E * M) per forward pass
- Output is independent of w and d (always w=∞, d=1)
- Same output format as BCGrounder (ForwardResult)
- Only filter: query exclusion (body atom != query)

Compatible with torch.compile(fullgraph=True, mode='reduce-overhead').
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from grounder.grounders.base import Grounder
from grounder.types import ForwardResult
from grounder.compilation import (
    CompiledRule,
    compile_rules,
    build_rule_clustering,
    tensorize_rules,
    BINDING_FREE_VAR_OFFSET,
)


class FullBCGrounder(Grounder):
    """Full backward chaining grounder BC_{∞,1} — enumerates ALL constants.

    For each 2-body rule with a free variable, generates E candidates instead
    of D (KG degree). All groundings are accepted regardless of whether body
    atoms exist in the fact base (w=∞). No depth cascading (d=1).

    Args:
        facts_idx:           [F, 3] fact triples (pred, arg0, arg1)
        rules_heads_idx:     [R, 3] rule head atoms
        rules_bodies_idx:    [R, Bmax, 3] rule body atoms (padded)
        rule_lens:           [R] number of body atoms per rule
        constant_no:         highest constant index (variables start above)
        padding_idx:         padding value
        device:              target device
        max_total_groundings: static cap on total groundings per query
        predicate_no:        total number of predicates (exclusive upper bound)
        num_entities:        total entities in the KG
        max_facts_per_query: K_f for fact index (passed to base)
        fact_index_type:     fact index type (passed to base)
    """

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
        max_total_groundings: int = 64,
        predicate_no: int,
        num_entities: int,
        max_facts_per_query: int = 64,
        fact_index_type: str = "arg_key",
        **kwargs,
    ) -> None:
        super().__init__(
            facts_idx,
            rules_heads_idx,
            rules_bodies_idx,
            rule_lens,
            constant_no,
            padding_idx,
            device,
            predicate_no=predicate_no,
            num_entities=num_entities,
            max_facts_per_query=max_facts_per_query,
            fact_index_type=fact_index_type,
            **kwargs,
        )

        self.max_total_groundings = max_total_groundings
        self._num_entities = num_entities
        self._predicate_no = predicate_no

        # Compile rules from raw tensors
        self.compiled_rules = compile_rules(
            rules_heads_idx, rules_bodies_idx, rule_lens, constant_no,
        )

        # Max body atoms across all rules
        self.max_body_atoms = max(
            (cr.num_body for cr in self.compiled_rules), default=1,
        )

        E = num_entities
        dev = device

        # All entity indices for full enumeration
        self.register_buffer(
            "all_entities",
            torch.arange(E, dtype=torch.long, device=dev),
        )

        # Slot-0-only mask for bound-only rules (no in-place ops needed)
        slot0_mask = torch.zeros(E, dtype=torch.bool, device=dev)
        slot0_mask[0] = True
        self.register_buffer("slot0_mask", slot0_mask)

        # Tensorize rules
        head_preds, body_preds, num_body_atoms = tensorize_rules(
            self.compiled_rules, self.max_body_atoms, dev,
        )
        self.register_buffer("head_preds", head_preds)
        self.register_buffer("body_preds", body_preds)
        self.register_buffer("num_body_atoms", num_body_atoms)

        # Build vectorized metadata (only has_free and check_arg_source needed)
        self._build_full_metadata(dev)

        # Build rule clustering
        pred_rule_indices, pred_rule_mask, R_eff = build_rule_clustering(
            self.compiled_rules, predicate_no, dev,
        )
        self.R_eff = R_eff
        self.register_buffer("pred_rule_indices", pred_rule_indices)
        self.register_buffer("pred_rule_mask", pred_rule_mask)

        self.effective_total_G = min(max_total_groundings, R_eff * E)

        print(
            f"  FullBCGrounder(BC_∞,1): E={E}, R_eff={R_eff}, "
            f"slots_per_query=R_eff*E={R_eff * E}, "
            f"cap={self.effective_total_G}"
        )

    def _build_full_metadata(self, device: torch.device) -> None:
        """Build has_free and check_arg_source tables for body atom resolution.

        check_arg_source[r, m, a]: 0=query_subj, 1=query_obj, 2=candidate (free var)

        The ns_lib version maps binding >= 2 to source 2 (candidate).  We do
        the same here: head_var0 → 0, head_var1 → 1, any free var → 2.
        """
        R = max(len(self.compiled_rules), 1)
        M = self.max_body_atoms

        has_free = torch.zeros(R, dtype=torch.bool, device=device)
        check_arg_source = torch.zeros(R, M, 2, dtype=torch.long, device=device)

        for i, cr in enumerate(self.compiled_rules):
            # Check if any body atom references a free variable
            for bp in cr.body_patterns:
                if (
                    bp["arg0_binding"] >= BINDING_FREE_VAR_OFFSET
                    or bp["arg1_binding"] >= BINDING_FREE_VAR_OFFSET
                ):
                    has_free[i] = True
                    break

            # Build arg source mapping: 0=head_var0, 1=head_var1, 2=free var
            for j, bp in enumerate(cr.body_patterns):
                for a, key in enumerate(["arg0_binding", "arg1_binding"]):
                    b = bp[key]
                    if b == 0:
                        check_arg_source[i, j, a] = 0
                    elif b == 1:
                        check_arg_source[i, j, a] = 1
                    else:
                        check_arg_source[i, j, a] = 2

        self.register_buffer("has_free", has_free)
        self.register_buffer("check_arg_source", check_arg_source)

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,       # [B, 3]
        query_mask: Tensor,    # [B]
    ) -> ForwardResult:
        """Ground rules by enumerating ALL constants for free variables.

        BC_{∞,1}: all entity substitutions accepted (w=∞), single step (d=1).
        Only filter: query exclusion (body atom identical to query is rejected).

        Args:
            queries:    [B, 3] - [pred_idx, subj_idx, obj_idx]
            query_mask: [B] - valid queries

        Returns:
            ForwardResult with collected_body, collected_mask, collected_count,
            collected_ridx.
        """
        B = queries.size(0)
        Re = self.R_eff
        E = self._num_entities
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

        # == 1. RULE CLUSTERING ==
        active_idx = self.pred_rule_indices[query_preds]        # (B, Re)
        active_mask = (
            self.pred_rule_mask[query_preds]
            & query_mask.unsqueeze(1)
        )                                                        # (B, Re)

        has_free_q = self.has_free[active_idx]                   # (B, Re)
        body_preds_q = self.body_preds[active_idx]               # (B, Re, M)
        num_body_q = self.num_body_atoms[active_idx]             # (B, Re)
        check_src_q = self.check_arg_source[active_idx]          # (B, Re, M, 2)

        # == 2. CANDIDATE GENERATION: ALL entities ==
        # For free-var rules: all E slots valid. For bound-only: only slot 0.
        candidates = self.all_entities.view(1, 1, E).expand(B, Re, -1)

        # Build candidate mask without in-place ops
        free_mask = has_free_q.unsqueeze(2) & active_mask.unsqueeze(2)
        bound_mask = (~has_free_q).unsqueeze(2) & active_mask.unsqueeze(2)
        slot0_exp = self.slot0_mask.view(1, 1, E)
        cand_mask = free_mask.expand(-1, -1, E) | (bound_mask & slot0_exp)

        # == 3. RESOLVE BODY ATOMS ==
        q_subjs_exp = query_subjs.view(B, 1, 1).expand(-1, Re, E)
        q_objs_exp = query_objs.view(B, 1, 1).expand(-1, Re, E)
        source_3 = torch.stack(
            [q_subjs_exp, q_objs_exp, candidates], dim=3,
        )                                                        # (B, Re, E, 3)

        source_exp = source_3.unsqueeze(3).expand(
            -1, -1, -1, M, -1,
        )                                                        # (B, Re, E, M, 3)

        idx_0 = check_src_q[:, :, :, 0].view(
            B, Re, 1, M,
        ).expand(-1, -1, E, -1)
        arg0 = source_exp.gather(
            4, idx_0.unsqueeze(-1),
        ).squeeze(-1)                                            # (B, Re, E, M)

        idx_1 = check_src_q[:, :, :, 1].view(
            B, Re, 1, M,
        ).expand(-1, -1, E, -1)
        arg1 = source_exp.gather(
            4, idx_1.unsqueeze(-1),
        ).squeeze(-1)                                            # (B, Re, E, M)

        body_preds_exp = body_preds_q.unsqueeze(2).expand(-1, -1, E, -1)
        body_atoms = torch.stack(
            [body_preds_exp, arg0, arg1], dim=-1,
        )                                                        # (B, Re, E, M, 3)

        # == 4. QUERY EXCLUSION ==
        # Reject groundings where a body atom is identical to the query
        atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
        body_active = atom_idx < num_body_q.view(B, Re, 1, 1)

        query_exp = queries.view(B, 1, 1, 1, 3).expand(-1, Re, E, M, -1)
        is_query = (body_atoms == query_exp).all(dim=-1)
        has_query_atom = (
            is_query & body_active.expand(-1, -1, E, -1)
        ).any(dim=-1)
        grounding_mask = cand_mask & ~has_query_atom

        # == 5. FLATTEN + TOPK ==
        rule_indices = active_idx.unsqueeze(2).expand(-1, -1, E)
        total_slots = Re * E

        flat_body = body_atoms.reshape(B, total_slots, M, 3)
        flat_mask = grounding_mask.reshape(B, total_slots)
        flat_ridx = rule_indices.reshape(B, total_slots)

        if tG >= total_slots:
            pad = tG - total_slots
            if pad > 0:
                flat_body = torch.nn.functional.pad(
                    flat_body, (0, 0, 0, 0, 0, pad),
                )
                flat_mask = torch.nn.functional.pad(flat_mask, (0, pad))
                flat_ridx = torch.nn.functional.pad(flat_ridx, (0, pad))
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

    def __repr__(self) -> str:
        E = self._num_entities
        return (
            f"FullBCGrounder(BC_∞,1, num_rules={self.num_rules}, "
            f"R_eff={self.R_eff}, E={E}, "
            f"max_total_groundings={self.max_total_groundings})"
        )
