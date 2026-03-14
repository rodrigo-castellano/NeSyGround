"""BCPruneGrounder — K_max-capped BFS + PruneIncompleteProofs post-filter.

Ports BCStaticPrune from ns_lib/grounding/backward_chaining_prune.py.
Standalone grounder inheriting from Grounder base. Compatible with
torch.compile(fullgraph=True, mode='reduce-overhead').

Algorithm:
  Phase 1: BFS with K_max-capped fact_index.enumerate() across depths.
           Stores all candidate groundings in pre-allocated buffers.
  Phase 2: PruneIncompleteProofs fixed-point — binary search on proved
           head hashes. Iteratively marks groundings as proved when all
           their body atoms are facts or heads of already-proved groundings.
  Phase 3: Output depth-0 proved groundings as ForwardResult.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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
)


class BCPruneGrounder(Grounder):
    """K_max-capped BFS + PruneIncompleteProofs, torch.compile compatible.

    Collects ALL candidate groundings across depths into pre-allocated
    buffers, then runs a PruneIncompleteProofs fixed-point filter to keep
    only transitively-provable groundings.

    Args:
        facts_idx:              [F, 3] fact triples (pred, arg0, arg1).
        rules_heads_idx:        [R, 3] rule head atoms.
        rules_bodies_idx:       [R, Bmax, 3] rule body atoms (padded).
        rule_lens:              [R] number of body atoms per rule.
        constant_no:            highest constant index.
        padding_idx:            padding value.
        device:                 target device.
        depth:                  BFS depth (number of iterations).
        max_groundings_per_query: K_max per rule per goal (enumerate cap).
        max_total_groundings:   output budget for grounding collection (tG).
        max_goals:              max goals in the BFS queue per query.
        predicate_no:           total number of predicates.
        num_entities:           total entities.
        max_facts_per_query:    K_f for fact index.
        fact_index_type:        'inverted' | 'block_sparse'.
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
        depth: int = 2,
        max_groundings_per_query: int = 32,
        max_total_groundings: int = 64,
        max_goals: int = 256,
        predicate_no: Optional[int] = None,
        num_entities: Optional[int] = None,
        max_facts_per_query: int = 64,
        fact_index_type: str = "inverted",
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

        self.depth = depth
        self.max_groundings_per_query = max_groundings_per_query
        self.max_total_groundings = max_total_groundings
        self.max_goals = max_goals

        # Compile rules from raw tensors
        R = rules_heads_idx.size(0)
        if R > 0:
            self.compiled_rules: List[CompiledRule] = compile_rules(
                rules_heads_idx, rules_bodies_idx, rule_lens, constant_no,
            )
            max_body_atoms = max(cr.num_body for cr in self.compiled_rules)
        else:
            self.compiled_rules = []
            max_body_atoms = 1
        self.max_body_atoms = max_body_atoms

        # Tensorize rule metadata
        head_preds, body_preds, num_body_atoms = tensorize_rules(
            self.compiled_rules, max_body_atoms, device,
        )
        self.register_buffer("head_preds", head_preds)
        self.register_buffer("body_preds", body_preds)
        self.register_buffer("num_body_atoms", num_body_atoms)

        # Vectorized metadata for enumeration
        (
            has_free,
            enum_pred_a,
            enum_bound_binding_a,
            enum_direction_a,
            check_arg_source_a,
            _num_free_vars,
            _body_introduces_fv,
            _body_enum_bound_src,
            _body_enum_direction,
            _body_enum_pred,
            _F_max,
        ) = build_vectorized_metadata(
            self.compiled_rules, max_body_atoms, device,
        )
        self.register_buffer("has_free", has_free)
        self.register_buffer("enum_pred_a", enum_pred_a)
        self.register_buffer("enum_bound_binding_a", enum_bound_binding_a)
        self.register_buffer("enum_direction_a", enum_direction_a)
        self.register_buffer("check_arg_source_a", check_arg_source_a)

        # Rule clustering: predicate -> rule indices
        P = predicate_no if predicate_no is not None else (
            max(int(rules_heads_idx[:, 0].max().item()) + 1, 1) if R > 0 else 1
        )
        # Ensure P covers padding_idx so lookups on inactive goals are safe
        if P <= padding_idx:
            P = padding_idx + 1
        pred_rule_indices, pred_rule_mask, R_eff = build_rule_clustering(
            self.compiled_rules, P, device,
        )
        self.register_buffer("pred_rule_indices", pred_rule_indices)
        self.register_buffer("pred_rule_mask", pred_rule_mask)
        self.R_eff = R_eff

        # Effective output size
        self.effective_total_G = min(
            max_total_groundings,
            R_eff * max_groundings_per_query,
        )

        # Pre-compute BFS shape constants for static buffer allocation
        G_fi = max_facts_per_query
        self._G_use = min(max_groundings_per_query, G_fi)
        self._G_proc = min(max_goals, 32)
        self._n_flat = self._G_proc * R_eff * self._G_use

        print(
            f"  BCPruneGrounder: {self.num_rules} rules, R_eff={R_eff}, "
            f"depth={depth}, K_max={max_groundings_per_query}, "
            f"tG={self.effective_total_G}"
        )

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,        # [B, 3]
        query_mask: Tensor,     # [B]
    ) -> ForwardResult:
        """Ground rules via BFS + PruneIncompleteProofs post-filter.

        Phase 1: K_max-capped BFS collects ALL valid candidate groundings
                 across depths into pre-allocated buffers.
        Phase 2: Fixed-point PruneIncompleteProofs keeps only transitively-
                 provable groundings (body atoms are facts or proved heads).
        Phase 3: Select depth-0 proved groundings for output.

        Args:
            queries:    [B, 3] — [pred_idx, subj_idx, obj_idx]
            query_mask: [B] — valid queries

        Returns:
            ForwardResult with collected_body, collected_mask,
            collected_count, collected_ridx.
        """
        B = queries.size(0)
        M = self.max_body_atoms
        tG = self.effective_total_G
        dev = queries.device
        Re = self.R_eff
        max_G = self.max_goals
        E = self.fact_index._num_entities
        E2 = E * E

        if self.num_rules == 0:
            return ForwardResult(
                collected_body=queries.new_zeros(B, tG, M, 3),
                collected_mask=torch.zeros(B, tG, dtype=torch.bool, device=dev),
                collected_count=queries.new_zeros(B),
                collected_ridx=queries.new_zeros(B, tG),
            )

        G_proc = self._G_proc
        G_use = self._G_use
        n_flat = self._n_flat
        total_slots = self.depth * n_flat

        # Initialize goal buffer from queries
        goals = queries.new_zeros(B, max_G, 3)
        goal_mask = torch.zeros(B, max_G, dtype=torch.bool, device=dev)
        goals[:, 0, :] = queries
        goal_mask[:, 0] = query_mask

        # Pre-allocate buffers for all depths
        all_body = queries.new_zeros(B, total_slots, M, 3)
        all_body_hashes = queries.new_zeros(B, total_slots, M)
        all_is_fact = torch.zeros(B, total_slots, M, dtype=torch.bool, device=dev)
        all_valid = torch.zeros(B, total_slots, dtype=torch.bool, device=dev)
        all_head_hash = queries.new_zeros(B, total_slots)
        all_ridx = queries.new_zeros(B, total_slots)
        all_body_active = torch.zeros(B, total_slots, M, dtype=torch.bool, device=dev)

        # -- Phase 1: BFS --
        for depth_step in range(self.depth):
            slot_s = depth_step * n_flat

            proc_preds = goals[:, :G_proc, 0]
            proc_subjs = goals[:, :G_proc, 1]
            proc_objs = goals[:, :G_proc, 2]
            proc_mask = goal_mask[:, :G_proc]

            # Rule clustering
            flat_preds = proc_preds.reshape(-1)
            active_idx = self.pred_rule_indices[flat_preds].view(B, G_proc, Re)
            active_rule_mask = (
                self.pred_rule_mask[flat_preds].view(B, G_proc, Re)
                & proc_mask.unsqueeze(-1)
            )

            # Gather per-rule metadata
            flat_ridx = active_idx.reshape(-1)
            has_free_q = self.has_free[flat_ridx].view(B, G_proc, Re)
            enum_pred_q = self.enum_pred_a[flat_ridx].view(B, G_proc, Re)
            enum_bound_q = self.enum_bound_binding_a[flat_ridx].view(B, G_proc, Re)
            enum_dir_q = self.enum_direction_a[flat_ridx].view(B, G_proc, Re)
            check_src_q = self.check_arg_source_a[flat_ridx].view(B, G_proc, Re, M, 2)
            body_preds_q = self.body_preds[flat_ridx].view(B, G_proc, Re, M)
            num_body_q = self.num_body_atoms[flat_ridx].view(B, G_proc, Re)

            # Enumerate candidates (K_max capped via fact_index)
            source = torch.stack([proc_subjs, proc_objs], dim=-1)
            source_exp = source.unsqueeze(2).expand(-1, -1, Re, -1)
            enum_bound_vals = source_exp.gather(
                -1, enum_bound_q.unsqueeze(-1),
            ).squeeze(-1)

            candidates, cand_mask = self.fact_index.enumerate(
                enum_pred_q.reshape(-1),
                enum_bound_vals.reshape(-1),
                enum_dir_q.reshape(-1),
            )
            candidates = candidates[:, :G_use].reshape(B, G_proc, Re, G_use)
            cand_mask = cand_mask[:, :G_use].reshape(B, G_proc, Re, G_use)

            # Mask: only free-var rules get enumerated candidates
            cand_mask = cand_mask & has_free_q.unsqueeze(-1)
            bound_rules = ~has_free_q
            cand_mask[:, :, :, 0] = (
                cand_mask[:, :, :, 0]
                | (bound_rules & active_rule_mask)
            )

            # Resolve body atoms (clamp source indices to 3-slot source)
            q_subjs_exp = proc_subjs.view(B, G_proc, 1, 1).expand(-1, -1, Re, G_use)
            q_objs_exp = proc_objs.view(B, G_proc, 1, 1).expand(-1, -1, Re, G_use)
            source_3 = torch.stack([q_subjs_exp, q_objs_exp, candidates], dim=-1)
            source_exp4 = source_3.unsqueeze(4).expand(-1, -1, -1, -1, M, -1)

            idx_0 = check_src_q[..., 0].clamp(max=2).unsqueeze(3).expand(
                -1, -1, -1, G_use, -1)
            arg0 = source_exp4.gather(5, idx_0.unsqueeze(-1)).squeeze(-1)
            idx_1 = check_src_q[..., 1].clamp(max=2).unsqueeze(3).expand(
                -1, -1, -1, G_use, -1)
            arg1 = source_exp4.gather(5, idx_1.unsqueeze(-1)).squeeze(-1)

            body_preds_exp = body_preds_q.unsqueeze(3).expand(-1, -1, -1, G_use, -1)
            body_atoms = torch.stack([body_preds_exp, arg0, arg1], dim=-1)

            # Existence check (base facts)
            body_flat = body_atoms.reshape(-1, 3)
            exists_flat = self.fact_index.exists(body_flat)
            exists = exists_flat.view(B, G_proc, Re, G_use, M)

            # Body active mask
            atom_idx_m = torch.arange(M, device=dev).view(1, 1, 1, 1, M)
            body_active = atom_idx_m < num_body_q.view(B, G_proc, Re, 1, 1)

            # Grounding validity
            grounding_valid = active_rule_mask.unsqueeze(-1) & cand_mask

            # Query exclusion
            query_exp = queries.view(B, 1, 1, 1, 1, 3).expand(
                -1, G_proc, Re, G_use, M, -1)
            is_query = (body_atoms == query_exp).all(dim=-1)
            has_query = (
                is_query & body_active.expand(-1, -1, -1, G_use, -1)
            ).any(dim=-1)
            grounding_valid = grounding_valid & ~has_query

            # Body hashes
            body_hashes = (
                body_atoms[..., 0] * E2
                + body_atoms[..., 1] * E
                + body_atoms[..., 2]
            )  # (B, G_proc, Re, G_use, M)

            # Head hashes (hash of the goal being expanded)
            goal_hash = proc_preds * E2 + proc_subjs * E + proc_objs
            head_hash_exp = goal_hash.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, Re, G_use)  # (B, G_proc, Re, G_use)

            # Expand body_active for flatten
            body_active_exp = body_active.expand(-1, -1, -1, G_use, -1)

            # Store in big buffers
            all_body[:, slot_s:slot_s + n_flat] = body_atoms.reshape(B, n_flat, M, 3)
            all_body_hashes[:, slot_s:slot_s + n_flat] = body_hashes.reshape(B, n_flat, M)
            all_is_fact[:, slot_s:slot_s + n_flat] = exists.reshape(B, n_flat, M)
            all_valid[:, slot_s:slot_s + n_flat] = grounding_valid.reshape(B, n_flat)
            all_head_hash[:, slot_s:slot_s + n_flat] = head_hash_exp.reshape(B, n_flat)
            all_ridx[:, slot_s:slot_s + n_flat] = active_idx.unsqueeze(-1).expand(
                -1, -1, -1, G_use).reshape(B, n_flat)
            all_body_active[:, slot_s:slot_s + n_flat] = body_active_exp.reshape(
                B, n_flat, M)

            # Collect new goals: body atoms that are NOT base facts
            all_facts = (exists | ~body_active_exp).all(dim=-1)
            has_unproved = grounding_valid & ~all_facts
            unproved_atom_mask = (
                body_active_exp
                & ~exists
                & has_unproved.unsqueeze(-1)
            )

            new_goals_all = body_atoms.reshape(B, n_flat * M, 3)
            new_goal_mask = unproved_atom_mask.reshape(B, n_flat * M)

            n_new = new_goal_mask.size(1)
            n_sel = min(max_G, n_new)
            _, goal_topk = new_goal_mask.to(torch.int8).topk(
                n_sel, dim=1, largest=True, sorted=False)

            goals_sel = new_goals_all.gather(
                1, goal_topk.unsqueeze(-1).expand(-1, -1, 3))
            goal_mask_sel = new_goal_mask.gather(1, goal_topk)

            # Dedup via hash+sort+adjacent_diff
            sel_hashes = (
                goals_sel[..., 0] * E2
                + goals_sel[..., 1] * E
                + goals_sel[..., 2]
            )
            sentinel_goal = torch.tensor(-1, dtype=torch.long, device=dev)
            sel_hashes = torch.where(
                goal_mask_sel, sel_hashes,
                sentinel_goal.expand_as(sel_hashes))

            sorted_hashes, sort_idx = sel_hashes.sort(dim=1)
            prev_hashes = torch.nn.functional.pad(
                sorted_hashes[:, :-1], (1, 0), value=-2)
            is_dup = (sorted_hashes == prev_hashes)
            is_valid_sorted = goal_mask_sel.gather(1, sort_idx) & ~is_dup

            dedup_goals = goals_sel.gather(
                1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
            _, dedup_topk = is_valid_sorted.to(torch.int8).topk(
                min(max_G, n_sel), dim=1, largest=True, sorted=False)

            goals_next = dedup_goals.gather(
                1, dedup_topk.unsqueeze(-1).expand(-1, -1, 3))
            goal_mask_next = is_valid_sorted.gather(1, dedup_topk)

            n_got = goals_next.size(1)
            if n_got < max_G:
                pad_g = max_G - n_got
                goals_next = torch.nn.functional.pad(goals_next, (0, 0, 0, pad_g))
                goal_mask_next = torch.nn.functional.pad(goal_mask_next, (0, pad_g))
            else:
                goals_next = goals_next[:, :max_G]
                goal_mask_next = goal_mask_next[:, :max_G]

            goals = goals_next
            goal_mask = goal_mask_next

        # -- Phase 2: PruneIncompleteProofs fixed-point --
        # Initially proved: ALL active body atoms are base facts
        grounding_proved = (
            (all_is_fact | ~all_body_active).all(dim=-1) & all_valid
        )

        sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
        body_hashes_2d = all_body_hashes.reshape(B, total_slots * M)

        for _ in range(self.depth + 1):
            # Head hashes of proved groundings (sentinel for non-proved)
            proved_hashes = torch.where(
                grounding_proved, all_head_hash,
                sentinel.expand_as(all_head_hash),
            )  # (B, total_slots)

            # Sort per batch row (sentinels go to the front)
            proved_sorted, _ = proved_hashes.sort(dim=1)

            # Binary search each body atom hash in the proved heads
            pos = torch.searchsorted(proved_sorted, body_hashes_2d)
            pos = pos.clamp(max=total_slots - 1)
            found = proved_sorted.gather(1, pos) == body_hashes_2d
            body_in_proved = found.view(B, total_slots, M)

            # Atom OK if: base fact, OR proved head, OR inactive (padding)
            atom_ok = all_is_fact | body_in_proved | ~all_body_active
            grounding_proved = atom_ok.all(dim=-1) & all_valid

        # -- Phase 3: Output (depth-0 proved groundings) --
        depth0_proved = grounding_proved[:, :n_flat]
        depth0_body = all_body[:, :n_flat]
        depth0_ridx = all_ridx[:, :n_flat]

        n_keep = min(tG, n_flat)
        _, keep_idx = depth0_proved.to(torch.int8).topk(
            n_keep, dim=1, largest=True, sorted=False)

        if n_keep < tG:
            pad_n = tG - n_keep
            keep_body = depth0_body.gather(
                1, keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
            keep_mask = depth0_proved.gather(1, keep_idx)
            keep_ridx = depth0_ridx.gather(1, keep_idx)
            collected_body = torch.nn.functional.pad(
                keep_body, (0, 0, 0, 0, 0, pad_n))
            collected_mask = torch.nn.functional.pad(keep_mask, (0, pad_n))
            collected_ridx = torch.nn.functional.pad(keep_ridx, (0, pad_n))
        else:
            collected_body = depth0_body.gather(
                1, keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
            collected_mask = depth0_proved.gather(1, keep_idx)
            collected_ridx = depth0_ridx.gather(1, keep_idx)

        collected_count = collected_mask.sum(dim=1)
        return ForwardResult(
            collected_body=collected_body,
            collected_mask=collected_mask,
            collected_count=collected_count,
            collected_ridx=collected_ridx,
        )

    def __repr__(self) -> str:
        return (
            f"BCPruneGrounder(num_rules={self.num_rules}, R_eff={self.R_eff}, "
            f"depth={self.depth}, K_max={self.max_groundings_per_query}, "
            f"tG={self.effective_total_G})"
        )
