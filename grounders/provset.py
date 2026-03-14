"""BCProvsetGrounder — K_max-capped BFS + FC provable-set soundness check.

Ports BCStaticProvset from ns_lib/grounding/backward_chaining_provset.py.
Standalone grounder inheriting from Grounder base. Compatible with
torch.compile(fullgraph=True, mode='reduce-overhead').

Algorithm:
  At init: computes the provable set I_D via run_forward_chaining.
  Per step: enumerates candidates via K_max-capped fact_index.enumerate(),
            checks body atoms against facts UNION provable set.
  Collects proved groundings across depths, advances goals.
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
    check_in_provable,
)
from grounder.forward_chaining import run_forward_chaining


class BCProvsetGrounder(Grounder):
    """K_max-capped BFS + FC provable-set check, torch.compile compatible.

    Sound via FC provable-set check: body atoms are accepted if they are
    base facts OR in the provable set I_D computed at init via forward
    chaining. The provable set is stored as a sorted buffer and checked
    via torch.searchsorted (fullgraph-safe).

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
        fc_depth:               forward chaining depth for provable set
                                (defaults to depth if not specified).
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
        fc_depth: Optional[int] = None,
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

        # Compute provable set via forward chaining (CPU, at init time)
        fc_d = fc_depth if fc_depth is not None else depth
        assert num_entities is not None, "num_entities required for provable set"
        provable_hashes, n_prov = run_forward_chaining(
            self.compiled_rules, facts_idx, num_entities, P,
            depth=fc_d, device="cpu",
        )
        self.register_buffer("_provable_hashes", provable_hashes)

        print(
            f"  BCProvsetGrounder: {self.num_rules} rules, R_eff={R_eff}, "
            f"depth={depth}, K_max={max_groundings_per_query}, "
            f"tG={self.effective_total_G}, provable_atoms={n_prov}"
        )

    @torch.no_grad()
    def forward(
        self,
        queries: Tensor,        # [B, 3]
        query_mask: Tensor,     # [B]
    ) -> ForwardResult:
        """Ground rules via iterative BFS with provable-set check.

        Body atoms are accepted if they are base facts OR in the provable
        set I_D. This makes grounding sound: no false groundings from
        disconnected subgoals.

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
        G = self.max_groundings_per_query
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

        # Initialize goal buffer from queries
        goals = queries.new_zeros(B, max_G, 3)
        goal_mask = torch.zeros(B, max_G, dtype=torch.bool, device=dev)
        goals[:, 0, :] = queries
        goal_mask[:, 0] = query_mask

        # Output buffer
        collected_body = queries.new_zeros(B, tG, M, 3)
        collected_mask = torch.zeros(B, tG, dtype=torch.bool, device=dev)
        collected_ridx = queries.new_zeros(B, tG)

        for _depth in range(self.depth):
            G_proc = min(max_G, 32)

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
            check_src_q = self.check_arg_source_a[flat_ridx].view(
                B, G_proc, Re, M, 2)
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
            G_fi = candidates.size(1)
            G_use = min(G, G_fi)
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
            q_subjs_exp = proc_subjs.view(B, G_proc, 1, 1).expand(
                -1, -1, Re, G_use)
            q_objs_exp = proc_objs.view(B, G_proc, 1, 1).expand(
                -1, -1, Re, G_use)
            source_3 = torch.stack(
                [q_subjs_exp, q_objs_exp, candidates], dim=-1)
            source_exp4 = source_3.unsqueeze(4).expand(-1, -1, -1, -1, M, -1)

            idx_0 = check_src_q[..., 0].clamp(max=2).unsqueeze(3).expand(
                -1, -1, -1, G_use, -1)
            arg0 = source_exp4.gather(5, idx_0.unsqueeze(-1)).squeeze(-1)
            idx_1 = check_src_q[..., 1].clamp(max=2).unsqueeze(3).expand(
                -1, -1, -1, G_use, -1)
            arg1 = source_exp4.gather(5, idx_1.unsqueeze(-1)).squeeze(-1)

            body_preds_exp = body_preds_q.unsqueeze(3).expand(
                -1, -1, -1, G_use, -1)
            body_atoms = torch.stack([body_preds_exp, arg0, arg1], dim=-1)

            # Existence check (base facts)
            body_flat = body_atoms.reshape(-1, 3)
            exists_flat = self.fact_index.exists(body_flat)
            exists = exists_flat.view(B, G_proc, Re, G_use, M)

            # Provable-set check (FC I_D)
            body_hashes = (
                body_atoms[..., 0] * E2
                + body_atoms[..., 1] * E
                + body_atoms[..., 2]
            )  # (B, G_proc, Re, G_use, M)
            in_provable = check_in_provable(body_hashes, self._provable_hashes)

            atom_idx_m = torch.arange(M, device=dev).view(1, 1, 1, 1, M)
            body_active = atom_idx_m < num_body_q.view(B, G_proc, Re, 1, 1)

            grounding_valid = active_rule_mask.unsqueeze(-1) & cand_mask

            # Query exclusion
            query_exp = queries.view(B, 1, 1, 1, 1, 3).expand(
                -1, G_proc, Re, G_use, M, -1)
            is_query = (body_atoms == query_exp).all(dim=-1)
            has_query = (
                is_query & body_active.expand(-1, -1, -1, G_use, -1)
            ).any(dim=-1)
            grounding_valid = grounding_valid & ~has_query

            # Fully proved: base fact OR in provable set
            all_proved = (
                (exists | in_provable) | ~body_active.expand(-1, -1, -1, G_use, -1)
            ).all(dim=-1)
            fully_proved = grounding_valid & all_proved

            # Collect proved groundings
            n_flat = G_proc * Re * G_use
            proved_flat = fully_proved.reshape(B, n_flat)
            body_flat_all = body_atoms.reshape(B, n_flat, M, 3)
            ridx_flat = active_idx.unsqueeze(-1).expand(
                -1, -1, -1, G_use).reshape(B, n_flat)

            all_body = torch.cat([collected_body, body_flat_all], dim=1)
            all_mask = torch.cat([collected_mask, proved_flat], dim=1)
            all_ridx = torch.cat([collected_ridx, ridx_flat], dim=1)

            n_total = all_mask.size(1)
            n_keep = min(tG, n_total)
            _, keep_idx = all_mask.to(torch.int8).topk(
                n_keep, dim=1, largest=True, sorted=False)

            if n_keep < tG:
                pad_n = tG - n_keep
                keep_body = all_body.gather(
                    1, keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
                keep_mask = all_mask.gather(1, keep_idx)
                keep_ridx = all_ridx.gather(1, keep_idx)
                collected_body = torch.nn.functional.pad(
                    keep_body, (0, 0, 0, 0, 0, pad_n))
                collected_mask = torch.nn.functional.pad(keep_mask, (0, pad_n))
                collected_ridx = torch.nn.functional.pad(keep_ridx, (0, pad_n))
            else:
                collected_body = all_body.gather(
                    1, keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
                collected_mask = all_mask.gather(1, keep_idx)
                collected_ridx = all_ridx.gather(1, keep_idx)

            # Collect new goals
            has_unproved = grounding_valid & ~all_proved
            unproved_atom_mask = (
                body_active.expand(-1, -1, -1, G_use, -1)
                & ~(exists | in_provable)
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
            sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
            sel_hashes = torch.where(
                goal_mask_sel, sel_hashes,
                sentinel.expand_as(sel_hashes))

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
                goals_next = torch.nn.functional.pad(
                    goals_next, (0, 0, 0, pad_g))
                goal_mask_next = torch.nn.functional.pad(
                    goal_mask_next, (0, pad_g))
            else:
                goals_next = goals_next[:, :max_G]
                goal_mask_next = goal_mask_next[:, :max_G]

            goals = goals_next
            goal_mask = goal_mask_next

        collected_count = collected_mask.sum(dim=1)
        return ForwardResult(
            collected_body=collected_body,
            collected_mask=collected_mask,
            collected_count=collected_count,
            collected_ridx=collected_ridx,
        )

    def __repr__(self) -> str:
        return (
            f"BCProvsetGrounder(num_rules={self.num_rules}, R_eff={self.R_eff}, "
            f"depth={self.depth}, K_max={self.max_groundings_per_query}, "
            f"provable_atoms={self._provable_hashes.size(0)}, "
            f"tG={self.effective_total_G})"
        )
