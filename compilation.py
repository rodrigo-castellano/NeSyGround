"""Rule compilation from raw tensors for vectorized grounding.

Extracts rule metadata (head bindings, body patterns, enumeration order,
rule clustering) from raw tensor representations.  No ns_lib domain objects
required — works purely with integer tensors where variables are identified
by value > constant_no.

Key exports:
    CompiledRule  — per-rule pattern extracted from raw tensors
    compile_rules — batch-compile all rules
    build_vectorized_metadata — per-rule enum/check tensors
    build_rule_clustering — predicate → rule mapping
    tensorize_rules — head_preds / body_preds / num_body tensors
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

# Named constants for binding indices.
BINDING_HEAD_VAR0 = 0
BINDING_HEAD_VAR1 = 1
BINDING_FREE_VAR_OFFSET = 2
BINDING_NO_FREE_VAR = -1


class CompiledRule:
    """Pre-compiled rule pattern from raw tensor representation.

    Supports multiple free variables via cascaded enumeration.
    Body atoms are reordered so that each atom has at least one
    already-known argument (topological sort on variable binding).

    Args:
        rule_idx: Original rule index in the rule tensor.
        head: [3] tensor (pred, arg0, arg1) for the rule head.
        body: [M, 3] tensor (pred, arg0, arg1) for the rule body (padded).
        body_len: Number of actual body atoms.
        constant_no: Highest constant index — values above are variables.
    """

    def __init__(
        self,
        rule_idx: int,
        head: Tensor,
        body: Tensor,
        body_len: int,
        constant_no: int,
    ) -> None:
        self.rule_idx = rule_idx
        self.constant_no = constant_no

        # Head info
        self.head_pred_idx: int = head[0].item()
        self.head_var0: int = head[1].item()
        self.head_var1: int = head[2].item()

        self.num_body: int = body_len
        self.body_pred_indices: List[int] = [
            body[j, 0].item() for j in range(body_len)
        ]

        # Collect all variables from body atoms
        all_body_vars: set = set()
        for j in range(body_len):
            a0 = body[j, 1].item()
            a1 = body[j, 2].item()
            if a0 > constant_no:
                all_body_vars.add(a0)
            if a1 > constant_no:
                all_body_vars.add(a1)

        # Head variables
        head_vars: set = set()
        if self.head_var0 > constant_no:
            head_vars.add(self.head_var0)
        if self.head_var1 > constant_no:
            head_vars.add(self.head_var1)

        # Free variables = body vars not in head
        self.free_vars_list: List[int] = sorted(all_body_vars - head_vars)
        self.num_free: int = len(self.free_vars_list)
        self.free_var_to_idx: Dict[int, int] = {
            v: i for i, v in enumerate(self.free_vars_list)
        }

        # Build body_patterns with extended bindings
        self.body_patterns: List[dict] = []
        for j in range(body_len):
            a0 = body[j, 1].item()
            a1 = body[j, 2].item()
            pattern = {
                "pred_idx": body[j, 0].item(),
                "arg0_var": a0,
                "arg1_var": a1,
                "arg0_binding": self._get_binding(a0),
                "arg1_binding": self._get_binding(a1),
            }
            self.body_patterns.append(pattern)

        # Compute processing order and reorder body atoms
        self._compute_enum_order()

    def _get_binding(self, val: int) -> int:
        """Map a value to its binding source index."""
        if val > self.constant_no:
            if val == self.head_var0:
                return BINDING_HEAD_VAR0
            if val == self.head_var1:
                return BINDING_HEAD_VAR1
            if val in self.free_var_to_idx:
                return BINDING_FREE_VAR_OFFSET + self.free_var_to_idx[val]
        # Constant or head variable when it's the same as another head var
        # (fall through to head_var0 for constants bound to head)
        if val == self.head_var0:
            return BINDING_HEAD_VAR0
        if val == self.head_var1:
            return BINDING_HEAD_VAR1
        return BINDING_HEAD_VAR0  # constant fallback

    def _compute_enum_order(self) -> None:
        """Compute processing order for cascaded enumeration.

        Reorders body_patterns so that each atom has at least one known arg.
        Known sources start as {0, 1} (head vars) and grow as free vars
        are introduced.
        """
        known_sources = {BINDING_HEAD_VAR0, BINDING_HEAD_VAR1}
        remaining = list(range(self.num_body))
        order: List[int] = []
        meta_list: List[dict] = []

        while remaining:
            found = False
            for idx in remaining:
                bp = self.body_patterns[idx]
                b0, b1 = bp["arg0_binding"], bp["arg1_binding"]
                a0_known = b0 in known_sources
                a1_known = b1 in known_sources

                if a0_known or a1_known:
                    if a0_known and a1_known:
                        meta = {
                            "introduces_fv": BINDING_NO_FREE_VAR,
                            "enum_bound_src": BINDING_HEAD_VAR0,
                            "enum_direction": 0,
                            "enum_pred": bp["pred_idx"],
                        }
                    elif a0_known:
                        fv_idx = b1 - BINDING_FREE_VAR_OFFSET
                        meta = {
                            "introduces_fv": fv_idx,
                            "enum_bound_src": b0,
                            "enum_direction": 0,
                            "enum_pred": bp["pred_idx"],
                        }
                        known_sources.add(b1)
                    else:
                        fv_idx = b0 - BINDING_FREE_VAR_OFFSET
                        meta = {
                            "introduces_fv": fv_idx,
                            "enum_bound_src": b1,
                            "enum_direction": 1,
                            "enum_pred": bp["pred_idx"],
                        }
                        known_sources.add(b0)

                    order.append(idx)
                    meta_list.append(meta)
                    remaining.remove(idx)
                    found = True
                    break

            if not found:
                for idx in remaining:
                    order.append(idx)
                    meta_list.append(
                        {
                            "introduces_fv": BINDING_NO_FREE_VAR,
                            "enum_bound_src": BINDING_HEAD_VAR0,
                            "enum_direction": 0,
                            "enum_pred": self.body_patterns[idx]["pred_idx"],
                        }
                    )
                break

        self.body_order = order
        self.enum_meta = meta_list

        # Reorder body_patterns and body_pred_indices to match processing order
        self.body_patterns = [self.body_patterns[i] for i in order]
        self.body_pred_indices = [self.body_pred_indices[i] for i in order]


# ══════════════════════════════════════════════════════════════════════════════
# Batch compilation
# ══════════════════════════════════════════════════════════════════════════════


def compile_rules(
    rule_heads: Tensor,
    rule_bodies: Tensor,
    rule_lens: Tensor,
    constant_no: int,
) -> List[CompiledRule]:
    """Compile all rules from raw tensors.

    Args:
        rule_heads:  [R, 3] rule head atoms.
        rule_bodies: [R, M, 3] rule body atoms (padded).
        rule_lens:   [R] body lengths.
        constant_no: Highest constant index.

    Returns:
        List of CompiledRule instances.
    """
    R = rule_heads.size(0)
    return [
        CompiledRule(
            rule_idx=i,
            head=rule_heads[i],
            body=rule_bodies[i],
            body_len=int(rule_lens[i].item()),
            constant_no=constant_no,
        )
        for i in range(R)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Tensorization helpers
# ══════════════════════════════════════════════════════════════════════════════


def tensorize_rules(
    compiled_rules: List[CompiledRule],
    max_body_atoms: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert rule patterns to tensors.

    Returns:
        head_preds:     [R] head predicate indices.
        body_preds:     [R, M] body predicate indices.
        num_body_atoms: [R] body lengths.
    """
    R = max(len(compiled_rules), 1)
    M = max_body_atoms

    head_preds = torch.zeros(R, dtype=torch.long, device=device)
    body_preds = torch.zeros(R, M, dtype=torch.long, device=device)
    num_body = torch.zeros(R, dtype=torch.long, device=device)

    for i, cr in enumerate(compiled_rules):
        head_preds[i] = cr.head_pred_idx
        num_body[i] = cr.num_body
        for j, bp in enumerate(cr.body_patterns):
            body_preds[i, j] = bp["pred_idx"]

    return head_preds, body_preds, num_body


def build_vectorized_metadata(
    compiled_rules: List[CompiledRule],
    max_body_atoms: int,
    device: torch.device,
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, int,
]:
    """Build per-rule metadata for vectorized grounding.

    Returns:
        has_free:            [R] bool — any free vars
        enum_pred_a:         [R] long — first enum predicate
        enum_bound_binding_a:[R] long — first enum bound binding
        enum_direction_a:    [R] long — first enum direction
        check_arg_source_a:  [R, M, 2] long — 0=hd0, 1=hd1, 2+=fv_i
        num_free_vars:       [R] long — count of free vars per rule
        body_introduces_fv:  [R, M] long — -1 or fv_idx this atom introduces
        body_enum_bound_src: [R, M] long — source idx of bound arg for enum
        body_enum_direction: [R, M] long — 0=enumerate objects, 1=subjects
        body_enum_pred:      [R, M] long — predicate to enumerate on
        F_max:               int — max free vars across all rules
    """
    R = max(len(compiled_rules), 1)
    M = max_body_atoms

    has_free = torch.zeros(R, dtype=torch.bool, device=device)
    enum_pred_a = torch.zeros(R, dtype=torch.long, device=device)
    enum_bound_binding_a = torch.zeros(R, dtype=torch.long, device=device)
    enum_direction_a = torch.zeros(R, dtype=torch.long, device=device)
    check_arg_source_a = torch.zeros(R, M, 2, dtype=torch.long, device=device)

    num_free_vars = torch.zeros(R, dtype=torch.long, device=device)
    body_introduces_fv = torch.full((R, M), -1, dtype=torch.long, device=device)
    body_enum_bound_src = torch.zeros(R, M, dtype=torch.long, device=device)
    body_enum_direction = torch.zeros(R, M, dtype=torch.long, device=device)
    body_enum_pred = torch.zeros(R, M, dtype=torch.long, device=device)
    F_max = 0

    for i, cr in enumerate(compiled_rules):
        num_free_vars[i] = cr.num_free
        if cr.num_free > F_max:
            F_max = cr.num_free

        if cr.num_free > 0:
            has_free[i] = True

        first_enum_filled = False
        for j, meta in enumerate(cr.enum_meta):
            body_introduces_fv[i, j] = meta["introduces_fv"]
            body_enum_bound_src[i, j] = meta["enum_bound_src"]
            body_enum_direction[i, j] = meta["enum_direction"]
            body_enum_pred[i, j] = meta["enum_pred"]

            if meta["introduces_fv"] >= 0 and not first_enum_filled:
                enum_pred_a[i] = meta["enum_pred"]
                enum_bound_binding_a[i] = meta["enum_bound_src"]
                enum_direction_a[i] = meta["enum_direction"]
                first_enum_filled = True

        for j, bp in enumerate(cr.body_patterns):
            check_arg_source_a[i, j, 0] = bp["arg0_binding"]
            check_arg_source_a[i, j, 1] = bp["arg1_binding"]

    return (
        has_free,
        enum_pred_a,
        enum_bound_binding_a,
        enum_direction_a,
        check_arg_source_a,
        num_free_vars,
        body_introduces_fv,
        body_enum_bound_src,
        body_enum_direction,
        body_enum_pred,
        F_max,
    )


def build_rule_clustering(
    compiled_rules: List[CompiledRule],
    num_predicates: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, int]:
    """Build per-predicate rule mapping.

    Returns:
        pred_rule_indices: [P, R_eff] long — rule indices per predicate.
        pred_rule_mask:    [P, R_eff] bool — validity mask.
        R_eff:             int — max rules sharing a single head predicate.
    """
    P = num_predicates

    pred_to_rules: Dict[int, List[int]] = {}
    for i, cr in enumerate(compiled_rules):
        pred_to_rules.setdefault(cr.head_pred_idx, []).append(i)

    R_eff = max((len(v) for v in pred_to_rules.values()), default=1)

    pred_rule_indices = torch.zeros(P, R_eff, dtype=torch.long, device=device)
    pred_rule_mask = torch.zeros(P, R_eff, dtype=torch.bool, device=device)
    for p, rule_indices in pred_to_rules.items():
        for j, ri in enumerate(rule_indices[:R_eff]):
            pred_rule_indices[p, j] = ri
            pred_rule_mask[p, j] = True

    return pred_rule_indices, pred_rule_mask, R_eff


def check_in_provable(atom_hashes: Tensor, provable_hashes: Tensor) -> Tensor:
    """Check which atoms are in the provable set via searchsorted.

    Args:
        atom_hashes: Arbitrary-shape long tensor of atom hashes.
        provable_hashes: Sorted 1-D long tensor of provable atom hashes.

    Returns:
        Bool tensor of same shape as atom_hashes.
    """
    flat = atom_hashes.reshape(-1)
    pos = torch.searchsorted(provable_hashes, flat)
    pos = pos.clamp(max=provable_hashes.size(0) - 1)
    found = provable_hashes[pos] == flat
    return found.view(atom_hashes.shape)
