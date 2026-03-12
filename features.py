"""Variable-representation feature-encoding helpers.

These functions encode variable structure as features for policy networks
in neurosymbolic reasoning systems.
"""

from __future__ import annotations
from typing import Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor


def compute_shared_slot_indices(
    states: Tensor,  # [B, K, M, 3]
    constant_no: int,
    padding_idx: int = 0,
    n_slots: int = 128,
) -> Tuple[Tensor, Tensor]:
    """Compute per-argument slot IDs for shared+slot variable embeddings."""
    args = states[:, :, :, 1:3]
    is_var = (args > constant_no) & (args != padding_idx)

    large = constant_no + 10_000
    args_for_min = torch.where(is_var, args, large)
    min_var = args_for_min.amin(dim=-1).amin(dim=-1)  # [B, K]
    has_var = is_var.any(dim=-1).any(dim=-1)  # [B, K]
    base = torch.where(has_var, min_var, constant_no + 1)

    raw_slot = args - base.unsqueeze(-1).unsqueeze(-1)
    slot = torch.where(is_var, raw_slot, torch.zeros_like(raw_slot))
    slot_lower_ok = ((~is_var) | (slot >= 0)).all()
    slot_upper_ok = ((~is_var) | (slot < n_slots)).all()
    torch._assert_async(slot_lower_ok, "Shared+slot produced negative slot ID")
    torch._assert_async(slot_upper_ok, "Shared+slot slot ID exceeded configured slot table")
    slot = slot.long()
    return slot, is_var


def build_rule_feature_encoding(
    states: Tensor,  # [B, K, M, 3]
    rule_ids: Tensor,  # [B, K]
    rule_var_count_table: Tensor,  # [R+1]
    constant_no: int,
    padding_idx: int = 0,
    n_slots: int = 128,
) -> Tensor:
    """Build variable feature tensor for rule-aware feature-based encoding.

    Returns:
        Tensor [B, K, M, 2, 5] with features:
        ``[is_var, slot_norm, arg_pos, rule_var_count_norm, slot_in_rule]``.
    """
    slot, is_var = compute_shared_slot_indices(
        states=states,
        constant_no=constant_no,
        padding_idx=padding_idx,
        n_slots=n_slots,
    )

    B, K = rule_ids.shape
    slot_f = slot.float()
    is_var_f = is_var.float()

    rv = rule_var_count_table.float().index_select(0, rule_ids.reshape(-1)).view(B, K)  # [B,K]
    rv_exp = rv.unsqueeze(-1).unsqueeze(-1)  # [B,K,1,1]
    max_rule_vars = torch.clamp(rule_var_count_table.float().max(), min=1.0)

    slot_norm = slot_f / float(max(1, n_slots - 1))
    arg_pos = torch.tensor([0.0, 1.0], device=states.device).view(1, 1, 1, 2).expand_as(slot_f)
    rv_norm = (rv_exp / max_rule_vars).expand_as(slot_f)
    slot_in_rule = (slot_f < rv_exp).float()

    return torch.stack((is_var_f, slot_norm, arg_pos, rv_norm, slot_in_rule), dim=-1)


def _count_unique_vars(rule: object) -> int:
    """Count unique uppercase-initial variables in a rule."""
    unique_vars: set = set()
    for atom in [rule.head] + list(rule.body):
        for arg in atom.args:
            if isinstance(arg, str) and arg and arg[0].isupper():
                unique_vars.add(arg)
    return len(unique_vars)


def build_rule_var_count_table(rules: Sequence[object], device: Optional[torch.device] = None) -> Tensor:
    """Create ``rule_id -> unique_var_count`` lookup table for feature encoding.

    Table includes a padding row at index 0.
    """
    counts = [0] + [_count_unique_vars(r) for r in rules]
    return torch.tensor(counts, dtype=torch.float32, device=device)


def build_predicate_var_count_table(
    rules: Sequence[object],
    predicate_str2idx: Mapping[str, int],
    n_predicates: Optional[int] = None,
    device: Optional[torch.device] = None,
    reduce: str = "max",
) -> Tensor:
    """Create ``predicate_id -> unique_var_count`` table from rule set.

    For predicates with multiple rules, ``reduce='max'`` keeps the largest
    per-rule unique variable count (default).
    """
    if n_predicates is None:
        max_idx = max(predicate_str2idx.values()) if predicate_str2idx else 0
        n_predicates = int(max_idx)
    table = torch.zeros(int(n_predicates) + 1, dtype=torch.float32, device=device)
    denom = torch.zeros_like(table) if reduce == "mean" else None

    for rule in rules:
        p_idx = predicate_str2idx.get(rule.head.predicate, None)
        if p_idx is None or p_idx < 0 or p_idx >= table.numel():
            continue

        count = float(_count_unique_vars(rule))

        if reduce == "max":
            table[p_idx] = torch.maximum(table[p_idx], torch.tensor(count, dtype=table.dtype, device=table.device))
        elif reduce == "mean":
            table[p_idx] += count
            denom[p_idx] += 1.0
        else:
            raise ValueError("reduce must be 'max' or 'mean'")

    if reduce == "mean":
        nonzero = denom > 0
        table = torch.where(nonzero, table / torch.where(nonzero, denom, torch.ones_like(denom)), table)

    return table
