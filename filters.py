"""Derived-state filters: ground-child capping, dead-rule pruning.

Pure tensor operations with no external dependencies beyond PyTorch.
These filters are used by both the grounder and RL engines.
"""

from typing import Optional
import torch
from torch import Tensor


def cap_ground_children(fact_success: Tensor, max_ground: int) -> Tensor:
    """Cap fact-derived children to max_ground per parent. [B, K_f] -> [B, K_f]."""
    needs_cap = fact_success.sum(dim=1) > max_ground
    cumsum = fact_success.long().cumsum(dim=1)
    capped = fact_success & (cumsum <= max_ground)
    return torch.where(needs_cap.unsqueeze(1), capped, fact_success)


def prune_dead_nonground_rules(
    rule_states: Tensor,    # [B, K_r, M, 3]
    rule_success: Tensor,   # [B, K_r]
    arg0_lens: Tensor,      # [max_key0]
    arg1_lens: Tensor,      # [max_key1]
    key_scale: int,
    constant_no: int,
    pad: int,
) -> Tensor:
    """Kill rule-derived states containing dead non-ground atoms. [B, K_r] -> [B, K_r]."""
    B, K_r, M, _ = rule_states.shape
    preds = rule_states[:, :, :, 0]
    arg0s = rule_states[:, :, :, 1]
    arg1s = rule_states[:, :, :, 2]
    is_valid = preds != pad
    is_const_a0 = (arg0s <= constant_no) & (arg0s != 0)
    is_const_a1 = (arg1s <= constant_no) & (arg1s != 0)
    is_var_a0 = (arg0s > constant_no) & (arg0s != 0)
    is_var_a1 = (arg1s > constant_no) & (arg1s != 0)
    max_key0, max_key1 = arg0_lens.shape[0], arg1_lens.shape[0]

    # Branchless: compute keys for all atoms, mask with case predicates
    case_cv = is_valid & is_const_a0 & is_var_a1
    keys_cv = (preds.long() * key_scale + arg0s.long()).clamp(0, max_key0 - 1)
    dead_cv = case_cv & (arg0_lens[keys_cv] == 0)

    case_vc = is_valid & is_var_a0 & is_const_a1
    keys_vc = (preds.long() * key_scale + arg1s.long()).clamp(0, max_key1 - 1)
    dead_vc = case_vc & (arg1_lens[keys_vc] == 0)

    atom_dead = dead_cv | dead_vc
    return rule_success & ~atom_dead.any(dim=2)
