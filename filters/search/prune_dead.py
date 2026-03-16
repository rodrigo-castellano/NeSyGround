"""Dead-atom pruning — kill states with provably dead body atoms.

A body atom is dead if:
  - It has no matching facts (checked via fact_index or CSR segment lengths)
  - Its predicate is not a rule head (cannot be derived by any rule)

Sound heuristic: never kills valid proofs (dead atoms cannot contribute).
Only meaningful for SLD/RTF (enum atoms are always ground).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def filter_prune_dead(
    rule_goals: Tensor,      # [B, S, K_r, G, 3]
    rule_success: Tensor,    # [B, S, K_r]
    *,
    head_pred_mask: Tensor,  # [P] bool — True if predicate is a rule head
    fact_index,              # FactIndex with .exists()
    constant_no: int,
    padding_idx: int,
    M: int,
    a0_lens: Optional[Tensor] = None,  # CSR segment lengths for (pred, arg0) keys
    a1_lens: Optional[Tensor] = None,  # CSR segment lengths for (pred, arg1) keys
    p_lens: Optional[Tensor] = None,   # CSR segment lengths for pred-only keys
    key_scale: int = 0,
) -> Tensor:
    """Kill rule-derived states containing dead body atoms.

    Args:
        rule_goals:     [B, S, K_r, G, 3] resolved rule-derived goals.
        rule_success:   [B, S, K_r] validity mask for rule children.
        head_pred_mask: [P] bool — True if predicate appears as a rule head.
        fact_index:     FactIndex with .exists() method.
        constant_no:    max constant index (constants are <= constant_no).
        padding_idx:    padding value.
        M:              number of body atom slots.
        a0_lens:        CSR segment lengths for (pred, arg0) keys (optional).
        a1_lens:        CSR segment lengths for (pred, arg1) keys (optional).
        p_lens:         CSR segment lengths for pred-only keys (optional).
        key_scale:      multiplier for CSR key computation.

    Returns:
        [B, S, K_r] filtered rule_success mask.
    """
    P = head_pred_mask.shape[0]
    body = rule_goals[:, :, :, :M, :]                  # [B, S, K_r, M, 3]
    preds = body[..., 0]                                # [B, S, K_r, M]
    arg0s = body[..., 1]                                # [B, S, K_r, M]
    arg1s = body[..., 2]                                # [B, S, K_r, M]

    is_active = preds != padding_idx                    # [B, S, K_r, M]

    # Derivable: predicate is a rule head
    derivable = head_pred_mask[preds.clamp(0, P - 1)]  # [B, S, K_r, M]

    # --- Ground atoms ---
    is_const_a0 = arg0s <= constant_no                  # [B, S, K_r, M]
    is_const_a1 = arg1s <= constant_no                  # [B, S, K_r, M]
    is_ground = is_const_a0 & is_const_a1 & is_active   # [B, S, K_r, M]

    flat = body.reshape(-1, 3)
    is_fact_flat = fact_index.exists(flat)
    is_fact = is_fact_flat.view(body.shape[:-1]) & is_ground  # [B, S, K_r, M]

    ground_dead = is_ground & ~is_fact & ~derivable     # [B, S, K_r, M]

    # --- Non-ground atoms (CSR checks) ---
    is_var_a0 = (arg0s > constant_no) & is_active
    is_var_a1 = (arg1s > constant_no) & is_active

    nonground_dead = torch.zeros_like(is_active)        # [B, S, K_r, M]

    if a0_lens is not None:
        # const-var: arg0 constant, arg1 variable
        case_cv = is_const_a0 & is_var_a1 & is_active
        max_key0 = a0_lens.shape[0]
        keys_cv = (preds.long() * key_scale + arg0s.long()).clamp(0, max_key0 - 1)
        dead_cv = case_cv & (a0_lens[keys_cv] == 0) & ~derivable
        nonground_dead = nonground_dead | dead_cv

    if a1_lens is not None:
        # var-const: arg0 variable, arg1 constant
        case_vc = is_var_a0 & is_const_a1 & is_active
        max_key1 = a1_lens.shape[0]
        keys_vc = (preds.long() * key_scale + arg1s.long()).clamp(0, max_key1 - 1)
        dead_vc = case_vc & (a1_lens[keys_vc] == 0) & ~derivable
        nonground_dead = nonground_dead | dead_vc

    if p_lens is not None:
        # var-var: both args variable
        case_vv = is_var_a0 & is_var_a1 & is_active
        dead_vv = case_vv & (p_lens[preds.clamp(0, p_lens.shape[0] - 1)] == 0) & ~derivable
        nonground_dead = nonground_dead | dead_vv

    # Any dead body atom kills the state
    atom_dead = ground_dead | nonground_dead            # [B, S, K_r, M]
    return rule_success & ~atom_dead.any(dim=-1)
