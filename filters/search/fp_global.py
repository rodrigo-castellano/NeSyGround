"""Per-step fp_global filter — reject candidates with body atoms not in I_D.

This is the per-step (search) variant of fp_global. It checks candidate
body atoms against the precomputed fp_global set during resolution, allowing
early pruning of unprovable branches.

The terminal variant (filters/soundness/fp_global.py) performs the same check
on collected groundings after the full BC loop.

Both use the same fp_global set I_D built by fc/fc.py at init.
"""

from __future__ import annotations

from torch import Tensor

from grounder.filters.soundness import check_in_fp_global


def filter_fp_global_step(
    rule_goals: Tensor,         # [B, S, K_r, G, 3]
    rule_success: Tensor,       # [B, S, K_r]
    fp_global_hashes: Tensor,   # [I_max] sorted fp_global atom hashes
    fact_index,                 # FactIndex with .exists()
    constant_no: int,
    padding_idx: int,
    M: int,
) -> Tensor:
    """Per-step fp_global filter: reject rule children with unprovable body atoms.

    Checks body atoms (first M slots of rule_goals) against facts and fp_global.
    If any body atom is ground, not a fact, and not in fp_global, reject.

    Args:
        rule_goals:       [B, S, K_r, G, 3] derived goals from rule resolution.
        rule_success:     [B, S, K_r] current success mask.
        fp_global_hashes: [I_max] sorted hashes of fp_global set.
        fact_index:       FactIndex with .exists() method.
        constant_no:      highest constant index.
        padding_idx:      padding value.
        M:                max body atoms per rule.

    Returns:
        [B, S, K_r] updated success mask with unprovable children rejected.
    """
    B, S, K_r, G, _ = rule_goals.shape
    pad = padding_idx
    E = constant_no + 1

    # Extract body atoms (first M slots)
    body = rule_goals[:, :, :, :M, :]  # [B, S, K_r, M, 3]

    body_active = body[..., 0] != pad  # [B, S, K_r, M]
    is_ground = (body[..., 1:3] < E).all(dim=-1)  # [B, S, K_r, M]
    ground_active = body_active & is_ground

    # Check facts
    flat = body.reshape(-1, 3)
    is_fact = fact_index.exists(flat).view(B, S, K_r, M)

    # Check fp_global
    body_hashes = (body[..., 0].long() * (E * E)
                   + body[..., 1].long() * E
                   + body[..., 2].long())
    in_fp_global = check_in_fp_global(body_hashes, fp_global_hashes)

    # A ground active atom is provable if it's a fact or in fp_global
    provable = is_fact | in_fp_global | ~ground_active

    # Reject children where any ground body atom is unprovable
    all_provable = provable.all(dim=-1)  # [B, S, K_r]
    return rule_success & all_provable
