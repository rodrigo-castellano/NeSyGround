"""Cross-query Kleene T_P fixed-point filter for transitive provability.

The one soundness filter, in two layouts of the same fixed-point:
  * ``apply_fp_batch``        — padded body atoms ``[B,N,M,3]`` → proved mask
                                (terminal filter for ``CompletedTreeFirings``).
  * ``prune_rule_groundings`` — CSR pool ``RuleGroundings`` → row-dropped copy.
A grounding is proved when every body atom is a base fact or the head of an
already-proved grounding across the batch.
"""
from __future__ import annotations

import torch
from torch import Tensor

from grounder.base.types import RuleGroundings


def apply_fp_batch(
    body: Tensor,             # [B, N, M, 3]
    mask: Tensor,             # [B, N] bool
    fact_index,
    pack_base: int,
    padding_idx: int,
    depth: int,
    grounding_heads: Tensor,  # [B, N, D, 3] per-depth grounding heads
) -> Tensor:
    """Cross-query Kleene T_P fixed-point over the per-depth head pool → proved mask.

    A body atom is proved when every active atom is a base fact or the head of an
    already-proved grounding; iterated ``depth+1`` times to a fixed point."""
    B, N, M, _ = body.shape
    pb = pack_base
    dev = body.device
    D_h = grounding_heads.shape[2]
    M_per = M // D_h
    V = N * D_h

    vbody = body.reshape(B, V, M_per, 3)
    vhead = grounding_heads.reshape(B, V, 3)
    vhead_active = grounding_heads[..., 0] != padding_idx
    vmask = (mask.unsqueeze(-1).expand(-1, -1, D_h) & vhead_active).reshape(B, V)

    vbody_h = (vbody[..., 0].long() * (pb * pb)
               + vbody[..., 1].long() * pb + vbody[..., 2].long())
    vhead_h = (vhead[..., 0].long() * (pb * pb)
               + vhead[..., 1].long() * pb + vhead[..., 2].long())

    vis_fact = fact_index.exists(vbody.reshape(-1, 3)).view(B, V, M_per)
    vbody_active = vbody[..., 0] != padding_idx
    vproved = (vis_fact | ~vbody_active).all(dim=-1) & vmask

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    for _ in range(depth + 1):
        pool = torch.where(vproved, vhead_h, sentinel.expand(B, V))
        pool_sorted, _ = pool.reshape(B * V).sort()
        flat_vb = vbody_h.reshape(B * V * M_per)
        pos = torch.searchsorted(pool_sorted, flat_vb).clamp(max=B * V - 1)
        found = pool_sorted[pos] == flat_vb
        in_pool = found.view(B, V, M_per)
        vproved = (vis_fact | in_pool | ~vbody_active).all(dim=-1) & vmask

    vproved_per_depth = vproved.reshape(B, N, D_h)
    proved = (vproved_per_depth | ~vhead_active).all(dim=-1) & mask
    return proved


def prune_rule_groundings(rg, *, facts_idx: Tensor, depth: int,
                          padding_idx: int = None) -> RuleGroundings:
    """Snapshot-based ``num_steps``-iteration Kleene pruning over the CSR pool —
    drops rule applications whose body atoms aren't all proved, rebuilds offsets."""
    atom_table = rg.atom_table
    num_atoms = atom_table.size(0)
    device = atom_table.device

    if facts_idx.numel() > 0:
        fi_dev = facts_idx.to(device)
        base = int(torch.maximum(atom_table.max(), fi_dev.max()).item()) + 1
        bb = base * base
        atom_h = (atom_table[:, 0].long() * bb
                  + atom_table[:, 1].long() * base + atom_table[:, 2].long())
        fact_h = (fi_dev[:, 0].long() * bb
                  + fi_dev[:, 1].long() * base + fi_dev[:, 2].long())
        is_fact = torch.isin(atom_h, fact_h)
    else:
        is_fact = torch.zeros(num_atoms, dtype=torch.bool, device=device)

    if padding_idx is not None and num_atoms > 0:
        is_pad = atom_table[:, 0] == padding_idx
        is_fact = is_fact | is_pad

    A_in_all = rg.body_pool_idx
    A_out_all = rg.head_pool_idx
    body_atom_valid = rg.body_atom_valid
    rule_idx_all = rg.rule_idx

    if A_in_all.size(0) == 0:
        return rg

    proved = is_fact.to(torch.int8)
    for _ in range(max(1, depth)):
        body_proved = (proved[A_in_all] == 1).all(dim=-1).to(torch.int8)
        new_heads = torch.zeros_like(proved)
        new_heads.scatter_reduce_(
            0, A_out_all, body_proved, reduce="amax", include_self=False)
        proved = torch.maximum(proved, new_heads)

    keep_all = (proved[A_in_all] == 1).all(dim=-1)

    body_pool_idx_kept = A_in_all[keep_all]
    head_pool_idx_kept = A_out_all[keep_all]
    body_atom_valid_kept = body_atom_valid[keep_all]
    rule_idx_kept = rule_idx_all[keep_all].long()

    sizes = torch.bincount(rule_idx_kept, minlength=rg.num_rules)
    rule_offsets = torch.zeros(rg.num_rules + 1, dtype=torch.long, device=device)
    rule_offsets[1:] = torch.cumsum(sizes, dim=0)

    return RuleGroundings(
        atom_table=rg.atom_table,
        body_pool_idx=body_pool_idx_kept,
        body_atom_valid=body_atom_valid_kept,
        head_pool_idx=head_pool_idx_kept,
        rule_idx=rule_idx_kept,
        rule_offsets=rule_offsets,
        num_atoms=rg.num_atoms,
        num_rules=rg.num_rules,
        M_max=rg.M_max,
        query_pool_idx=rg.query_pool_idx,
    )


__all__ = ["apply_fp_batch", "prune_rule_groundings"]
