"""Cross-query Kleene T_P fixed-point filter for transitive provability.

The one soundness filter, in two layouts of the same fixed-point:
  * ``apply_fp_batch``        — padded body atoms ``[B,N,M,3]`` → proved mask
                                (terminal filter for ``CompletedTreeFirings``).
  * ``prune_rule_groundings`` — CSR pool ``RuleGroundings`` → row-dropped copy.
Ported from OLD ``filters/soundness/fp_batch.py`` + ``bc/pruning.py``. A grounding
is proved when every body atom is a base fact or the head of an already-proved
grounding across the batch.
"""
from __future__ import annotations

import torch
from torch import Tensor

from grounder.types import RuleGroundings


def apply_fp_batch(
    body: Tensor,           # [B, N, M, 3]
    mask: Tensor,           # [B, N] bool
    queries: Tensor,        # [B, 3]
    fact_index,
    pack_base: int,
    padding_idx: int,
    depth: int,
    grounding_heads: Tensor = None,  # [B, N, D, 3]
) -> Tensor:
    """Cross-query Kleene T_P fixed-point on collected groundings → proved mask."""
    B, N, M, _ = body.shape
    pb = pack_base
    dev = body.device

    body_hashes = (body[..., 0].long() * (pb * pb)
                   + body[..., 1].long() * pb
                   + body[..., 2].long())

    if grounding_heads is not None:
        D_h = grounding_heads.shape[2]
        gh = grounding_heads.long()
        all_head_hashes = (gh[..., 0] * (pb * pb) + gh[..., 1] * pb + gh[..., 2])
        head_valid = grounding_heads[..., 0] != padding_idx
        head_hashes_flat = all_head_hashes.reshape(B, N * D_h)
        head_valid_flat = head_valid.reshape(B, N * D_h)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D_h).reshape(B, N * D_h)
        head_valid_flat = head_valid_flat & mask_expanded
        N_pool = N * D_h
    else:
        q_hash = (queries[:, 0].long() * (pb * pb)
                  + queries[:, 1].long() * pb + queries[:, 2].long())
        head_hashes_flat = q_hash.unsqueeze(1).expand(-1, N)
        head_valid_flat = mask
        N_pool = N

    if grounding_heads is not None:
        M_per = M // D_h
        body_structured = body.reshape(B, N, D_h, M_per, 3)
        V = N * D_h
        vbody = body_structured.reshape(B, V, M_per, 3)
        vhead = grounding_heads.reshape(B, V, 3)
        vmask_base = mask.unsqueeze(-1).expand(-1, -1, D_h)
        vhead_active = grounding_heads[..., 0] != padding_idx
        vmask = (vmask_base & vhead_active).reshape(B, V)

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
            pos = torch.searchsorted(pool_sorted, flat_vb)
            pos = pos.clamp(max=B * V - 1)
            found = pool_sorted[pos] == flat_vb
            in_pool = found.view(B, V, M_per)
            vproved = (vis_fact | in_pool | ~vbody_active).all(dim=-1) & vmask

        vproved_per_depth = vproved.reshape(B, N, D_h)
        depth_inactive = ~vhead_active
        proved = (vproved_per_depth | depth_inactive).all(dim=-1) & mask
        return proved

    is_fact = fact_index.exists(body.reshape(-1, 3)).view(B, N, M)
    body_active = body[..., 0] != padding_idx
    proved = (is_fact | ~body_active).all(dim=-1) & mask

    sentinel = torch.tensor(-1, dtype=torch.long, device=dev)
    for _ in range(depth + 1):
        all_heads = head_hashes_flat.reshape(B * N_pool)
        all_valid = proved.reshape(B * N_pool)
        proved_pool = torch.where(all_valid, all_heads, sentinel.expand(B * N_pool))
        proved_pool_sorted, _ = proved_pool.sort()
        flat_body = body_hashes.reshape(B * N * M)
        pos = torch.searchsorted(proved_pool_sorted, flat_body)
        pos = pos.clamp(max=B * N_pool - 1)
        found = proved_pool_sorted[pos] == flat_body
        in_proved = found.view(B, N, M)
        atom_ok = is_fact | in_proved | ~body_active
        proved = atom_ok.all(dim=-1) & mask

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
