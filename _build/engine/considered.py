"""Per-step "considered" rule-application accumulator (PRIMARY for RuleGroundings).

Ported from OLD ``bc/considered.py``. Captures every rule application the BFS
proposes at each step, BEFORE pack/prune drops dead candidates. Firings are
run-scoped (``RunState.firings``); ``capture_step`` appends one ``FiringSet``
emission per step (query_idx lifted by chunk offset; finalize IGNORES it so the
global concat+single finalize stays order-invariant). ``capture_step`` runs AFTER
resolve, BEFORE pack. Bodies stored in canonical rule order. ``finalize`` builds
the NEW ``RuleGroundings`` via injective-int64 atom dedup + (orig_rule, head,
body) row dedup + binding-consistency filter + CSR sort.

(Tabling / subgoal memo are default-OFF and not ported — they are not on the
fingerprint path.)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from torch import Tensor

from grounder._build.state import FiringSet
from grounder._build.types import FlatResolvedChildren, RuleGroundings


def _extract_considered_rows(plan, resolved, fr):
    """``(resolved, fr) -> valid considered rows`` (body in canonical order)."""
    pad = plan.kb.padding_idx
    M = plan.kb.M

    if isinstance(resolved, FlatResolvedChildren):
        rule_idx = resolved.flat_rule_idx
        body = resolved.flat_goals[:, :M, :]
        b_idx = resolved.flat_batch_idx
        s_idx = resolved.flat_state_idx
    else:
        ridx = resolved.sub_rule_idx                  # [B, S, K_r]
        success = resolved.rule_success               # [B, S, K_r]
        goals = resolved.rule_goals[..., :M, :]       # [B, S, K_r, M, 3]
        B, S, K_r = ridx.shape
        dev = ridx.device
        rule_idx = ridx.reshape(-1)
        body = goals.reshape(-1, M, 3)
        rule_idx = torch.where(success.reshape(-1), rule_idx,
                               torch.full_like(rule_idx, -1))
        bi = (torch.arange(B, device=dev).view(B, 1, 1)
              .expand(B, S, K_r).reshape(-1))
        si = (torch.arange(S, device=dev).view(1, S, 1)
              .expand(B, S, K_r).reshape(-1))
        b_idx = bi
        s_idx = si

    rule_idx = rule_idx.long()
    active_atom = body[..., 0] != pad
    has_body = active_atom.any(dim=-1)
    valid = has_body & (rule_idx >= 0)

    v2o = plan.variant_to_orig
    if v2o is not None:
        rule_idx = v2o[rule_idx.clamp(min=0)]

    sel = fr.selected_goal  # always set when capture_step runs
    head = sel[b_idx.long(), s_idx.long()]

    return (rule_idx[valid], head[valid], body[valid], b_idx[valid].long())


def capture_step(plan, resolved, fr, run, query_offset: int):
    """Append one ``FiringSet`` emission (rule_idx, head, body, global query_idx)."""
    r_rule, r_head, r_body, r_bidx = _extract_considered_rows(
        plan, resolved, fr)
    emission = FiringSet.from_emission(r_rule, r_head, r_body,
                                       r_bidx + query_offset)
    return replace(run, firings=run.firings.extend(emission))


def finalize(plan, firings) -> Optional[RuleGroundings]:
    """Build a ``RuleGroundings`` from the firings (no pruning). query_idx IGNORED."""
    if firings is None or not firings.rule_idx:
        return None
    rule_idx = torch.cat(firings.rule_idx, 0)
    head = torch.cat(firings.head, 0)
    body = torch.cat(firings.body, 0)
    if rule_idx.size(0) == 0:
        return None

    pad = plan.kb.padding_idx
    num_rules = plan.kb.num_rules
    T = rule_idx.size(0)
    M = body.size(1)

    rule_idx_safe = torch.where(
        rule_idx < 0, torch.full_like(rule_idx, num_rules), rule_idx).long()
    all_atoms_flat = torch.cat(
        [head.long().unsqueeze(1), body.long()], dim=1).reshape(-1, 3)
    abase = int(all_atoms_flat.max().item()) + 1  # T>=1, never empty
    akey = (all_atoms_flat[:, 0] * abase + all_atoms_flat[:, 1]) * abase \
        + all_atoms_flat[:, 2]
    uniq_akey, ainv = torch.unique(akey, return_inverse=True)
    n_atoms = uniq_akey.size(0)
    rep_at = torch.zeros(n_atoms, dtype=torch.long, device=akey.device)
    rep_at.scatter_reduce_(
        0, ainv, torch.arange(akey.size(0), device=akey.device),
        reduce="amax", include_self=False)
    atom_table = all_atoms_flat[rep_at]
    ainv = ainv.reshape(T, M + 1)
    row = torch.cat([rule_idx_safe.unsqueeze(1), ainv], dim=1)
    A = max(int(num_rules) + 1, int(n_atoms))
    if A ** (M + 1) * (int(num_rules) + 1) < (1 << 62):
        key = rule_idx_safe.clone()
        for c in range(M + 1):
            key = key * A + ainv[:, c]
        _, inv_row = torch.unique(key, return_inverse=True)
        n_uniq = int(inv_row.max().item()) + 1  # T>=1, never empty
        rep = torch.zeros(n_uniq, dtype=torch.long, device=row.device)
        rep.scatter_reduce_(0, inv_row, torch.arange(T, device=row.device),
                            reduce="amax", include_self=False)
        uniq_row = row[rep]
    else:
        uniq_row, _ = torch.unique(row, dim=0, return_inverse=True)
    u_rule = uniq_row[:, 0].long()
    head_atom_idx = uniq_row[:, 1].long()
    body_atom_idx = uniq_row[:, 2:].long()
    u_head = atom_table[head_atom_idx]
    u_body = atom_table[body_atom_idx]

    bt = plan.kb.binding_tables(M, pad)  # KB-cached (pure fn of kb)
    rule_c = u_rule.clamp(min=0, max=num_rules - 1)
    ent = torch.cat([u_head[:, 1:3], u_body[..., 1:].reshape(-1, 2 * M)], dim=1)
    pred_ok = (u_head[:, 0] == bt["head_pred"][rule_c])
    pred_ok = pred_ok & (u_body[..., 0] == bt["body_pred"][rule_c]).all(dim=1)
    cs = bt["canon_src"][rule_c]
    sa = bt["slot_active"][rule_c]
    bind_ok = ((ent == ent.gather(1, cs)) | ~sa).all(dim=1)
    guard_keep = pred_ok & bind_ok

    keep = (u_rule < num_rules) & guard_keep
    u_rule_keep = u_rule[keep]
    head_atom_keep = head_atom_idx[keep]
    body_atom_keep = body_atom_idx[keep]

    sort_idx = torch.argsort(u_rule_keep, stable=True)
    rule_idx_sorted = u_rule_keep[sort_idx]
    body_atom_sorted = body_atom_keep[sort_idx]
    head_atom_sorted = head_atom_keep[sort_idx]
    sizes = torch.bincount(rule_idx_sorted, minlength=num_rules)
    rule_offsets = torch.zeros(
        num_rules + 1, dtype=torch.long, device=rule_idx_sorted.device)
    rule_offsets[1:] = torch.cumsum(sizes, dim=0)

    body_atoms_gathered = atom_table[body_atom_sorted]
    body_atom_valid = body_atoms_gathered[..., 0] != pad

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        body_pool_idx=body_atom_sorted,
        body_atom_valid=body_atom_valid,
        head_pool_idx=head_atom_sorted,
        rule_idx=rule_idx_sorted.long(),
        rule_offsets=rule_offsets,
        num_atoms=int(atom_table.size(0)),
        num_rules=num_rules,
        M_max=int(M),
    )


__all__ = ["capture_step", "finalize"]
