"""Accumulated-body sync — gather from parents, apply subs, write at depth d.

The structured accumulator ``accumulated_body[B,G,D,M,3]`` writes each depth's
body atoms at slot ``d``. ``apply_substitutions`` is the identity when
``subs_noop`` (flat enum path) and takes ``enc`` (Encoding). ``d`` is always a
Python int on the eager fingerprint path.
"""
from __future__ import annotations

import torch
from torch import Tensor

from grounder.resolution.primitives import apply_substitutions
from grounder.backward.state import Frontier


def advance_tree_tape(plan, fr: Frontier, sync, dsel) -> Frontier:
    """Propagate accumulated_body: gather from parents, apply subs, write at depth d."""
    parent_map = sync["parent_map"]
    winning_subs = sync["winning_subs"]
    has_new_body = sync["has_new_body"]
    parent_body_count = sync["parent_body_count"]

    if not plan.collect_evidence:
        return fr.replace(body_count=parent_body_count)

    B, S_out = parent_map.shape
    D_dim = fr.accumulated_body.shape[2]
    M_acc = fr.accumulated_body.shape[3]
    M_work = fr.grounding_body.shape[2]
    pad = plan.kb.padding_idx
    enc = plan.kb.encoding
    dev = parent_map.device

    subs_noop = sync["subs_noop"]

    pi = parent_map[:, :, None, None, None].expand(-1, -1, D_dim, M_acc, 3)
    acc = fr.accumulated_body.gather(1, pi)

    rpi = parent_map[:, :, None].expand(-1, -1, D_dim)
    rule_idx = fr.rule_idx_per_depth.gather(1, rpi)
    bc = fr.body_count.gather(1, rpi)

    if subs_noop:
        subs_flat = winning_subs.reshape(B * S_out, 2, 2)
    else:
        acc_flat = acc.reshape(B * S_out, D_dim * M_acc, 3)
        subs_flat = winning_subs.reshape(B * S_out, 2, 2)
        acc_flat = apply_substitutions(acc_flat, subs_flat, enc)
        acc = acc_flat.reshape(B, S_out, D_dim, M_acc, 3)

    new_atoms = fr.grounding_body
    if M_work > M_acc:
        write_atoms = new_atoms[:, :, :M_acc, :]
    elif M_work < M_acc:
        write_atoms = torch.full(
            (B, S_out, M_acc, 3), pad, dtype=torch.long, device=dev)
        write_atoms[:, :, :M_work, :] = new_atoms
    else:
        write_atoms = new_atoms
    write_mask = has_new_body[:, :, None, None]

    cur_rule_idx = sync["current_rule_idx"]
    if plan.all_anchors:
        v2o = plan.variant_to_orig
        cur_rule_idx = torch.where(cur_rule_idx >= 0, v2o[cur_rule_idx.clamp(min=0)], cur_rule_idx)

    d = dsel.d
    acc = dsel.write_slot(acc, torch.where(write_mask, write_atoms, acc[:, :, d, :, :]))
    rule_idx = dsel.write_slot(rule_idx, torch.where(has_new_body, cur_rule_idx, rule_idx[:, :, d]))
    new_active = (write_atoms[:, :, :, 0] != pad)
    new_lens = new_active.long().sum(dim=-1)
    bc = dsel.write_slot(bc, torch.where(has_new_body, new_lens, bc[:, :, d]))

    hpi = parent_map[:, :, None, None].expand(-1, -1, D_dim, 3)
    head = fr.head_per_depth.gather(1, hpi)
    if not subs_noop:
        head_flat = head.reshape(B * S_out, D_dim, 3)
        head_flat = apply_substitutions(head_flat, subs_flat, enc)
        head = head_flat.reshape(B, S_out, D_dim, 3)
    if fr.selected_atom is not None:
        sel = fr.selected_atom
        sel_parent = sel.gather(1, parent_map.unsqueeze(-1).expand(-1, -1, 3))
        if not subs_noop:
            sel_flat = sel_parent.reshape(B * S_out, 1, 3)
            sel_flat = apply_substitutions(sel_flat, subs_flat, enc)
            sel_parent = sel_flat.reshape(B, S_out, 3)
        head = dsel.write_slot(head, torch.where(
            has_new_body.unsqueeze(-1), sel_parent, head[:, :, d, :]))

    return fr.replace(
        accumulated_body=acc, body_count=bc,
        rule_idx_per_depth=rule_idx, head_per_depth=head)


__all__ = ["advance_tree_tape"]
