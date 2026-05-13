"""Per-step "considered" rule-application accumulator.

Captures every rule application the BFS proposes at each step, BEFORE
the pack/prune phase drops dead candidates. Mirrors keras-ns
:meth:`ApproximateBackwardChainingGrounder.ground`'s ``rule2groundings``
accumulator: every fired rule application is recorded, regardless of
whether the proof tree it lives in completes within the depth budget.
``filters/soundness/fp_batch`` pruning runs at the end to drop apps
whose body atoms aren't transitively groundable.

Without this accumulator, ``rule_groundings`` is built from
``ProofEvidence`` only — which only sees firings inside completed
proof trees, undercounting by ~3× for paper-equivalent BC_{w,d,u=0}
configs (e.g. ablation_d3 BC13: 80 vs keras's 252).

Hook point: in :func:`bc.step.step`, after ``_apply_hooks`` and
before ``pack``. At that moment ``flat_resolved`` carries every
candidate child the BFS just generated, with body atoms substituted
by THIS step's ``flat_subs``. Subsequent steps' substitutions to
cross-rule shared variables are NOT applied to the captured atoms —
this is exact for single-rule programs (every var bound at the step
that introduces it) and correct-on-aggregate for multi-rule programs
(since the same firing is re-captured at later depths after its
shared vars get bound).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from grounder.types import FlatResolvedChildren, RuleGroundings


def reset_accumulator(grounder) -> None:
    """Reset per-step accumulator at the start of a BFS forward pass."""
    grounder._considered_acc_rule: List[Tensor] = []
    grounder._considered_acc_head: List[Tensor] = []
    grounder._considered_acc_body: List[Tensor] = []


def capture_step(grounder, resolved, states: Dict[str, Tensor]) -> None:
    """Append (rule_idx, head, sorted_body) for each candidate child.

    Called between :func:`resolve` and :func:`pack` in :func:`step`.
    Filters out rows with ``rule_idx < 0`` (no rule fired) or empty
    body. Sorts body atoms by per-atom hash so anchor variants of
    the same logical app share a key (= match keras-ns dedup).
    """
    pad = grounder.kb.padding_idx
    M = grounder.kb.M

    # Bring both ResolvedChildren shapes onto a flat layout.
    if isinstance(resolved, FlatResolvedChildren):
        rule_idx = resolved.flat_rule_idx
        body = resolved.flat_goals[:, :M, :]
        b_idx = resolved.flat_b_idx
        s_idx = resolved.flat_s_idx
    else:
        # Dense layout: [B, S, K_r, ...].
        ridx = resolved.sub_rule_idx                  # [B, S, K_r]
        success = resolved.rule_success               # [B, S, K_r]
        goals = resolved.rule_goals[..., :M, :]       # [B, S, K_r, M, 3]
        B, S, K_r = ridx.shape
        dev = ridx.device
        rule_idx = ridx.reshape(-1)
        body = goals.reshape(-1, M, 3)
        # Mask invalid as -1 so the filter below drops them.
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

    # Variant → orig mapping: ``all_anchors=True`` splits each rule
    # into one variant per body anchor; the dedup hash already
    # collapses them, but ``rule_idx`` would otherwise hold the
    # variant index (out of range vs ``num_rules``).
    v2o = getattr(grounder, "_variant_to_orig_t", None)
    if v2o is not None:
        rule_idx = v2o[rule_idx.clamp(min=0)]

    # Head: gather selected goal per (b, s). ``_selected_goal`` was
    # snapshotted in ``step()`` at the start of this depth.
    sel = states.get("_selected_goal")
    if sel is not None:
        head = sel[b_idx.long(), s_idx.long()]                   # [T, 3]
    else:
        head = torch.full(
            (rule_idx.size(0), 3), pad,
            dtype=torch.long, device=rule_idx.device)

    # Body atoms are stored in RULE ORDER (matches keras-ns
    # ``rule2groundings`` storage). With ``all_anchors=True``, the
    # different anchor variants are stored as distinct internal rules
    # (one per body atom) but their resolved body atoms are written
    # back into the canonical rule body order, so anchor variants of
    # the same logical app produce identical (rule_idx, head, body)
    # tuples without needing a sort. (The earlier sort-by-hash here
    # collapsed firings keras-ns kept distinct on multi-rule programs
    # like countries_s3.)
    grounder._considered_acc_rule.append(rule_idx[valid])
    grounder._considered_acc_head.append(head[valid])
    grounder._considered_acc_body.append(body[valid])


def finalize(grounder) -> Optional[RuleGroundings]:
    """Build a ``RuleGroundings`` from the accumulator.

    Returns ``None`` if no firings were captured (empty BFS).
    Does not run ``prune_rule_groundings``; the caller should apply
    that filter when ``filter_mode == 'fp_batch'``.
    """
    if not grounder._considered_acc_rule:
        return None
    rule_idx = torch.cat(grounder._considered_acc_rule, 0)
    head = torch.cat(grounder._considered_acc_head, 0)
    body = torch.cat(grounder._considered_acc_body, 0)
    if rule_idx.size(0) == 0:
        return None

    pad = grounder.kb.padding_idx
    num_rules = grounder.kb.num_rules
    T = rule_idx.size(0)
    M = body.size(1)

    # Encode (rule, head, body) as a row and dedup. Hash-based 1D
    # unique replaces ``torch.unique(combined, dim=0)`` (slow per-row
    # sort on a (1 + 3 + 3M)-wide int64 row): polynomial hash over
    # column projections + 1D ``unique``. The same collision-rarity
    # trade-off the rest of this module's ``atom_hash`` already accepts.
    from grounder.groundings import atom_hash, _HASH_P0
    rule_idx_safe = torch.where(
        rule_idx < 0, torch.full_like(rule_idx, num_rules), rule_idx)
    combined = torch.cat([
        rule_idx_safe.unsqueeze(-1),
        head.long(),
        body.long().reshape(T, M * 3),
    ], dim=-1)
    head_h = atom_hash(head)                                    # [T]
    body_h = atom_hash(body.long())                             # [T, M]
    P = _HASH_P0
    row_hash = rule_idx_safe.long() * P + head_h
    for m in range(M):
        row_hash = row_hash * P + body_h[:, m]
    uniq_row_h, inv_row = torch.unique(row_hash, return_inverse=True)
    n_uniq = uniq_row_h.size(0)
    uniq = torch.empty(
        n_uniq, combined.size(1), dtype=combined.dtype, device=combined.device)
    uniq[inv_row] = combined
    u_rule = uniq[:, 0].long()
    u_head = uniq[:, 1:4].long()
    u_body = uniq[:, 4:].reshape(-1, M, 3).long()

    # Build atom_table = unique union of head + body atoms. 1D hash
    # dedup over (p, a0, a1) triples — same speed-up vs ``unique_dim``.
    all_atoms = torch.cat([u_head.unsqueeze(1), u_body], dim=1)   # [U, M+1, 3]
    all_atoms_flat = all_atoms.reshape(-1, 3)
    atom_h = atom_hash(all_atoms_flat)                            # [U*(M+1)]
    uniq_atom_h, inverse = torch.unique(atom_h, return_inverse=True)
    n_uniq_atom = uniq_atom_h.size(0)
    atom_table = torch.empty(
        n_uniq_atom, 3, dtype=all_atoms_flat.dtype, device=all_atoms_flat.device)
    atom_table[inverse] = all_atoms_flat
    inverse = inverse.reshape(-1, M + 1)
    head_atom_idx = inverse[:, 0]
    body_atom_idx = inverse[:, 1:]

    # Bucket per rule. Skip rule_idx == num_rules (= invalid sentinel).
    A_in: Dict[int, Tensor] = {}
    A_out: Dict[int, Tensor] = {}
    for r in range(num_rules):
        mask = (u_rule == r)
        A_in[r] = body_atom_idx[mask]
        A_out[r] = head_atom_idx[mask].unsqueeze(-1)

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        A_in=A_in, A_out=A_out,
        num_atoms=int(atom_table.size(0)),
        num_rules=num_rules,
    )


__all__ = ["reset_accumulator", "capture_step", "finalize"]
