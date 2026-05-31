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
    # Global query index per captured firing — used by the per-query
    # tabling cache (task #47) to partition the accumulator rows by the
    # query that produced them. ``b_idx`` is batch-local; adding
    # ``grounder._chunk_query_offset`` makes it a global query index that
    # is stable across chunks. Always maintained (cheap) so the cache can
    # be toggled on without re-running a forward pass.
    grounder._considered_acc_bidx: List[Tensor] = []


def _extract_considered_rows(grounder, resolved, states: Dict[str, Tensor]):
    """Pure extraction: ``(resolved, states) -> valid considered rows``.

    Returns ``(rule_idx[V], head[V,3], body[V,M,3], b_idx[V])`` restricted
    to the VALID candidate rows (a rule fired with a non-empty body), with
    ``rule_idx`` already mapped through ``_variant_to_orig`` and ``head``
    set to the selected-goal atom of each row's ``(b, s)``. ``b_idx`` is
    the batch-local query index (NOT yet lifted by
    ``_chunk_query_offset``).

    This is the shared core of the considered accumulator: :func:`capture_step`
    appends these rows directly, and the per-subgoal memo (task #48) keys /
    replays them. Body atoms are in canonical RULE ORDER.
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
    return (rule_idx[valid], head[valid], body[valid],
            b_idx[valid].long())


def capture_step(grounder, resolved, states: Dict[str, Tensor],
                 d: Optional[int] = None) -> None:
    """Append (rule_idx, head, body) for each valid candidate child.

    Called between :func:`resolve` and :func:`pack` in :func:`step`.
    Filters out rows with ``rule_idx < 0`` (no rule fired) or empty body.

    When the per-subgoal memo (task #48) is active, the valid rows are
    routed through :mod:`grounder.bc.subgoal` (memoize MISS-goal rows,
    replay HIT-goal rows) before being appended; otherwise they are
    appended directly. ``d`` is the current depth (needed to compute the
    ``is_last`` component of the subgoal memo key).
    """
    r_rule, r_head, r_body, r_bidx = _extract_considered_rows(
        grounder, resolved, states)

    from grounder.bc import subgoal
    if subgoal.subgoal_active(grounder):
        subgoal.route_rows(grounder, r_rule, r_head, r_body, r_bidx, d)
        return

    grounder._considered_acc_rule.append(r_rule)
    grounder._considered_acc_head.append(r_head)
    grounder._considered_acc_body.append(r_body)
    # Global query index per firing (for the per-query tabling cache).
    # ``b_idx`` is batch-local; ``_chunk_query_offset`` lifts it to a
    # global index stable across chunks.
    grounder._considered_acc_bidx.append(
        r_bidx + grounder._chunk_query_offset)


def finalize(grounder) -> Optional[RuleGroundings]:
    """Build a ``RuleGroundings`` from the accumulator.

    Returns ``None`` if no firings were captured (empty BFS).
    Does not run ``prune_rule_groundings``; the caller should apply
    that filter when ``filter_mode == 'fp_batch'``.

    Fixed-shape padding for compile-friendly downstream consumers is
    a separate post-pruning step — see
    :func:`grounder.groundings.pad_rule_groundings`.
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
    # Deterministic representative per hash group. ``uniq[inv_row] = combined``
    # is a last-write-wins scatter: single-threaded the highest-index row per
    # group wins, but the CPU multi-thread scatter races, so the surviving
    # row (hence the grounding set) varied run-to-run once depth>=3 produced
    # collisions. ``scatter_reduce(amax)`` over the row index picks the same
    # highest-index representative deterministically, matching single-thread.
    T_rows = combined.size(0)
    rep_idx = torch.zeros(n_uniq, dtype=torch.long, device=combined.device)
    rep_idx.scatter_reduce_(
        0, inv_row, torch.arange(T_rows, device=combined.device),
        reduce="amax", include_self=False)
    uniq = combined[rep_idx]
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
    # Same deterministic-representative fix as the row dedup above:
    # ``atom_table[inverse] = all_atoms_flat`` races across CPU threads.
    n_at = all_atoms_flat.size(0)
    rep_at = torch.zeros(n_uniq_atom, dtype=torch.long, device=all_atoms_flat.device)
    rep_at.scatter_reduce_(
        0, inverse, torch.arange(n_at, device=all_atoms_flat.device),
        reduce="amax", include_self=False)
    atom_table = all_atoms_flat[rep_at]
    inverse = inverse.reshape(-1, M + 1)
    head_atom_idx = inverse[:, 0]
    body_atom_idx = inverse[:, 1:]

    # Sort firings by rule_idx so each rule's slice in the flat tensors
    # is contiguous. Drop the sentinel ``rule_idx == num_rules``: those
    # are the invalid-rule rows from the dedup pipeline. ``bincount``
    # gives per-rule sizes; cumsum gives the flat offset table.
    keep = u_rule < num_rules
    u_rule_keep = u_rule[keep]
    head_atom_keep = head_atom_idx[keep]
    body_atom_keep = body_atom_idx[keep]

    sort_idx = torch.argsort(u_rule_keep, stable=True)
    rule_idx_sorted = u_rule_keep[sort_idx]
    body_atom_sorted = body_atom_keep[sort_idx]                 # [N, M]
    head_atom_sorted = head_atom_keep[sort_idx]                 # [N]
    sizes = torch.bincount(rule_idx_sorted, minlength=num_rules)
    rule_offsets = torch.zeros(
        num_rules + 1, dtype=torch.long,
        device=rule_idx_sorted.device)
    rule_offsets[1:] = torch.cumsum(sizes, dim=0)

    # Body-atom validity: a body slot is invalid when it equals the
    # padding sentinel in the atom_table. For consistency with
    # ``evidence_to_rule_groundings`` we recompute from the body
    # predicate (column 0 of the gathered atom row).
    body_atoms_gathered = atom_table[body_atom_sorted]           # [N, M, 3]
    body_atom_valid = body_atoms_gathered[..., 0] != pad        # [N, M]
    firing_valid = torch.ones(
        rule_idx_sorted.size(0), dtype=torch.bool,
        device=rule_idx_sorted.device)

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        body_pool_idx=body_atom_sorted,
        body_atom_valid=body_atom_valid,
        head_pool_idx=head_atom_sorted,
        rule_idx=rule_idx_sorted.long(),
        rule_offsets=rule_offsets,
        firing_valid=firing_valid,
        num_atoms=int(atom_table.size(0)),
        num_rules=num_rules,
        M_max=int(M),
    )


__all__ = ["reset_accumulator", "capture_step", "finalize"]
