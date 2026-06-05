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


def _binding_tables(grounder, num_rules: int, M: int, pad: int, device):
    """Precompute (once, cached) the variable-binding constraints per rule.

    A recorded firing ``(rule r, head Ha, body B[M])`` is a VALID rule application
    iff there is ONE substitution making the rule's head = Ha and its body = B. We
    encode that as: a slot vector ``[head_arg0, head_arg1, body0_arg0, body0_arg1,
    ...]`` over entities, where every two slots holding the SAME rule variable must
    carry the same entity. ``head`` slots are included, so this also enforces that
    the recorded head is exactly the one the body entails — catching the all_anchors
    leak where a body grounding for one head is stamped with the query goal's head.

    Returns tensors keyed by orig rule index:
      head_pred [R]      — rule head predicate
      body_pred [R, M]   — rule body predicates in canonical order (pad past len)
      slot_active [R, Nslot] bool — slot participates (head always; body atom m iff m<len)
      canon_src  [R, Nslot] long  — first active slot sharing this slot's variable
    Nslot = 2 + 2*M.
    """
    ri = grounder.kb.rule_index
    heads = ri.rules_heads.to("cpu")        # [R, 3] (pred, v0, v1)
    bodies = ri.rules_bodies.to("cpu")      # [R, M, 3]
    lens = ri.rule_lens.to("cpu")
    Nslot = 2 + 2 * M
    head_pred = torch.full((num_rules,), pad, dtype=torch.long)
    body_pred = torch.full((num_rules, M), pad, dtype=torch.long)
    slot_active = torch.zeros((num_rules, Nslot), dtype=torch.bool)
    canon_src = torch.arange(Nslot).unsqueeze(0).repeat(num_rules, 1).long()
    for r in range(num_rules):
        L = int(lens[r])
        head_pred[r] = int(heads[r, 0])
        # slot variable ids; inactive slots get a unique sentinel so they never match.
        var = [-(s + 1) for s in range(Nslot)]
        var[0] = int(heads[r, 1]); var[1] = int(heads[r, 2])
        slot_active[r, 0] = slot_active[r, 1] = True
        for m in range(M):
            body_pred[r, m] = int(bodies[r, m, 0]) if m < L else pad
            if m < L:
                var[2 + 2 * m] = int(bodies[r, m, 1])
                var[3 + 2 * m] = int(bodies[r, m, 2])
                slot_active[r, 2 + 2 * m] = slot_active[r, 3 + 2 * m] = True
        # canon_src[slot] = first active slot carrying the same variable.
        first = {}
        for s in range(Nslot):
            if not bool(slot_active[r, s]):
                continue
            v = var[s]
            if v in first:
                canon_src[r, s] = first[v]
            else:
                first[v] = s
                canon_src[r, s] = s
    return {
        "num_rules": num_rules, "M": M,
        "head_pred": head_pred.to(device), "body_pred": body_pred.to(device),
        "slot_active": slot_active.to(device), "canon_src": canon_src.to(device),
    }


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

    # COLLISION-FREE dedup of (rule, head, body) firings + atom_table.
    # The legacy ``atom_hash`` (three near-equal primes) collides distinct atoms on
    # large accumulators, which (a) merges unrelated firings and (b) corrupts the
    # atom_table so ``atom_table[head_pool_idx]`` reads back ANOTHER atom — a firing
    # recorded under one rule reports a different rule's head predicate. Both are exact
    # set operations, so do them exactly: dedup ATOMS by an injective int64 key (one
    # atom fits int64), then dedup ROWS by their (rule, atom-index) tuple via an exact
    # ``unique(dim=0)`` over a narrow (M+2)-column tensor (no overflow, no collisions).
    rule_idx_safe = torch.where(
        rule_idx < 0, torch.full_like(rule_idx, num_rules), rule_idx).long()
    all_atoms_flat = torch.cat(
        [head.long().unsqueeze(1), body.long()], dim=1).reshape(-1, 3)  # [T*(M+1), 3]
    abase = int(all_atoms_flat.max().item()) + 1 if all_atoms_flat.numel() else 1
    akey = (all_atoms_flat[:, 0] * abase + all_atoms_flat[:, 1]) * abase \
        + all_atoms_flat[:, 2]
    uniq_akey, ainv = torch.unique(akey, return_inverse=True)
    n_atoms = uniq_akey.size(0)
    rep_at = torch.zeros(n_atoms, dtype=torch.long, device=akey.device)
    rep_at.scatter_reduce_(
        0, ainv, torch.arange(akey.size(0), device=akey.device),
        reduce="amax", include_self=False)
    atom_table = all_atoms_flat[rep_at]                               # [n_atoms, 3]
    ainv = ainv.reshape(T, M + 1)                                    # [T, M+1] atom indices
    # Row = (rule, head_atom_idx, body_atom_idx[0..M-1]); dedup over small int indices.
    # Pack injectively into one int64 (base > every column) so a FAST 1D unique replaces
    # the O(T·log T) lexicographic ``unique(dim=0)`` row-sort (which blew BC13 to ~40s).
    # Indices are < n_atoms and rule < num_rules+1, so the pack fits int64 whenever
    # ``A**(M+1) * (num_rules+1) < 2**63``; otherwise fall back to the exact row-sort.
    row = torch.cat([rule_idx_safe.unsqueeze(1), ainv], dim=1)        # [T, M+2]
    A = max(int(num_rules) + 1, int(n_atoms))
    if A ** (M + 1) * (int(num_rules) + 1) < (1 << 62):
        key = rule_idx_safe.clone()
        for c in range(M + 1):
            key = key * A + ainv[:, c]
        _, inv_row = torch.unique(key, return_inverse=True)
        n_uniq = int(inv_row.max().item()) + 1 if inv_row.numel() else 0
        rep = torch.zeros(n_uniq, dtype=torch.long, device=row.device)
        rep.scatter_reduce_(0, inv_row, torch.arange(T, device=row.device),
                            reduce="amax", include_self=False)
        uniq_row = row[rep]
    else:
        uniq_row, _ = torch.unique(row, dim=0, return_inverse=True)
    u_rule = uniq_row[:, 0].long()
    head_atom_idx = uniq_row[:, 1].long()                            # [U] -> atom_table
    body_atom_idx = uniq_row[:, 2:].long()                          # [U, M] -> atom_table
    u_head = atom_table[head_atom_idx]                              # [U, 3]
    u_body = atom_table[body_atom_idx]                             # [U, M, 3]

    # Binding-consistency: keep ONLY firings that are valid rule applications.
    # The ``all_anchors`` enumeration can stamp a body grounding for one head with the
    # QUERY goal's head (``_extract_considered_rows`` gathers the head from the selected
    # goal, trusting resolution to be head-consistent — which the anchored variants are
    # not). Result: firings like ``nephew(766,887) <- aunt(8,39), brother(86,39)`` whose
    # body actually entails a DIFFERENT head. keras-ns never emits these. A firing is a
    # valid application iff the rule's head predicate + body predicates match AND every
    # rule variable binds to a single entity across all its occurrences (head slots
    # included), i.e. the recorded head is the one the body entails. This is the grounder
    # being correct by construction; there is no opt-out. Runs PRE-prune so a leaked seed
    # can't chain. Correct applications are untouched (head==entailed, vars consistent),
    # so gold-query counts / keras parity are preserved.
    dev = u_rule.device
    bt = getattr(grounder, "_binding_tables_cache", None)
    if bt is None or bt["num_rules"] != num_rules or bt["M"] != M:
        bt = _binding_tables(grounder, num_rules, M, pad, dev)
        grounder._binding_tables_cache = bt
    rule_c = u_rule.clamp(min=0, max=num_rules - 1)
    # slot entity vector [U, 2+2M]: head args then body args, canonical rule order.
    ent = torch.cat([u_head[:, 1:3], u_body[..., 1:].reshape(-1, 2 * M)], dim=1)
    # predicate match (head + each body slot; inactive body slots are pad on both sides)
    pred_ok = (u_head[:, 0] == bt["head_pred"][rule_c])
    pred_ok = pred_ok & (u_body[..., 0] == bt["body_pred"][rule_c]).all(dim=1)
    # variable consistency: every active slot must equal its canonical (first) occurrence,
    # i.e. the recorded head is exactly the one the body entails under this rule.
    cs = bt["canon_src"][rule_c]                                       # [U, Nslot]
    sa = bt["slot_active"][rule_c]                                     # [U, Nslot]
    bind_ok = ((ent == ent.gather(1, cs)) | ~sa).all(dim=1)           # [U]
    guard_keep = pred_ok & bind_ok

    # Sort firings by rule_idx so each rule's slice in the flat tensors
    # is contiguous. Drop the sentinel ``rule_idx == num_rules``: those
    # are the invalid-rule rows from the dedup pipeline. ``bincount``
    # gives per-rule sizes; cumsum gives the flat offset table.
    keep = u_rule < num_rules
    if guard_keep is not None:
        keep = keep & guard_keep
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
