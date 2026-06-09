"""Per-step rule-application accumulator (PRIMARY for RuleGroundings).

``capture_step`` runs AFTER resolve, BEFORE pack: it appends one ``FiringSet``
emission of every rule application the BFS proposed (bodies in canonical rule
order), capturing candidates before pack/prune drop them. Firings are run-scoped;
query_idx is lifted by the chunk offset but finalize IGNORES it, so the global
concat + single finalize is order-invariant. ``finalize`` builds the
``RuleGroundings`` by: injective-int64 atom dedup → (orig_rule, head, body) row
dedup → binding-consistency filter → CSR sort.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from torch import Tensor

from grounder.backward.state import FiringSet
from grounder.base.types import FlatResolvedChildren, RuleGroundings


def _extract_considered_rows(plan, resolved, fr):
    """``(resolved, fr) -> valid considered rows`` (body in canonical order)."""
    pad = plan.kb.padding_idx
    M = plan.kb.M

    if isinstance(resolved, FlatResolvedChildren):
        rule_idx = resolved.flat_rule_idx
        body = resolved.flat_child_goals[:, :M, :]
        b_idx = resolved.flat_batch_idx
        s_idx = resolved.flat_state_idx
    else:
        rule_idx = resolved.sub_rule_idx                  # [B, G, K_r]
        success = resolved.rule_success               # [B, G, K_r]
        goals = resolved.rule_child_goals[..., :M, :]  # [B, G, K_r, M, 3]
        B, S, K_r = rule_idx.shape
        dev = rule_idx.device
        rule_idx = rule_idx.reshape(-1)
        body = goals.reshape(-1, M, 3)
        rule_idx = torch.where(success.reshape(-1), rule_idx,
                               torch.full_like(rule_idx, -1))
        b_idx = (torch.arange(B, device=dev).view(B, 1, 1)
                 .expand(B, S, K_r).reshape(-1))
        s_idx = (torch.arange(S, device=dev).view(1, S, 1)
                 .expand(B, S, K_r).reshape(-1))

    rule_idx = rule_idx.long()
    active_atom = body[..., 0] != pad
    has_body = active_atom.any(dim=-1)
    valid = has_body & (rule_idx >= 0)

    v2o = plan.variant_to_orig
    if v2o is not None:
        rule_idx = v2o[rule_idx.clamp(min=0)]

    sel = fr.selected_atom  # always set when capture_step runs
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


def _atom_hash(atoms: Tensor) -> Tensor:
    """Injective int64 key for atom triples ``(p, a0, a1)`` (..., 3) -> (...)."""
    a = atoms.long()
    base = int(a.max().item()) + 1 if a.numel() else 1
    return (a[..., 0] * base + a[..., 1]) * base + a[..., 2]


def populate_query_pool_idx(
    rg: RuleGroundings, queries: Tensor, padding_idx: int,
) -> RuleGroundings:
    """Extend ``rg.atom_table`` to cover every query atom and set ``query_pool_idx``.

    Pool-iter reasoners (SBR/DCR/R2N via tkk ``_rule_loop``) gather
    ``pool[query_pool_idx]`` regardless of provability, so each query
    ``(pred, h, t)`` needs a slot in the atom table even when no firing
    produced it as a head. Atoms already present keep their slot; novel
    queries are appended (the 0..N-1 prefix stays stable so existing
    body/head pool indices remain valid). ``query_pool_idx`` is ``[B]``."""
    queries = queries.long()
    device = queries.device
    pool = rg.atom_table.to(device)

    # Hash over the union so query and pool keys share one base — atom_hash's
    # per-call max base would diverge between the two tensors otherwise.
    both = torch.cat([pool, queries], dim=0) if pool.numel() else queries
    base = int(both.max().item()) + 1 if both.numel() else 1

    def _h(a: Tensor) -> Tensor:
        a = a.long()
        return (a[..., 0] * base + a[..., 1]) * base + a[..., 2]

    pool_h = _h(pool)
    query_h = _h(queries)
    # Membership via searchsorted on sorted pool hashes (perfect hash → exact),
    # not an O(Q*P) [Q,P] pairwise that OOMs at exhaustive-eval scale.
    if pool.numel():
        pool_sorted, _ = torch.sort(pool_h)
        ins = torch.searchsorted(pool_sorted, query_h).clamp(max=pool_sorted.numel() - 1)
        in_pool = pool_sorted[ins] == query_h
    else:
        in_pool = torch.zeros_like(query_h, dtype=torch.bool)

    novel = queries[~in_pool]
    novel_h = _h(novel)
    nuniq_h, ninv = torch.unique(novel_h, return_inverse=True)
    rep = torch.zeros(nuniq_h.shape[0], dtype=torch.long, device=device)
    rep.scatter_reduce_(0, ninv, torch.arange(novel.size(0), device=device),
                        reduce="amax", include_self=False)
    new_pool = torch.cat([pool, novel[rep]], dim=0)

    new_h = _h(new_pool)
    sort_idx = new_h.argsort()
    pos = torch.searchsorted(new_h[sort_idx], query_h)
    query_pool_idx = sort_idx[pos]

    return replace(rg, atom_table=new_pool.contiguous(),
                   num_atoms=int(new_pool.shape[0]),
                   query_pool_idx=query_pool_idx)


def next_pow2(n: int) -> int:
    """Smallest power of 2 >= n. Returns 1 for n <= 1."""
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def pad_rule_groundings(
    rg: RuleGroundings, *,
    pad_per_rule_to: Optional[int] = None,
    pad_atom_table_to: Optional[int] = None,
    pad_idx_for_atoms: int = 0,
) -> RuleGroundings:
    """Pad each rule's firing slice to ``pad_per_rule_to`` rows and ``atom_table``
    to ``pad_atom_table_to`` rows (``run_bc(pad_outputs=True)``).

    Re-added in the redesign (REDESIGN.md says pad_outputs was "eliminated") because
    torch-ns's countries_s3 BC12/BC13 baseline cells need fixed-shape output.

    Gives the downstream compiled reasoner a fixed-shape input on cells whose
    flat-path output would otherwise oscillate per batch (countries_s3+BC13,
    family+BC{12,13}) and blow the reduce-overhead CUDA-graph pool. Padding rows
    point at the sentinel pool slot ``pad_idx_for_atoms`` and carry
    ``firing_valid=False`` so the rule loop masks them out. No-op when both pads
    are ``None``."""
    if pad_per_rule_to is None and pad_atom_table_to is None:
        return rg

    atom_table = rg.atom_table
    device = atom_table.device
    M_max = rg.M_max
    num_rules = rg.num_rules

    if pad_per_rule_to is not None:
        G = pad_per_rule_to
        rule_offsets_in = rg.rule_offsets                              # [num_rules+1]
        rule_sizes = (rule_offsets_in[1:] - rule_offsets_in[:-1]).long()
        K_clamped = rule_sizes.clamp(max=G)
        new_N = num_rules * G
        if new_N == 0:
            body_pool_idx = torch.empty((0, M_max), dtype=rg.body_pool_idx.dtype, device=device)
            body_atom_valid = torch.empty((0, M_max), dtype=torch.bool, device=device)
            head_pool_idx = torch.empty((0,), dtype=rg.head_pool_idx.dtype, device=device)
            firing_valid = torch.empty((0,), dtype=torch.bool, device=device)
        else:
            rule_idx_flat = torch.arange(num_rules, device=device, dtype=torch.long).repeat_interleave(G)
            local_idx = torch.arange(new_N, device=device, dtype=torch.long) % G
            K_per_slot = K_clamped.repeat_interleave(G)
            valid = local_idx < K_per_slot
            src_idx_raw = rule_offsets_in[rule_idx_flat] + local_idx
            src_size = rg.body_pool_idx.size(0)
            pad_atom_id = torch.tensor(pad_idx_for_atoms, dtype=rg.body_pool_idx.dtype, device=device)
            if src_size > 0:
                src_idx_safe = src_idx_raw.clamp(max=src_size - 1)
                gathered_body = rg.body_pool_idx[src_idx_safe]
                gathered_body_valid = rg.body_atom_valid[src_idx_safe]
                gathered_head = rg.head_pool_idx[src_idx_safe]
                gathered_firing_valid = rg.firing_valid[src_idx_safe]
            else:
                gathered_body = torch.empty((new_N, M_max), dtype=rg.body_pool_idx.dtype, device=device).fill_(pad_atom_id)
                gathered_body_valid = torch.zeros((new_N, M_max), dtype=torch.bool, device=device)
                gathered_head = torch.empty((new_N,), dtype=rg.head_pool_idx.dtype, device=device).fill_(pad_atom_id)
                gathered_firing_valid = torch.zeros((new_N,), dtype=torch.bool, device=device)
            valid_2d = valid.unsqueeze(-1)
            body_pool_idx = torch.where(valid_2d, gathered_body, pad_atom_id)
            body_atom_valid = gathered_body_valid & valid_2d
            head_pool_idx = torch.where(valid, gathered_head, pad_atom_id)
            firing_valid = gathered_firing_valid & valid
        rule_idx = torch.repeat_interleave(torch.arange(num_rules, device=device, dtype=torch.long), G)
        rule_offsets = torch.arange(num_rules + 1, device=device, dtype=torch.long) * G
    else:
        body_pool_idx = rg.body_pool_idx
        body_atom_valid = rg.body_atom_valid
        head_pool_idx = rg.head_pool_idx
        rule_idx = rg.rule_idx
        rule_offsets = rg.rule_offsets
        firing_valid = rg.firing_valid

    if pad_atom_table_to is not None and pad_atom_table_to > atom_table.size(0):
        extra = pad_atom_table_to - atom_table.size(0)
        pad_rows = torch.zeros(extra, 3, dtype=atom_table.dtype, device=device)
        atom_table = torch.cat([atom_table, pad_rows], dim=0)

    return RuleGroundings(
        atom_table=atom_table.contiguous(),
        body_pool_idx=body_pool_idx, body_atom_valid=body_atom_valid,
        head_pool_idx=head_pool_idx, rule_idx=rule_idx, rule_offsets=rule_offsets,
        firing_valid=firing_valid, num_atoms=int(atom_table.size(0)),
        num_rules=num_rules, M_max=M_max, query_pool_idx=rg.query_pool_idx)


__all__ = ["capture_step", "finalize", "populate_query_pool_idx",
           "pad_rule_groundings", "next_pow2"]
