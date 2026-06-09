"""PBC candidate generation: enumerate free-var bindings + fill body atoms.

No MGU — bindings are pre-compiled (PbcRuleIndex). Two materializations:
  ``*_dense`` — padded [B,K_r,Y_r,M,3], static shape for compile/CUDA-graph
  ``*_flat``  — compact [T,...] via nonzero, eager zero-waste
``_gather_body_atoms`` / ``_cartesian_expand_one_fv`` are the shared inner ops.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor


# ── stage 1: cluster (query predicate → candidate rules + per-rule view) ──

class ClusteredRules(NamedTuple):
    active_idx: Tensor      # [N, K_r] candidate rule rows per query
    active_mask: Tensor     # [N, K_r] rule exists AND query active
    K_r: int
    has_free_q: Tensor      # [N, K_r]


def cluster(query_preds: Tensor, query_valid: Tensor, tables) -> ClusteredRules:
    """Map each query predicate to its candidate rules (shared dense/flat)."""
    active_idx = tables.pred_rule_indices[query_preds]               # [N, K_r]
    active_mask = tables.pred_rule_mask[query_preds] & query_valid.unsqueeze(1)
    return ClusteredRules(active_idx, active_mask, active_idx.size(1),
                          tables.has_free[active_idx])


# Cached read-only arange (eager); fresh under compile (CUDA-graph aliasing).
_ARANGE_CACHE: Dict[Tuple[int, str], Tensor] = {}


def arange_cached(n: int, device) -> Tensor:
    """Cached read-only ``arange(n)`` (eager); fresh under torch.compile."""
    if torch.compiler.is_compiling():
        return torch.arange(n, device=device)
    key = (int(n), str(device))
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device)
        _ARANGE_CACHE[key] = t
    return t


def cumcount_flat(keys: Tensor) -> Tensor:
    """0-based position within each group of equal keys (sort + cummax).

    [A,A,B,A,B,C] → [0,1,0,2,1,0]. Used by the flat pack for per-batch slots.
    """
    T = keys.size(0)
    if T == 0:
        return keys.new_empty(0, dtype=torch.long)
    dev = keys.device
    sort_perm = torch.argsort(keys, stable=True)
    sorted_keys = keys[sort_perm]
    running_idx = torch.arange(T, device=dev)
    ne = sorted_keys[1:] != sorted_keys[:-1]
    group_change = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), ne], dim=0)
    group_starts = (running_idx * group_change).cummax(0).values
    result = torch.zeros(T, dtype=torch.long, device=dev)
    result[sort_perm] = running_idx - group_starts
    return result


# ── body-atom fill ──

def _gather_body_atoms(source_m: Tensor, check_arg_m: Tensor,
                       body_preds_m: Tensor) -> Tensor:
    """Shared gather (dense+flat): ``source_m [...,M,W]``, ``check_arg_m [...,M,2]``
    (indexes W, clamped), ``body_preds_m [...,M]`` → ``[...,M,3]`` (pred,a0,a1)."""
    W = source_m.size(-1)
    arg0 = source_m.gather(-1, check_arg_m[..., 0].clamp(max=W - 1).unsqueeze(-1)).squeeze(-1)
    arg1 = source_m.gather(-1, check_arg_m[..., 1].clamp(max=W - 1).unsqueeze(-1)).squeeze(-1)
    return torch.stack([body_preds_m, arg0, arg1], dim=-1)


def fill_body_dense(source: Tensor, check_arg_source_q: Tensor,
                    body_preds_q: Tensor) -> Tensor:
    """Dense fill: ``source [B,K_r,Y_r,W]`` → ``[B,K_r,Y_r,M,3]``."""
    Y_r = source.size(2)
    M = body_preds_q.size(2)
    source_m = source.unsqueeze(3).expand(-1, -1, -1, M, -1)
    check_m = check_arg_source_q.unsqueeze(2).expand(-1, -1, Y_r, -1, -1)
    preds_m = body_preds_q.unsqueeze(2).expand(-1, -1, Y_r, -1)
    return _gather_body_atoms(source_m, check_m, preds_m)


def fill_body_flat(flat_source: Tensor, check_arg_source_flat: Tensor,
                   body_preds_flat: Tensor) -> Tensor:
    """Flat fill: ``flat_source [T,W]`` → ``[T,M,3]``."""
    M = body_preds_flat.size(1)
    source_m = flat_source.unsqueeze(1).expand(-1, M, -1)
    return _gather_body_atoms(source_m, check_arg_source_flat, body_preds_flat)


# ── enumeration ──

def enumerate_single_dense(B: int, K_r: int, Y_r: int, query_subjs: Tensor,
                    query_objs: Tensor, enum_pred_q: Tensor, enum_bound_q: Tensor,
                    enum_dir_q: Tensor, fact_index, cartesian_product: bool = False,
                    E: int = 0) -> Tuple[Tensor, Tensor]:
    """Dense enumerate for ≤1 free var (one bound→free fact-index lookup).

    ``cartesian_product`` → all E entities. Returns
    ``(candidates[B,K_r,G_actual], cand_mask[B,K_r,G_actual])``.
    """
    if cartesian_product:
        dev = query_subjs.device
        candidates = torch.arange(E, device=dev).view(1, 1, E).expand(B * K_r, 1, -1).reshape(B, K_r, E)
        return candidates, torch.ones(B, K_r, E, dtype=torch.bool, device=dev)

    source = torch.stack([query_subjs, query_objs], dim=1)   # [B, 2]
    enum_bound_vals = source.gather(1, enum_bound_q)          # [B, K_r]
    candidates, cand_mask = fact_index.enumerate(
        enum_pred_q.reshape(-1), enum_bound_vals.reshape(-1), enum_dir_q.reshape(-1))
    G_actual = min(Y_r, candidates.size(1))
    return (candidates[:, :G_actual].reshape(B, K_r, G_actual),
            cand_mask[:, :G_actual].reshape(B, K_r, G_actual))


def _cartesian_expand_one_fv(B, K_r, query_subjs, query_objs, fact_index,
                             ep, eb, ed, ev, all_cands, all_masks, G_current, k_cap):
    """Shared inner step: enumerate one free var + interleaved Cartesian-expand.

    ``k_cap`` caps K_use to K_v (dense) or None (flat, full K_f). Interleaved
    layout lets a later topk pick diverse fv0 candidates before repeats.
    """
    qs_exp = query_subjs.view(B, 1, 1).expand(B, K_r, G_current)
    qo_exp = query_objs.view(B, 1, 1).expand(B, K_r, G_current)
    src = torch.stack([qs_exp, qo_exp] + all_cands, dim=3)    # [B,K_r,G_current,2+]
    W_cur = src.size(3)
    eb_idx = eb.clamp(max=W_cur - 1).view(B, K_r, 1, 1).expand(B, K_r, G_current, 1)
    bound_vals = src.gather(3, eb_idx).squeeze(3)             # [B,K_r,G_current]

    flat_pred = ep.view(B, K_r, 1).expand(B, K_r, G_current).reshape(-1)
    flat_bound = bound_vals.reshape(-1)
    flat_dir = ed.view(B, K_r, 1).expand(B, K_r, G_current).reshape(-1)
    new_cands, new_mask = fact_index.enumerate(flat_pred, flat_bound, flat_dir)
    K_fi = new_cands.size(1)
    K_use = K_fi if k_cap is None else min(K_fi, k_cap)
    new_cands = new_cands[:, :K_use].reshape(B, K_r, G_current, K_use)
    new_mask = (new_mask[:, :K_use].reshape(B, K_r, G_current, K_use) & ev.view(B, K_r, 1, 1))

    G_new = G_current * K_use
    expanded_cands = [prev.unsqueeze(3).expand(B, K_r, G_current, K_use)
                          .transpose(2, 3).reshape(B, K_r, G_new) for prev in all_cands]
    expanded_masks = [prev.unsqueeze(3).expand(B, K_r, G_current, K_use)
                          .transpose(2, 3).reshape(B, K_r, G_new) for prev in all_masks]
    expanded_cands.append(new_cands.transpose(2, 3).reshape(B, K_r, G_new))
    expanded_masks.append(new_mask.transpose(2, 3).reshape(B, K_r, G_new))
    return expanded_cands, expanded_masks, G_new


def enumerate_cartesian_dense(B, K_r, query_subjs, query_objs, fv_pred_q, fv_bound_q,
                              fv_dir_q, fv_valid_q, has_free_q, active_mask, fact_index,
                              K_v, V, G_cap=0, fv_any_valid=None,
                              check_arg_source_q=None, body_preds_q=None,
                              num_body_q=None, M=0) -> Tuple[Tensor, Tensor, int]:
    """Dense Cartesian enumerate of ≥2 free vars; topk-caps G to G_cap each step
    (keeps a static shape). Returns ``(source[B,K_r,G,2+V], mask[B,K_r,G], G)``."""
    dev = query_subjs.device
    if G_cap <= 0:
        G_cap = K_v
    all_cands, all_masks, G_current = [], [], 1

    for fv_idx in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv_idx]:
            all_cands.append(torch.zeros(B, K_r, G_current, dtype=torch.long, device=dev))
            all_masks.append(torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev))
            continue
        all_cands, all_masks, G_current = _cartesian_expand_one_fv(
            B, K_r, query_subjs, query_objs, fact_index,
            fv_pred_q[:, :, fv_idx], fv_bound_q[:, :, fv_idx], fv_dir_q[:, :, fv_idx],
            fv_valid_q[:, :, fv_idx], all_cands, all_masks, G_current, K_v)
        if G_current > G_cap:
            combined = torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev)
            for fi in range(len(all_masks)):
                combined = combined & (all_masks[fi] | ~fv_valid_q[:, :, fi].unsqueeze(2))
            _, top_idx = combined.to(torch.int8).topk(G_cap, dim=2, largest=True, sorted=False)
            all_cands = [c.gather(2, top_idx) for c in all_cands]
            all_masks = [m.to(torch.long).gather(2, top_idx).bool() for m in all_masks]
            G_current = G_cap

    G_final = G_current
    combined_mask = torch.ones(B, K_r, G_final, dtype=torch.bool, device=dev)
    for fv_idx in range(V):
        combined_mask = combined_mask & (all_masks[fv_idx] | ~fv_valid_q[:, :, fv_idx].unsqueeze(2))
    combined_mask = combined_mask & has_free_q.unsqueeze(2)
    combined_mask[:, :, 0] = combined_mask[:, :, 0] | (~has_free_q & active_mask)

    qs_final = query_subjs.view(B, 1, 1).expand(B, K_r, G_final)
    qo_final = query_objs.view(B, 1, 1).expand(B, K_r, G_final)
    source = torch.stack([qs_final, qo_final] + all_cands, dim=3)
    return source, combined_mask, G_final


def enumerate_cartesian_flat(B, K_r, query_subjs, query_objs, fv_pred_q, fv_bound_q,
                             fv_dir_q, fv_valid_q, has_free_q, active_mask, fact_index,
                             V, fv_any_valid=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Flat Cartesian enumerate (uncapped, full K_f) → compact rows via nonzero.

    Returns ``(flat_source[T,2+V], b_idx[T], r_idx[T])`` for surviving cells.
    """
    dev = query_subjs.device
    all_cands, all_masks, G_current = [], [], 1
    for fv_idx in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv_idx]:
            all_cands.append(torch.zeros(B, K_r, G_current, dtype=torch.long, device=dev))
            all_masks.append(torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev))
            continue
        all_cands, all_masks, G_current = _cartesian_expand_one_fv(
            B, K_r, query_subjs, query_objs, fact_index,
            fv_pred_q[:, :, fv_idx], fv_bound_q[:, :, fv_idx], fv_dir_q[:, :, fv_idx],
            fv_valid_q[:, :, fv_idx], all_cands, all_masks, G_current, None)

    combined_mask: Optional[Tensor] = None
    for fv_idx in range(V):
        term = all_masks[fv_idx] | ~fv_valid_q[:, :, fv_idx].unsqueeze(2)
        combined_mask = term if combined_mask is None else combined_mask & term
    if combined_mask is None:
        combined_mask = torch.ones(B, K_r, G_current, dtype=torch.bool, device=dev)
    combined_mask = combined_mask & has_free_q.unsqueeze(2)
    combined_mask[:, :, 0] = combined_mask[:, :, 0] | (~has_free_q & active_mask)
    # Padded K_r slots must yield no candidates, else rule_idx-0 leaks spurious apps.
    combined_mask = combined_mask & active_mask.unsqueeze(2)

    valid_idx = torch.nonzero(combined_mask, as_tuple=False)  # [T, 3]
    b_idx, r_idx, g_idx = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
    flat_parts = [query_subjs[b_idx], query_objs[b_idx]]
    for c in all_cands:
        flat_parts.append(c[b_idx, r_idx, g_idx])
    return torch.stack(flat_parts, dim=1), b_idx, r_idx


# ── L3 join enumeration (semantics-preserving width branch-pruning) ──

def _determined_unknown_count(flat_source, ready_now, arg_src_dep, bpreds_dep,
                              nbody, fact_index, M):
    """Count UNKNOWN (non-fact) body atoms that are FULLY DETERMINED so far.

    ``ready_now [T,M]`` marks atoms whose every free-var ref is already bound.
    Padded/undetermined atoms are excluded → a sound LOWER BOUND on the final
    ``num_unknown`` (monotone as more fv bind), so a row exceeding ``width`` here
    can never survive the final width filter. Returns ``[T]`` counts.
    """
    T = flat_source.size(0)
    if T == 0:
        return flat_source.new_zeros(0, dtype=torch.long)
    body = _gather_body_atoms(
        flat_source.unsqueeze(1).expand(-1, M, -1), arg_src_dep, bpreds_dep)  # [T,M,3]
    exists = fact_index.exists(body.reshape(-1, 3)).reshape(T, M)
    return (ready_now & ~exists).sum(dim=-1)


def enumerate_join_flat(B, K_r, query_subjs, query_objs, fv_pred_q, fv_bound_src_q,
                        fv_dir_q, fv_valid_q, active_mask, fact_index,
                        V, fv_any_valid, arg_source_dep_q, body_preds_dep_q,
                        num_body_q, ready_after_q, width, w_is_capped):
    """L3 join: incremental free-var expansion with a width BRANCH PRUNER.

    Same row contract as ``enumerate_cartesian_flat`` (``flat_source[T,2+V]``,
    ``b_idx[T]``, ``r_idx[T]``) but expands compactly (nonzero per step, no dense
    padded cartesian) and — when width is bounded — drops partial rows whose
    DETERMINED unknown count already exceeds ``width``. The prune only removes
    rows the final ``apply_filters_flat`` would also reject, so the survivor SET
    is identical to the flat cartesian path.
    """
    dev = query_subjs.device
    # seed: every (active query, valid rule slot) — flat row indices into [B*K_r].
    seed = torch.nonzero(active_mask, as_tuple=False)      # [T0, 2] (b, r)
    n_idx, r_idx = seed[:, 0], seed[:, 1]
    if n_idx.numel() == 0:
        return (torch.empty(0, 2, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    # flat_source columns: [subj, obj, fv0, fv1, ...] (free-var cols filled per step).
    cols = [query_subjs[n_idx], query_objs[n_idx]]
    for _ in range(V):
        cols.append(torch.zeros(n_idx.size(0), dtype=torch.long, device=dev))
    src = torch.stack(cols, dim=1)                          # [T, 2+V]
    M = arg_source_dep_q.size(2)

    for fv in range(V):
        if fv_any_valid is not None and not fv_any_valid[fv]:
            continue
        # rows whose rule actually enumerates this fv (else pass through unchanged).
        does_fv = fv_valid_q[n_idx, r_idx, fv]
        if bool(does_fv.any()):
            act = torch.nonzero(does_fv, as_tuple=False).squeeze(1)
            ep = fv_pred_q[n_idx[act], r_idx[act], fv]
            bsrc = fv_bound_src_q[n_idx[act], r_idx[act], fv].clamp(max=src.size(1) - 1)
            bound = src[act].gather(1, bsrc.unsqueeze(1)).squeeze(1)
            ed = fv_dir_q[n_idx[act], r_idx[act], fv]
            cands, cmask = fact_index.enumerate(ep, bound, ed)   # [Ta, Kf]
            Kf = cands.size(1)
            rep = torch.repeat_interleave(act, Kf)
            keep = cmask.reshape(-1)
            new_src = src[rep].clone()
            new_src[:, 2 + fv] = cands.reshape(-1)
            keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            exp_n = n_idx[rep][keep_idx]; exp_r = r_idx[rep][keep_idx]
            exp_src = new_src[keep_idx]
            # pass-through rows (rule doesn't enumerate this fv at all).
            pas = torch.nonzero(~does_fv, as_tuple=False).squeeze(1)
            n_idx = torch.cat([exp_n, n_idx[pas]])
            r_idx = torch.cat([exp_r, r_idx[pas]])
            src = torch.cat([exp_src, src[pas]])
        # BRANCH PRUNE: drop rows whose determined-unknown count already > width.
        if w_is_capped and width is not None:
            asd = arg_source_dep_q[n_idx, r_idx]; bpd = body_preds_dep_q[n_idx, r_idx]
            nb = num_body_q[n_idx, r_idx]
            atom_idx = arange_cached(M, dev).unsqueeze(0)
            ready = (ready_after_q[n_idx, r_idx] <= fv) & (atom_idx < nb.unsqueeze(1))  # [T,M]
            du = _determined_unknown_count(src, ready, asd, bpd, nb, fact_index, M)
            surv = torch.nonzero(du <= width, as_tuple=False).squeeze(1)
            n_idx, r_idx, src = n_idx[surv], r_idx[surv], src[surv]
        if n_idx.numel() == 0:
            return (torch.empty(0, 2 + V, dtype=torch.long, device=dev),
                    torch.empty(0, dtype=torch.long, device=dev),
                    torch.empty(0, dtype=torch.long, device=dev))

    return src, n_idx, r_idx


__all__ = [
    "cluster", "ClusteredRules", "arange_cached", "cumcount_flat",
    "fill_body_dense", "fill_body_flat",
    "enumerate_single_dense", "enumerate_cartesian_dense", "enumerate_cartesian_flat",
    "enumerate_join_flat",
]
