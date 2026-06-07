"""PBC resolve — the per-step driver + the two materializers.

Fixed, shared pipeline:  depth_gate → prepare → cluster → enumerate → fill →
width → emit.  The dense and flat MATERIALIZERS implement the stages that differ
by layout (chosen once per run — no hot-loop branching, each traces its own
graph); ``cluster``/``depth_gate`` and the inner cores (`_gather_body_atoms`,
`_cartesian_expand_one_fv`, the width reductions) are shared.

  DenseMaterializer — static padded [B,S,K,…]; compile/CUDA-graph; topk-caps to K.
  FlatMaterializer  — compact [T,…] via nonzero; eager, zero-waste; optional dedup_goals.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor

from grounder._build.resolution.pbc import candidates as C
from grounder._build.resolution.pbc.candidates import cluster
from grounder._build.resolution.pbc.width import apply_filters_dense, apply_filters_flat, depth_gate
from grounder._build.types import FlatResolvedChildren, Layout, ResolvedChildren


class StepInputs(NamedTuple):
    """One proof step's request (per-step tensors + scalars/flags)."""
    queries: Tensor          # [B, S, 3]
    remaining: Tensor        # [B, S, G, 3]
    grounding_body: Tensor   # [B, S, G_body, 3]
    state_valid: Tensor      # [B, S]
    active_mask: Tensor      # [B, S]
    padding_idx: int
    d: object                # int (eager) | 0-dim tensor (compiled)
    depth: int
    is_last: Optional[Tensor]
    width: object            # int | None
    w_last_depth: int
    collect_evidence: bool
    dedup_goals: bool


def resolve_step(inp: StepInputs, plan, fact_index, materializer):
    """Drive the pbc pipeline for one step → ResolvedChildren | FlatResolvedChildren."""
    w_d, hpm_d = depth_gate(inp.d, inp.depth, inp.width, inp.w_last_depth,
                            plan.tables.head_pred_mask, inp.is_last)
    work = materializer.prepare(inp, plan)
    if work is None:
        return materializer.empty(inp, plan)
    cl = cluster(work.q_preds, work.q_valid, plan.tables)
    cand = materializer.enumerate(work, cl, plan, fact_index)
    if cand is None:
        return materializer.empty(inp, plan)
    body, exists, body_active = materializer.fill(cand, cl, plan, fact_index, work)
    keep = materializer.width(body, exists, body_active, cand, cl, work, w_d, hpm_d, inp)
    return materializer.emit(body, keep, cand, cl, work, inp, plan)


# ══════════════════════════ dense ══════════════════════════

class _DenseWork(NamedTuple):
    q: Tensor; q_preds: Tensor; q_valid: Tensor; B: int; S: int

class _DenseCand(NamedTuple):
    source: Tensor; check_arg: Tensor; body_preds: Tensor
    cmask: Tensor; G_r: int; num_body_q: Tensor


class DenseMaterializer:
    """Static padded [B,S,K,…] children (compile/CUDA-graph)."""

    def prepare(self, inp: StepInputs, plan) -> _DenseWork:
        B, S, _ = inp.queries.shape
        q = inp.queries.reshape(B * S, 3)
        q_valid = (inp.active_mask & inp.state_valid).reshape(B * S)
        return _DenseWork(q, q[:, 0], q_valid, B, S)

    def enumerate(self, work: _DenseWork, cl, plan, fact_index) -> _DenseCand:
        t, ai, K_r = plan.tables, cl.active_idx, cl.K_r
        N = work.q.size(0)
        qs, qo = work.q[:, 1], work.q[:, 2]
        body_preds_q = t.body_preds[ai]
        num_body_q = t.num_body_atoms[ai]
        G_r = plan.E if plan.cartesian_product else min(plan.G_r, fact_index._max_facts_per_query)

        if plan.V >= 2 and not plan.cartesian_product:           # ≥2 free vars: cartesian
            dep_src, dep_bpreds = t.arg_source_dep[ai], t.body_preds_dep[ai]
            source, cmask, G_r = C.enumerate_cartesian_dense(
                N, K_r, qs, qo, t.fv_enum_pred[ai], t.fv_enum_bound_src[ai],
                t.fv_enum_direction[ai], t.fv_enum_valid[ai], cl.has_free_q, cl.active_mask,
                fact_index, K_v=plan.K_v, V=plan.V, G_cap=G_r, fv_any_valid=plan.fv_any_valid,
                check_arg_source_q=dep_src, body_preds_q=dep_bpreds, num_body_q=num_body_q, M=plan.M)
            return _DenseCand(source, dep_src, dep_bpreds, cmask, G_r, num_body_q)

        cands, cmask = C.enumerate_single_dense(                  # ≤1 free var: single lookup
            N, K_r, G_r, qs, qo, t.enum_pred[ai], t.enum_bound[ai], t.enum_dir[ai],
            fact_index, cartesian_product=plan.cartesian_product, E=plan.E)
        G_r = cands.size(2)
        cmask = cmask & cl.has_free_q.unsqueeze(2)
        cmask[:, :, 0] = cmask[:, :, 0] | (~cl.has_free_q & cl.active_mask)
        source = torch.stack([qs.view(N, 1, 1).expand(-1, K_r, G_r),
                              qo.view(N, 1, 1).expand(-1, K_r, G_r), cands], dim=3)
        return _DenseCand(source, t.check_arg_source[ai], body_preds_q, cmask, G_r, num_body_q)

    def fill(self, cand: _DenseCand, cl, plan, fact_index, work: _DenseWork):
        N, K_r, M, dev = work.q.size(0), cl.K_r, plan.M, work.q.device
        body = C.fill_body_dense(cand.source, cand.check_arg, cand.body_preds)
        exists = fact_index.exists(body.reshape(-1, 3)).view(N, K_r, cand.G_r, M)
        atom_idx = torch.arange(M, device=dev).view(1, 1, 1, M)
        body_active = atom_idx < cand.num_body_q.view(N, K_r, 1, 1)
        body = body.masked_fill(~body_active.unsqueeze(-1), fact_index._padding_idx)
        return body, exists, body_active

    def width(self, body, exists, body_active, cand: _DenseCand, cl, work, w_d, hpm_d, inp):
        return apply_filters_dense(body, exists, body_active, cl.active_mask, cand.cmask,
                                   work.q, cand.G_r, w_d, hpm_d, is_last=inp.is_last)

    def emit(self, body, mask, cand: _DenseCand, cl, work: _DenseWork, inp: StepInputs, plan):
        B, S, K_r, M, dev = work.B, work.S, cl.K_r, plan.M, work.q.device
        N, G, pad = B * S, inp.remaining.shape[2], inp.padding_idx
        G_use = body.size(2)
        K_total = K_r * G_use
        body_flat = body.reshape(N, K_total, M, 3)
        success_flat = mask.reshape(N, K_total)
        ridx_flat = cl.active_idx.unsqueeze(2).expand(-1, -1, G_use).reshape(N, K_total)

        K = plan.K
        if K_total > K:
            _, top = success_flat.to(torch.int8).topk(K, dim=1, largest=True, sorted=False)
            body_flat = body_flat.gather(1, top.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3))
            success_flat = success_flat.gather(1, top)
            ridx_flat = ridx_flat.gather(1, top)
        else:
            K = K_total

        body_flat = body_flat.reshape(B, S, K, M, 3)
        success_flat = success_flat.reshape(B, S, K)
        ridx_flat = ridx_flat.reshape(B, S, K)

        rule_goals = torch.full((B, S, K, G, 3), pad, dtype=torch.long, device=dev)
        rule_goals[:, :, :, :M, :] = body_flat
        n_rem = min(G - M, G - 1)
        if n_rem > 0:
            rule_goals[:, :, :, M:M + n_rem, :] = \
                inp.remaining[:, :, 1:1 + n_rem, :].unsqueeze(2).expand(-1, -1, K, -1, -1)

        G_body = inp.grounding_body.shape[2]
        rule_gbody = (inp.grounding_body.unsqueeze(2).expand(-1, -1, K, -1, -1) if inp.collect_evidence
                      else torch.zeros(B, S, K, G_body, 3, dtype=torch.long, device=dev))
        z_goals = torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev)
        z_gbody = torch.zeros(B, S, 0, G_body, 3, dtype=torch.long, device=dev)
        z_succ = torch.zeros(B, S, 0, dtype=torch.bool, device=dev)
        z_subs = torch.full((B, S, 0, 2, 2), pad, dtype=torch.long, device=dev)
        rule_subs = torch.full((B, S, K, 2, 2), pad, dtype=torch.long, device=dev)
        return ResolvedChildren(Layout.DENSE, z_goals, z_gbody, z_succ,
                                rule_goals, rule_gbody, success_flat, ridx_flat, z_subs, rule_subs)

    def empty(self, inp, plan):   # dense is fixed-shape; never reached
        raise RuntimeError("DenseMaterializer.empty is unreachable")


# ══════════════════════════ flat ══════════════════════════

class _FlatWork(NamedTuple):
    flat_q: Tensor; q_preds: Tensor; q_valid: Tensor
    active_pos: Tensor; state_to_goal: Optional[Tensor]; B: int; S: int

class _FlatCand(NamedTuple):
    flat_source: Tensor; n_idx_eff: Tensor; r_idx: Tensor
    rule_global_idx: Tensor; check_flat: Tensor; bpreds_flat: Tensor; nbody_flat: Tensor


class FlatMaterializer:
    """Compact [T,…] children via nonzero (eager, zero-waste)."""

    def prepare(self, inp: StepInputs, plan) -> Optional[_FlatWork]:
        B, S, _ = inp.queries.shape
        flat_q_full = inp.queries.reshape(B * S, 3)
        flat_valid = (inp.active_mask & inp.state_valid).reshape(B * S)
        active_pos = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if active_pos.size(0) == 0:
            return None
        flat_q_active = flat_q_full[active_pos]
        if inp.dedup_goals:
            flat_q, state_to_goal = torch.unique(flat_q_active, dim=0, return_inverse=True)
        else:
            flat_q, state_to_goal = flat_q_active, None
        ones = torch.ones(flat_q.size(0), dtype=torch.bool, device=flat_q.device)
        return _FlatWork(flat_q, flat_q[:, 0], ones, active_pos, state_to_goal, B, S)

    def enumerate(self, work: _FlatWork, cl, plan, fact_index) -> Optional[_FlatCand]:
        t, ai = plan.tables, cl.active_idx
        qs, qo = work.flat_q[:, 1], work.flat_q[:, 2]
        flat_source, n_idx, r_idx = C.enumerate_cartesian_flat(
            work.flat_q.size(0), cl.K_r, qs, qo, t.fv_enum_pred[ai], t.fv_enum_bound_src[ai],
            t.fv_enum_direction[ai], t.fv_enum_valid[ai], cl.has_free_q, cl.active_mask,
            fact_index, V=plan.V, fv_any_valid=plan.fv_any_valid)
        if flat_source.size(0) == 0:
            return None
        rule_global = ai[n_idx, r_idx]
        return _FlatCand(flat_source, n_idx, r_idx, rule_global,
                         t.arg_source_dep[rule_global], t.body_preds_dep[rule_global],
                         t.num_body_atoms[rule_global])

    def fill(self, cand: _FlatCand, cl, plan, fact_index, work: _FlatWork):
        M, dev = plan.M, work.flat_q.device
        body = C.fill_body_flat(cand.flat_source, cand.check_flat, cand.bpreds_flat)
        T = body.size(0)
        exists = fact_index.exists(body.reshape(-1, 3)).reshape(T, M)
        body_active = C.arange_cached(M, dev).unsqueeze(0) < cand.nbody_flat.unsqueeze(1)
        body = body.masked_fill(~body_active.unsqueeze(-1), fact_index._padding_idx)
        return body, exists, body_active

    def width(self, body, exists, body_active, cand: _FlatCand, cl, work, w_d, hpm_d, inp):
        M = body.size(1)
        return apply_filters_flat(body, exists, cand.n_idx_eff, cand.nbody_flat,
                                  work.flat_q, w_d, hpm_d, M, body_active=body_active)

    def emit(self, body, mask, cand: _FlatCand, cl, work: _FlatWork, inp: StepInputs, plan):
        B, S, M, dev = work.B, work.S, plan.M, body.device
        G, G_body, pad = inp.remaining.shape[2], inp.grounding_body.shape[2], inp.padding_idx
        surv = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if surv.size(0) == 0:
            return self.empty(inp, plan)
        surv_body, surv_g, surv_rule = body[surv], cand.n_idx_eff[surv], cand.rule_global_idx[surv]

        if inp.dedup_goals:
            D = work.flat_q.size(0)
            st_order = torch.argsort(work.state_to_goal, stable=True)
            st_cnt = torch.bincount(work.state_to_goal, minlength=D)
            st_off = torch.zeros(D + 1, dtype=torch.long, device=dev); st_off[1:] = torch.cumsum(st_cnt, 0)
            sv_order = torch.argsort(surv_g, stable=True)
            sv_cnt = torch.bincount(surv_g, minlength=D)
            sv_off = torch.zeros(D + 1, dtype=torch.long, device=dev); sv_off[1:] = torch.cumsum(sv_cnt, 0)
            per_goal = sv_cnt * st_cnt
            E_tot = int(per_goal.sum().item())
            if E_tot == 0:
                return self.empty(inp, plan)
            e_goal = torch.repeat_interleave(torch.arange(D, device=dev), per_goal)
            e_within = (torch.arange(E_tot, device=dev)
                        - torch.repeat_interleave(torch.cumsum(per_goal, 0) - per_goal, per_goal))
            g_st = st_cnt[e_goal]
            sv_rows = sv_order[sv_off[e_goal] + e_within // g_st]
            st_rows = st_order[st_off[e_goal] + e_within % g_st]
            surv_body, surv_rule, surv_n_eff = surv_body[sv_rows], surv_rule[sv_rows], st_rows
        else:
            surv_n_eff = surv_g

        T_surv = surv_body.size(0)
        surv_n = work.active_pos[surv_n_eff]
        b_idx, s_idx = surv_n // S, surv_n % S
        flat_goals = torch.full((T_surv, G, 3), pad, dtype=torch.long, device=dev)
        flat_goals[:, :M, :] = surv_body
        n_rem = min(G - M, G - 1)
        if n_rem > 0:
            flat_goals[:, M:M + n_rem, :] = inp.remaining[b_idx, s_idx, 1:1 + n_rem, :]
        flat_gbody = torch.zeros(T_surv, G_body, 3, dtype=torch.long, device=dev)
        flat_subs = torch.full((T_surv, 2, 2), pad, dtype=torch.long, device=dev)
        flat_is_fact = torch.zeros(T_surv, dtype=torch.bool, device=dev)
        flat_top_ridx = torch.zeros(T_surv, dtype=torch.long, device=dev)
        return FlatResolvedChildren(Layout.FLAT, flat_goals, flat_gbody, surv_rule,
                                    b_idx, s_idx, flat_subs, flat_is_fact, flat_top_ridx,
                                    B, S, True)

    def empty(self, inp: StepInputs, plan) -> FlatResolvedChildren:
        B, S, _ = inp.queries.shape
        G, G_body, pad, dev = (inp.remaining.shape[2], inp.grounding_body.shape[2],
                               inp.padding_idx, inp.queries.device)
        z = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        return FlatResolvedChildren(
            Layout.FLAT, torch.full((0, G, 3), pad, dtype=torch.long, device=dev),
            z(0, G_body, 3), z(0), z(0), z(0),
            torch.full((0, 2, 2), pad, dtype=torch.long, device=dev),
            torch.zeros(0, dtype=torch.bool, device=dev), z(0), B, S, True)


__all__ = ["StepInputs", "resolve_step", "DenseMaterializer", "FlatMaterializer"]
