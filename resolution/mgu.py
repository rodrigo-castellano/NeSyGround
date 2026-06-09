"""MGU resolution primitives — ``resolve_facts`` and ``resolve_rules``.

Shared by sld and rtf. All ops are fixed-shape / CUDA-graph-safe (no ``.item()``,
no data-dependent branching). Classification goes through ``Encoding``.

  resolve_facts — targeted_lookup (ArgKey) OR enumerate (Inverted/BlockSparse)
                  → unify → substitute remaining.
  resolve_rules — segment lookup → standardize-apart (per state/rule var
                  namespace) → unify head → substitute [body, remaining].

Bit-exact (fingerprint-locked): standardize-apart base ``next_var + goal*V``,
template_start ``E``; rule child goals = body ``[:Bmax]`` then remaining.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.data.encoding import Encoding
from grounder.data.fact_index.arg_key import ArgKeyFactIndex
from grounder.resolution.primitives import apply_substitutions, unify_one_to_one


def empty_rule_results(B: int, S: int, G: int, M_g: int, pad: int, dev
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The 5 empty rule tensors for the ``num_rules == 0`` fast path:
    ``(rule_goals, rule_grounding_body, rule_success, sub_rule_idx, rule_subs)``."""
    return (torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
            torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            torch.full((B, S, 0, 2, 2), pad, dtype=torch.long, device=dev))


def resolve_facts(
    goals: Tensor,                 # [B, G, 3]
    remaining: Tensor,             # [B, G, L, 3]
    fact_index,
    facts_idx: Tensor,             # [F, 3]
    enc: Encoding,
    K_f: int,
    goal_valid: Tensor,           # [B, G]
    active_mask: Tensor,           # [B, G]
    excluded_queries: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve goal atoms against facts via MGU. Dispatches on index type.

    Returns ``(fact_child_goals[B,G,K_f,L,3], fact_grounding_body[B,G,K_f,M_g,3],
    fact_success[B,G,K_f], fact_subs[B,G,K_f,2,2])``.
    """
    if isinstance(fact_index, ArgKeyFactIndex):
        return _resolve_facts_argkey(goals, remaining, fact_index, facts_idx, enc,
                                     K_f, goal_valid, active_mask, excluded_queries)
    return _resolve_facts_enumerate(goals, remaining, fact_index, enc,
                                    goal_valid, active_mask, excluded_queries)


def resolve_rules(
    goals: Tensor,                 # [B, G, 3]
    remaining: Tensor,             # [B, G, L, 3]
    rule_index,
    enc: Encoding,
    K_r: int,
    max_vars_per_rule: int,
    num_rules: int,
    goal_valid: Tensor,           # [B, G]
    active_mask: Tensor,           # [B, G]
    next_var: Tensor,      # [B]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor]:
    """Resolve goals against rule heads via MGU + standardization-apart.

    Returns ``(rule_child_goals[B,G,K_r,L,3], rule_grounding_body[B,G,K_r,M_g,3],
    rule_success[B,G,K_r], sub_rule_idx[B,G,K_r], sub_lens[B,G,K_r], Bmax,
    rule_subs[B,G,K_r,2,2])``. rule child goals = body at ``[:Bmax]``, remaining after.
    """
    B, S, _ = goals.shape
    G = remaining.shape[2]
    M_g = 0
    dev = goals.device
    pad, E, V = enc.pad, enc.E, max_vars_per_rule
    Bmax = rule_index.rules_bodies_sorted.shape[1]

    # ── segment rule lookup ──
    sorted_pos_flat, sub_rule_mask_flat, _ = rule_index.lookup(goals[:, :, 0].reshape(-1), K_r)
    sub_rule_mask = sub_rule_mask_flat.view(B, S, K_r)
    R = rule_index.rules_heads_sorted.shape[0]
    safe_pos = sorted_pos_flat.clamp(0, max(R - 1, 0))
    sub_rule_idx = rule_index.rules_idx_sorted[safe_pos].view(B, S, K_r)

    flat_pos = safe_pos.reshape(-1)
    sub_heads = rule_index.rules_heads_sorted[flat_pos]            # [N_r, 3]
    sub_bodies = rule_index.rules_bodies_sorted[flat_pos]          # [N_r, Bmax, 3]
    sub_lens_flat = rule_index.rule_lens_sorted[flat_pos]          # [N_r]
    N_r = B * S * K_r

    # ── standardization apart: each (state, rule) gets a unique var namespace ──
    nv_exp = next_var.view(B, 1, 1).expand(B, S, K_r)
    state_offsets = torch.arange(S, device=dev).view(1, S, 1).expand(1, S, K_r) * V
    rule_var_base = (nv_exp + state_offsets).reshape(N_r)
    template_start = E

    is_var_h = sub_heads[:, 1:] >= template_start
    h_off = rule_var_base.unsqueeze(1).expand(N_r, 2)
    std_heads = torch.cat([sub_heads[:, 0:1],
                           torch.where(is_var_h, sub_heads[:, 1:] - template_start + h_off,
                                       sub_heads[:, 1:])], dim=1)

    is_var_b = sub_bodies[:, :, 1:] >= template_start
    b_off = rule_var_base.view(N_r, 1, 1).expand(N_r, Bmax, 2)
    std_bodies = torch.cat([sub_bodies[:, :, 0:1],
                            torch.where(is_var_b, sub_bodies[:, :, 1:] - template_start + b_off,
                                        sub_bodies[:, :, 1:])], dim=2)

    # ── unify head with goal ──
    flat_q = goals.unsqueeze(2).expand(B, S, K_r, 3).reshape(N_r, 3)
    ok_flat, subs_flat = unify_one_to_one(flat_q, std_heads, enc)
    rule_success = (ok_flat.view(B, S, K_r) & sub_rule_mask
                    & goal_valid.unsqueeze(-1) & active_mask.unsqueeze(-1))
    rule_subs = subs_flat.view(B, S, K_r, 2, 2)

    # ── apply subs to [body, remaining] ──
    rem_exp = remaining.unsqueeze(2).expand(B, S, K_r, G, 3).reshape(N_r, G, 3)
    combined = apply_substitutions(torch.cat([std_bodies, rem_exp], dim=1), subs_flat, enc)
    rule_body_subst = combined[:, :Bmax, :].view(B, S, K_r, Bmax, 3)
    rule_remaining = combined[:, Bmax:, :].view(B, S, K_r, G, 3)

    rule_grounding_body_out = torch.zeros(B, S, K_r, M_g, 3, dtype=torch.long, device=dev)

    # ── mask body atoms beyond each rule's length ──
    sub_lens_v = sub_lens_flat.view(B, S, K_r)
    atom_idx = torch.arange(Bmax, device=dev).view(1, 1, 1, Bmax)
    inactive = atom_idx >= sub_lens_v.unsqueeze(-1)
    pad_t = torch.tensor(pad, dtype=torch.long, device=dev)
    rule_body_subst = torch.where(inactive.unsqueeze(-1).expand(B, S, K_r, Bmax, 3),
                                  pad_t, rule_body_subst)

    # ── assemble rule_goals: body then remaining ──
    rule_goals = torch.full((B, S, K_r, G, 3), pad, dtype=torch.long, device=dev)
    rule_goals[:, :, :, :Bmax, :] = rule_body_subst
    n_rem = min(G - Bmax, G)
    if n_rem > 0:
        rule_goals[:, :, :, Bmax:Bmax + n_rem, :] = rule_remaining[:, :, :, :n_rem, :]

    return rule_goals, rule_grounding_body_out, rule_success, sub_rule_idx, sub_lens_v, Bmax, rule_subs


def _resolve_facts_argkey(goals, remaining, fact_index, facts_idx, enc, K_f,
                          goal_valid, active_mask, excluded_queries):
    """ArgKey fact resolution: targeted_lookup → unify → substitute remaining."""
    B, S, _ = goals.shape
    G = remaining.shape[2]
    M_g = 0
    dev = goals.device
    pad = enc.pad
    N = B * S
    flat_q = goals.reshape(N, 3)
    flat_active = (active_mask & goal_valid).reshape(N)

    fact_item_idx, fact_valid = fact_index.targeted_lookup(flat_q, K_f)
    F = facts_idx.shape[0]
    safe_idx = fact_item_idx.clamp(0, max(F - 1, 0))
    fact_atoms = facts_idx[safe_idx.view(-1)].view(N, K_f, 3)
    q_exp = flat_q.unsqueeze(1).expand(-1, K_f, -1)
    ok_flat, subs_flat = unify_one_to_one(q_exp.reshape(-1, 3), fact_atoms.reshape(-1, 3), enc)
    ok = ok_flat.view(N, K_f)
    subs = subs_flat.view(N, K_f, 2, 2)
    success = ok & fact_valid & flat_active.unsqueeze(1)

    if excluded_queries is not None and facts_idx.numel() > 0:
        excl_flat = excluded_queries[:, 0, :].unsqueeze(1).expand(B, S, 3).reshape(N, 1, 3)
        match_excl = (fact_atoms == excl_flat).all(dim=-1)
        success = success & ~match_excl

    rem_exp = remaining.reshape(N, G, 3).unsqueeze(1).expand(-1, K_f, -1, -1).reshape(N * K_f, G, 3)
    fact_goals = apply_substitutions(rem_exp, subs_flat, enc).view(B, S, K_f, G, 3)
    fact_grounding_body = torch.zeros(B, S, K_f, max(M_g, 1), 3, dtype=torch.long, device=dev)

    pad_t = torch.tensor(pad, dtype=torch.long, device=dev)
    fact_goals = torch.where(success.view(B, S, K_f, 1, 1), fact_goals, pad_t)
    return fact_goals, fact_grounding_body, success.view(B, S, K_f), subs.view(B, S, K_f, 2, 2)


def _resolve_facts_enumerate(goals, remaining, fact_index, enc,
                             goal_valid, active_mask, excluded_queries):
    """Enumerate fact resolution (Inverted/BlockSparse): build free-var subs."""
    B, S, _ = goals.shape
    G = remaining.shape[2]
    M_g = 0
    dev = goals.device
    pad, c_no = enc.pad, enc.E - 1
    pred, arg0, arg1 = goals[:, :, 0], goals[:, :, 1], goals[:, :, 2]

    arg0_ground = arg0 <= c_no
    arg1_ground = arg1 <= c_no
    has_ground = arg0_ground | arg1_ground
    both_ground = arg0_ground & arg1_ground
    is_active = goal_valid & active_mask

    use_arg0 = arg0_ground
    direction = torch.where(use_arg0, torch.zeros_like(arg0), torch.ones_like(arg0))
    bound_arg = torch.where(has_ground & is_active, torch.where(use_arg0, arg0, arg1),
                            torch.zeros_like(arg0))
    safe_pred = torch.where(is_active, pred, torch.zeros_like(pred))
    free_var = torch.where(use_arg0, arg1, arg0)

    cands, cand_mask = fact_index.enumerate(safe_pred.reshape(-1), bound_arg.reshape(-1),
                                            direction.reshape(-1))
    K_f = cands.shape[1]
    cands = cands.view(B, S, K_f)
    cand_mask = cand_mask.view(B, S, K_f)

    N_f = B * S * K_f
    free_var_exp = free_var.unsqueeze(2).expand(B, S, K_f)
    subs = torch.full((N_f, 2, 2), pad, dtype=torch.long, device=dev)
    subs[:, 0, 0] = free_var_exp.reshape(-1)
    subs[:, 0, 1] = cands.reshape(-1)

    other_arg = torch.where(use_arg0, arg1, arg0)
    both_filter = torch.where(both_ground.unsqueeze(2).expand(B, S, K_f),
                              cands == other_arg.unsqueeze(2).expand(B, S, K_f),
                              torch.ones(B, S, K_f, dtype=torch.bool, device=dev))
    fact_success = (cand_mask & has_ground.unsqueeze(-1)
                    & goal_valid.unsqueeze(-1) & active_mask.unsqueeze(-1) & both_filter)

    rem_exp = remaining.unsqueeze(2).expand(B, S, K_f, G, 3).reshape(N_f, G, 3)
    fact_goals = apply_substitutions(rem_exp, subs, enc).view(B, S, K_f, G, 3)
    fact_grounding_body = torch.zeros(B, S, K_f, max(M_g, 1), 3, dtype=torch.long, device=dev)

    return fact_goals, fact_grounding_body, fact_success, subs.view(B, S, K_f, 2, 2)


def init_mgu(
    resolution: str,
    K_f: int,
    K_r: int,
    rule_index,
    max_total_groundings: int,
    *,
    max_children: int,                   # the children cap K (shell owns the default)
    K_f_min_budget: int = 10,            # internal guard: min fact slots; not user config
    max_groundings_per_rule: Optional[int] = None,
) -> dict:
    """Compute MGU shape params (K, K_f, max_vars_per_rule, Y_q) for sld/rtf.

    S is NOT returned — the shell (BackwardGrounder.__init__) is the single owner."""
    import warnings
    K_uncapped = K_f * K_r if resolution == "rtf" else K_f + K_r
    K = min(K_uncapped, max_children)

    if resolution == "sld":
        # require room for min(K_f_min_budget, available facts) — never more fact slots
        # than the KB actually has facts (so small KBs don't trip the guard).
        min_facts = min(K_f_min_budget, K_f)
        K_f_budget = max(K - K_r, min_facts)
        if K_r > K - min_facts:
            raise ValueError(f"K_r={K_r} leaves fewer than {min_facts} fact slots in K={K}.")
        if K_f > K_f_budget:
            warnings.warn(f"[init_mgu] K_f capped {K_f}->{K_f_budget} (K_r={K_r}, K={K}).", stacklevel=2)
            K_f = K_f_budget
    elif K_f > K:
        K_f = K

    max_vars_per_rule = (int(rule_index.rule_lens_sorted.max().item()) + 2
                         if rule_index.rule_lens_sorted.numel() > 0 else 3)
    Y_q = (min(max_total_groundings, rule_index.max_rule_pairs * max(max_groundings_per_rule, 1))
           if max_groundings_per_rule is not None else max_total_groundings)
    return {"K": K, "K_f": K_f, "max_vars_per_rule": max_vars_per_rule,
            "Y_q": Y_q, "max_fact_pairs_body": K_f}


__all__ = ["resolve_facts", "resolve_rules", "empty_rule_results", "init_mgu"]
