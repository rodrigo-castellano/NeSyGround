"""RTFGrounder — two-level Rule-Then-Fact resolution.

K = K_f * K_r.  First resolves queries against rule heads, then resolves
body atoms against facts.  Supports body_order_agnostic and rtf_cascade.

The heavy lifting (body-fact resolution helpers) lives as module-level
functions so they can be referenced by both RTFGrounder and the RL-layer
RTFEngine without method-resolution conflicts.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from grounder.grounders.backward import BCGrounder
from grounder.operations import unify_with_facts


class RTFGrounder(BCGrounder):
    """Two-level Rule-Then-Fact: K = K_f * K_r.

    First resolves queries against rule heads, then resolves body atoms
    against facts. Supports body_order_agnostic and rtf_cascade options.
    """

    def __init__(
        self, *args,
        body_order_agnostic: bool = False,
        rtf_cascade: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.body_order_agnostic = body_order_agnostic
        self.rtf_cascade = rtf_cascade
        self._max_fact_pairs_body = self.K_f

    def _compute_K_uncapped(self) -> int:
        return self.K_f * self.K_r

    # --- Fact resolution: empty for RTF (all work done in _resolve_rules) ---

    def _resolve_facts(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        return (
            torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
            torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
        )

    # --- Rule resolution: full two-level (rules then body-fact) ---

    def _resolve_rules(
        self,
        queries: Tensor,            # [B, S, 3]
        remaining: Tensor,          # [B, S, G, 3]
        grounding_body: Tensor,     # [B, S, M, 3]
        state_valid: Tensor,        # [B, S]
        active_mask: Tensor,        # [B, S]
        next_var_indices: Tensor,   # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Two-level resolution: rule head unification then body-fact resolution.

        Returns:
            rule_goals:   [B, S, K_rtf, G, 3]
            rule_gbody:   [B, S, K_rtf, M_g, 3]
            rule_success: [B, S, K_rtf]
            sub_rule_idx: [B, S, K_rtf]
        """
        B, S, _ = queries.shape
        G = remaining.shape[2]
        M_g = grounding_body.shape[2]
        dev = queries.device
        pad = self.padding_idx
        K_r = self.K_r

        if self.num_rules == 0:
            return (
                torch.full((B, S, 0, G, 3), pad, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, M_g, 3, dtype=torch.long, device=dev),
                torch.zeros(B, S, 0, dtype=torch.bool, device=dev),
                torch.zeros(B, S, 0, dtype=torch.long, device=dev),
            )

        # ================================================================
        # Level 1: Shared rule head unification
        # ================================================================
        rule_body_subst, rule_remaining, rule_gbody_l1, rule_success_l1, \
            sub_rule_idx_l1, _, Bmax = self._resolve_rule_heads(
                queries, remaining, grounding_body,
                state_valid, active_mask, next_var_indices)

        # Build intermediate states: [body | remaining] shape [B, S, K_r, Bmax+G, 3]
        M_inter = Bmax + G
        rule_states_l1 = torch.full(
            (B, S, K_r, M_inter, 3), pad, dtype=torch.long, device=dev)
        rule_states_l1[:, :, :, :Bmax, :] = rule_body_subst
        n_rem = min(G, M_inter - Bmax)
        if n_rem > 0:
            rule_states_l1[:, :, :, Bmax:Bmax + n_rem, :] = rule_remaining[:, :, :, :n_rem, :]

        # ================================================================
        # Level 2: Body-fact resolution
        # ================================================================
        # Flatten S into batch for body-fact helpers: [B*S, K_r, M_inter, 3]
        rule_states_flat = rule_states_l1.reshape(B * S, K_r, M_inter, 3)
        rule_success_flat = rule_success_l1.reshape(B * S, K_r)

        resolved_flat, resolved_ok_flat = resolve_rules_with_facts(
            self, rule_states_flat, rule_success_flat, excluded_queries=None)

        K_rtf = resolved_ok_flat.shape[1]
        G_out = resolved_flat.shape[2]

        # Reshape back to [B, S, K_rtf, ...]
        resolved = resolved_flat.view(B, S, K_rtf, G_out, 3)
        resolved_ok = resolved_ok_flat.view(B, S, K_rtf)

        # Pad/trim G dimension to match expected G
        if G_out < G:
            resolved = torch.nn.functional.pad(
                resolved, (0, 0, 0, G - G_out), value=pad)
        elif G_out > G and self.track_grounding_body:
            # Only trim in TS mode; RL mode keeps wider output so
            # get_derived_states_compiled can truncate to M properly.
            resolved = resolved[:, :, :, :G, :]

        # Build rule_gbody: propagate from level-1
        # Each level-1 rule match spawns K_f children → expand gbody
        K_f_body = self._max_fact_pairs_body
        rule_gbody_out = rule_gbody_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_body, M_g, 3).reshape(B, S, K_r * K_f_body, M_g, 3)
        # Trim/pad to match K_rtf
        if rule_gbody_out.shape[2] < K_rtf:
            rule_gbody_out = torch.nn.functional.pad(
                rule_gbody_out,
                (0, 0, 0, 0, 0, K_rtf - rule_gbody_out.shape[2]),
                value=0)
        elif rule_gbody_out.shape[2] > K_rtf:
            rule_gbody_out = rule_gbody_out[:, :, :K_rtf]

        # Build sub_rule_idx: expand level-1 rule indices
        sub_rule_idx_out = sub_rule_idx_l1.unsqueeze(3).expand(
            B, S, K_r, K_f_body).reshape(B, S, K_r * K_f_body)
        if sub_rule_idx_out.shape[2] < K_rtf:
            sub_rule_idx_out = torch.nn.functional.pad(
                sub_rule_idx_out, (0, K_rtf - sub_rule_idx_out.shape[2]), value=0)
        elif sub_rule_idx_out.shape[2] > K_rtf:
            sub_rule_idx_out = sub_rule_idx_out[:, :, :K_rtf]

        return resolved, rule_gbody_out, resolved_ok, sub_rule_idx_out


# ---------------------------------------------------------------------------
# Module-level RTF helpers (shared with RTFEngine in the RL layer)
# ---------------------------------------------------------------------------

def _sort_slice_valid(
    states: Tensor,     # [B, K, ..., 3]
    mask: Tensor,       # [B, K]
    K_cap: int,
) -> Tuple[Tensor, Tensor]:
    """Sort valid entries to front, then slice to K_cap."""
    K = mask.shape[1]
    if K <= K_cap:
        return states, mask
    sort_idx = torch.argsort(~mask, dim=1, stable=True)
    idx_exp = sort_idx
    for _ in range(states.dim() - 2):
        idx_exp = idx_exp.unsqueeze(-1)
    idx_exp = idx_exp.expand_as(states)
    states = torch.gather(states, 1, idx_exp)[:, :K_cap]
    mask = torch.gather(mask, 1, sort_idx)[:, :K_cap]
    return states, mask


@torch.no_grad()
def resolve_body_atom_with_facts(
    eng: RTFGrounder,
    atoms: Tensor,              # [B, K_r, 3]
    remaining: Tensor,          # [B, K_r, G, 3]
    rule_success: Tensor,       # [B, K_r]
    excluded_queries: Optional[Tensor] = None,  # [B, 1, 3]
) -> Tuple[Tensor, Tensor]:
    """Resolve a set of atoms against facts, returning derived states."""
    B, K_r, _ = atoms.shape
    G = remaining.shape[2]
    pad = eng.padding_idx
    max_fp = eng._max_fact_pairs_body
    N = B * K_r

    flat_atoms = atoms.reshape(N, 3)
    flat_remaining = remaining.reshape(N, G, 3)
    flat_remaining_counts = (flat_remaining[:, :, 0] != pad).sum(dim=1)
    flat_rule_ok = rule_success.reshape(N)

    flat_fact_idx, flat_fact_valid = eng.fact_index.targeted_lookup(
        flat_atoms, max_fp)
    flat_fact_valid = flat_fact_valid & flat_rule_ok.unsqueeze(1)

    flat_derived, flat_ok, _ = unify_with_facts(
        flat_atoms, flat_remaining, flat_remaining_counts,
        flat_fact_idx, flat_fact_valid, eng.facts_idx,
        eng.constant_no, pad,
    )

    if excluded_queries is not None and eng.facts_idx.numel() > 0:
        excl = excluded_queries[:, 0, :].unsqueeze(1).expand(-1, K_r, -1).reshape(N, 3)
        safe_idx = flat_fact_idx.clamp(0, max(eng.facts_idx.shape[0] - 1, 0))
        matched = eng.facts_idx[safe_idx.view(-1)].view(N, max_fp, 3)
        flat_ok = flat_ok & ~(matched == excl.unsqueeze(1)).all(dim=-1)

    total_per_batch = K_r * max_fp
    resolved_states = flat_derived.reshape(B, total_per_batch, G, 3)
    resolved_ok = flat_ok.reshape(B, total_per_batch)
    return resolved_states, resolved_ok


@torch.no_grad()
def cascade_step_chunked(
    eng: RTFGrounder,
    cap_states: Tensor,        # [B, K_cap, G_cur, 3]
    cap_ok: Tensor,            # [B, K_cap]
    excluded_queries: Optional[Tensor],
    K_budget: int,
) -> Tuple[Tensor, Tensor]:
    """B-chunked cascade step with output compaction."""
    B, K_cap, G_cur, _ = cap_states.shape
    G_step = G_cur - 1
    max_fp = eng._max_fact_pairs_body
    pad = eng.padding_idx
    device = cap_states.device

    mem_per_b = K_cap * max_fp * max(G_step, 1) * 24 * 4
    B_chunk = max(1, min(B, (1 << 30) // max(mem_per_b, 1)))

    out_states = torch.full(
        (B, K_budget, max(G_step, 1), 3), pad, dtype=torch.long, device=device)
    out_ok = torch.zeros((B, K_budget), dtype=torch.bool, device=device)

    for b_off in range(0, B, B_chunk):
        b_end = min(b_off + B_chunk, B)
        b_sl = slice(b_off, b_end)

        c_atoms = cap_states[b_sl, :, 0, :]
        c_rem = cap_states[b_sl, :, 1:, :]
        c_ok_in = cap_ok[b_sl]
        c_excl = excluded_queries[b_sl] if excluded_queries is not None else None

        c_res, c_ok_out = resolve_body_atom_with_facts(
            eng, c_atoms, c_rem, c_ok_in, c_excl)

        b_size = b_end - b_off
        K_src = c_ok_out.shape[1]
        cumsum = c_ok_out.long().cumsum(dim=1)
        target_idx = torch.where(
            c_ok_out,
            cumsum - 1,
            K_budget
        ).clamp(min=0, max=K_budget)

        buf = torch.full(
            (b_size, K_budget + 1, G_step, 3), pad,
            dtype=torch.long, device=device)
        target_exp = target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, G_step, 3)
        buf.scatter_(1, target_exp, c_res)
        out_states[b_off:b_end] = buf[:, :K_budget]
        out_ok[b_off:b_end] = torch.arange(K_budget, device=device).unsqueeze(0) < cumsum[:, -1:].clamp(max=K_budget)

        del c_res, c_ok_out, buf
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return out_states, out_ok


@torch.no_grad()
def resolve_rules_with_facts(
    eng: RTFGrounder,
    rule_states: Tensor,       # [B, K_r, M, 3]
    rule_success: Tensor,      # [B, K_r]
    excluded_queries: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Two-level 'rtf' resolution: resolve body atom(s) with facts."""
    B, K_r, M, _ = rule_states.shape
    G = M - 1

    if not eng.body_order_agnostic or M <= 1:
        rule_states, rule_success = _sort_slice_valid(rule_states, rule_success, eng.K)
        K_r = rule_success.shape[1]

        first_atoms = rule_states[:, :, 0, :]
        remaining = rule_states[:, :, 1:, :]
        step0_states, step0_ok = resolve_body_atom_with_facts(
            eng, first_atoms, remaining, rule_success, excluded_queries)

        if not eng.rtf_cascade or G <= 1:
            return step0_states, step0_ok

        max_body = (eng.rule_index.rules_bodies_sorted.shape[1]
                    if eng.rule_index.rules_bodies_sorted.numel() else 0)
        if max_body <= 1:
            return step0_states, step0_ok

        step0_states, step0_ok = _sort_slice_valid(step0_states, step0_ok, eng.K)

        all_states = [step0_states]
        all_ok = [step0_ok]

        cur_states = step0_states
        cur_ok = step0_ok

        for step in range(1, max_body):
            G_cur = cur_states.shape[2]
            if G_cur == 0:
                break

            K_cur = cur_states.shape[1]
            K_cap = min(K_cur, eng.K)

            cap_states, cap_ok = _sort_slice_valid(cur_states, cur_ok, K_cap)

            max_fp = eng._max_fact_pairs_body
            G_rem = max(G_cur - 1, 1)

            est_bytes = B * K_cap * max_fp * G_rem * 24 * 4
            use_oneshot = est_bytes <= 2 * (1 << 30) or torch.compiler.is_compiling()

            if use_oneshot:
                step_atoms = cap_states[:, :, 0, :]
                step_remaining = cap_states[:, :, 1:, :]
                step_resolved, step_ok_out = resolve_body_atom_with_facts(
                    eng, step_atoms, step_remaining, cap_ok, excluded_queries)
            else:
                step_resolved, step_ok_out = cascade_step_chunked(
                    eng, cap_states, cap_ok, excluded_queries, K_cap)

            G_step = step_resolved.shape[2]
            step_resolved, step_ok_out = _sort_slice_valid(step_resolved, step_ok_out, eng.K)

            if G_step < G:
                step_resolved = torch.nn.functional.pad(
                    step_resolved, (0, 0, 0, G - G_step), value=eng.padding_idx)

            all_states.append(step_resolved)
            all_ok.append(step_ok_out)

            cur_states = step_resolved
            cur_ok = step_ok_out

        return torch.cat(list(reversed(all_states)), dim=1), \
               torch.cat(list(reversed(all_ok)), dim=1)

    # Body-order-agnostic: try each body atom position against facts
    all_resolved = []
    all_ok = []
    for pos in range(M):
        atoms_at_pos = rule_states[:, :, pos, :]
        if pos == 0:
            rem = rule_states[:, :, 1:, :]
        elif pos == M - 1:
            rem = rule_states[:, :, :pos, :]
        else:
            rem = torch.cat([rule_states[:, :, :pos, :],
                             rule_states[:, :, pos+1:, :]], dim=2)

        res_states, res_ok = resolve_body_atom_with_facts(
            eng, atoms_at_pos, rem, rule_success, excluded_queries)
        all_resolved.append(res_states)
        all_ok.append(res_ok)

    resolved_states = torch.cat(all_resolved, dim=1)
    resolved_ok = torch.cat(all_ok, dim=1)
    return resolved_states, resolved_ok
