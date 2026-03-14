"""High-level unification operations: query-vs-facts and query-vs-rules.

Also includes MGU-based resolution building blocks that combine index lookup
with unification.  These are used by strategy modules (Prolog, RTF).

Depends on ``primitives`` (apply_substitutions, unify_one_to_one).
"""

from __future__ import annotations
from typing import Optional, Protocol, Tuple

import torch
from torch import Tensor

from grounder.primitives import apply_substitutions, unify_one_to_one


# ---------------------------------------------------------------------------
# Protocol types for index objects (documented interface contracts)
# ---------------------------------------------------------------------------

class FactIndex(Protocol):
    """Protocol for fact index objects used by mgu_resolve_atom_facts.

    Must provide:
        targeted_lookup(query_atoms, max_results) -> (fact_idx [B, K], valid [B, K])
    """
    def targeted_lookup(
        self, query_atoms: Tensor, max_results: int,
    ) -> Tuple[Tensor, Tensor]: ...


class RuleIndex(Protocol):
    """Protocol for rule index objects used by mgu_resolve_atom_rules.

    Must provide:
        lookup_by_segments(query_preds, max_pairs, device)
            -> (item_idx [B, K], valid_mask [B, K], query_idx [B, K])
        rules_heads:  Tensor  [R, 3]
        rules_bodies: Tensor  [R, Bmax, 3]
        rule_lens:    Tensor  [R]
    """
    rules_heads: Tensor
    rules_bodies: Tensor
    rule_lens: Tensor

    def lookup_by_segments(
        self, query_preds: Tensor, max_pairs: int, device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...


# ---------------------------------------------------------------------------
# Core unification operations (copied from BE _operations.py)
# ---------------------------------------------------------------------------

def unify_with_facts(
    queries: Tensor,                # [B, 3] query atoms
    remaining: Tensor,              # [B, G, 3] remaining atoms
    remaining_counts: Tensor,       # [B] valid remaining count
    item_idx: Tensor,               # [B, max_pairs] fact indices
    valid_mask: Tensor,             # [B, max_pairs] valid pairs
    facts: Tensor,                  # [F, 3] all facts
    constant_no: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Vectorized fact unification with fixed output shape.

    Returns:
        derived_states: [B, max_pairs, G, 3] successor states
        success_mask: [B, max_pairs] which unifications succeeded
        subs: [B, max_pairs, 2, 2] substitutions applied
    """
    B, max_pairs = item_idx.shape
    G = remaining.shape[1]
    device = queries.device
    pad = padding_idx

    if B == 0 or facts.numel() == 0:
        return (
            torch.full((B, max_pairs, G, 3), pad, dtype=torch.long, device=device),
            torch.zeros((B, max_pairs), dtype=torch.bool, device=device),
            torch.full((B, max_pairs, 2, 2), pad, dtype=torch.long, device=device),
        )

    # Clamp indices to valid range (mask handles invalids)
    safe_idx = item_idx.clamp(0, facts.shape[0] - 1)  # [B, K]

    # Gather facts for all pairs
    fact_atoms = facts[safe_idx.view(-1)].view(B, max_pairs, 3)  # [B*K] -> [B, K, 3]

    # Expand queries for pairwise unification
    q_expanded = queries.unsqueeze(1).expand(-1, max_pairs, -1)  # [B, 3] -> [B, K, 3]

    # Flatten for unify_one_to_one
    flat_q = q_expanded.reshape(-1, 3)  # [B*K, 3]
    flat_f = fact_atoms.reshape(-1, 3)  # [B*K, 3]

    ok_flat, subs_flat = unify_one_to_one(flat_q, flat_f, constant_no, pad)
    # ok_flat: [B*K], subs_flat: [B*K, 2, 2]

    ok = ok_flat.view(B, max_pairs)           # [B, K]
    subs = subs_flat.view(B, max_pairs, 2, 2)  # [B, K, 2, 2]

    success_mask = ok & valid_mask

    # Apply substitutions to remaining atoms
    rem_flat = remaining.unsqueeze(1).expand(-1, max_pairs, -1, -1).reshape(B * max_pairs, G, 3)
    subs_for_apply = subs.reshape(B * max_pairs, 2, 2)  # [B*K, 2, 2]

    rem_subst = apply_substitutions(rem_flat, subs_for_apply, pad)  # [B*K, G, 3]
    derived_states = rem_subst.view(B, max_pairs, G, 3)  # [B, K, G, 3]

    # Zero out invalid entries (compile-friendly, no boolean fancy indexing)
    pad_t = torch.tensor(pad, dtype=derived_states.dtype, device=derived_states.device)
    derived_states = torch.where(
        success_mask.unsqueeze(-1).unsqueeze(-1),
        derived_states,
        pad_t,
    )

    return derived_states, success_mask, subs


def unify_with_rules(
    queries: Tensor,                    # [B, 3]
    remaining: Tensor,                  # [B, G, 3]
    remaining_counts: Tensor,           # [B]
    item_idx: Tensor,                   # [B, max_pairs] rule indices
    valid_mask: Tensor,                 # [B, max_pairs]
    rules_heads: Tensor,                # [R, 3]
    rules_bodies: Tensor,              # [R, Bmax, 3]
    rule_lens: Tensor,                  # [R]
    next_var_indices: Tensor,           # [B]
    constant_no: int,
    padding_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Vectorized rule unification with fixed output shape.

    Includes standardization apart: template variables in rules are renamed
    to fresh runtime variables starting at next_var_indices.

    Returns:
        derived_states: [B, max_pairs, M, 3] where M = Bmax + G
        success_mask: [B, max_pairs]
        subs: [B, max_pairs, 2, 2]
        body_lens: [B, max_pairs] length of rule body for each pair
    """
    B, max_pairs = item_idx.shape
    G = remaining.shape[1]
    device = queries.device
    pad = padding_idx

    if B == 0 or rules_heads.numel() == 0:
        M = G + 1
        empty_states = torch.full((B, max_pairs, M, 3), pad, dtype=torch.long, device=device)
        return (
            empty_states,
            torch.zeros((B, max_pairs), dtype=torch.bool, device=device),
            torch.full((B, max_pairs, 2, 2), pad, dtype=torch.long, device=device),
            torch.zeros((B, max_pairs), dtype=torch.long, device=device),
        )

    Bmax = rules_bodies.shape[1]
    M = Bmax + G

    # Clamp indices
    safe_idx = item_idx.clamp(0, rules_heads.shape[0] - 1)  # [B, K]

    # Gather rule heads
    rule_heads_sel = rules_heads[safe_idx.view(-1)].view(B, max_pairs, 3)

    # Gather rule bodies
    rule_bodies_sel = rules_bodies[safe_idx.view(-1)].view(B, max_pairs, Bmax, 3)

    # Gather rule lengths
    rule_lens_sel = rule_lens[safe_idx.view(-1)].view(B, max_pairs)

    # -------------------------------------------------------------------------
    # Standardization Apart: Rename template variables to runtime variables
    # -------------------------------------------------------------------------
    template_start = constant_no + 1
    next_per_pair = next_var_indices.unsqueeze(1).expand(-1, max_pairs)  # [B] -> [B, K]

    # Rename head args
    h_args = rule_heads_sel[:, :, 1:3]  # [B, K, 2]
    is_template_h = (h_args >= template_start) & (h_args != pad)
    h_args_renamed = torch.where(
        is_template_h,
        next_per_pair.unsqueeze(-1) + (h_args - template_start),
        h_args,
    )
    heads_renamed = torch.cat([rule_heads_sel[:, :, :1], h_args_renamed], dim=-1)

    # Rename body args
    b_args = rule_bodies_sel[:, :, :, 1:3]  # [B, K, Bmax, 2]
    is_template_b = (b_args >= template_start) & (b_args != pad)
    b_args_renamed = torch.where(
        is_template_b,
        next_per_pair.unsqueeze(-1).unsqueeze(-1) + (b_args - template_start),
        b_args,
    )
    bodies_renamed = torch.cat([rule_bodies_sel[:, :, :, :1], b_args_renamed], dim=-1)

    # -------------------------------------------------------------------------
    # Unification
    # -------------------------------------------------------------------------
    q_expanded = queries.unsqueeze(1).expand(-1, max_pairs, -1)

    flat_q = q_expanded.reshape(-1, 3)   # [B*K, 3]
    flat_h = heads_renamed.reshape(-1, 3)  # [B*K, 3]

    ok_flat, subs_flat = unify_one_to_one(flat_q, flat_h, constant_no, pad)
    ok = ok_flat.view(B, max_pairs)
    subs = subs_flat.view(B, max_pairs, 2, 2)

    # -------------------------------------------------------------------------
    # Apply substitutions and combine body + remaining
    # -------------------------------------------------------------------------
    remaining_exp = remaining.unsqueeze(1).expand(-1, max_pairs, -1, -1)
    subs_flat_for_apply = subs.reshape(B * max_pairs, 2, 2)

    combined_flat = torch.cat([bodies_renamed, remaining_exp], dim=2).reshape(
        B * max_pairs, Bmax + G, 3)
    combined_subst = apply_substitutions(combined_flat, subs_flat_for_apply, pad)
    derived_states = combined_subst.view(B, max_pairs, Bmax + G, 3)

    success_mask = ok & valid_mask

    # Zero out invalid entries
    pad_t = torch.tensor(pad, dtype=derived_states.dtype, device=derived_states.device)
    derived_states = torch.where(
        success_mask.unsqueeze(-1).unsqueeze(-1),
        derived_states,
        pad_t,
    )

    return derived_states, success_mask, subs, rule_lens_sel


# ---------------------------------------------------------------------------
# MGU-based resolution building blocks (adapted from BE _resolve_mgu.py)
# ---------------------------------------------------------------------------

def mgu_resolve_atom_facts(
    queries: Tensor,              # [B, 3]
    remaining: Tensor,            # [B, R, 3]
    remaining_counts: Tensor,     # [B]
    active_mask: Tensor,          # [B] bool
    excluded_queries: Optional[Tensor],  # [B, 1, 3] or None
    facts_idx: Tensor,            # [F, 3] all facts
    fact_index: FactIndex,        # index object with targeted_lookup()
    constant_no: int,
    padding_idx: int,
    max_fact_pairs: int,
) -> Tuple[Tensor, Tensor]:
    """Single-atom fact resolution via MGU.

    targeted_lookup -> unify_with_facts -> active_mask -> cycle prevention.

    Args:
        queries: [B, 3] query atoms to resolve
        remaining: [B, R, 3] remaining goal atoms
        remaining_counts: [B] valid remaining count per batch
        active_mask: [B] bool mask for non-terminal states
        excluded_queries: [B, 1, 3] atoms to exclude from cycle prevention
        facts_idx: [F, 3] all facts tensor
        fact_index: index object providing targeted_lookup(query_atoms, max_results)
        constant_no: highest constant index
        padding_idx: padding index
        max_fact_pairs: maximum fact candidates per query

    Returns:
        fact_states: [B, K_f, G, 3] derived states from fact unification
        fact_success: [B, K_f] validity mask
    """
    B = queries.shape[0]
    pad = padding_idx

    fact_item_idx, fact_valid = fact_index.targeted_lookup(queries, max_fact_pairs)

    fact_states, fact_success, _ = unify_with_facts(
        queries, remaining, remaining_counts,
        fact_item_idx, fact_valid, facts_idx, constant_no, pad,
    )

    fact_success = fact_success & active_mask.unsqueeze(1)

    # Cycle prevention
    if excluded_queries is not None and facts_idx.numel() > 0:
        excl_first = excluded_queries[:, 0, :]  # [B, 3]
        K_f = fact_states.shape[1]
        safe_idx = fact_item_idx.clamp(0, max(facts_idx.shape[0] - 1, 0))
        matched_facts = facts_idx[safe_idx.view(-1)].view(B, K_f, 3)
        fact_success = fact_success & ~(matched_facts == excl_first.unsqueeze(1)).all(dim=-1)

    return fact_states, fact_success


def mgu_resolve_atom_rules(
    queries: Tensor,              # [B, 3]
    remaining: Tensor,            # [B, R, 3]
    remaining_counts: Tensor,     # [B]
    active_mask: Tensor,          # [B] bool
    next_var_indices: Tensor,     # [B]
    rule_index: RuleIndex,        # index object with lookup_by_segments()
    constant_no: int,
    padding_idx: int,
    max_rule_pairs: int,
) -> Tuple[Tensor, Tensor]:
    """Single-atom rule resolution via MGU.

    lookup_by_segments -> unify_with_rules -> active_mask.

    Args:
        queries: [B, 3] query atoms to resolve
        remaining: [B, R, 3] remaining goal atoms
        remaining_counts: [B] valid remaining count per batch
        active_mask: [B] bool mask for non-terminal states
        next_var_indices: [B] next available variable index per batch
        rule_index: index object providing lookup_by_segments() and rule tensors
        constant_no: highest constant index
        padding_idx: padding index
        max_rule_pairs: maximum rule candidates per query

    Returns:
        rule_states: [B, K_r, M, 3] derived states from rule unification
        rule_success: [B, K_r] validity mask
    """
    pad = padding_idx
    device = queries.device
    query_preds = queries[:, 0]

    rule_item_idx, rule_valid, _ = rule_index.lookup_by_segments(
        query_preds, max_rule_pairs,
    )
    rule_states, rule_success, _, _rule_lens = unify_with_rules(
        queries, remaining, remaining_counts,
        rule_item_idx, rule_valid,
        rule_index.rules_heads, rule_index.rules_bodies, rule_index.rule_lens,
        next_var_indices, constant_no, pad,
    )
    rule_success = rule_success & active_mask.unsqueeze(1)

    return rule_states, rule_success


