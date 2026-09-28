"""SLD resolution: backward chaining over proof states, conjunctions of goals whose variables are shared.

    sld = SLD(kb, depth=3)
    proofs = sld.prove(queries)                                   # [Q, W] -> the completed proofs of each query
    children, count, next_var, rule = sld.derive(states, next_var)   # one step, from each state [B, A, W]

A step selects each state's leftmost goal and resolves it against the facts and the rule heads in parallel: a fact
child is the other goals under the fact's bindings, a rule child the rule's body (its variables renamed apart) then the
other goals, under the head's bindings; the goals that are ground facts drop out. ``derive`` packs each state's
children into ``G`` slots and renumbers their variables (the RL environment's step; LOGIC2RL's engine, reproduced).
``prove`` packs each query's children into ``S`` states, keeps each state's proof trail and collects the states left
with no goal and a ground trail — at most ``P`` distinct proofs per query — then, with ``prune="fp_batch"``, keeps the
proofs whose every step is proved (``state.prove_mask``). The budgets: ``K_f`` fact and ``K_r`` rule children per goal,
``K_f + K_r <= max_children`` (rules first, facts the rest, at least ``min(10, K_f)``).
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.sld.resolve import resolve_facts, resolve_rules
from grounder.sld.state import Proofs, Trail, compact, drop_facts, harvest, pack, pack_children, prove_mask, rename


class SLD(nn.Module):
    """SLD resolution over ``kb`` (see the module docstring). ``max_atoms`` (L) bounds a state's goals (default
    ``M + (M - 1) * depth``; a child that would not fit is dropped), ``max_states`` the states per query (``prove``) and
    the children per state (``derive``: at most ``K_f + K_r``), ``max_proofs`` the proofs per query."""

    def __init__(self, kb, *, depth: int, prune: Optional[str] = "fp_batch", drop_facts: bool = True,
                 max_states: int = 256, max_children: int = 550, max_proofs: int = 64,
                 max_atoms: Optional[int] = None, full_fact_slots: bool = False) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if prune not in ("fp_batch", None):
            raise ValueError(f"prune: 'fp_batch' or None, got {prune!r}")
        self.kb, self.depth, self.prune, self.drop_facts = kb, depth, prune, drop_facts
        K_f, K_r = kb.facts.max_lookup, kb.rules.max_per_pred
        K = min(K_f + K_r, max_children)
        least = min(10, K_f)
        if K_r > K - least:
            raise ValueError(f"K_r={K_r} leaves fewer than {least} fact slots in K={K}")
        if K_f > max(K - K_r, least):
            warnings.warn(f"K_f capped {K_f} -> {max(K - K_r, least)} (K_r={K_r}, K={K})", stacklevel=2)
            K_f = max(K - K_r, least)
        self.K_f, self.K_r, self.max_children = (K if full_fact_slots else K_f), K_r, K
        M = kb.M
        self.L = max(max_atoms if max_atoms is not None else M + (M - 1) * depth, M)
        self.S, self.G, self.P = max_states, min(max_states, self.K_f + K_r), max_proofs
        self.V = int(kb.rules.lens.max()) + 2          # prove's variable namespace per (state, rule)

    @torch.no_grad()
    def derive(self, states: Tensor, next_var: Tensor, excluded: Optional[Tensor] = None
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """One step from each state ``[B, A, W]`` (``next_var [B]``: the first free variable id; ``excluded [B, 1, W]``:
        the root query, never read as a fact) → ``(children [B, G, L, W], count [B], next_var [B], rule [B, G])``. A
        child with no goal left is a completed proof; ``count == 0``, a dead end; a fact child's rule is -1."""
        kb, pad = self.kb, self.kb.pad
        B, A, W = states.shape
        goals = torch.full((B, self.L, W), pad, dtype=torch.long, device=states.device)
        goals[:, :A] = states
        active = goals[:, 0, 0] != pad
        selected = goals[:, 0] * active.unsqueeze(-1)
        remaining = goals.clone()
        remaining[:, 0] = pad
        f_goals, f_ok, _ = resolve_facts(kb, selected, remaining, active, self.K_f, excluded)
        r_goals, r_ok, r_rule, _ = resolve_rules(kb, selected, remaining, active, self.K_r, next_var)
        children, count, rule = pack_children(f_goals, f_ok, r_goals, r_ok, r_rule, self.G, pad)
        children = compact(children, drop_facts(kb, children, excluded), pad)
        children, next_var = rename(children, next_var, kb.E, pad)
        valid = torch.arange(self.G, device=states.device) < count.unsqueeze(1)
        return torch.where(valid[..., None, None], children, pad), count, next_var, torch.where(valid, rule, 0)

    @torch.no_grad()
    def prove(self, queries: Tensor, mask: Optional[Tensor] = None, excluded: Optional[Tensor] = None) -> Proofs:
        """The completed proofs of each query ``[Q, W]`` (``mask [Q]``: the real ones), at most ``P`` per query."""
        kb, pad, E = self.kb, self.kb.pad, self.kb.E
        B, W = queries.shape
        D, M, L, dev = self.depth, kb.M, self.L, queries.device
        goals = torch.full((B, 1, L, W), pad, dtype=torch.long, device=dev)
        goals[:, 0, 0] = queries
        valid = (mask if mask is not None else torch.ones(B, dtype=torch.bool, device=dev)).unsqueeze(1)
        top = torch.full((B, 1), -1, dtype=torch.long, device=dev)
        next_var = torch.full((B,), E, dtype=torch.long, device=dev)
        trail, proofs = Trail.empty(B, D, M, W, pad, dev), Proofs.empty(B, self.P, D, M, W, pad, dev)
        for d in range(D):
            S = goals.shape[1]
            selected = goals[:, :, 0]                                            # [B, S, W]
            active = selected[..., 0] != pad
            flat = (selected * active.unsqueeze(-1)).reshape(B * S, W)
            remaining = goals.clone()
            remaining[:, :, 0] = pad
            remaining = remaining.reshape(B * S, L, W)
            live = (active & valid).reshape(B * S)
            ex = excluded.repeat_interleave(S, 0) if excluded is not None else None      # [B * S, 1, W]
            f_goals, f_ok, f_subs = resolve_facts(kb, flat, remaining, live, self.K_f, ex)
            base = (next_var.unsqueeze(1) + torch.arange(S, device=dev) * self.V).reshape(B * S)
            r_goals, r_ok, r_rule, r_subs = resolve_rules(kb, flat, remaining, live, self.K_r, base)
            view = lambda t: t.view(B, S, *t.shape[1:])                          # noqa: E731
            started = (trail.count.sum(-1) == 0) & (top != -1)
            packed = pack((view(f_goals), view(f_ok), view(f_subs)), (view(r_goals), view(r_ok), view(r_rule),
                          view(r_subs)), top, started, self.S, M, pad)
            next_var = next_var + self.S * self.V
            keep = drop_facts(kb, packed.goals, excluded) if self.drop_facts else packed.goals[..., 0] != pad
            goals = compact(packed.goals, keep, pad)
            trail = trail.advance(packed, selected, d, pad)
            proofs, valid = harvest(proofs, trail, goals, packed.valid, E, pad)
            top = packed.top
        if self.prune == "fp_batch":
            proofs = proofs._replace(mask=prove_mask(kb, proofs, D + 1))
        return proofs

    def __repr__(self) -> str:
        return (f"SLD(depth={self.depth}, prune={self.prune!r}, K_f={self.K_f}, K_r={self.K_r}, L={self.L}, "
                f"S={self.S}, P={self.P})")


__all__ = ["SLD", "Proofs"]
