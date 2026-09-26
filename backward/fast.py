"""The FIRINGS of a PBC backward grounder, fast: width <= 1, any depth, few host syncs.

Same output as the general engine (``backward.loop.run_backward`` + ``finalize_rule_groundings``) on the
configurations it covers — flat pbc, unguided, width <= 1, last-step width 0, dense fact blocks — and it reuses
the engine's compiled rule tables (every anchor variant), its fact index and its ``finalize`` / fp_batch prune /
query pinning, so the firing set and its canonical order are the engine's by construction.

Per chunk of queries and step, the goals (the queries, then the unknown body atoms of the previous step's kept
groundings) are deduplicated; the live (goal, rule) pairs enumerate their free variables through the fact index,
compacted after each variable; the candidate groundings are filled, tested (width, cycle, head-predicate prune) and
the kept ones recorded as firings. With width <= 1 a kept grounding has at most one unknown atom, the next step's goal. The
atom table is then compacted to the atoms the kept firings reference (the engine keeps every considered atom).
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import torch
from torch import Tensor

from grounder.backward.considered import finalize, populate_query_pool_idx
from grounder.base.types import Layout, RuleGroundings
from grounder.filters.fp_batch import prune_rule_groundings


def supports(g) -> bool:
    """Whether :func:`ground` covers the BackwardGrounder ``g``'s configuration."""
    fi = g.kb.fact_index
    return (g.resolution == "pbc" and g._exec_layout is Layout.FLAT and g.guided_topk is None
            and g.guided_stats is None and g.guided_query_topk is None and g.guided_query_depth is None
            and g.width is not None and g.width <= 1 and g.w_last_depth == 0 and not g._cartesian_product
            and getattr(fi, "_use_dense", False) and g.filter_mode in ("fp_batch", "none"))


def _step(g, goals: Tensor, width: int, head_pred_mask: Optional[Tensor], want_next: bool):
    """One step over ``goals`` ``[N, 3]``: the kept groundings as (rule, head, body) and, with ``want_next``, their
    unknown body atoms."""
    fi, pad, M, dev = g.kb.fact_index, g.kb.padding_idx, g.kb.M, goals.device
    n, r = torch.nonzero(g.pred_rule_mask[goals[:, 0]], as_tuple=True)       # the live (goal, rule) pairs
    rule = g.pred_rule_indices[goals[n, 0], r]
    src = goals[n, 1:]                                                          # [T, 2 + free vars]
    no_free = ~g.has_free[rule]
    for fv in range(g.V):
        if not g._fv_any_valid[fv]:
            src = torch.cat([src, src.new_zeros(src.shape[0], 1)], 1)
            continue
        bound = src.gather(1, g.fv_enum_bound_src[rule, fv].clamp(max=src.shape[1] - 1).unsqueeze(1)).squeeze(1)
        cands, cmask = fi.enumerate(g.fv_enum_pred[rule, fv], bound, g.fv_enum_direction[rule, fv])
        # a rule without free variables keeps one row; a rule not using this variable keeps every slot
        keep = torch.where(no_free.unsqueeze(1), torch.arange(cands.shape[1], device=dev) == 0,
                           cmask | ~g.fv_enum_valid[rule, fv].unsqueeze(1))
        row, slot = torch.nonzero(keep, as_tuple=True)
        n, rule, no_free = n[row], rule[row], no_free[row]
        src = torch.cat([src[row], cands[row, slot].unsqueeze(1)], 1)
    args = src.unsqueeze(1).expand(-1, M, -1).gather(-1, g.arg_source_dep[rule].clamp(max=src.shape[1] - 1))
    body = torch.cat([g.body_preds_dep[rule].unsqueeze(-1), args], -1)            # [T, M, 3]
    active = torch.arange(M, device=dev) < g.num_body_atoms[rule].unsqueeze(1)
    body = body.masked_fill(~active.unsqueeze(-1), pad)
    exists = fi.exists(body.reshape(-1, 3)).view(-1, M)
    unknown = active & ~exists
    goal = goals[n]
    keep = ((unknown.sum(-1) <= width) & active.any(-1)
            & ~((body == goal.unsqueeze(1)).all(-1) & active).any(-1))            # no body atom is the goal
    if head_pred_mask is not None:                                              # an unknown atom must be provable
        P = head_pred_mask.shape[0]
        keep &= (exists | head_pred_mask[body[..., 0].clamp(max=P - 1)] | ~active).all(-1)
    if width == 0:
        keep &= (exists | ~active).all(-1)
    kept = torch.nonzero(keep, as_tuple=True)[0]
    nxt = None
    if want_next:
        t, m = torch.nonzero(unknown[kept], as_tuple=True)
        nxt = body[kept[t], m]
    return g._variant_to_orig_t[rule[kept]], goal[kept], body[kept], nxt


def _unique_atoms(atoms: Tensor, base: int) -> Tensor:
    key = torch.unique((atoms[:, 0] * base + atoms[:, 1]) * base + atoms[:, 2])
    return torch.stack([key // (base * base), key // base % base, key % base], 1)


def compact_atoms(rg: RuleGroundings) -> RuleGroundings:
    """``rg`` with only the atoms its firings reference, renumbered in order (order-preserving: every per-atom
    reduction a reasoner runs over it is unchanged)."""
    if rg.atom_table.shape[0] == 0:
        return rg
    used = torch.zeros(rg.atom_table.shape[0], dtype=torch.bool, device=rg.atom_table.device)
    used[rg.head_pool_idx] = True
    used[rg.body_pool_idx.reshape(-1)] = True
    new = torch.cumsum(used, 0) - 1
    table = rg.atom_table[used]
    return replace(rg, atom_table=table, num_atoms=int(table.shape[0]), head_pool_idx=new[rg.head_pool_idx],
                   body_pool_idx=new[rg.body_pool_idx])


@torch.no_grad()
def ground(g, queries: Tensor, query_mask: Tensor, chunk_size: Optional[int] = None) -> RuleGroundings:
    """The FIRINGS ``RuleGroundings`` of ``queries`` ``[B, 3]`` (``query_pool_idx`` set, atom table compacted);
    the queries are grounded ``chunk_size`` at a time, every depth per chunk (``None`` / ``<= 0``: all at once)."""
    kb, pad, dev = g.kb, g.kb.padding_idx, queries.device
    qs = queries.long()
    live = qs[query_mask & (qs[:, 0] != pad)]
    base = max(pad, kb.constant_no, int(g._P)) + 2
    rules, heads, bodies = [], [], []
    step = chunk_size if chunk_size and chunk_size > 0 else max(live.shape[0], 1)
    for start in range(0, live.shape[0], step):
        goals = live[start:start + step]
        for d in range(g.depth):
            if goals.shape[0] == 0:
                break
            last = d == g.depth - 1
            r, h, b, goals = _step(g, _unique_atoms(goals, base), g.w_last_depth if last else g.width,
                                   None if last else g.head_pred_mask, not last)
            rules.append(r)
            heads.append(h)
            bodies.append(b)
            if goals is None:
                break
    rg = finalize(SimpleNamespace(kb=kb), SimpleNamespace(rule_idx=rules, head=heads, body=bodies)) if rules else None
    if rg is not None and g.filter_mode == "fp_batch":
        rg = prune_rule_groundings(rg, facts_idx=kb.fact_index.facts_idx, depth=g.depth, padding_idx=pad)
    if rg is None:
        rg = RuleGroundings.empty(num_rules=kb.num_rules, M=kb.M, device=dev)
    return populate_query_pool_idx(compact_atoms(rg), queries, pad)


__all__ = ["supports", "ground", "compact_atoms"]
