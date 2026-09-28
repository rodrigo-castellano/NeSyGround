"""SLD proof states: packing the children, dropping the goals that are facts, the proof trail, the proofs.

    pack           a query's children (of all its states) into ``S`` states: fact children first, then rule children
    pack_children  a state's children into ``G`` slots (``SLD.derive``)
    drop_facts     which goals to keep: not padding, not a ground fact
    compact        left-align the kept goals
    rename         renumber a state's variables down to ``E`` (``SLD.derive``: bounded variable ids)
    Trail          each state's proof so far: per depth, the rule, the goal it resolved, its body under the bindings
    harvest        move the states with no goals left and a ground trail into the proofs (per query, at most ``P``)
    prove_mask     fp_batch over the proofs: keep those whose every body atom is a fact or proved by a kept proof
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from grounder.ops import key
from grounder.sld.resolve import substitute
from grounder.types import Proofs


class Packed(NamedTuple):
    goals: Tensor       # [B, S, L, W]
    valid: Tensor       # [B, S]
    parent: Tensor      # [B, S] the state each came from
    subs: Tensor        # [B, S, W - 1, 2] its resolution's bindings
    rule: Tensor        # [B, S] the rule it applied (-1: a fact)
    top: Tensor         # [B, S] the first rule of its proof (-1: none yet)
    body: Tensor        # [B, S, M, W] the applied rule's body (pad: a fact child)
    new: Tensor         # [B, S] it applied a rule


def pack(fact: Tuple[Tensor, Tensor, Tensor], rule: Tuple[Tensor, Tensor, Tensor, Tensor], top: Tensor,
         started: Tensor, S: int, M: int, pad: int) -> Packed:
    """A query's children into ``S`` states: every fact child (state-major), then every rule child, the first ``S``.
    ``fact = (children [B, S_in, K_f, L, W], ok, subs)``, ``rule = (children [B, S_in, K_r, L, W], ok, rule, subs)``;
    ``top [B, S_in]`` each state's first rule; ``started [B, S_in]``: a state whose proof has begun takes no fact
    child unless it has resolved a body (its trail's body count is not 0)."""
    f_goals, f_ok, f_subs = fact
    r_goals, r_ok, r_rule, r_subs = rule
    B, S_in, K_f, L, W = f_goals.shape
    K_r = r_goals.shape[2]
    dev = f_goals.device
    f_ok = (f_ok & ~started.unsqueeze(-1)).reshape(B, S_in * K_f)
    r_ok = r_ok.reshape(B, S_in * K_r)
    cf = f_ok.long().cumsum(1)
    cr = r_ok.long().cumsum(1) + cf[:, -1:]
    trash = torch.tensor(S, device=dev)
    tf = torch.where(f_ok, cf - 1, trash).clamp(0, S)                  # slot S: the discarded ones
    tr = torch.where(r_ok, cr - 1, trash).clamp(0, S)
    count = cr[:, -1].clamp(max=S)

    def scatter(out: Tensor, target: Tensor, src: Tensor) -> Tensor:
        idx = target.view(B, -1, *([1] * (src.dim() - 2))).expand(-1, -1, *src.shape[2:])
        return out.scatter_(1, idx, src)
    goals = torch.full((B, S + 1, L, W), pad, dtype=torch.long, device=dev)
    subs = torch.full((B, S + 1, W - 1, 2), pad, dtype=torch.long, device=dev)
    parent = torch.zeros(B, S + 1, dtype=torch.long, device=dev)
    rid = torch.full((B, S + 1), -1, dtype=torch.long, device=dev)
    body = torch.full((B, S + 1, M, W), pad, dtype=torch.long, device=dev)
    new = torch.zeros(B, S + 1, dtype=torch.bool, device=dev)
    states = torch.arange(S_in, device=dev)
    scatter(goals, tr, r_goals.reshape(B, S_in * K_r, L, W))
    scatter(subs, tr, r_subs.reshape(B, S_in * K_r, W - 1, 2))
    scatter(parent, tr, states.repeat_interleave(K_r).expand(B, -1))
    scatter(rid, tr, r_rule.reshape(B, S_in * K_r))
    scatter(body, tr, r_goals[:, :, :, :M].reshape(B, S_in * K_r, M, W))
    scatter(new, tr, r_ok)
    scatter(goals, tf, f_goals.reshape(B, S_in * K_f, L, W))
    scatter(subs, tf, f_subs.reshape(B, S_in * K_f, W - 1, 2))
    scatter(parent, tf, states.repeat_interleave(K_f).expand(B, -1))
    scatter(rid, tf, torch.full((B, S_in * K_f), -1, dtype=torch.long, device=dev))
    valid = torch.arange(S, device=dev) < count.unsqueeze(1)
    parent, rid = parent[:, :S], rid[:, :S]
    top_in = top.gather(1, parent)
    top = torch.where(valid, torch.where(top_in == -1, rid, top_in), 0)
    return Packed(goals[:, :S], valid, parent, subs[:, :S], rid, top, body[:, :S], new[:, :S])


def pack_children(f_goals: Tensor, f_ok: Tensor, r_goals: Tensor, r_ok: Tensor, r_rule: Tensor, G: int,
                  pad: int) -> Tuple[Tensor, Tensor, Tensor]:
    """A state's children into ``G`` slots, facts first then rules, the first ``G``: ``(children [B, G, L, W],
    count [B], rule [B, G])`` (a fact child's rule -1, an empty slot's 0)."""
    B, K_f = f_ok.shape
    dev = r_goals.device
    trash = torch.tensor(G, device=dev)
    cf = f_ok.long().cumsum(1)
    cr = r_ok.long().cumsum(1) + cf[:, -1:]
    target = torch.cat([torch.where(r_ok, cr - 1, trash), torch.where(f_ok, cf - 1, trash)], 1).clamp(0, G)
    count = cr[:, -1].clamp(max=G)
    out = torch.full((B, G + 1, *r_goals.shape[2:]), pad, dtype=torch.long, device=dev)
    rid = torch.zeros(B, G + 1, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev).unsqueeze(1)
    out.index_put_((rows, target), torch.cat([r_goals, f_goals], 1))
    rid.index_put_((rows, target), torch.cat([r_rule, torch.full((B, K_f), -1, dtype=torch.long, device=dev)], 1))
    valid = torch.arange(G, device=dev) < count.unsqueeze(1)
    return out[:, :G], count, torch.where(valid, rid[:, :G], 0)


def drop_facts(kb, goals: Tensor, excluded: Optional[Tensor] = None) -> Tensor:
    """``[..., L, W] -> [..., L]``: the goals to keep — not padding and not a ground fact (``excluded [B, 1, W]``, the
    root query, is never one; ``goals`` ``[B, ...]``)."""
    present = goals[..., 0] != kb.pad
    fact = kb.facts.contains(goals) & (goals[..., 1:] < kb.E).all(-1)
    if excluded is not None:
        ex = excluded[:, 0].view(excluded.shape[0], *([1] * (goals.dim() - 2)), -1)
        fact = fact & ~(goals == ex).all(-1)
    return present & ~fact


def compact(goals: Tensor, keep: Tensor, pad: int) -> Tensor:
    """``goals [..., L, W]`` with the ``keep`` ones left-aligned (the rest padding)."""
    if goals.numel() == 0:
        return goals
    *lead, L, W = goals.shape
    flat, keep = goals.reshape(-1, L, W), keep.reshape(-1, L)
    target = torch.where(keep, keep.long().cumsum(1) - 1, L)
    out = flat.new_full((flat.shape[0], L + 1, W), pad)
    out.index_put_((torch.arange(flat.shape[0], device=flat.device).unsqueeze(1), target), flat)
    return out[:, :L].reshape(*lead, L, W)


def rename(states: Tensor, next_var: Tensor, E: int, pad: int) -> Tuple[Tensor, Tensor]:
    """Each row's variables ``states [B, K, L, W]`` shifted so the smallest is ``E`` (a uniform shift: distinct stay
    distinct), and ``next_var [B]`` past the largest (the next renaming apart's base)."""
    B = states.shape[0]
    if B == 0 or states.numel() == 0:
        return states, next_var
    args = states[..., 1:]
    var = (args >= E) & (args != pad)
    low = torch.where(var, args, torch.tensor(1_000_000, dtype=args.dtype, device=args.device)).amin((1, 2, 3))
    shift = torch.where(var.any((1, 2, 3)), E - low, torch.zeros_like(next_var))
    args = torch.where(var, args + shift.view(B, 1, 1, 1), args)
    out = states.clone()
    out[..., 1:] = args
    top = torch.where(var, args, torch.zeros_like(args)).amax((1, 2, 3))
    return out, torch.maximum(top + 1, torch.full_like(next_var, E))


class Trail(NamedTuple):
    """Each state's proof so far, per depth ``d``: the body its rule resolved to (``body``, under every binding since),
    the rule, the goal it resolved (``head``) and the body's length."""
    body: Tensor        # [B, S, D, M, W]
    rule: Tensor        # [B, S, D]
    head: Tensor        # [B, S, D, W]
    count: Tensor       # [B, S, D]

    @staticmethod
    def empty(B: int, D: int, M: int, W: int, pad: int, device) -> "Trail":
        return Trail(torch.full((B, 1, D, M, W), pad, dtype=torch.long, device=device),
                     torch.full((B, 1, D), -1, dtype=torch.long, device=device),
                     torch.full((B, 1, D, W), pad, dtype=torch.long, device=device),
                     torch.zeros(B, 1, D, dtype=torch.long, device=device))

    def advance(self, p: Packed, selected: Tensor, d: int, pad: int) -> "Trail":
        """The packed states' trails: their parents', under the new bindings, with depth ``d`` set where a rule was
        applied (``selected [B, S_in, W]``: each parent's resolved goal)."""
        B, S = p.parent.shape
        D, M, W = self.body.shape[2:]
        subs = p.subs.reshape(B * S, W - 1, 2)
        body = self.body.gather(1, p.parent.view(B, S, 1, 1, 1).expand(-1, -1, D, M, W))
        body = substitute(body.reshape(B * S, D * M, W), subs, pad).view(B, S, D, M, W)
        rule = self.rule.gather(1, p.parent.unsqueeze(-1).expand(-1, -1, D))
        count = self.count.gather(1, p.parent.unsqueeze(-1).expand(-1, -1, D))
        head = self.head.gather(1, p.parent.view(B, S, 1, 1).expand(-1, -1, D, W))
        head = substitute(head.reshape(B * S, D, W), subs, pad).view(B, S, D, W)
        sel = substitute(selected.gather(1, p.parent.unsqueeze(-1).expand(-1, -1, W)).reshape(B * S, 1, W), subs,
                         pad).view(B, S, W)
        new = p.new
        body, rule, head, count = body.clone(), rule.clone(), head.clone(), count.clone()
        body[:, :, d] = torch.where(new.view(B, S, 1, 1), p.body, body[:, :, d])
        rule[:, :, d] = torch.where(new, p.rule, rule[:, :, d])
        count[:, :, d] = torch.where(new, (p.body[..., 0] != pad).sum(-1), count[:, :, d])
        head[:, :, d] = torch.where(new.unsqueeze(-1), sel, head[:, :, d])
        return Trail(body, rule, head, count)


_PRIMES = (1_000_003, 999_983, 999_979, 999_961, 999_959, 999_953, 999_931)


def _distinct(body: Tensor, rule: Tensor, mask: Tensor) -> Tensor:
    """``mask`` less the repeats: two proofs with the same rules and, per depth, the same body atoms (in any order)."""
    B, N, D, M, W = body.shape
    atom = sum(body[..., j].long() * _PRIMES[j] for j in range(W))              # [B, N, D, M]
    pw = torch.tensor(_PRIMES[3], device=body.device) ** torch.arange(M - 1, -1, -1, device=body.device)
    pd = torch.tensor(_PRIMES[4], device=body.device) ** torch.arange(D - 1, -1, -1, device=body.device)
    body_h = ((atom.sort(-1)[0] * pw).sum(-1) * pd).sum(-1)
    rule_h = (rule.long() * (torch.tensor(_PRIMES[3], device=body.device)
                             ** torch.arange(D - 1, -1, -1, device=body.device))).sum(-1)
    h = torch.where(mask, rule_h * _PRIMES[0] + body_h, -1)
    sh, order = h.sort(1)
    dup = sh == torch.nn.functional.pad(sh[:, :-1], (1, 0), value=-2)
    return mask & ~dup.gather(1, order.argsort(1))


def harvest(proofs: Proofs, trail: Trail, goals: Tensor, valid: Tensor, E: int, pad: int) -> Tuple[Proofs, Tensor]:
    """``(proofs, valid)``: the states with no goals left and a ground trail added to the proofs (distinct ones; per
    query the first ``P`` slots holding one, as ``topk`` picks them), and no longer valid."""
    B, S, D, M, W = trail.body.shape
    P = proofs.mask.shape[1]
    flat = trail.body.reshape(B, S, D * M, W)
    ground = ((flat[..., 1:] < E) | (flat[..., :1] == pad)).all(-1).all(-1)
    done = (goals[..., 0] == pad).all(2) & ground & valid
    body = torch.cat([proofs.body, trail.body], 1)
    mask = _distinct(body, torch.cat([proofs.rule, trail.rule], 1), torch.cat([proofs.mask, done], 1))
    k = mask.to(torch.int8).topk(P, dim=1, largest=True, sorted=False)[1]
    pick = lambda t: t.gather(1, k.view(B, P, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:]))  # noqa: E731
    out = Proofs(pick(body), pick(torch.cat([proofs.rule, trail.rule], 1)), pick(torch.cat([proofs.head, trail.head], 1)),
                 pick(torch.cat([proofs.count, trail.count], 1)), mask.gather(1, k))
    return out, valid & ~done


def prove_mask(kb, proofs: Proofs, rounds: int) -> Tensor:
    """fp_batch over the proofs: ``mask [Q, P]`` less the proofs one of whose steps has a body atom that is not a fact
    nor the head of a proved step, after ``rounds`` rounds from the steps whose body atoms are all facts."""
    body, head, mask = proofs.body, proofs.head, proofs.mask
    B, N, D, M, W = body.shape
    V = N * D
    vbody, vhead = body.reshape(B, V, M, W), head.reshape(B, V, W)
    head_on = head[..., 0] != kb.pad
    vmask = (mask.unsqueeze(-1) & head_on).reshape(B, V)
    body_k, head_k = key(vbody, kb.base), key(vhead, kb.base)
    fact = kb.facts.contains(vbody)
    on = vbody[..., 0] != kb.pad
    proved = (fact | ~on).all(-1) & vmask
    for _ in range(rounds):
        pool = torch.where(proved, head_k, -1).reshape(-1).sort()[0]
        pos = torch.searchsorted(pool, body_k.reshape(-1)).clamp(max=pool.numel() - 1)
        found = (pool[pos] == body_k.reshape(-1)).view(B, V, M)
        proved = (fact | found | ~on).all(-1) & vmask
    return (proved.view(B, N, D) | ~head_on).all(-1) & mask


__all__ = ["Packed", "pack", "pack_children", "drop_facts", "compact", "rename", "Trail", "harvest", "prove_mask"]
