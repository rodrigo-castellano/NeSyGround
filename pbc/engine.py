"""PBC's grounding of pools of queries: the steps, the prune, the canonical output.

Per pool and step, the goals (the queries, then the unknown body atoms of the last step's kept groundings) are made
unique; the live (goal, variant) pairs enumerate their free variables through the fact lookups. The last free variable
is enumerated in a fused kernel (``kernels.last_stage``) that also tests every candidate grounding (width, cycle,
head-predicate provability; existence through a fact hash set) and writes only the kept ones. A kept grounding's
unknown atoms (at most the width) are the next step's goals; a (goal, variant) row none of whose candidates can be kept
is not walked (``kernels.live_count``). With fp_batch, the step before the last keeps only the groundings whose unknown
atom can still be proved (a fact, an earlier step's goal, or a goal some rule could ground at the last step:
``kernels.live_atoms``). With keras on a large pool, the last step writes only the groundings whose unknown atoms can be
proved at all (``_keras_last_step``). The groundings are then filtered by the rules' variable bindings and pruned:
fp_batch to the provable ones (``depth`` rounds of propagation from the facts), keras first to those whose unknown atom
can be proved at all (``_keras_provable``), then by keras-ns's proof walk (``_keras_proved``). They are made canonical
(unique atoms and groundings, sorted) and each pool's queries are added to its atoms.

Every atom carries its pool as the most significant digit of its key, so pools never share an atom: a pool's output is
what grounding it alone gives.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grounder.ops import decode, key, unique_rows
from grounder.pbc.guide import select
from grounder.pbc.kernels import HashSet, expand, last_stage, live_atoms
from grounder.types import Groundings


class FactTable:
    """The fact lookups as one CSR over slots ``side * P * B + pred * B + value`` (side 0: by subject, the objects;
    side 1: by object, the subjects; ``B`` the key base), and a hash set of the facts. ``every`` (the Full grounder):
    a free variable ranges over every entity ``0 .. n_entities - 1`` instead (the slot counts, which the tests of
    unknown atoms read, stay the facts')."""

    def __init__(self, facts, every: bool = False, n_entities: int = 0) -> None:
        F, self.P, self.E = len(facts), facts.P, facts.base
        self.every, self.n_entities = every, n_entities
        self.values = torch.cat([facts.atoms[facts.order[0], 2], facts.atoms[facts.order[1], 1],
                                 torch.arange(n_entities if every else 0, device=facts.atoms.device)])
        self.start = torch.cat([facts.offsets[0][:-1], facts.offsets[1][:-1] + F])
        self.count = torch.cat([facts.offsets[0].diff(), facts.offsets[1].diff()])
        self.facts = self.seen = HashSet(facts.atoms, facts.base)
        self.every_start = 2 * F

    def slots(self, pred: Tensor, bound: Tensor, direction: Tensor, one: Tensor):
        """Each row's candidates as a CSR slice ``(start, count)``; a row with ``one`` set gets exactly one."""
        if self.every:                                   # the entities' block: values[2F : 2F + n_entities]
            return torch.full_like(pred, self.every_start), torch.where(one, 1, self.n_entities)
        ok = (pred < self.P) & (bound < self.E)
        slot = (direction != 0).long() * (self.P * self.E) + pred.clamp(max=self.P - 1) * self.E + bound.clamp(
            max=self.E - 1)
        return self.start[slot], torch.where(one, 1, torch.where(ok, self.count[slot], 0))

    def enumerate(self, pred: Tensor, bound: Tensor, direction: Tensor, one: Tensor):
        """Each row's candidates as ``(row, value)`` pairs, row-major; a row with ``one`` set gets exactly one
        (value 0)."""
        start, count = self.slots(pred, bound, direction, one)
        row = torch.repeat_interleave(count)
        j = torch.arange(row.shape[0], device=row.device) - (count.cumsum(0) - count)[row]
        value = self.values[(start[row] + j).clamp(max=self.values.shape[0] - 1)]
        return row, torch.where(one[row], 0, value)


def _cheapest_variant(g, facts: FactTable, goals: Tensor, n: Tensor, rule: Tensor):
    """At a width-0 step every body atom must be a fact, so each anchor variant of a rule keeps the same groundings (the
    canonical output merges them): keep one (goal, variant) row per goal and rule, the variant whose first free variable
    has the fewest facts to enumerate for this goal (ties: the first). Same groundings, a fraction of the rows on a hub
    (YAGO3-10: a person's facts through a country's thousands)."""
    t = g.tables
    if n.numel() == 0:
        return n, rule
    orig = t.rule[rule]
    has = t.has_free[rule] & t.fv_valid[rule, 0]
    bound = goals[n, 1:].gather(1, t.fv_src[rule, 0].clamp(max=1).unsqueeze(1)).squeeze(1)
    _, count = facts.slots(t.fv_pred[rule, 0], bound, t.fv_dir[rule, 0], ~has)
    # a rule's variants are consecutive (rule-major), so a (goal, rule)'s rows are one run: no sort needed
    k = n * (len(t.rule) + 1) + orig
    run = torch.cat([k.new_zeros(1), (k[1:] != k[:-1]).long()]).cumsum(0)
    best = torch.full((int(run[-1]) + 1,), torch.iinfo(count.dtype).max, dtype=count.dtype,
                      device=count.device).scatter_reduce(0, run, count, "amin")
    row = torch.arange(n.shape[0], device=n.device)
    first = torch.full_like(best, row.shape[0]).scatter_reduce(
        0, run, torch.where(count == best[run], row, row.shape[0]), "amin")
    keep = row == first[run]
    return n[keep], rule[keep]


def _candidates(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor]):
    """The rows of a step over ``goals`` ``[N, 3]`` whose last free variable ``_step`` walks: ``(src, variant, n,
    start, count, one)``, the free variables but the last enumerated."""
    t, V = g.tables, g.tables.V
    n, r = torch.nonzero(t.by_pred_mask[goals[:, 0]], as_tuple=True)        # the live (goal, variant) pairs
    rule = t.by_pred[goals[n, 0], r]
    if width == 0 and g.prune != "keras":
        n, rule = _cheapest_variant(g, facts, goals, n, rule)
    src = goals[n, 1:]                                                         # [T, 2 + bound free vars]
    no_free = ~t.has_free[rule]
    enumerated = [fv for fv in range(V) if t.fv_used[fv]]
    last = enumerated[-1] if enumerated else V
    arg_src = t.arg_src.clamp(max=1 + V)
    for fv in range(last):
        if not t.fv_used[fv]:
            src = torch.cat([src, src.new_zeros(src.shape[0], 1)], 1)
            continue
        bound = src.gather(1, t.fv_src[rule, fv].clamp(max=src.shape[1] - 1).unsqueeze(1)).squeeze(1)
        one = no_free | ~t.fv_valid[rule, fv]            # a variant without it keeps its row once
        if fv == last - 1 and not facts.every:  # the variable before the last: one kernel, with the last one's slices
            start, count = facts.slots(t.fv_pred[rule, fv], bound, t.fv_dir[rule, fv], one)
            src, rule, n, start, count, one = expand(
                src, rule, n, start, count, one, facts.values, t.fv_src[:, last], t.fv_pred[:, last],
                t.fv_dir[:, last], ~t.has_free | ~t.fv_valid[:, last], facts.start, facts.count, facts.P, facts.E,
                arg_src, t.body_pred, t.n_body, g.enumerated, heads, facts.seen, width)
            break
        row, value = facts.enumerate(t.fv_pred[rule, fv], bound, t.fv_dir[rule, fv], one)
        n, rule, no_free = n[row], rule[row], no_free[row]
        src = torch.cat([src[row], value.unsqueeze(1)], 1)
    else:
        if last < V:
            bound = src.gather(1, t.fv_src[rule, last].clamp(max=src.shape[1] - 1).unsqueeze(1)).squeeze(1)
            one = no_free | ~t.fv_valid[rule, last]
            start, count = facts.slots(t.fv_pred[rule, last], bound, t.fv_dir[rule, last], one)
        else:
            start, one = torch.zeros_like(rule), torch.ones_like(rule, dtype=torch.bool)
            count = one.long()
    return src, rule, n, start, count, one


def _step(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor], want_next: bool,
          cand=None, live=None, pool: Optional[Tensor] = None):
    """One step over ``goals`` ``[N, 3]``: the kept groundings as (rule, goal row, body) and, with ``want_next``,
    their unknown body atoms as (kept grounding, body slot). The free variables but the last are enumerated first
    (``_candidates``, or ``cand``); the last one, with every test of the groundings, in ``kernels.last_stage``
    (``live``: an unknown atom must also be in this key set, keyed with its goal's ``pool``)."""
    t, pad, M, V = g.tables, g.kb.pad, g.tables.M, g.tables.V
    src, rule, n, start, count, one = cand if cand is not None else _candidates(g, facts, goals, width, heads)
    arg_src = t.arg_src.clamp(max=1 + V)
    rows, vals = last_stage(src, rule, n, goals, start, count, one, facts.values, arg_src, t.body_pred, t.n_body,
                            g.enumerated, heads, facts.seen, pad, width, facts.count, facts.P, facts.E,
                            cycle="unknown" if g.prune == "keras" else "all", live_set=live, seg=pool)
    n, rule = n[rows], rule[rows]
    pad_cols = src.new_zeros(rows.shape[0], max(1 + V - src.shape[1], 0))       # (none when no rule has a free var)
    src = torch.cat([src[rows], vals.unsqueeze(1), pad_cols], 1)[:, :2 + V]
    args = src.unsqueeze(1).expand(-1, M, -1).gather(-1, arg_src[rule])
    body = torch.cat([t.body_pred[rule].unsqueeze(-1), args], -1)                 # [K, M, 3]
    active = torch.arange(M, device=goals.device) < t.n_body[rule].unsqueeze(1)
    body = body.masked_fill(~active.unsqueeze(-1), pad)
    nxt = torch.nonzero(active & ~facts.seen.contains(body), as_tuple=True) if want_next else None
    return t.rule[rule], n, body, nxt


def _live_tables(g, facts: FactTable):
    """``(ok_s, ok_o)`` ``[R', E + 1]`` int8, per variant and entity: every body atom with the head's subject
    (``ok_s``; object: ``ok_o``) and a free variable has a fact of its predicate on that side (column ``E``: an
    entity with no facts; a body atom whose predicate has none fails both); cached."""
    if g._live is None:
        t, P, E, V = g.tables, facts.P, facts.E, g.tables.V
        arg, pred = t.arg_src.clamp(max=1 + V), t.body_pred                          # [R', M, 2], [R', M]
        active = torch.arange(pred.shape[1], device=pred.device) < t.n_body.unsqueeze(1)
        by_s = torch.cat([facts.count[:P * E].view(P, E) > 0, torch.zeros(P, 1, dtype=torch.bool,
                                                                          device=pred.device)], 1)   # [P, E + 1]
        by_o = torch.cat([facts.count[P * E:].view(P, E) > 0, torch.zeros(P, 1, dtype=torch.bool,
                                                                          device=pred.device)], 1)
        R = pred.shape[0]
        ok = [torch.ones(R, E + 1, dtype=torch.bool, device=pred.device) for _ in range(2)]
        for m in range(pred.shape[1]):
            a0, a1, q = arg[:, m, 0], arg[:, m, 1], pred[:, m]
            known = q < P
            qc = q.clamp(max=P - 1)
            for side, own in ((0, ok[0]), (1, ok[1])):
                subj = active[:, m] & (a0 == side) & (a1 >= 2)          # (own, free): facts with it as subject
                obj = active[:, m] & (a0 >= 2) & (a1 == side)           # (free, own): as object
                own &= ~subj.unsqueeze(1) | (by_s[qc] & known.unsqueeze(1))
                own &= ~obj.unsqueeze(1) | (by_o[qc] & known.unsqueeze(1))
        g._live = (ok[0].to(torch.int8).contiguous(), ok[1].to(torch.int8).contiguous())
    return g._live


def _live_goals(g, facts: FactTable, goals: Tensor) -> Tensor:
    """``[N]`` bool: whether some rule could ground each goal ``[N, 3]`` at the last step (width 0: every body atom a
    fact) — ``kernels.live_atoms`` on per-variant tables."""
    t = g.tables
    ok_s, ok_o = _live_tables(g, facts)
    return live_atoms(goals, t.by_pred_mask, t.by_pred, ok_s, ok_o, t.arg_src.clamp(max=1 + t.V), t.body_pred,
                      t.n_body, facts.seen)


def _provable_next(g, facts: FactTable, pool: Tensor, r: Tensor, n: Tensor, b: Tensor, nxt, prior: Tensor,
                   base: int):
    """The groundings of the step before the last, less those whose unknown atom cannot be proved, so fp_batch would
    drop them: the atom is not a fact, no rule could ground it at the last step, and it is no goal of an earlier step
    (its only groundings would be the last step's). Their unknown atoms are the last step's goals."""
    row, slot = nxt
    atoms = b[row, slot]
    k = key(atoms, base, pool[n[row]])
    ok = HashSet.of_keys(prior).contains_keys(k)
    ok |= _live_goals(g, facts, atoms)
    keep = torch.ones(r.shape[0], dtype=torch.bool, device=r.device)
    keep[row[~ok]] = False
    new = torch.cumsum(keep, 0) - 1
    return r[keep], n[keep], b[keep], (new[row[ok]], slot[ok])


KERAS_CLOSURE_MIN_ROWS = 1 << 18       # below it (training batches, Countries) writing every candidate costs less


def _keras_last_step(g, facts: FactTable, goals: Tensor, pool: Tensor, width: int, heads: Optional[Tensor],
                     atoms: List[Tensor], bodies: List[Tensor], pools: List[Tensor], base: int):
    """The keras filter's last step over ``goals`` (of ``pool``), grounding only what ``_keras_provable`` keeps: its
    groundings whose unknown atom lies in the least fixed point of the provable atoms (reached from the all-fact
    groundings of every step), found by grounding again as the set grows (Kleene iteration from the empty set).
    ``atoms`` / ``bodies`` / ``pools``: the earlier steps' groundings' heads, bodies and pools."""
    cand = _candidates(g, facts, goals, width, heads)
    if cand[1].shape[0] < KERAS_CLOSURE_MIN_ROWS:      # few candidates: writing them all costs less than the passes
        return _step(g, facts, goals, width, heads, False, cand=cand)[:3]
    head, body, bpool = torch.cat(atoms), torch.cat(bodies), torch.cat(pools)
    row, slot = torch.nonzero((body[..., 0] != g.kb.pad) & ~facts.facts.contains(body), as_tuple=True)
    ukey, hkey = key(body[row, slot], base, bpool[row]), key(head, base, bpool)
    proved = hkey.new_zeros(0)
    while True:
        live = HashSet.of_keys(proved)
        r, n, b, _ = _step(g, facts, goals, width, heads, False, cand=cand, live=live, pool=pool)
        ok = torch.ones_like(hkey, dtype=torch.bool)
        ok[row[~live.contains_keys(ukey)]] = False
        grown = torch.unique(torch.cat([hkey[ok], key(goals[n], base, pool[n])]))
        if grown.shape[0] == proved.shape[0]:
            return r, n, b
        proved = grown


def _ground_steps(g, facts: FactTable, goals: Tensor, pool: Tensor, base: int, depth: int, stats: Optional[list]):
    """Every step's kept groundings of the queries ``goals`` (of ``pool``): ``(rule [T], head [T, 3],
    body [T, M, 3], pool [T])``, possibly repeated; each step's (depth, goals, kept groundings) appended to ``stats``."""
    rules, heads_, bodies, pools = [], [], [], []
    k = key(goals, base, pool)
    prior = []                                          # each step's goals
    for d in range(depth):
        if k.shape[0] == 0:
            break
        last = d == depth - 1
        k = torch.unique(k)
        prior.append(k)
        p, atoms = decode(k, base, pool=True)
        width = g.last_width if last else g.width
        hpm = None if last and g.prune != "keras" else g.tables.heads
        if last and d > 0 and g.prune == "keras" and g.width <= 1:
            (r, n, b), nxt = _keras_last_step(g, facts, atoms, p, width, hpm, heads_, bodies, pools, base), None
        else:
            r, n, b, nxt = _step(g, facts, atoms, width, hpm, not last)
        if d == depth - 2 and g.prune == "fp_batch" and nxt[0].numel():
            r, n, b, nxt = _provable_next(g, facts, p, r, n, b, nxt, torch.cat(prior), base)
        if g.guide is not None and r.numel():            # the guide's selection, then the next goals of what it kept
            fact = facts.facts.contains(b)
            keep = select(g.guide, fact, r, n, b, p[n], d, g.kb.pad)
            r, n, b, fact = r[keep], n[keep], b[keep], fact[keep]
            if nxt is not None:
                nxt = torch.nonzero((b[..., 0] != g.kb.pad) & ~fact, as_tuple=True)
        if stats is not None:
            stats.append((d, int(atoms.shape[0]), int(r.shape[0])))
        rules.append(r)
        heads_.append(atoms[n])
        bodies.append(b)
        pools.append(p[n])
        if nxt is None:
            break
        k = key(b[nxt[0], nxt[1]], base, pools[-1][nxt[0]])
    if not rules:
        z = goals.new_zeros(0)
        return z, goals.new_zeros(0, 3), goals.new_zeros(0, g.tables.M, 3), z
    return torch.cat(rules), torch.cat(heads_), torch.cat(bodies), torch.cat(pools)


def _keras_proved(g, facts: FactTable, rule: Tensor, head: Tensor, body: Tensor, pool: Tensor, keep: Tensor,
                  qkey: Tensor, base: int, depth: int) -> Tensor:
    """``[T]``: whether each raw grounding survives the pruning of keras-ns's ``ApproximateBackwardChainingGrounder``
    (the IJCAI-25 code's BC_{w,d}): every body atom a fact or proved.

    Its goals: the queries, then at each step the non-fact atoms of the groundings so far, less, for a grounding's own
    rule, the goals already of that rule's head predicate (a query can be a goal again). Every grounding of a goal is a
    proof of it, its unknown atoms the proof's atoms (a rule with one body atom records none). After the last step,
    ``depth - 1`` rounds (``g.rounds`` if set: the later keras-ns walks ``depth``) walk the proofs in the rules' input
    order, a rule's step-1 goals' proofs before its step-2 goals', ..., each proving its head if its atoms are proved by
    then. Keras walks a (rule, step) block's goals in the order of a Python set, which varies with the string hash seed
    (so does its output, a few groundings in thousands); here in atom order, one of the orders it can take. An atom's
    earliest proof (its position in the walk) is found by propagation to a fixed point."""
    if rule.numel() == 0:
        return keep
    pad, R = g.kb.pad, len(g.kb.rules)
    rc = rule.clamp(min=0, max=R - 1)
    valid = body[..., 0] != pad
    unknown = valid & ~facts.facts.contains(body)                                        # [T, M]
    keys, inv = torch.unique(torch.cat([key(head, base, pool).unsqueeze(1), key(body, base, pool.unsqueeze(1))], 1),
                             return_inverse=True)
    hid, bid, n = inv[:, 0], inv[:, 1:], keys.shape[0]
    at = torch.searchsorted(keys, qkey).clamp(max=max(n - 1, 0))
    goals = [torch.zeros(n, dtype=torch.bool, device=rule.device)]
    goals[0][at[keys[at] == qkey]] = True                                                # step 1: the queries
    seen = goals[0].clone()
    for _ in range(depth - 1):                     # step s + 1: the non-fact atoms of the groundings so far, less ...
        found = keep & seen[hid]
        new = unknown & found.unsqueeze(1) & ~(seen[bid] & (body[..., 0] == head[:, :1]))   # ... their rule's own
        nxt = torch.zeros_like(seen)
        nxt[bid[new]] = True
        goals.append(nxt)
        seen |= nxt
    rounds = g.rounds or depth - 1
    if rounds < 1:              # (IJCAI-25's depth 1: no walk, so every body atom a fact)
        return (~unknown).all(1)
    proof = keep & (g.kb.rules.lens[rc] >= 2)
    order = g.kb.rules.order[rc]
    walk, never = R * depth * n, torch.iinfo(torch.long).max // 4
    t = torch.full((n,), never, dtype=torch.long, device=rule.device)          # each atom's earliest proof
    while True:
        latest = torch.where(unknown, t[bid], -1).amax(1)                               # its atoms' latest proof
        best = torch.full_like(latest, never)
        for s in range(depth):
            at = (order * depth + s) * n + hid                  # the walk: (rule, step) blocks, a block's goals in order
            k = ((latest + 1 - at).clamp(min=0) + walk - 1) // walk                     # the first round after them
            ok = proof & goals[s][hid] & (k < rounds)
            best = torch.where(ok, torch.minimum(best, k * walk + at), best)
        t_next = t.scatter_reduce(0, hid, best, reduce="amin")
        if torch.equal(t_next, t):
            break
        t = t_next
    return (~unknown | (t[bid] < never)).all(1)


KERAS_PREFILTER_ROWS = 1 << 20      # below it the walk's sorts cost less than the fixed point's rounds (a sync each)


def _keras_provable(g, facts: FactTable, rule: Tensor, head: Tensor, body: Tensor, pool: Tensor, base: int):
    """The raw groundings less those with an unknown atom that no chain of groundings from the all-fact ones reaches:
    keras-ns proves an atom only by a grounding of it whose unknown atoms are proved before, so what it proves lies in
    that least fixed point, and ``_keras_proved`` never keeps a grounding outside it; it then walks thousands of
    groundings instead of millions. Exact with at most one unknown atom per grounding (width <= 1; else nothing is
    dropped); the atoms left keep their order, so the walk's positions compare the same."""
    unknown = (body[..., 0] != g.kb.pad) & ~facts.facts.contains(body)          # [T, M]
    if bool((unknown.sum(1) > 1).any()):
        return rule, head, body, pool
    row, slot = torch.nonzero(unknown, as_tuple=True)
    ukey, hkey = key(body[row, slot], base, pool[row]), key(head, base, pool)
    ok = ~unknown.any(1)
    while True:
        ok_next = torch.ones_like(ok)
        ok_next[row[~HashSet.of_keys(hkey[ok]).contains_keys(ukey)]] = False
        if torch.equal(ok_next, ok):
            return rule[ok], head[ok], body[ok], pool[ok]
        ok = ok_next


def _kept(g, facts: FactTable, rule: Tensor, head: Tensor, body: Tensor, pool: Tensor, base: int, depth: int,
          qkey: Tensor):
    """The groundings ``(rule, head [T, 3], body [T, M, 3], pool)`` whose atoms fit their rule's variable bindings (a
    repeated variable binds one constant) and, with fp_batch, whose body is proved within ``depth`` rounds of
    propagation from the facts (an atom is proved by a fact, or as the head of a grounding whose body was proved the
    round before); with keras, as the keras-ns grounder prunes them (``_keras_proved``; ``qkey``: the queries) —
    before making them canonical, on the raw groundings (repeats keep or drop together)."""
    t, pad, M, R = g.tables, g.kb.pad, g.tables.M, len(g.kb.rules)
    if g.prune == "keras" and depth >= 2 and rule.shape[0] >= KERAS_PREFILTER_ROWS:
        rule, head, body, pool = _keras_provable(g, facts, rule, head, body, pool, base)
    rc = rule.clamp(min=0, max=R - 1)
    ent = torch.cat([head.unsqueeze(1), body], 1)[..., 1:].reshape(-1, 2 * (M + 1))
    keep = ((rule >= 0) & (rule < R) & (head[:, 0] == t.bind_head[rc]) & (body[..., 0] == t.bind_body[rc]).all(1)
            & ((ent == ent.gather(1, t.bind_first[rc])) | ~t.bind_on[rc]).all(1))
    if g.prune == "fp_batch":
        fact = facts.facts.contains(body) | (body[..., 0] == pad)                  # [T, M]
        head_key, body_key = key(head, base, pool), key(body, base, pool.unsqueeze(1))
        proved = fact
        for _ in range(max(1, depth)):
            proved = fact | HashSet.of_keys(head_key[proved.all(1) & keep]).contains_keys(body_key)
        keep &= proved.all(1)
    elif g.prune == "keras":
        keep &= _keras_proved(g, facts, rule, head, body, pool, keep, qkey, base, depth)
    return rule[keep], head[keep], body[keep], pool[keep]


@torch.no_grad()
def ground(g, queries: Tensor, mask: Tensor, depth: int, stats: Optional[list] = None) -> Groundings:
    """The groundings of the pools ``queries [S, Q, 3]`` (``mask [S, Q]``: the real queries), ``depth`` steps."""
    kb, pad, dev = g.kb, g.kb.pad, queries.device
    S, Q, M, R = queries.shape[0], queries.shape[1], g.tables.M, len(kb.rules)
    base = kb.base
    fit = max(((1 << 62) - 1) // base ** 3, 1)
    if S > fit:                                                     # the pool digit would overflow the key
        return concat([ground(g, queries[s:s + fit], mask[s:s + fit], depth, stats) for s in range(0, S, fit)])
    facts = g.facts()
    qs = queries.long().reshape(-1, 3)
    qpool = torch.arange(S, device=dev).repeat_interleave(Q)
    live = mask.reshape(-1) & (qs[:, 0] != pad)
    rule, head, body, pool = _ground_steps(g, facts, qs[live], qpool[live], base, depth, stats)
    qkey = torch.unique(key(qs[live], base, qpool[live]))
    rule, head, body, pool = _kept(g, facts, rule, head, body, pool, base, depth, qkey)

    # canonical: unique atoms (sorted by pool, then key), unique groundings (sorted by pool, rule, atoms)
    akey, ainv = torch.unique(key(torch.cat([head.unsqueeze(1), body], 1), base, pool.unsqueeze(1)),
                              return_inverse=True)                                  # ainv [T, 1 + M]
    apool, table = decode(akey, base, pool=True)
    start = torch.searchsorted(apool, torch.arange(S + 1, device=dev))              # each pool's first atom
    local = ainv - start[pool].unsqueeze(1)
    A = max(int(start.diff().max()), 1) if S > 1 else max(int(akey.shape[0]), 1)     # atoms per pool
    fpool, rule, *local = unique_rows([pool, rule, *local.unbind(1)], [S, R] + [A] * (M + 1))
    idx = torch.stack(local, 1)                                                     # [N, 1 + M] rows local to pools

    # each pool's queries that are none of its atoms, after them
    qk = key(qs, base, qpool)
    at = torch.searchsorted(akey, qk).clamp(max=max(akey.numel() - 1, 0))
    in_pool = (akey[at] == qk) if akey.numel() else torch.zeros_like(qk, dtype=torch.bool)
    nkey = torch.unique(qk[~in_pool])
    npool, ntable = decode(nkey, base, pool=True)
    nstart = torch.searchsorted(npool, torch.arange(S + 1, device=dev))
    n_a, n_n = start.diff(), nstart.diff()
    atom_offsets = torch.zeros(S + 1, dtype=torch.long, device=dev)
    atom_offsets[1:] = (n_a + n_n).cumsum(0)
    arow = torch.arange(akey.shape[0], device=dev)
    nrow = torch.arange(nkey.shape[0], device=dev)
    atoms = torch.empty(akey.shape[0] + nkey.shape[0], 3, dtype=torch.long, device=dev)
    atoms[arow - start[apool] + atom_offsets[apool]] = table
    atoms[n_a[npool] + nrow - nstart[npool] + atom_offsets[npool]] = ntable
    query = torch.where(in_pool, at - start[qpool], n_a[qpool] + torch.searchsorted(nkey, qk) - nstart[qpool])
    query = (query + atom_offsets[qpool]).view(S, Q)
    rows = idx + atom_offsets[fpool].unsqueeze(1)
    offsets = torch.zeros(S, R + 1, dtype=torch.long, device=dev)
    offsets[:, 1:] = torch.bincount(fpool * R + rule, minlength=S * R).view(S, R).cumsum(1)
    offsets = offsets + torch.cat([offsets.new_zeros(1), offsets[:-1, -1].cumsum(0)]).unsqueeze(1)
    return Groundings(atoms, rule, rows[:, 0], rows[:, 1:], table[idx[:, 1:] + start[fpool].unsqueeze(1), 0] != pad,
                      offsets, atom_offsets, query)


def concat(parts: List[Groundings]) -> Groundings:
    """The pools of ``parts``, in order, as one ``Groundings``."""
    atoms_at = torch.cumsum(torch.tensor([0] + [len(p.atoms) for p in parts[:-1]]), 0).tolist()
    rows_at = torch.cumsum(torch.tensor([0] + [len(p.rule) for p in parts[:-1]]), 0).tolist()
    return Groundings(
        torch.cat([p.atoms for p in parts]), torch.cat([p.rule for p in parts]),
        torch.cat([p.head + a for p, a in zip(parts, atoms_at)]), torch.cat([p.body + a for p, a in zip(parts, atoms_at)]),
        torch.cat([p.body_mask for p in parts]), torch.cat([p.offsets + n for p, n in zip(parts, rows_at)]),
        torch.cat([parts[0].atom_offsets[:1]] + [p.atom_offsets[1:] + a for p, a in zip(parts, atoms_at)]),
        torch.cat([p.query + a for p, a in zip(parts, atoms_at)]))


__all__ = ["FactTable", "ground", "concat"]
