"""PBC's grounding of pools of queries: the steps, the prune, the canonical output.

Per pool and step, the goals (the queries, then the unknown body atoms of the last step's kept groundings) are made
unique; the live (goal, variant) pairs enumerate their free variables through the fact lookups. The last free variable
is enumerated in a fused kernel (``kernels.last_stage``) that also tests every candidate grounding (width, cycle,
head-predicate provability; existence through a fact hash set) and writes only the kept ones. A kept grounding's
unknown atoms (at most the width) are the next step's goals; a (goal, variant) row none of whose candidates can be kept
is not walked (``kernels.live_count``). A step walks its candidates in chunks of at most ``ROWS`` rows. With fp_batch,
the step before the last keeps only the groundings whose unknown atom can still be proved (an earlier step's goal, or
a goal some rule could ground at the last step: ``kernels.live_atoms``). With keras (width <= 1), a step writing more
than ``KERAS_STORE_ROWS`` groundings keeps only those whose unknown atom can be proved at all, grounding it again as
the proved atoms grow (``_keras_closure``). The groundings are then filtered by the rules' variable bindings and pruned:
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

from grounder.ops import canonical, concat, decode, groups, key
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


ROWS = 1 << 24       # about the most candidate rows a step holds at once: it runs in chunks of goals and of rows
WALK = 1 << 27       # the most candidates one walk of the fused kernel may keep (its output buffers: 8 bytes each)



def _candidates(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor]):
    """The rows of a step over ``goals`` ``[N, 3]`` whose last free variable ``_last`` walks, in chunks of at most
    ``ROWS`` rows: ``(src, variant, n, start, count, one)`` each, the free variables but the last enumerated."""
    t, V = g.tables, g.tables.V
    n, r = torch.nonzero(t.by_pred_mask[goals[:, 0]], as_tuple=True)        # the live (goal, variant) pairs
    rule = t.by_pred[goals[n, 0], r]
    if width == 0:
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
            a = 0
            for b in groups(count, ROWS):
                yield expand(src[a:b], rule[a:b], n[a:b], start[a:b], count[a:b], one[a:b], facts.values,
                             t.fv_src[:, last], t.fv_pred[:, last], t.fv_dir[:, last],
                             ~t.has_free | ~t.fv_valid[:, last], facts.start, facts.count, facts.P, facts.E,
                             arg_src, t.body_pred, t.n_body, g.enumerated, heads, facts.seen, width)
                a = b
            return
        row, value = facts.enumerate(t.fv_pred[rule, fv], bound, t.fv_dir[rule, fv], one)
        n, rule, no_free = n[row], rule[row], no_free[row]
        src = torch.cat([src[row], value.unsqueeze(1)], 1)
    if last < V:
        bound = src.gather(1, t.fv_src[rule, last].clamp(max=src.shape[1] - 1).unsqueeze(1)).squeeze(1)
        one = no_free | ~t.fv_valid[rule, last]
        start, count = facts.slots(t.fv_pred[rule, last], bound, t.fv_dir[rule, last], one)
    else:
        start, one = torch.zeros_like(rule), torch.ones_like(rule, dtype=torch.bool)
        count = one.long()
    yield src, rule, n, start, count, one


def _chunks(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor]):
    """``(first goal, end goal, cand)``: a step's candidate rows (``_candidates``; ``n`` local to the chunk), a chunk
    of goals at a time (about ``ROWS`` (goal, variant) rows)."""
    s = 0
    for e in groups(g.tables.n_by_pred[goals[:, 0]], ROWS):
        for cand in _candidates(g, facts, goals[s:e], width, heads):
            yield s, e, cand
        s = e


def _walk(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor], cands=None, live=None,
          pool: Optional[Tensor] = None, admit=None):
    """One step over ``goals`` ``[N, 3]``, a chunk of candidate rows (``_chunks``, or ``cands``) at a time, walked
    ``WALK`` candidates at a time: the kept groundings, ``ROWS`` at a time, as (rule [K], goal row [K], body
    [K, M, 3]). The last free variable, with every test of the groundings, in ``kernels.last_stage`` (``live``: an
    unknown atom must also be in this key set, keyed with its goal's ``pool``), then ``admit(goal row, body)``, a
    ``[K]`` mask, if given. Only the kept groundings outlive their turn."""
    t, pad, M, V = g.tables, g.kb.pad, g.tables.M, g.tables.V
    arg_src = t.arg_src.clamp(max=1 + V)
    for s, e, (src, rule, n, start, count, one) in (cands if cands is not None else _chunks(g, facts, goals, width,
                                                                                            heads)):
        chunk = goals[s:e]
        for rows, vals in last_stage(src, rule, n, chunk, start, count, one, facts.values, arg_src, t.body_pred,
                                     t.n_body, g.enumerated, heads, facts.seen, pad, width, facts.count, facts.P,
                                     facts.E, cycle="unknown" if g.prune == "keras" else "all", live_set=live,
                                     seg=None if pool is None else pool[s:e], limit=WALK):
            for a in range(0, rows.shape[0], ROWS):                    # its kept groundings, ROWS at a time
                row, val = rows[a:a + ROWS], vals[a:a + ROWS]
                nk, rk = n[row], rule[row]
                pad_cols = src.new_zeros(row.shape[0], max(1 + V - src.shape[1], 0))   # (none: no rule has a free var)
                bound = torch.cat([src[row], val.unsqueeze(1), pad_cols], 1)[:, :2 + V]
                args = bound.unsqueeze(1).expand(-1, M, -1).gather(-1, arg_src[rk])
                body = torch.cat([t.body_pred[rk].unsqueeze(-1), args], -1)            # [K, M, 3]
                active = torch.arange(M, device=goals.device) < t.n_body[rk].unsqueeze(1)
                body = body.masked_fill(~active.unsqueeze(-1), pad)
                if admit is not None:
                    ok = admit(nk + s, body)
                    nk, rk, body = nk[ok], rk[ok], body[ok]
                yield t.rule[rk], nk + s, body


def _cat(parts) -> Tensor:
    """``torch.cat`` of the parts, the one part itself (no copy) when there is one."""
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _unknown(g, facts: FactTable, body: Tensor):
    """The (grounding, slot) pairs of the body atoms ``[K, M, 3]`` that are no facts."""
    return torch.nonzero((body[..., 0] != g.kb.pad) & ~facts.seen.contains(body), as_tuple=True)


def _step(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor], want_next: bool, **walk):
    """``_walk``'s groundings of a step, together: (rule, goal row, body) and, with ``want_next``, their unknown body
    atoms as (kept grounding, body slot)."""
    parts = list(_walk(g, facts, goals, width, heads, **walk))
    if not parts:
        z = goals.new_zeros(0)
        return z, z, goals.new_zeros(0, g.tables.M, 3), ((z, z) if want_next else None)
    r, n, b = (_cat(x) for x in zip(*parts))
    return r, n, b, (_unknown(g, facts, b) if want_next else None)


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


def _provable(g, facts: FactTable, pool: Tensor, prior: Tensor, base: int):
    """fp_batch's ``admit`` for the step before the last: a grounding only if each unknown atom can be proved — a goal
    of an earlier step or this one (``prior``, keyed with ``pool``), or one some rule could ground at the last step
    (width 0). Else no grounding of the atom is kept (its only ones would be the last step's), so neither is this one;
    the atom is no goal of the last step."""
    known = HashSet.of_keys(prior)

    def admit(n: Tensor, body: Tensor) -> Tensor:
        row, slot = _unknown(g, facts, body)
        atoms = body[row, slot]
        ok = known.contains_keys(key(atoms, base, pool[n[row]])) | _live_goals(g, facts, atoms)
        keep = torch.ones(body.shape[0], dtype=torch.bool, device=body.device)
        keep[row[~ok]] = False
        return keep
    return admit


KERAS_STORE_ROWS = 1 << 24             # a step writing more groundings is grounded again in each round of the closure
KERAS_CLOSURE_MIN_ROWS = 1 << 18       # a last step with fewer candidate rows (training batches, Countries) writes all
KERAS_CACHE_ROWS = 1 << 25             # a step grounded again keeps up to this many candidate rows for the next round


def _cached(g, facts: FactTable, goals: Tensor, width: int, heads: Optional[Tensor]):
    """A step's chunks of candidate rows as a list, to walk them again, when they are at most ``KERAS_CACHE_ROWS`` rows
    in all; else None (made again each walk)."""
    chunks, rows = [], 0
    for c in _chunks(g, facts, goals, width, heads):
        rows += c[2][1].shape[0]
        if rows > KERAS_CACHE_ROWS:
            return None
        chunks.append(c)
    return chunks


def _keras_closure(g, facts: FactTable, stored: List[tuple], streamed: List[tuple], base: int):
    """The keras filter (width <= 1) without writing what it drops: each ``streamed`` step's groundings whose unknown
    atom lies in the least fixed point of the provable atoms — the heads of the groundings, ``stored`` ``(head, body,
    pool)`` or streamed, whose unknown atom is in it — found by grounding the streamed steps ``(goals, pool, width,
    heads, cands)`` (``cands``: ``_cached``'s) again as it grows (Kleene iteration from the empty set).
    ``_keras_provable`` keeps the same."""
    if stored:
        head, body, bpool = (_cat(x) for x in zip(*stored))
        row, slot = _unknown(g, facts, body)
        ukey, hkey = key(body[row, slot], base, bpool[row]), key(head, base, bpool)
    proved = torch.zeros(0, dtype=torch.long, device=facts.values.device)
    while True:
        live = HashSet.of_keys(proved)
        out = [_step(g, facts, goals, width, heads, False, cands=c, live=live, pool=p)[:3]
               for goals, p, width, heads, c in streamed]
        grown = [key(goals[n], base, p[n]) for (_, n, _), (goals, p, *_) in zip(out, streamed)]
        if stored:
            ok = torch.ones_like(hkey, dtype=torch.bool)
            ok[row[~live.contains_keys(ukey)]] = False
            grown.append(hkey[ok])
        grown = torch.unique(torch.cat(grown))
        if grown.shape[0] == proved.shape[0]:
            return out
        proved = grown


def _keras_steps(g, facts: FactTable, k: Tensor, base: int, depth: int, stats: Optional[list]):
    """``_ground_steps`` with keras (width <= 1): each step's goals, the unknown atoms of the last step's
    groundings, found a chunk at a time; a step writing at most ``KERAS_STORE_ROWS`` groundings keeps them, a larger one
    only those the keras filter can keep (``_keras_closure``), so it never sits in memory whole. So does the last step
    unless it has fewer than ``KERAS_CLOSURE_MIN_ROWS`` candidate rows: what it drops would be most of what the proof
    walk walks."""
    steps = []                                          # (goals, pool, width, cached chunks, groundings or None)
    for d in range(depth):
        k = torch.unique(k)
        if k.shape[0] == 0:
            break
        p, atoms = decode(k, base, pool=True)
        last = d == depth - 1
        width = g.last_width if last else g.width
        if last and d > 0:
            cands = _cached(g, facts, atoms, width, g.tables.heads)
            if cands is None or sum(c[2][1].shape[0] for c in cands) >= KERAS_CLOSURE_MIN_ROWS:
                steps.append((atoms, p, width, cands, None))
                break
            steps.append((atoms, p, width, None, _step(g, facts, atoms, width, g.tables.heads, False, cands=cands)[:3]))
            break
        parts, nxt, rows = [], [], 0
        for part in _walk(g, facts, atoms, width, g.tables.heads):
            if not last:
                row, slot = _unknown(g, facts, part[2])
                nxt.append(key(part[2][row, slot], base, p[part[1][row]]))
                del row, slot
            rows += part[0].shape[0]
            parts = parts if parts is not None and rows <= KERAS_STORE_ROWS else None
            if parts is not None:
                parts.append(part)
            del part                                    # (no chunk outlives its turn)
        if parts is not None:
            z = atoms.new_zeros(0)
            parts = (tuple(_cat(x) for x in zip(*parts)) if parts else (z, z, atoms.new_zeros(0, g.tables.M, 3)))
        steps.append((atoms, p, width, None if parts is not None else _cached(g, facts, atoms, width, g.tables.heads),
                      parts))
        k = _cat(nxt) if nxt else atoms.new_zeros(0)
    if not steps:
        z = k.new_zeros(0)
        return z, k.new_zeros(0, 3), k.new_zeros(0, g.tables.M, 3), z
    out = [None if s[4] is None else (s[4][0], s[0][s[4][1]], s[4][2], s[1][s[4][1]]) for s in steps]
    big = [i for i, s in enumerate(steps) if s[4] is None]               # (rule, head, body, pool) per step
    if big:
        stored = [o[1:] for o in out if o is not None]                   # (head, body, pool)
        streamed = [(atoms, p, width, g.tables.heads, cands) for atoms, p, width, cands, _ in (steps[i] for i in big)]
        for i, (r, n, b) in zip(big, _keras_closure(g, facts, stored, streamed, base)):
            out[i] = r, steps[i][0][n], b, steps[i][1][n]
    if stats is not None:
        stats.extend((d, int(s[0].shape[0]), int(o[0].shape[0])) for d, (s, o) in enumerate(zip(steps, out)))
    return tuple(_cat(x) for x in zip(*out))


def _ground_steps(g, facts: FactTable, goals: Tensor, pool: Tensor, base: int, depth: int, stats: Optional[list]):
    """Every step's kept groundings of the queries ``goals`` (of ``pool``): ``(rule [T], head [T, 3],
    body [T, M, 3], pool [T])``, possibly repeated; each step's (depth, goals, kept groundings) appended to ``stats``."""
    k = key(goals, base, pool)
    if g.prune == "keras" and g.width <= 1 and depth >= 2:
        return _keras_steps(g, facts, k, base, depth, stats)
    rules, heads_, bodies, pools = [], [], [], []
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
        admit = _provable(g, facts, p, torch.cat(prior), base) if d == depth - 2 and g.prune == "fp_batch" else None
        r, n, b, nxt = _step(g, facts, atoms, width, hpm, not last, admit=admit)
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
    S, Q, R = queries.shape[0], queries.shape[1], len(kb.rules)
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

    return canonical(rule, head, body, pool, qs, qpool, S, R, base, pad)


__all__ = ["FactTable", "ground"]
