"""The FIRINGS of a PBC backward grounder, fast: width <= 1 (the keras filter: <= 2), any depth, few host syncs, many
batches per call.

Same output as the general engine (``backward.loop.run_backward`` + ``finalize`` + the fp_batch prune + query pinning)
on the configurations it covers — flat pbc, unguided, width <= 1 and last-step width 0 (the keras filter, which the
engine does not run: width <= 2 at every step, as keras-ns's BC_{w,d}), a dense-block or offset-table fact index —
with its atom table compacted to the atoms the kept firings reference (the engine keeps every considered atom; the
compaction keeps the order, so every per-atom reduction a reasoner runs over it is unchanged). It reuses the engine's
compiled rule tables (every anchor variant), its fact index and its per-rule binding tables.

:func:`ground_many` grounds ``S`` batches of queries in one pass: every atom carries its batch as the most significant
digit of its key, so the batches never share an atom and batch ``s``'s output is exactly ``ground(queries[s])``, at a
fraction of the per-call cost (the kernels and host syncs are shared).

Per chunk of queries and step, the goals (the queries, then the unknown body atoms of the previous step's kept
groundings) are deduplicated; the live (goal, rule) pairs enumerate their free variables through the fact index. The
last free variable is enumerated in a fused kernel (``fast_kernels.last_stage``) that also tests every candidate
grounding (width, cycle, head-predicate prune; existence through a fact hash set) and writes only the kept ones, so the
rejected candidates (most of them) never reach memory. A kept grounding's unknown atoms (at most the width) are the
next step's goals; a (goal, rule) row none of whose candidates can be kept (a body atom with no fact on its bound side,
...: ``fast_kernels.live_count``) is not walked. With fp_batch, the step before the last keeps only the groundings
whose unknown atom can still be proved (a fact, an earlier step's goal, or a goal some rule could ground at the last
step: ``fast_kernels.live_atoms``), so the last step grounds only those. With keras on a large pool, the last step
writes only the groundings whose unknown atoms can be proved at all (:func:`_keras_last_step`). The groundings are then
filtered by the rules' variable bindings and pruned: fp_batch to the provable ones (``depth`` rounds of Kleene
propagation from the facts, over hash sets of the raw groundings); keras first to those whose unknown atom can be
proved at all (the least fixed point from the all-fact groundings: :func:`_keras_provable`, a few thousand of the
millions a keras step writes), then by keras-ns's proof walk. They are canonicalised (unique atoms and firings,
sorted), and the queries pinned into the atom table.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grounder.backward.fast_kernels import HashSet, expand, last_stage, live_atoms
from grounder.base.types import Layout, RuleGroundings
from grounder.data.fact_index.inverted import InvertedFactIndex


def supports(g) -> bool:
    """Whether :func:`ground` covers the BackwardGrounder ``g``'s configuration (on the GPU: its kernels are
    Triton's)."""
    return (torch.device(g.kb.device_).type == "cuda" and g.resolution == "pbc" and g._exec_layout is Layout.FLAT
            and g.guided_topk is None
            and g.guided_stats is None and g.guided_query_topk is None and g.guided_query_depth is None
            and g.width is not None and g.width <= (2 if g.filter_mode == "keras" else 1) and not g._cartesian_product
            and (g.w_last_depth == 0 or (g.filter_mode == "keras" and g.w_last_depth <= 2))
            and isinstance(g.kb.fact_index, InvertedFactIndex) and g.filter_mode in ("fp_batch", "none", "keras"))


class _Facts:
    """The fact index's enumeration tables as one CSR (a free variable's candidates are the first ``count`` values
    from ``start``, exactly the index's own ``enumerate``), and hash sets of the facts and of those ``exists`` sees."""

    def __init__(self, fi, base: int) -> None:
        self.base, self.P, self.E = base, fi._num_predicates, fi._num_entities
        if getattr(fi, "_use_dense", False):              # [P, E, K] blocks: slot i holds its values at i * K
            K = fi._K
            self.values = torch.cat([fi._ps_blocks.reshape(-1), fi._po_blocks.reshape(-1)])
            self.count = torch.cat([fi._ps_counts.reshape(-1), fi._po_counts.reshape(-1)])
            self.start = torch.arange(self.count.shape[0], device=self.count.device) * K
            ps, j = torch.nonzero(torch.arange(K, device=self.count.device) < fi._ps_counts.reshape(-1, 1),
                                  as_tuple=True)                  # exists sees the facts the blocks hold
            seen = torch.stack([ps // self.E, ps % self.E, fi._ps_blocks.reshape(-1, K)[ps, j]], 1)
        else:                                               # offset tables, each slot capped at max_facts_per_query
            F = fi._ps_values.shape[0]
            self.values = torch.cat([fi._ps_values, fi._po_values])
            self.start = torch.cat([fi._ps_offsets[:-1], fi._po_offsets[:-1] + F])
            self.count = torch.cat([fi._ps_offsets.diff(), fi._po_offsets.diff()]).clamp(max=fi._max_facts_per_query)
            seen = fi.facts_idx
        self.facts = HashSet(fi.facts_idx, base)
        self.seen = self.facts if seen is fi.facts_idx else HashSet(seen, base)
        self.seen_all = seen is fi.facts_idx or seen.shape[0] == torch.unique(fi.facts_idx, dim=0).shape[0]

    @staticmethod
    def get(g, base: int) -> "_Facts":
        """The grounder's tables, built once."""
        f = getattr(g, "_fast_facts", None)
        if f is None or f.base != base:
            f = g._fast_facts = _Facts(g.kb.fact_index, base)
        return f

    def slots(self, pred: Tensor, bound: Tensor, direction: Tensor, one: Tensor):
        """Each row's candidates as a CSR slice ``(start, count)``; a row with ``one`` set gets exactly one."""
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


def _enumerated_atoms(g) -> Tensor:
    """``[R, M]``: the body atoms a free variable's enumeration draws from the fact index (facts by construction),
    cached on the grounder."""
    known = getattr(g, "_fast_enumerated", None)
    if known is None:
        arg = g.arg_source_dep.clamp(max=1 + g.V)                              # [R, M, 2]
        known = torch.zeros(arg.shape[:2], dtype=torch.bool, device=arg.device)
        for fv in range(g.V):
            by_subject = g.fv_enum_direction[:, fv] == 0
            pair = torch.stack([torch.where(by_subject, g.fv_enum_bound_src[:, fv], 2 + fv),
                                torch.where(by_subject, 2 + fv, g.fv_enum_bound_src[:, fv])], -1)   # [R, 2]
            known |= (g.fv_enum_valid[:, fv].unsqueeze(1) & (g.body_preds_dep == g.fv_enum_pred[:, fv].unsqueeze(1))
                      & (arg == pair.unsqueeze(1)).all(-1))
        known &= torch.arange(arg.shape[1], device=arg.device) < g.num_body_atoms.unsqueeze(1)
        g._fast_enumerated = known
    return known


def _cheapest_variant(g, facts: _Facts, goals: Tensor, n: Tensor, rule: Tensor):
    """At a width-0 step every body atom must be a fact, so each anchor variant of a rule keeps the same groundings (the
    dedup merges them): keep one (goal, rule) row per goal and original rule, the variant whose first free variable
    has the fewest facts to enumerate for this goal (ties: the first variant). Same groundings, a fraction of the rows
    on a hub (YAGO3-10: a person's facts through a country's thousands)."""
    orig = g._variant_to_orig_t[rule]
    has = g.has_free[rule] & g.fv_enum_valid[rule, 0]
    bound = goals[n, 1:].gather(1, g.fv_enum_bound_src[rule, 0].clamp(max=1).unsqueeze(1)).squeeze(1)
    _, count = facts.slots(g.fv_enum_pred[rule, 0], bound, g.fv_enum_direction[rule, 0], ~has)
    if n.numel() == 0:
        return n, rule
    # a rule's variants are consecutive (rule-major expansion), so a (goal, rule)'s rows are one run: no sort (a run
    # split in two would keep two variants, which the dedup still merges)
    key = n * (g._variant_to_orig_t.shape[0] + 1) + orig
    inv = torch.cat([key.new_zeros(1), (key[1:] != key[:-1]).long()]).cumsum(0)
    best = torch.full((int(inv[-1]) + 1,), torch.iinfo(count.dtype).max, dtype=count.dtype,
                      device=count.device).scatter_reduce(0, inv, count, "amin")
    row = torch.arange(n.shape[0], device=n.device)
    cheapest = torch.where(count == best[inv], row, row.shape[0])
    first = torch.full_like(best, row.shape[0]).scatter_reduce(0, inv, cheapest, "amin")
    keep = row == first[inv]                                                    # original row order kept
    return n[keep], rule[keep]


def _candidates(g, facts: _Facts, goals: Tensor, width: int, head_pred_mask: Optional[Tensor]):
    """The rows of a step over ``goals`` ``[N, 3]`` whose last free variable :func:`_step` walks: ``(src, rule, n,
    start, count, one)``, the free variables but the last enumerated."""
    V = g.V
    n, r = torch.nonzero(g.pred_rule_mask[goals[:, 0]], as_tuple=True)       # the live (goal, rule) pairs
    rule = g.pred_rule_indices[goals[n, 0], r]
    if width == 0 and getattr(g, "filter_mode", None) != "keras":
        n, rule = _cheapest_variant(g, facts, goals, n, rule)
    src = goals[n, 1:]                                                          # [T, 2 + bound free vars]
    no_free = ~g.has_free[rule]
    enumerated = [fv for fv in range(V) if g._fv_any_valid[fv]]
    last = enumerated[-1] if enumerated else V
    arg_src = g.arg_source_dep.clamp(max=1 + V)                    # the source columns: head args, free variables
    for fv in range(last):
        if not g._fv_any_valid[fv]:
            src = torch.cat([src, src.new_zeros(src.shape[0], 1)], 1)
            continue
        bound = src.gather(1, g.fv_enum_bound_src[rule, fv].clamp(max=src.shape[1] - 1).unsqueeze(1)).squeeze(1)
        # a rule without free variables, or not using this one, keeps its row once
        one = no_free | ~g.fv_enum_valid[rule, fv]
        if fv == last - 1:      # the variable before the last: expanded in one kernel, with the last one's slices
            start, count = facts.slots(g.fv_enum_pred[rule, fv], bound, g.fv_enum_direction[rule, fv], one)
            src, rule, n, start, count, one = expand(
                src, rule, n, start, count, one, facts.values, g.fv_enum_bound_src[:, last], g.fv_enum_pred[:, last],
                g.fv_enum_direction[:, last], ~g.has_free | ~g.fv_enum_valid[:, last], facts.start, facts.count,
                facts.P, facts.E, arg_src, g.body_preds_dep, g.num_body_atoms, _enumerated_atoms(g), head_pred_mask,
                facts.seen, width)
            break
        row, value = facts.enumerate(g.fv_enum_pred[rule, fv], bound, g.fv_enum_direction[rule, fv], one)
        n, rule, no_free = n[row], rule[row], no_free[row]
        src = torch.cat([src[row], value.unsqueeze(1)], 1)
    else:
        if last < V:
            bound = src.gather(1, g.fv_enum_bound_src[rule, last].clamp(max=src.shape[1] - 1).unsqueeze(1)).squeeze(1)
            one = no_free | ~g.fv_enum_valid[rule, last]
            start, count = facts.slots(g.fv_enum_pred[rule, last], bound, g.fv_enum_direction[rule, last], one)
        else:
            start, one = torch.zeros_like(rule), torch.ones_like(rule, dtype=torch.bool)
            count = one.long()
    return src, rule, n, start, count, one


def _step(g, facts: _Facts, goals: Tensor, width: int, head_pred_mask: Optional[Tensor], want_next: bool,
          cand=None, live=None, seg: Optional[Tensor] = None):
    """One step over ``goals`` ``[N, 3]``: the kept groundings as (rule, goal row, body) and, with ``want_next``,
    their unknown body atoms as (kept grounding, body slot). The free variables but the last are enumerated first
    (:func:`_candidates`, or ``cand``); the last one, with every test of the groundings, in
    :func:`~grounder.backward.fast_kernels.last_stage` (``live``: an unknown atom must also be in this key set, keyed
    with its goal's batch ``seg``)."""
    pad, M, V, dev = g.kb.padding_idx, g.kb.M, g.V, goals.device
    src, rule, n, start, count, one = cand if cand is not None else _candidates(g, facts, goals, width, head_pred_mask)
    arg_src = g.arg_source_dep.clamp(max=1 + V)
    rows, vals = last_stage(src, rule, n, goals, start, count, one, facts.values, arg_src, g.body_preds_dep,
                            g.num_body_atoms, _enumerated_atoms(g), head_pred_mask, facts.seen, pad, width,
                            facts.count, facts.P, facts.E,
                            cycle="unknown" if getattr(g, "filter_mode", None) == "keras" else "all", live_set=live,
                            seg=seg)
    n, rule = n[rows], rule[rows]
    pad_cols = src.new_zeros(rows.shape[0], max(1 + V - src.shape[1], 0))       # (none when no rule has a free var)
    src = torch.cat([src[rows], vals.unsqueeze(1), pad_cols], 1)[:, :2 + V]
    args = src.unsqueeze(1).expand(-1, M, -1).gather(-1, arg_src[rule])
    body = torch.cat([g.body_preds_dep[rule].unsqueeze(-1), args], -1)            # [K, M, 3]
    active = torch.arange(M, device=dev) < g.num_body_atoms[rule].unsqueeze(1)
    body = body.masked_fill(~active.unsqueeze(-1), pad)
    nxt = torch.nonzero(active & ~facts.seen.contains(body), as_tuple=True) if want_next else None
    return g._variant_to_orig_t[rule], n, body, nxt


def _live_tables(g, facts: _Facts):
    """``(ok_s, ok_o)`` ``[R, E + 1]`` bool, per rule (variant) and entity: every body atom with the head's subject
    (``ok_s``; object: ``ok_o``) and a free variable has a fact of its predicate on that side (column ``E``: an
    entity with no facts; a body atom whose predicate has none fails both); cached on the grounder."""
    got = getattr(g, "_fast_live_tables", None)
    if got is None:
        P, E, V = facts.P, facts.E, g.V
        arg, pred = g.arg_source_dep.clamp(max=1 + V), g.body_preds_dep                    # [R, M, 2], [R, M]
        active = torch.arange(pred.shape[1], device=pred.device) < g.num_body_atoms.unsqueeze(1)
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
        got = g._fast_live_tables = (ok[0].to(torch.int8).contiguous(), ok[1].to(torch.int8).contiguous())
    return got


def _live_goals(g, facts: _Facts, goals: Tensor) -> Tensor:
    """``[N]`` bool: whether some rule could ground each goal ``[N, 3]`` at the last step (width 0: every body atom a
    fact) — :func:`~grounder.backward.fast_kernels.live_atoms` on per-rule tables."""
    ok_s, ok_o = _live_tables(g, facts)
    return live_atoms(goals, g.pred_rule_mask, g.pred_rule_indices, ok_s, ok_o, g.arg_source_dep.clamp(max=1 + g.V),
                      g.body_preds_dep, g.num_body_atoms, facts.seen)


def _key(seg: Tensor, atoms: Tensor, base: int) -> Tensor:
    """The injective int64 key of (batch, atom): lexicographic in (batch, predicate, subject, object)."""
    return ((seg * base + atoms[..., 0]) * base + atoms[..., 1]) * base + atoms[..., 2]


def _unique_rows(cols: List[Tensor], radix: List[int]) -> List[Tensor]:
    """The distinct rows of the digit columns ``cols`` (``0 <= cols[i] < radix[i]``), sorted lexicographically: the
    digits packed into as few int64 keys as fit, radix-sorted from the least significant key (stable), and the
    first row of each run kept."""
    keys, key, span = [], None, 1
    for c, r in zip(cols, radix):
        if key is not None and span * r >= 1 << 62:
            keys.append(key)
            key, span = None, 1
        key, span = (c if key is None else key * r + c), span * r
    keys.append(key)
    order = torch.arange(cols[0].shape[0], device=cols[0].device)
    for k in reversed(keys):
        order = order[torch.sort(k[order], stable=True)[1]]
    first = torch.zeros_like(order, dtype=torch.bool)
    first[:1] = True
    for k in keys:
        k = k[order]
        first[1:] |= k[1:] != k[:-1]
    keep = order[first]
    return [c[keep] for c in cols]


def _decode(key: Tensor, base: int):
    return key // base ** 3, torch.stack([key // (base * base) % base, key // base % base, key % base], -1)


def _provable_next(g, facts: _Facts, s: Tensor, r: Tensor, n: Tensor, b: Tensor, nxt, prior: Tensor, base: int):
    """The groundings of the step before the last, less those whose unknown atom cannot be proved, so fp_batch would
    drop them: the atom is not a fact, no rule could ground it at the last step (width ``w_last_depth``), and it is no
    goal of an earlier step (its only groundings would be the last step's). Their unknown atoms are the last step's
    goals, so it grounds only atoms that can be proved; the kept firings are fp_batch's."""
    row, slot = nxt
    atoms = b[row, slot]
    key = _key(s[n[row]], atoms, base)
    ok = HashSet.of_keys(prior).contains_keys(key)
    if facts.seen is not facts.facts:                  # an unknown atom (not seen) may still be a fact
        ok |= facts.facts.contains(atoms)
    ok |= _live_goals(g, facts, atoms)
    keep = torch.ones(r.shape[0], dtype=torch.bool, device=r.device)
    keep[row[~ok]] = False
    new = torch.cumsum(keep, 0) - 1
    return r[keep], n[keep], b[keep], (new[row[ok]], slot[ok])


KERAS_CLOSURE_MIN_ROWS = 1 << 18       # below it (training batches, Countries) writing every candidate costs less


def _keras_last_step(g, facts: _Facts, goals: Tensor, seg: Tensor, width: int, head_pred_mask: Optional[Tensor],
                     heads: List[Tensor], bodies: List[Tensor], segs: List[Tensor], base: int):
    """The keras filter's last step over ``goals`` (batches ``seg``), grounding only what :func:`_keras_provable`
    keeps: its groundings whose unknown atom lies in the least fixed point of the provable atoms (reached from the
    all-fact groundings of every step), which the step finds by grounding again as the set grows (Kleene iteration
    from the empty set: few rounds, each writing few groundings, instead of the millions a keras step considers).
    ``heads`` / ``bodies`` / ``segs``: the earlier steps' groundings."""
    cand = _candidates(g, facts, goals, width, head_pred_mask)
    if cand[1].shape[0] < KERAS_CLOSURE_MIN_ROWS:      # few candidates: writing them all costs less than the passes
        return _step(g, facts, goals, width, head_pred_mask, False, cand=cand)[:3]
    head, body, bseg = torch.cat(heads), torch.cat(bodies), torch.cat(segs)
    row, slot = torch.nonzero((body[..., 0] != g.kb.padding_idx) & ~facts.facts.contains(body), as_tuple=True)
    ukey, hkey = _key(bseg[row], body[row, slot], base), _key(bseg, head, base)
    proved = hkey.new_zeros(0)
    while True:
        live = HashSet.of_keys(proved)
        r, n, b, _ = _step(g, facts, goals, width, head_pred_mask, False, cand=cand, live=live, seg=seg)
        ok = torch.ones_like(hkey, dtype=torch.bool)
        ok[row[~live.contains_keys(ukey)]] = False
        grown = torch.unique(torch.cat([hkey[ok], _key(seg[n], goals[n], base)]))
        if grown.shape[0] == proved.shape[0]:
            return r, n, b
        proved = grown


def _ground_steps(g, goals: Tensor, seg: Tensor, base: int, chunk_size: Optional[int], depth: int,
                  stats: Optional[list]):
    """Every depth's kept groundings of the (batch-tagged) query atoms: ``(rule [T], head [T, 3], body [T, M, 3],
    batch [T])``, possibly with duplicates; each step's (goals, kept groundings) appended to ``stats``."""
    facts = _Facts.get(g, base)
    rules, heads, bodies, segs = [], [], [], []
    step = chunk_size if chunk_size and chunk_size > 0 else max(goals.shape[0], 1)
    # the keras filter's last step grounds only what it can prove, with every goal in one chunk (a chunk's are not
    # all) and a fact table holding every fact (an atom it misses would count as unknown)
    keras_last = g.filter_mode == "keras" and g.width <= 1 and step >= goals.shape[0] and facts.seen_all
    for start in range(0, goals.shape[0], step):
        key = _key(seg[start:start + step], goals[start:start + step], base)
        prior = []                                      # each step's goals
        for d in range(depth):
            if key.shape[0] == 0:
                break
            last = d == depth - 1
            key = torch.unique(key)
            prior.append(key)
            s, atoms = _decode(key, base)
            width = g.w_last_depth if last else g.width
            hpm = None if last and g.filter_mode != "keras" else g.head_pred_mask
            if last and d > 0 and keras_last:
                (r, n, b), nxt = _keras_last_step(g, facts, atoms, s, width, hpm, heads, bodies, segs, base), None
            else:
                r, n, b, nxt = _step(g, facts, atoms, width, hpm, not last)
            if d == depth - 2 and g.filter_mode == "fp_batch" and nxt[0].numel():
                r, n, b, nxt = _provable_next(g, facts, s, r, n, b, nxt, torch.cat(prior), base)
            if stats is not None:
                stats.append((d, int(atoms.shape[0]), int(r.shape[0])))
            rules.append(r)
            heads.append(atoms[n])
            bodies.append(b)
            segs.append(s[n])
            if nxt is None:
                break
            key = _key(segs[-1][nxt[0]], b[nxt[0], nxt[1]], base)
    if not rules:
        z = goals.new_zeros(0)
        return z, goals.new_zeros(0, 3), goals.new_zeros(0, g.kb.M, 3), z
    return torch.cat(rules), torch.cat(heads), torch.cat(bodies), torch.cat(segs)


def _keras_proved(g, rule: Tensor, head: Tensor, body: Tensor, seg: Tensor, keep: Tensor, qkey: Tensor, base: int,
                  depth: int) -> Tensor:
    """``[T]``: whether each raw grounding survives the pruning of keras-ns's ``ApproximateBackwardChainingGrounder``
    (the IJCAI-25 code's BC_{w,d}): every body atom a fact or proved.

    Its goals: the queries, then at each step the non-fact atoms of the groundings so far, less, for a grounding's own
    rule, the goals already of that rule's head predicate (a query can be a goal again). Every grounding of a goal is a
    proof of it, its unknown atoms the proof's atoms (a rule with one body atom records none). After the last step,
    ``depth - 1`` rounds (``g.keras_rounds`` if set: the later keras-ns walks ``depth``) walk the proofs in the rules'
    order (``g.keras_rule_order``), a rule's step-1 goals' proofs before its step-2 goals', ..., each proving its head
    if its atoms are proved by then. Keras walks a (rule, step) block's goals in the order of a Python set, which varies
    with the string hash seed (so does its output, a few groundings in thousands); here in atom order, one of the
    orders it can take. An atom's earliest proof (its position in the walk) is found by propagation to a fixed
    point."""
    if rule.numel() == 0:
        return keep
    pad, R = g.kb.padding_idx, g.kb.num_rules
    rc = rule.clamp(min=0, max=R - 1)
    valid = body[..., 0] != pad
    unknown = valid & ~_Facts.get(g, base).facts.contains(body)                          # [T, M]
    keys, inv = torch.unique(torch.cat([_key(seg, head, base).unsqueeze(1), _key(seg.unsqueeze(1), body, base)], 1),
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
    rounds = getattr(g, "keras_rounds", None) or depth - 1
    if rounds < 1:              # (IJCAI-25's depth 1: no walk, so every body atom a fact)
        return (~unknown).all(1)
    proof = keep & (g.kb.rule_lens.to(rule.device)[rc] >= 2)
    order = getattr(g, "keras_rule_order", torch.arange(R, device=rule.device)).to(rule.device)[rc]
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


def _keras_provable(g, rule: Tensor, head: Tensor, body: Tensor, seg: Tensor, base: int):
    """The raw groundings less those with an unknown atom that no chain of groundings from the all-fact ones reaches:
    keras-ns proves an atom only by a grounding of it whose unknown atoms are proved before, so what it proves lies in
    that least fixed point (cheap to find: few atoms), and :func:`_keras_proved` never keeps a grounding outside it; it
    then walks thousands of groundings instead of millions. Exact with at most one unknown atom per grounding (the
    fast path's width <= 1; else nothing is dropped): a dropped grounding's unknown atom is outside the fixed point, so
    are all its groundings' (dropped too), and the goals it would mark matter to no grounding left; the atoms left keep
    their order, so the walk's positions compare the same."""
    unknown = (body[..., 0] != g.kb.padding_idx) & ~_Facts.get(g, base).facts.contains(body)          # [T, M]
    if bool((unknown.sum(1) > 1).any()):
        return rule, head, body, seg
    row, slot = torch.nonzero(unknown, as_tuple=True)
    ukey, hkey = _key(seg[row], body[row, slot], base), _key(seg, head, base)
    ok = ~unknown.any(1)
    while True:
        ok_next = torch.ones_like(ok)
        ok_next[row[~HashSet.of_keys(hkey[ok]).contains_keys(ukey)]] = False
        if torch.equal(ok_next, ok):
            return rule[ok], head[ok], body[ok], seg[ok]
        ok = ok_next


def _kept(g, rule: Tensor, head: Tensor, body: Tensor, seg: Tensor, base: int, depth: int, qkey: Tensor):
    """The groundings ``(rule, head [T, 3], body [T, M, 3], batch)`` whose atoms fit their rule's variable bindings (a
    repeated variable binds one constant) and, with fp_batch, whose body is proved within ``depth`` rounds of
    propagation from the facts (an atom is proved by a fact, or as the head of a grounding whose body was proved the
    round before); with keras, as the keras-ns grounder prunes them (:func:`_keras_proved`; ``qkey``: the queries) —
    before canonicalising, on the raw groundings (repeats keep or drop together)."""
    kb, pad, M, R = g.kb, g.kb.padding_idx, g.kb.M, g.kb.num_rules
    if g.filter_mode == "keras" and depth >= 2 and rule.shape[0] >= KERAS_PREFILTER_ROWS:
        rule, head, body, seg = _keras_provable(g, rule, head, body, seg, base)
    bt = kb.binding_tables(M, pad)
    rc = rule.clamp(min=0, max=R - 1)
    ent = torch.cat([head.unsqueeze(1), body], 1)[..., 1:].reshape(-1, 2 * (M + 1))
    keep = ((rule >= 0) & (rule < R) & (head[:, 0] == bt["head_pred"][rc]) & (body[..., 0] == bt["body_pred"][rc]).all(1)
            & ((ent == ent.gather(1, bt["canon_src"][rc])) | ~bt["slot_active"][rc]).all(1))
    if g.filter_mode == "fp_batch":
        fact = _Facts.get(g, base).facts.contains(body) | (body[..., 0] == pad)        # [T, M]
        head_key, body_key = _key(seg, head, base), _key(seg.unsqueeze(1), body, base)
        proved = fact
        for _ in range(max(1, depth)):
            proved = fact | HashSet.of_keys(head_key[proved.all(1) & keep]).contains_keys(body_key)
        keep &= proved.all(1)
    elif g.filter_mode == "keras":
        keep &= _keras_proved(g, rule, head, body, seg, keep, qkey, base, depth)
    return rule[keep], head[keep], body[keep], seg[keep]


@torch.no_grad()
def ground_many(g, queries: Tensor, query_mask: Tensor, chunk_size: Optional[int] = None,
                depth: Optional[int] = None, stats: Optional[list] = None) -> List[RuleGroundings]:
    """``[ground(g, queries[s], query_mask[s]) for s in range(S)]`` for ``queries`` ``[S, Q, 3]``, in one pass;
    ``chunk_size`` bounds the queries grounded at a time (``None`` / ``<= 0``: all at once). ``depth`` (default
    the grounder's) grounds fewer steps; ``stats`` collects each step's ``(depth, goals, kept groundings)``."""
    depth = g.depth if depth is None else int(depth)
    kb, pad, dev = g.kb, g.kb.padding_idx, queries.device
    S, Q, M, R = queries.shape[0], queries.shape[1], kb.M, kb.num_rules
    base = max(pad, kb.constant_no, int(g._P)) + 2
    if S > 1 and S * base ** 3 >= 1 << 62:                          # the batch digit would overflow the key
        return [r for s in range(S) for r in ground_many(g, queries[s:s + 1], query_mask[s:s + 1], chunk_size,
                                                          depth, stats)]
    qs = queries.long().reshape(-1, 3)
    qseg = torch.arange(S, device=dev).repeat_interleave(Q)
    live = query_mask.reshape(-1) & (qs[:, 0] != pad)
    rule, head, body, seg = _ground_steps(g, qs[live], qseg[live], base, chunk_size, depth, stats)

    qkey = torch.unique(_key(qseg[live], qs[live], base))
    rule, head, body, seg = _kept(g, rule, head, body, seg, base, depth, qkey)

    # canonical firings: unique atoms (sorted by batch, then atom), unique firings (sorted by batch, rule, atoms)
    akey, ainv = torch.unique(_key(seg.unsqueeze(1), torch.cat([head.unsqueeze(1), body], 1), base),
                              return_inverse=True)                                  # ainv [T, 1 + M]
    aseg, table = _decode(akey, base)
    start = torch.searchsorted(aseg, torch.arange(S + 1, device=dev))              # each batch's first atom
    local = ainv - start[seg].unsqueeze(1)
    A = max(int(start.diff().max()), 1) if S > 1 else max(int(akey.shape[0]), 1)     # atoms per batch
    fseg, rule, *local = _unique_rows([seg, rule, *local.unbind(1)], [S, R] + [A] * (M + 1))
    idx = torch.stack(local, 1) + start[fseg].unsqueeze(1)                          # [F, 1 + M] global atom rows

    # every query pinned into the atom table (new ones appended, sorted)
    body_valid = table[idx[:, 1:], 0] != pad
    pstart = start
    qkey = _key(qseg, qs, base)
    at = torch.searchsorted(akey, qkey).clamp(max=max(akey.numel() - 1, 0))
    in_pool = (akey[at] == qkey) if akey.numel() else torch.zeros_like(qkey, dtype=torch.bool)
    nkey = torch.unique(qkey[~in_pool])
    nseg, ntable = _decode(nkey, base)
    nstart = torch.searchsorted(nseg, torch.arange(S + 1, device=dev))
    qidx = torch.where(in_pool, at - pstart[qseg],
                       (pstart[qseg + 1] - pstart[qseg]) + torch.searchsorted(nkey, qkey) - nstart[qseg])
    idx = idx - pstart[fseg].unsqueeze(1)
    offsets = torch.zeros(S, R + 1, dtype=torch.long, device=dev)
    offsets[:, 1:] = torch.bincount(fseg * R + rule, minlength=S * R).view(S, R).cumsum(1)

    n_f, n_a, n_n = torch.stack([offsets[:, -1], pstart.diff(), nstart.diff()]).tolist()
    return [RuleGroundings(atom_table=torch.cat([p, t]), body_pool_idx=b, body_atom_valid=v, head_pool_idx=h,
                           rule_idx=r, rule_offsets=o, num_atoms=a + n, num_rules=R, M_max=M, query_pool_idx=q)
            for p, t, h, b, v, r, o, a, n, q in zip(
                torch.split(table, n_a), torch.split(ntable, n_n), torch.split(idx[:, 0], n_f),
                torch.split(idx[:, 1:], n_f), torch.split(body_valid, n_f), torch.split(rule, n_f), offsets, n_a,
                n_n, qidx.view(S, Q))]


def ground(g, queries: Tensor, query_mask: Tensor, chunk_size: Optional[int] = None) -> RuleGroundings:
    """The FIRINGS ``RuleGroundings`` of ``queries`` ``[B, 3]`` (``query_pool_idx`` set, atom table compacted);
    the queries are grounded ``chunk_size`` at a time, every depth per chunk (``None`` / ``<= 0``: all at once)."""
    return ground_many(g, queries.unsqueeze(0), query_mask.unsqueeze(0), chunk_size)[0]


__all__ = ["supports", "ground", "ground_many"]
