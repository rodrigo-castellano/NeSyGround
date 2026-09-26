"""The FIRINGS of a PBC backward grounder, fast: width <= 1, any depth, few host syncs, many batches per call.

Same output as the general engine (``backward.loop.run_backward`` + ``finalize`` + the fp_batch prune + query pinning)
on the configurations it covers — flat pbc, unguided, width <= 1, last-step width 0, a dense-block or offset-table fact
index — with its atom table compacted to the atoms the kept firings reference (the engine keeps every considered atom;
the compaction keeps the order, so every per-atom reduction a reasoner runs over it is unchanged). It reuses the
engine's compiled rule tables (every anchor variant), its fact index and its per-rule binding tables.

:func:`ground_many` grounds ``S`` batches of queries in one pass: every atom carries its batch as the most significant
digit of its key, so the batches never share an atom and batch ``s``'s output is exactly ``ground(queries[s])``, at a
fraction of the per-call cost (the kernels and host syncs are shared).

Per chunk of queries and step, the goals (the queries, then the unknown body atoms of the previous step's kept
groundings) are deduplicated; the live (goal, rule) pairs enumerate their free variables through the fact index. The
last free variable is enumerated in a fused kernel (``fast_kernels.last_stage``) that also tests every candidate
grounding (width, cycle, head-predicate prune; existence through a fact hash set) and writes only the kept ones, so
the rejected candidates (most of them) never reach memory. With width <= 1 a kept grounding has at most one unknown
atom, the next step's goal. The firings are then canonicalised (unique atoms and firings, sorted), filtered by the
rules' variable bindings, pruned to the provable ones (fp_batch: ``depth`` rounds of Kleene propagation from the
facts), and the queries pinned into the atom table.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grounder.backward.fast_kernels import HashSet, expand, last_stage
from grounder.base.types import Layout, RuleGroundings
from grounder.data.fact_index.inverted import InvertedFactIndex


def supports(g) -> bool:
    """Whether :func:`ground` covers the BackwardGrounder ``g``'s configuration (on the GPU: its kernels are
    Triton's)."""
    return (torch.device(g.kb.device_).type == "cuda" and g.resolution == "pbc" and g._exec_layout is Layout.FLAT
            and g.guided_topk is None
            and g.guided_stats is None and g.guided_query_topk is None and g.guided_query_depth is None
            and g.width is not None and g.width <= 1 and g.w_last_depth == 0 and not g._cartesian_product
            and isinstance(g.kb.fact_index, InvertedFactIndex) and g.filter_mode in ("fp_batch", "none"))


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


def _step(g, facts: _Facts, goals: Tensor, width: int, head_pred_mask: Optional[Tensor], want_next: bool):
    """One step over ``goals`` ``[N, 3]``: the kept groundings as (rule, goal row, body) and, with ``want_next``,
    their unknown body atoms as (kept grounding, body slot). The free variables but the last are enumerated here;
    the last one, with every test of the groundings, in :func:`~grounder.backward.fast_kernels.last_stage`."""
    pad, M, V, dev = g.kb.padding_idx, g.kb.M, g.V, goals.device
    n, r = torch.nonzero(g.pred_rule_mask[goals[:, 0]], as_tuple=True)       # the live (goal, rule) pairs
    rule = g.pred_rule_indices[goals[n, 0], r]
    src = goals[n, 1:]                                                          # [T, 2 + bound free vars]
    no_free = ~g.has_free[rule]
    enumerated = [fv for fv in range(V) if g._fv_any_valid[fv]]
    last = enumerated[-1] if enumerated else V
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
                facts.P, facts.E)
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
    arg_src = g.arg_source_dep.clamp(max=1 + V)                    # the source columns: head args, free variables
    rows, vals = last_stage(src, rule, n, goals, start, count, one, facts.values, arg_src, g.body_preds_dep,
                            g.num_body_atoms, _enumerated_atoms(g), head_pred_mask, facts.seen, pad, width)
    n, rule = n[rows], rule[rows]
    src = torch.cat([src[rows], vals.unsqueeze(1), src.new_zeros(rows.shape[0], 1 + V - src.shape[1])], 1)[:, :2 + V]
    args = src.unsqueeze(1).expand(-1, M, -1).gather(-1, arg_src[rule])
    body = torch.cat([g.body_preds_dep[rule].unsqueeze(-1), args], -1)            # [K, M, 3]
    active = torch.arange(M, device=dev) < g.num_body_atoms[rule].unsqueeze(1)
    body = body.masked_fill(~active.unsqueeze(-1), pad)
    nxt = torch.nonzero(active & ~facts.seen.contains(body), as_tuple=True) if want_next else None
    return g._variant_to_orig_t[rule], n, body, nxt


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


def _ground_steps(g, goals: Tensor, seg: Tensor, base: int, chunk_size: Optional[int], depth: int,
                  stats: Optional[list]):
    """Every depth's kept groundings of the (batch-tagged) query atoms: ``(rule [T], head [T, 3], body [T, M, 3],
    batch [T])``, possibly with duplicates; each step's (goals, kept groundings) appended to ``stats``."""
    facts = _Facts.get(g, base)
    rules, heads, bodies, segs = [], [], [], []
    step = chunk_size if chunk_size and chunk_size > 0 else max(goals.shape[0], 1)
    for start in range(0, goals.shape[0], step):
        key = _key(seg[start:start + step], goals[start:start + step], base)
        for d in range(depth):
            if key.shape[0] == 0:
                break
            last = d == depth - 1
            s, atoms = _decode(torch.unique(key), base)
            r, n, b, nxt = _step(g, facts, atoms, g.w_last_depth if last else g.width,
                                 None if last else g.head_pred_mask, not last)
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

    # canonical firings: unique atoms (sorted by batch, then atom), unique firings (sorted by batch, rule, atoms)
    akey, ainv = torch.unique(_key(seg.unsqueeze(1), torch.cat([head.unsqueeze(1), body], 1), base),
                              return_inverse=True)                                  # ainv [T, 1 + M]
    aseg, table = _decode(akey, base)
    start = torch.searchsorted(aseg, torch.arange(S + 1, device=dev))              # each batch's first atom
    local = ainv - start[seg].unsqueeze(1)
    rule = torch.where(rule < 0, torch.full_like(rule, R), rule)
    A = max(int(start.diff().max()), 1) if S > 1 else max(int(akey.shape[0]), 1)     # atoms per batch
    fseg, rule, *local = _unique_rows([seg, rule, *local.unbind(1)], [S, R + 1] + [A] * (M + 1))
    local = torch.stack(local, 1)
    idx = local + start[fseg].unsqueeze(1)                                          # [F, 1 + M] global atom rows

    # the rules' variable bindings (a repeated variable binds one constant)
    bt = kb.binding_tables(M, pad)
    rc = rule.clamp(max=R - 1)
    fatoms = table[idx]                                                             # [F, 1 + M, 3]
    ent = fatoms[..., 1:].reshape(-1, 2 * (M + 1))
    keep = ((rule < R) & (fatoms[:, 0, 0] == bt["head_pred"][rc]) & (fatoms[:, 1:, 0] == bt["body_pred"][rc]).all(1)
            & ((ent == ent.gather(1, bt["canon_src"][rc])) | ~bt["slot_active"][rc]).all(1))

    # fp_batch: keep the firings whose body is proved within depth rounds from the facts
    if g.filter_mode == "fp_batch":
        proved = (_Facts.get(g, base).facts.contains(table) | (table[:, 0] == pad)).to(torch.int8)
        body_idx, head_idx = idx[:, 1:], idx[:, 0]
        for _ in range(max(1, depth)):
            fired = ((proved[body_idx] == 1).all(1) & keep).to(torch.int8)
            proved = torch.maximum(proved, torch.zeros_like(proved).scatter_reduce_(
                0, head_idx, fired, reduce="amax", include_self=False))
        keep &= (proved[body_idx] == 1).all(1)
    fseg, rule, idx = fseg[keep], rule[keep], idx[keep]

    # the atom table compacted to the kept firings' atoms, then every query pinned (new ones appended, sorted)
    used = torch.zeros(table.shape[0], dtype=torch.bool, device=dev)
    used[idx.reshape(-1)] = True
    new = torch.cumsum(used, 0) - 1
    akey, aseg, table = akey[used], aseg[used], table[used]
    idx = new[idx]
    body_valid = table[idx[:, 1:], 0] != pad
    pstart = torch.searchsorted(aseg, torch.arange(S + 1, device=dev))
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
