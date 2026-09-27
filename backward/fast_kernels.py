"""Triton kernels of the fast path (``backward.fast``): a fact hash set, and the fused last enumeration + test.

:class:`HashSet` stores atom keys ``(p * base + s) * base + o`` in an open-addressing table (linear probing, twice the
keys' size), so a membership test is one or two memory probes instead of a binary search's ~20.

:func:`last_stage` fuses a step's last free-variable enumeration with every test of the candidate groundings: each
row (a (goal, rule) pair with its earlier free variables bound) walks its candidates for the last variable (a slot
of the fact index's CSR), builds the body atoms from the rule's argument table, probes their existence and applies
the width, cycle and head-predicate tests; only the kept (row, value) pairs are written (a per-block atomic append:
their order varies, the grounder canonicalises the firings anyway). Most candidates fail (on YAGO3-10, 23.8M
candidates -> 0.5M kept), so the kept ones are the only ones that reach memory.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

EMPTY = -1       # an empty slot (keys are non-negative)
BLOCK = 128
UNROLL = 2       # candidates a last-stage row walks at a time


@triton.jit
def _slot(key, LOG2T: tl.constexpr):
    return ((key.to(tl.uint64) * 0x9E3779B97F4A7C15) >> (64 - LOG2T)).to(tl.int64)


@triton.jit
def _probe(table_ptr, key, live, LOG2T: tl.constexpr):
    """Whether each live lane's ``key`` is in the table."""
    s = _slot(key, LOG2T)
    found = key < 0                                    # all False, the key's shape
    todo = live
    while tl.max(todo.to(tl.int32), 0) > 0:
        v = tl.load(table_ptr + s, mask=todo, other=-1)
        found = found | (todo & (v == key))
        todo = todo & (v != key) & (v != -1)
        s = (s + 1) & ((1 << LOG2T) - 1)
    return found


@triton.jit
def _insert(table_ptr, keys_ptr, n, LOG2T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    todo = i < n
    key = tl.load(keys_ptr + i, mask=todo, other=0)
    s = _slot(key, LOG2T)
    none = tl.full([BLOCK], -2, tl.int64)                  # never in the table: an idle lane's swap fails
    while tl.max(todo.to(tl.int32), 0) > 0:
        prev = tl.atomic_cas(table_ptr + s, tl.where(todo, tl.full([BLOCK], -1, tl.int64), none),
                             tl.where(todo, key, none), sem="relaxed")
        todo = todo & (prev != -1) & (prev != key)       # placed, or already there: done; else the next slot
        s = tl.where(todo, (s + 1) & ((1 << LOG2T) - 1), s)


@triton.jit
def _contains(table_ptr, atoms_ptr, out_ptr, n, stride, base, LOG2T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    a = atoms_ptr + i.to(tl.int64) * stride
    key = (tl.load(a, mask=live, other=0) * base + tl.load(a + 1, mask=live, other=0)) * base \
        + tl.load(a + 2, mask=live, other=0)
    tl.store(out_ptr + i, _probe(table_ptr, key, live, LOG2T), mask=live)


@triton.jit
def _contains_keys(table_ptr, keys_ptr, out_ptr, n, LOG2T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    tl.store(out_ptr + i, _probe(table_ptr, tl.load(keys_ptr + i, mask=live, other=0), live, LOG2T), mask=live)


class HashSet:
    """The set of atom keys ``(p * base + s) * base + o`` of ``atoms [N, 3]`` (or of any non-negative int64 keys:
    :meth:`of_keys`)."""

    def __init__(self, atoms: Tensor, base: int) -> None:
        self._build(torch.unique((atoms[:, 0].long() * base + atoms[:, 1]) * base + atoms[:, 2]), base)

    @classmethod
    def of_keys(cls, keys: Tensor) -> "HashSet":
        """The set of ``keys`` (repeats allowed)."""
        out = cls.__new__(cls)
        out._build(keys.contiguous(), 0)
        return out

    def _build(self, keys: Tensor, base: int) -> None:
        self.base, self.log2t = base, max(4, (2 * keys.numel() - 1).bit_length())
        self.table = torch.full((1 << self.log2t,), EMPTY, dtype=torch.int64, device=keys.device)
        if keys.numel():
            _insert[(triton.cdiv(keys.numel(), 1024),)](self.table, keys, keys.numel(), LOG2T=self.log2t, BLOCK=1024)

    def contains_keys(self, keys: Tensor) -> Tensor:
        """``[...] -> [...]`` membership of int64 keys."""
        flat = keys.reshape(-1).contiguous()
        out = torch.empty(flat.shape[0], dtype=torch.bool, device=flat.device)
        if flat.shape[0]:
            _contains_keys[(triton.cdiv(flat.shape[0], 1024),)](self.table, flat, out, flat.shape[0],
                                                                 LOG2T=self.log2t, BLOCK=1024)
        return out.view(keys.shape)

    def contains(self, atoms: Tensor) -> Tensor:
        """``[..., 3] -> [...]`` membership."""
        flat = atoms.reshape(-1, 3).long().contiguous()
        out = torch.empty(flat.shape[0], dtype=torch.bool, device=flat.device)
        if flat.shape[0]:
            _contains[(triton.cdiv(flat.shape[0], 1024),)](self.table, flat, out, flat.shape[0], 3, self.base,
                                                            LOG2T=self.log2t, BLOCK=1024)
        return out.view(atoms.shape[:-1])


@triton.jit
def _expand(row_ptr, src_ptr, W, rule_ptr, n_ptr, at_ptr, start_ptr, one_ptr, values_ptr, n_values,
            nb_src_ptr, nb_pred_ptr, nb_dir_ptr, nb_one_ptr, slot_start_ptr, slot_count_ptr, P, E,
            arg_src_ptr, body_pred_ptr, n_body_ptr, known_ptr, hpm_ptr, HP, table_ptr, base, width,
            out_src_ptr, out_rule_ptr, out_n_ptr, out_start_ptr, out_count_ptr, out_one_ptr, counter_ptr, total,
            M: tl.constexpr, HPM: tl.constexpr, LOG2T: tl.constexpr, BLOCK: tl.constexpr):
    """Candidate rows ``[pid * BLOCK, +BLOCK)``, one lane each: candidate ``t`` is candidate ``t - at[row]`` of source
    row ``row[t]`` for its next free variable (``values[start + j]``, or 0 with ``one``), appended to the row's
    ``src [., W]``; with the row's rule and goal, and its CSR slice for the variable after it, the last (its bound
    source column ``nb_src[rule]``, predicate, direction; ``nb_one[rule]``: one row, value 0). A candidate whose
    groundings cannot be kept whatever the last variable takes (:func:`_live_count`'s test) is dropped; the others are
    appended (a per-block atomic append)."""
    t = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    on = t < total
    r = tl.load(row_ptr + t, mask=on, other=0)
    rule = tl.load(rule_ptr + r, mask=on, other=0).to(tl.int64)
    one = tl.load(one_ptr + r, mask=on, other=0) != 0
    j = t - tl.load(at_ptr + r, mask=on, other=0)
    v = tl.load(values_ptr + tl.minimum(tl.load(start_ptr + r, mask=on, other=0) + j, n_values - 1),
                mask=on & ~one, other=0)
    v = tl.where(one, 0, v)
    b = tl.load(nb_src_ptr + rule, mask=on, other=0)
    p = tl.load(nb_pred_ptr + rule, mask=on, other=0)
    d = tl.load(nb_dir_ptr + rule, mask=on, other=0)
    one_next = tl.load(nb_one_ptr + rule, mask=on, other=1) != 0
    bound = tl.where(b >= W, v, tl.load(src_ptr + r * W + tl.minimum(b, W - 1), mask=on, other=0))
    ok = (p < P) & (bound < E)
    slot = (d != 0).to(tl.int64) * (P * E) + tl.minimum(p, P - 1) * E + tl.minimum(bound, E - 1)
    s_start = tl.load(slot_start_ptr + slot, mask=on & ~one_next, other=0)
    s_count = tl.where(one_next, 1, tl.load(slot_count_ptr + slot, mask=on & ~one_next & ok, other=0))
    # the live test of the new row: columns <= W bound (column W is v), the last variable's free
    n_body = tl.load(n_body_ptr + rule, mask=on, other=0)
    unknown = tl.zeros([BLOCK], dtype=tl.int32)
    bad = on & (n_body < 0)
    for m in tl.static_range(M):
        active = on & (m < n_body) & (tl.load(known_ptr + rule * M + m, mask=on, other=1) == 0)
        a0 = tl.load(arg_src_ptr + (rule * M + m) * 2, mask=active, other=0)
        a1 = tl.load(arg_src_ptr + (rule * M + m) * 2 + 1, mask=active, other=0)
        q = tl.load(body_pred_ptr + rule * M + m, mask=active, other=0)
        x0 = tl.where(a0 == W, v, tl.load(src_ptr + r * W + tl.minimum(a0, W - 1), mask=active & (a0 < W), other=0))
        x1 = tl.where(a1 == W, v, tl.load(src_ptr + r * W + tl.minimum(a1, W - 1), mask=active & (a1 < W), other=0))
        absent = _absent(q, x0, x1, a0 <= W, a1 <= W, active, slot_count_ptr, P, E, table_ptr, base, LOG2T)
        unknown += absent.to(tl.int32)
        if HPM:
            bad = bad | (absent & (tl.load(hpm_ptr + tl.minimum(q, HP - 1), mask=absent, other=1) == 0))
    keep = on & ~bad & (unknown <= width) & (s_count > 0)
    k = keep.to(tl.int32)
    at = (tl.atomic_add(counter_ptr, tl.sum(k, 0)) + tl.cumsum(k, 0) - k).to(tl.int64)
    for c in range(W):
        tl.store(out_src_ptr + at * (W + 1) + c, tl.load(src_ptr + r * W + c, mask=keep, other=0), mask=keep)
    tl.store(out_src_ptr + at * (W + 1) + W, v, mask=keep)
    tl.store(out_rule_ptr + at, rule, mask=keep)
    tl.store(out_n_ptr + at, tl.load(n_ptr + r, mask=keep, other=0), mask=keep)
    tl.store(out_start_ptr + at, s_start, mask=keep)
    tl.store(out_count_ptr + at, s_count, mask=keep)
    tl.store(out_one_ptr + at, one_next.to(tl.int8), mask=keep)


def expand(src: Tensor, rule: Tensor, n: Tensor, start: Tensor, count: Tensor, one: Tensor, values: Tensor,
           next_src: Tensor, next_pred: Tensor, next_dir: Tensor, next_one: Tensor, slot_start: Tensor,
           slot_count: Tensor, P: int, E: int, arg_src: Tensor, body_pred: Tensor, n_body: Tensor, known: Tensor,
           head_pred_mask: Optional[Tensor], facts: HashSet, width: int):
    """Rows ``src [T, W]`` expanded by their candidates for the free variable before the last (``values[start : start
    + count]``, ``one``: a single 0): ``(src [T', W + 1], rule, n)`` and each new row's CSR slice ``(start, count,
    one)`` for the last variable (per rule: its bound source column, predicate, direction and ``next_one``), only the
    rows whose groundings could be kept (:func:`last_stage`'s test: ``arg_src``, ``body_pred``, ``n_body``, ``known``,
    ``head_pred_mask``, ``width``), in no fixed order."""
    T, W = src.shape
    total = int(count.sum())
    row = torch.repeat_interleave(torch.arange(T, device=src.device), count, output_size=total)
    dev = src.device
    out = (torch.empty(total, W + 1, dtype=src.dtype, device=dev), torch.empty(total, dtype=rule.dtype, device=dev),
           torch.empty(total, dtype=n.dtype, device=dev), torch.empty(total, dtype=slot_start.dtype, device=dev),
           torch.empty(total, dtype=slot_count.dtype, device=dev), torch.empty(total, dtype=torch.int8, device=dev))
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    if total:
        hpm = (head_pred_mask.to(torch.int8) if head_pred_mask is not None
               else torch.zeros(1, dtype=torch.int8, device=dev))
        _expand[(triton.cdiv(total, 1024),)](
            row, src.contiguous(), W, rule.contiguous(), n.contiguous(), count.cumsum(0) - count, start.contiguous(),
            one.to(torch.int8).contiguous(), values, values.numel(), next_src.contiguous(), next_pred.contiguous(),
            next_dir.contiguous(), next_one.to(torch.int8).contiguous(), slot_start, slot_count, P, E,
            arg_src.contiguous(), body_pred.contiguous(), n_body.contiguous(), known.to(torch.int8).contiguous(), hpm,
            hpm.numel(), facts.table, facts.base, width, *out, counter, total, M=body_pred.shape[1],
            HPM=head_pred_mask is not None, LOG2T=facts.log2t, BLOCK=1024)
    k = int(counter.item())
    return out[0][:k], out[1][:k], out[2][:k], out[3][:k], out[4][:k], out[5][:k].bool()


@triton.jit
def _absent(p, x0, x1, bound0, bound1, active, slot_count_ptr, PF, EF, table_ptr, base, LOG2T: tl.constexpr):
    """Whether body atom ``p(x0, x1)`` is surely not a fact, whatever its free arguments take: both bound and not a
    fact, or one free and the other bound to a constant with no fact of ``p`` on that side (the fact index's slot
    counts: by object, direction 1, when the subject is free; by subject when the object is)."""
    both = active & bound0 & bound1
    absent = both & ~_probe(table_ptr, (p * base + x0) * base + x1, both, LOG2T)
    by_obj = active & ~bound0 & bound1
    one_side = by_obj | (active & bound0 & ~bound1)
    side = tl.where(by_obj, x1, x0)
    ok = (p < PF) & (side < EF)
    slot = by_obj.to(tl.int64) * (PF * EF) + tl.minimum(p, PF - 1) * EF + tl.minimum(side, EF - 1)
    return absent | (one_side & (tl.load(slot_count_ptr + slot, mask=one_side & ok, other=0) == 0))


@triton.jit
def _live_count(src_ptr, W, rule_ptr, count_ptr, arg_src_ptr, body_pred_ptr, n_body_ptr, known_ptr, hpm_ptr, P,
                slot_count_ptr, PF, EF, table_ptr, base, width, out_ptr, T,
                M: tl.constexpr, HPM: tl.constexpr, SKIP_KNOWN: tl.constexpr, LOG2T: tl.constexpr,
                BLOCK: tl.constexpr):
    """Each row's candidate count, or 0 when none of its candidates can be kept: more body atoms than ``width`` are
    surely unknown (:func:`_absent`), or one of them has an unprovable predicate. Columns ``>= W`` of a row are free.
    ``SKIP_KNOWN``: the atoms an enumeration draws from the facts are not tested (their variable is enumerated: a fact
    whatever it takes)."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = i < T
    rule = tl.load(rule_ptr + i, mask=live, other=0).to(tl.int64)
    n_body = tl.load(n_body_ptr + rule, mask=live, other=0)
    unknown = tl.zeros([BLOCK], dtype=tl.int32)
    bad = live & (n_body < 0)
    for m in tl.static_range(M):
        active = live & (m < n_body)
        if SKIP_KNOWN:
            active = active & (tl.load(known_ptr + rule * M + m, mask=live, other=1) == 0)
        a0 = tl.load(arg_src_ptr + (rule * M + m) * 2, mask=active, other=0)
        a1 = tl.load(arg_src_ptr + (rule * M + m) * 2 + 1, mask=active, other=0)
        p = tl.load(body_pred_ptr + rule * M + m, mask=active, other=0)
        x0 = tl.load(src_ptr + i * W + a0, mask=active & (a0 < W), other=0)
        x1 = tl.load(src_ptr + i * W + a1, mask=active & (a1 < W), other=0)
        absent = _absent(p, x0, x1, a0 < W, a1 < W, active, slot_count_ptr, PF, EF, table_ptr, base, LOG2T)
        unknown += absent.to(tl.int32)
        if HPM:
            bad = bad | (absent & (tl.load(hpm_ptr + tl.minimum(p, P - 1), mask=absent, other=1) == 0))
    count = tl.load(count_ptr + i, mask=live, other=0)
    tl.store(out_ptr + i, tl.where(bad | (unknown > width), 0, count), mask=live)


@triton.jit
def _probe3(table_ptr, key, live, LOG2T: tl.constexpr):
    """:func:`_probe` of a 3-D block of keys."""
    s = _slot(key, LOG2T)
    found = key < 0
    todo = live
    while tl.max(tl.max(tl.max(todo.to(tl.int32), 2), 1), 0) > 0:
        v = tl.load(table_ptr + s, mask=todo, other=-1)
        found = found | (todo & (v == key))
        todo = todo & (v != key) & (v != -1)
        s = (s + 1) & ((1 << LOG2T) - 1)
    return found


@triton.jit
def _last_stage(order_ptr, src_ptr, W, rule_ptr, n_ptr, goal_ptr, start_ptr, count_ptr, one_ptr, values_ptr, n_values,
                arg_src_ptr, body_pred_ptr, n_body_ptr, known_ptr, hpm_ptr, P, table_ptr, base, pad, width,
                out_row_ptr, out_val_ptr, counter_ptr, T,
                M: tl.constexpr, MP: tl.constexpr, U: tl.constexpr, HPM: tl.constexpr, LOG2T: tl.constexpr,
                BLOCK: tl.constexpr):
    """Rows ``order[pid * BLOCK : +BLOCK]`` (rows ordered by candidate count: a block's lanes walk as many): each
    walks its ``count`` candidates (``values[start + j]``, or one 0 with ``one``), ``U`` at a time, as the source
    column ``W`` (columns past it read 0); the kept (row, value) pairs are appended. A row's body atoms (``MP``, the
    body length's power of two) are read once; those without the candidate are tested once, the others per
    candidate (``[BLOCK, U, MP]`` blocks: ``U`` probes in flight per row)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < T
    r = tl.load(order_ptr + i, mask=live, other=0)
    r64 = r.to(tl.int64)
    rule = tl.load(rule_ptr + r64, mask=live, other=0).to(tl.int64)
    count = tl.load(count_ptr + r64, mask=live, other=0)
    start = tl.load(start_ptr + r64, mask=live, other=0)
    one = tl.load(one_ptr + r64, mask=live, other=0) != 0
    n_body = tl.load(n_body_ptr + rule, mask=live, other=0)
    goal = tl.load(n_ptr + r64, mask=live, other=0).to(tl.int64) * 3
    g0 = tl.load(goal_ptr + goal, mask=live, other=0)[:, None, None]
    g1 = tl.load(goal_ptr + goal + 1, mask=live, other=0)[:, None, None]
    g2 = tl.load(goal_ptr + goal + 2, mask=live, other=0)[:, None, None]
    m = tl.arange(0, MP)[None, None, :]
    act = live[:, None, None] & (m < n_body[:, None, None]) & (m < M)          # [BLOCK, 1, MP] the body atoms
    rm = rule[:, None, None] * M + m
    a0 = tl.load(arg_src_ptr + rm * 2, mask=act, other=0)
    a1 = tl.load(arg_src_ptr + rm * 2 + 1, mask=act, other=0)
    p = tl.load(body_pred_ptr + rm, mask=act, other=0)
    known = tl.load(known_ptr + rm, mask=act, other=0) != 0                  # enumerated: a fact
    b0 = tl.load(src_ptr + r64[:, None, None] * W + a0, mask=act & (a0 < W), other=0)
    b1 = tl.load(src_ptr + r64[:, None, None] * W + a1, mask=act & (a1 < W), other=0)
    uses_v = (a0 == W) | (a1 == W)
    fixed = act & ~uses_v                                                     # the same atom for every candidate
    exists_fixed = known | _probe3(table_ptr, (p * base + b0) * base + b1, fixed & ~known, LOG2T)
    unknown_fixed = tl.sum((fixed & ~exists_fixed).to(tl.int32), 2)                         # [BLOCK, 1]
    bad_fixed = (live & (n_body <= 0))[:, None]
    bad_fixed = bad_fixed | (tl.max((fixed & (p == g0) & (b0 == g1) & (b1 == g2)).to(tl.int32), 2) > 0)
    varying = act & uses_v
    if HPM:                                        # an unknown atom must be provable
        unprovable = tl.load(hpm_ptr + tl.minimum(p, P - 1), mask=act, other=1) == 0
        bad_fixed = bad_fixed | (tl.max((fixed & ~exists_fixed & unprovable).to(tl.int32), 2) > 0)
    u = tl.arange(0, U)[None, :]
    j_end = tl.max(tl.where(live, count, 0), 0)
    j = 0
    while j < j_end:
        jj = j + u                                                                           # [1, U]
        on = live[:, None] & (jj < count[:, None])                                           # [BLOCK, U]
        v = tl.load(values_ptr + tl.minimum(start[:, None] + jj, n_values - 1), mask=on & ~one[:, None], other=0)
        v = tl.where(one[:, None], 0, v)
        x0 = tl.where(a0 == W, v[:, :, None], b0)
        x1 = tl.where(a1 == W, v[:, :, None], b1)
        test = varying & on[:, :, None]
        exists = known | _probe3(table_ptr, (p * base + x0) * base + x1, test & ~known, LOG2T)
        missing = test & ~exists
        unknown = unknown_fixed + tl.sum(missing.to(tl.int32), 2)
        bad = bad_fixed | (tl.max((test & (p == g0) & (x0 == g1) & (x1 == g2)).to(tl.int32), 2) > 0)
        if HPM:
            bad = bad | (tl.max((missing & unprovable).to(tl.int32), 2) > 0)
        keep = on & ~bad & (unknown <= width)
        keep = tl.reshape(keep, [BLOCK * U])
        k = keep.to(tl.int32)
        at = tl.atomic_add(counter_ptr, tl.sum(k, 0)) + tl.cumsum(k, 0) - k
        tl.store(out_row_ptr + at, tl.reshape(r[:, None] + tl.zeros([BLOCK, U], dtype=tl.int32), [BLOCK * U]),
                 mask=keep)
        tl.store(out_val_ptr + at, tl.reshape(v.to(tl.int32), [BLOCK * U]), mask=keep)
        j += U


@triton.jit
def _live_atoms(atoms_ptr, n, rmask_ptr, ridx_ptr, PR, RMAX, ok_s_ptr, ok_o_ptr, E, arg_src_ptr, body_pred_ptr,
                n_body_ptr, table_ptr, base, out_ptr, M: tl.constexpr, LOG2T: tl.constexpr, BLOCK: tl.constexpr):
    """Whether some rule of each atom's predicate could ground it with every body atom a fact (:func:`live_atoms`)."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    p = tl.load(atoms_ptr + i * 3, mask=live, other=0)
    s = tl.load(atoms_ptr + i * 3 + 1, mask=live, other=0)
    o = tl.load(atoms_ptr + i * 3 + 2, mask=live, other=0)
    pc = tl.minimum(p, PR - 1).to(tl.int64)
    se, oe = tl.minimum(s, E).to(tl.int64), tl.minimum(o, E).to(tl.int64)
    res = live & (p < 0)                                                    # all False
    for j in range(RMAX):
        ok = live & (p < PR) & (tl.load(rmask_ptr + pc * RMAX + j, mask=live, other=0) != 0)
        r = tl.load(ridx_ptr + pc * RMAX + j, mask=ok, other=0).to(tl.int64)
        n_body = tl.load(n_body_ptr + r, mask=ok, other=0)
        ok = ok & (n_body > 0) & (tl.load(ok_s_ptr + r * (E + 1) + se, mask=ok, other=0) != 0) \
            & (tl.load(ok_o_ptr + r * (E + 1) + oe, mask=ok, other=0) != 0)
        for m in tl.static_range(M):
            a0 = tl.load(arg_src_ptr + (r * M + m) * 2, mask=ok, other=2)
            a1 = tl.load(arg_src_ptr + (r * M + m) * 2 + 1, mask=ok, other=2)
            q = tl.load(body_pred_ptr + r * M + m, mask=ok, other=0)
            both = ok & (m < n_body) & (a0 < 2) & (a1 < 2)                      # both arguments the atom's own
            x0 = tl.where(a0 == 0, s, o)
            x1 = tl.where(a1 == 0, s, o)
            ok = ok & (~both | _probe(table_ptr, (q * base + x0) * base + x1, both, LOG2T))
        res = res | ok
    tl.store(out_ptr + i, res.to(tl.int8), mask=live)


def live_atoms(atoms: Tensor, rule_mask: Tensor, rule_idx: Tensor, ok_s: Tensor, ok_o: Tensor, arg_src: Tensor,
               body_pred: Tensor, n_body: Tensor, facts: HashSet) -> Tensor:
    """``[N]`` bool: whether some rule (of ``rule_idx [P, RMAX]`` where ``rule_mask``) could ground each atom
    ``(p, s, o)`` of ``atoms [N, 3]`` with every body atom a fact — the necessary tests of :func:`_live_count` at width
    0 with every free variable unbound, from per-rule tables: ``ok_s [R, E + 1]`` / ``ok_o`` (every body atom with the
    head's subject / object and a free variable has a fact on that side for that entity; column ``E``: an entity with
    no facts), and a probe of the body atoms whose arguments are both the head's."""
    out = torch.empty(atoms.shape[0], dtype=torch.int8, device=atoms.device)
    if atoms.shape[0]:
        PR, RMAX = rule_mask.shape
        _live_atoms[(triton.cdiv(atoms.shape[0], 1024),)](
            atoms.contiguous(), atoms.shape[0], rule_mask.to(torch.int8).contiguous(), rule_idx.contiguous(), PR, RMAX,
            ok_s, ok_o, ok_s.shape[1] - 1, arg_src.contiguous(), body_pred.contiguous(), n_body.contiguous(),
            facts.table, facts.base, out, M=body_pred.shape[1], LOG2T=facts.log2t, BLOCK=1024)
    return out.bool()


def live_count(src: Tensor, rule: Tensor, count: Tensor, arg_src: Tensor, body_pred: Tensor, n_body: Tensor,
               known: Tensor, head_pred_mask: Optional[Tensor], facts: HashSet, width: int, slot_count: Tensor, P: int,
               E: int, *, skip_known: bool) -> Tensor:
    """``[T]`` int32: each row's ``count``, or 0 when no grounding of it can be kept (:func:`_live_count`; columns
    ``>= W`` of ``src [T, W]`` free)."""
    T, W = src.shape
    live = torch.empty(T, dtype=torch.int32, device=src.device)
    if T:
        hpm = (head_pred_mask.to(torch.int8) if head_pred_mask is not None
               else torch.zeros(1, dtype=torch.int8, device=src.device))
        _live_count[(triton.cdiv(T, 1024),)](src.contiguous(), W, rule.contiguous(), count.contiguous(),
                                             arg_src.contiguous(), body_pred.contiguous(), n_body.contiguous(),
                                             known.to(torch.int8).contiguous(), hpm, hpm.numel(), slot_count, P, E,
                                             facts.table, facts.base, width, live, T, M=body_pred.shape[1],
                                             HPM=head_pred_mask is not None, SKIP_KNOWN=skip_known,
                                             LOG2T=facts.log2t, BLOCK=1024)
    return live


def last_stage(src: Tensor, rule: Tensor, n: Tensor, goals: Tensor, start: Tensor, count: Tensor, one: Tensor,
               values: Tensor, arg_src: Tensor, body_pred: Tensor, n_body: Tensor, known: Tensor,
               head_pred_mask: Optional[Tensor], facts: HashSet, pad: int, width: int, slot_count: Tensor, P: int,
               E: int) -> Tuple[Tensor, Tensor]:
    """The kept ``(row, value)`` pairs of rows ``src [T, W]`` (rule ``rule [T]``, goal ``goals[n] [T, 3]``) whose
    source column ``W`` walks ``values[start : start + count]`` (``one``: a single 0); ``arg_src [R, M, 2]`` holds
    each body argument's source column (past ``W``: 0), ``known [R, M]`` the body atoms an enumeration drew from
    the facts (not probed). Rows that cannot keep a candidate (:func:`_live_count`, from the fact index's slot
    counts ``slot_count`` over ``P`` predicates and ``E`` entities) are not walked."""
    T, W = src.shape
    dev = src.device
    src, rule = src.contiguous(), rule.contiguous()
    live = live_count(src, rule, count, arg_src, body_pred, n_body, known, head_pred_mask, facts, width, slot_count, P,
                      E, skip_known=True)
    total, n_live = torch.stack([live.sum(), (live > 0).sum()]).tolist() if T else (0, 0)
    known8, M = known.to(torch.int8).contiguous(), body_pred.shape[1]
    hpm = (head_pred_mask.to(torch.int8) if head_pred_mask is not None
           else torch.zeros(1, dtype=torch.int8, device=dev))
    rows = torch.empty(total, dtype=torch.int32, device=dev)
    vals = torch.empty(total, dtype=torch.int32, device=dev)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    if n_live:
        # the live rows by candidate count (a block's lanes walk as many; an 8-bit key: one radix pass)
        order = torch.sort(live.clamp(max=255).to(torch.uint8))[1][T - n_live:].int()
        _last_stage[(triton.cdiv(n_live, BLOCK),)](
            order, src, W, rule, n.contiguous(), goals.contiguous(), start.contiguous(), live,
            one.to(torch.int8).contiguous(), values, values.numel(), arg_src.contiguous(), body_pred.contiguous(),
            n_body.contiguous(), known8, hpm, hpm.numel(), facts.table, facts.base, pad, width, rows, vals, counter,
            n_live, M=M, MP=triton.next_power_of_2(M), U=UNROLL, HPM=head_pred_mask is not None,
            LOG2T=facts.log2t, BLOCK=BLOCK)
    k = int(counter.item())
    rows, vals = rows[:k], vals[:k]
    return rows.long(), vals.long()
