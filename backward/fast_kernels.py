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
BLOCK = 256


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


class HashSet:
    """The set of atom keys ``(p * base + s) * base + o`` of ``atoms [N, 3]``."""

    def __init__(self, atoms: Tensor, base: int) -> None:
        keys = torch.unique((atoms[:, 0].long() * base + atoms[:, 1]) * base + atoms[:, 2])
        self.base, self.log2t = base, max(4, (2 * keys.numel() - 1).bit_length())
        self.table = torch.full((1 << self.log2t,), EMPTY, dtype=torch.int64, device=atoms.device)
        if keys.numel():
            _insert[(triton.cdiv(keys.numel(), 1024),)](self.table, keys, keys.numel(), LOG2T=self.log2t, BLOCK=1024)

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
            out_src_ptr, out_rule_ptr, out_n_ptr, out_start_ptr, out_count_ptr, out_one_ptr, total,
            BLOCK: tl.constexpr):
    """Output rows ``[pid * BLOCK, +BLOCK)``, one lane each: output row ``t`` is candidate ``t - at[row]`` of source
    row ``row[t]`` for its next free variable (``values[start + j]``, or 0 with ``one``), appended to the row's
    ``src [., W]``; with the row's rule and goal, and its CSR slice for the variable after it (its bound source column
    ``nb_src[rule]``, predicate, direction; ``nb_one[rule]``: one row, value 0)."""
    t = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    on = t < total
    r = tl.load(row_ptr + t, mask=on, other=0)
    rule = tl.load(rule_ptr + r, mask=on, other=0).to(tl.int64)
    one = tl.load(one_ptr + r, mask=on, other=0) != 0
    j = t - tl.load(at_ptr + r, mask=on, other=0)
    v = tl.load(values_ptr + tl.minimum(tl.load(start_ptr + r, mask=on, other=0) + j, n_values - 1),
                mask=on & ~one, other=0)
    v = tl.where(one, 0, v)
    for c in range(W):
        tl.store(out_src_ptr + t * (W + 1) + c, tl.load(src_ptr + r * W + c, mask=on, other=0), mask=on)
    tl.store(out_src_ptr + t * (W + 1) + W, v, mask=on)
    b = tl.load(nb_src_ptr + rule, mask=on, other=0)
    p = tl.load(nb_pred_ptr + rule, mask=on, other=0)
    d = tl.load(nb_dir_ptr + rule, mask=on, other=0)
    one_next = tl.load(nb_one_ptr + rule, mask=on, other=1) != 0
    bound = tl.where(b >= W, v, tl.load(src_ptr + r * W + tl.minimum(b, W - 1), mask=on, other=0))
    ok = (p < P) & (bound < E)
    slot = (d != 0).to(tl.int64) * (P * E) + tl.minimum(p, P - 1) * E + tl.minimum(bound, E - 1)
    s_start = tl.load(slot_start_ptr + slot, mask=on & ~one_next, other=0)
    s_count = tl.where(one_next, 1, tl.load(slot_count_ptr + slot, mask=on & ~one_next & ok, other=0))
    tl.store(out_rule_ptr + t, rule, mask=on)
    tl.store(out_n_ptr + t, tl.load(n_ptr + r, mask=on, other=0), mask=on)
    tl.store(out_start_ptr + t, s_start, mask=on)
    tl.store(out_count_ptr + t, s_count, mask=on)
    tl.store(out_one_ptr + t, one_next.to(tl.int8), mask=on)


def expand(src: Tensor, rule: Tensor, n: Tensor, start: Tensor, count: Tensor, one: Tensor, values: Tensor,
           next_src: Tensor, next_pred: Tensor, next_dir: Tensor, next_one: Tensor, slot_start: Tensor,
           slot_count: Tensor, P: int, E: int):
    """Rows ``src [T, W]`` expanded by their candidates for a free variable (``values[start : start + count]``,
    ``one``: a single 0), row-major: ``(src [T', W + 1], rule, n)`` and each new row's CSR slice ``(start, count,
    one)`` for the next variable (per rule: its bound source column, predicate, direction and ``next_one``)."""
    T, W = src.shape
    total = int(count.sum())
    row = torch.repeat_interleave(torch.arange(T, device=src.device), count, output_size=total)
    dev = src.device
    out = (torch.empty(total, W + 1, dtype=src.dtype, device=dev), torch.empty(total, dtype=rule.dtype, device=dev),
           torch.empty(total, dtype=n.dtype, device=dev), torch.empty(total, dtype=slot_start.dtype, device=dev),
           torch.empty(total, dtype=slot_count.dtype, device=dev), torch.empty(total, dtype=torch.int8, device=dev))
    if total:
        _expand[(triton.cdiv(total, 1024),)](
            row, src.contiguous(), W, rule.contiguous(), n.contiguous(), count.cumsum(0) - count, start.contiguous(),
            one.to(torch.int8).contiguous(), values, values.numel(), next_src.contiguous(), next_pred.contiguous(),
            next_dir.contiguous(), next_one.to(torch.int8).contiguous(), slot_start, slot_count, P, E, *out, total,
            BLOCK=1024)
    return out[0], out[1], out[2], out[3], out[4], out[5].bool()


@triton.jit
def _last_stage(order_ptr, src_ptr, W, rule_ptr, n_ptr, goal_ptr, start_ptr, count_ptr, one_ptr, values_ptr, n_values,
                arg_src_ptr, body_pred_ptr, n_body_ptr, known_ptr, hpm_ptr, P, table_ptr, base, pad, width,
                out_row_ptr, out_val_ptr, counter_ptr, T,
                M: tl.constexpr, HPM: tl.constexpr, LOG2T: tl.constexpr, BLOCK: tl.constexpr):
    """Rows ``order[pid * BLOCK : +BLOCK]`` (rows ordered by candidate count: a block's lanes walk as many): each
    walks its ``count`` candidates (``values[start + j]``, or one 0 with ``one``) as the source column ``W`` (columns
    past it read 0); the kept (row, value) pairs are appended."""
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
    g0 = tl.load(goal_ptr + goal, mask=live, other=0)
    g1 = tl.load(goal_ptr + goal + 1, mask=live, other=0)
    g2 = tl.load(goal_ptr + goal + 2, mask=live, other=0)
    j_end = tl.max(tl.where(live, count, 0), 0)
    j = 0
    while j < j_end:
        on = live & (j < count)
        v = tl.load(values_ptr + tl.minimum(start + j, n_values - 1), mask=on & ~one, other=0)
        v = tl.where(one, 0, v)
        unknown = tl.zeros([BLOCK], dtype=tl.int32)
        bad = on & (n_body <= 0)
        for m in tl.static_range(M):
            active = on & (m < n_body)
            a0 = tl.load(arg_src_ptr + (rule * M + m) * 2, mask=active, other=0)
            a1 = tl.load(arg_src_ptr + (rule * M + m) * 2 + 1, mask=active, other=0)
            p = tl.load(body_pred_ptr + rule * M + m, mask=active, other=0)
            x0 = tl.where(a0 == W, v, tl.load(src_ptr + r64 * W + a0, mask=active & (a0 < W), other=0))
            x1 = tl.where(a1 == W, v, tl.load(src_ptr + r64 * W + a1, mask=active & (a1 < W), other=0))
            known = tl.load(known_ptr + rule * M + m, mask=active, other=0) != 0     # enumerated: a fact
            exists = known | _probe(table_ptr, (p * base + x0) * base + x1, active & ~known, LOG2T)
            unknown += (active & ~exists).to(tl.int32)
            bad = bad | (active & (p == g0) & (x0 == g1) & (x1 == g2))           # a body atom is the goal
            if HPM:                                        # an unknown atom must be provable
                h = tl.load(hpm_ptr + tl.minimum(p, P - 1), mask=active & ~exists, other=1)
                bad = bad | (active & ~exists & (h == 0))
        keep = on & ~bad & (unknown <= width)
        k = keep.to(tl.int32)
        at = tl.atomic_add(counter_ptr, tl.sum(k, 0)) + tl.cumsum(k, 0) - k
        tl.store(out_row_ptr + at, r, mask=keep)
        tl.store(out_val_ptr + at, v.to(tl.int32), mask=keep)
        j += 1


def last_stage(src: Tensor, rule: Tensor, n: Tensor, goals: Tensor, start: Tensor, count: Tensor, one: Tensor,
               values: Tensor, arg_src: Tensor, body_pred: Tensor, n_body: Tensor, known: Tensor,
               head_pred_mask: Optional[Tensor], facts: HashSet, pad: int, width: int) -> Tuple[Tensor, Tensor]:
    """The kept ``(row, value)`` pairs of rows ``src [T, W]`` (rule ``rule [T]``, goal ``goals[n] [T, 3]``) whose
    source column ``W`` walks ``values[start : start + count]`` (``one``: a single 0); ``arg_src [R, M, 2]`` holds
    each body argument's source column (past ``W``: 0), ``known [R, M]`` the body atoms an enumeration drew from
    the facts (not probed)."""
    T, W = src.shape
    total = int(count.sum())
    rows = torch.empty(total, dtype=torch.int32, device=src.device)
    vals = torch.empty(total, dtype=torch.int32, device=src.device)
    counter = torch.zeros(1, dtype=torch.int32, device=src.device)
    if T and total:
        hpm = head_pred_mask.to(torch.int8) if head_pred_mask is not None else counter
        order = torch.sort(count.int())[1].int()
        _last_stage[(triton.cdiv(T, BLOCK),)](
            order, src.contiguous(), W, rule.contiguous(), n.contiguous(), goals.contiguous(), start.contiguous(),
            count.contiguous(), one.to(torch.int8).contiguous(), values, values.numel(), arg_src.contiguous(),
            body_pred.contiguous(), n_body.contiguous(), known.to(torch.int8).contiguous(), hpm, hpm.numel(), facts.table, facts.base, pad, width, rows,
            vals, counter, T, M=body_pred.shape[1], HPM=head_pred_mask is not None, LOG2T=facts.log2t, BLOCK=BLOCK)
    k = int(counter.item())
    return rows[:k].long(), vals[:k].long()
