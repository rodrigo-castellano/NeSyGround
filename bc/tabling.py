"""Per-query grounding tabling cache (task #47, Design #2).

Grounding is **weight-independent**: for a fixed KB (facts + rules), each
query atom's RAW (pre-dedup, pre-prune) "considered" firing set is a pure
function of the atom. So recurring queries — train positives re-ground
every epoch, dev/test fixed across runs — can be served from a cache
instead of re-running the depth loop, byte-identically.

This module owns the cache hooks that wrap the single-batch forward chain
(:func:`grounder.bc.forward.forward_one_batch` /
:func:`grounder.bc.forward.forward_one_batch_inner`). It is a no-op unless
``grounder._tabling_enabled`` is True (default OFF), so the existing path
is untouched.

Cache layout
------------
``grounder._grounding_cache[key]`` maps an ``atom_hash`` key to a tuple
``(rule_idx[F], head[F, 3], body[F, M, 3], query_triple[3])`` of RAW
considered rows produced by that query atom. The stored ``query_triple``
is the TRIPLE-VERIFY guard: ``atom_hash`` has real collisions (e.g. on the
family w1d3 gate cell), so a hash HIT is only honoured when the stored
triple equals the live query triple; otherwise it is treated as a MISS.

Why RAW (pre-prune)
-------------------
The end-of-BFS ``prune_rule_groundings`` (fp_batch) is a CROSS-QUERY
fixpoint: it can resurrect a query's body atoms via firings from *other*
queries in the same batch. Caching post-prune rows would bake in one
batch's neighbours and corrupt a different batch. The considered
accumulator captured by :mod:`grounder.bc.considered` is exactly the RAW
pre-prune set, so we cache and replay that; the UNCHANGED finalize + prune
then runs over the reconstituted (miss + replayed-hit) accumulator.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from grounder.groundings import atom_hash


def tabling_active(grounder) -> bool:
    """Return True when the per-query tabling cache should run.

    Guarded hard (no silent fallback): the cache only applies to the eager
    enum FLAT considered path, never the compiled / dense path (which
    derives rule_groundings from evidence, not the considered accumulator).
    """
    if not getattr(grounder, "_tabling_enabled", False):
        return False
    # MANDATORY guardrails — assert rather than silently degrade.
    assert grounder.resolution == "enum", (
        "tabling requires resolution=='enum', "
        f"got {grounder.resolution!r}")
    assert getattr(grounder, "_flat_intermediate", False) \
        or getattr(grounder, "_flat_intermediate_flag", False), (
        "tabling requires the enum FLAT path (flat_intermediate=True)")
    assert not torch.compiler.is_compiling(), (
        "tabling must not run inside torch.compile tracing")
    assert not getattr(grounder, "_compiled", False), (
        "tabling is incompatible with the compiled/dense step path "
        "(it uses evidence-derived rule_groundings, not considered)")
    assert grounder._collect_rule_groundings, (
        "tabling requires collect_rule_groundings=True (it caches the "
        "considered accumulator)")
    # KB-immutability: the cache is only valid while facts + rules are
    # unchanged for the grounder's lifetime.
    stamp = (id(grounder.kb.fact_index), int(grounder.kb.num_rules))
    assert stamp == grounder._tabling_kb_stamp, (
        "tabling KB-immutability stamp changed "
        f"(was {grounder._tabling_kb_stamp}, now {stamp}); the fact index "
        "or rule count was mutated, invalidating every cached entry")
    return True


def split_queries(
    grounder, queries: Tensor, query_mask: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, List[int], List[int], List[int]]:
    """Partition queries into cache HITs and MISSes.

    Returns
    -------
    miss_queries : ``[Nm, 3]`` the MISS query atoms (depth loop runs on
        these only).
    miss_mask : ``[Nm]`` validity mask for the miss subset.
    keys : ``[N]`` per-query ``atom_hash`` keys (long, on CPU as a list-
        backing tensor for indexing the python dict).
    miss_positions : original query indices of the MISS rows (so the
        depth loop's miss-local results map back to global positions).
    hit_positions : original query indices of the HIT rows.
    hit_keys : the cache key for each HIT (parallel to ``hit_positions``).

    Only ACTIVE queries (``query_mask`` True) participate; inactive
    (padded) rows are routed to the MISS subset with mask=False so the
    depth loop skips them and they never enter the cache.
    """
    N = int(queries.size(0))
    # ``queries[:, :3]`` — the (pred, subj, obj) triple. Some callers pass
    # wider rows; the grounding key is the triple only.
    triples = queries[:, :3].long()
    keys_t = atom_hash(triples)                          # [N] long
    keys = keys_t.tolist()
    mask_list = query_mask.tolist()

    miss_positions: List[int] = []
    hit_positions: List[int] = []
    hit_keys: List[int] = []
    cache = grounder._grounding_cache
    for i in range(N):
        if not mask_list[i]:
            # Inactive / padded row — always a MISS (mask=False so the
            # depth loop skips it and writeback won't store it).
            miss_positions.append(i)
            continue
        k = keys[i]
        entry = cache.get(k)
        if entry is not None:
            stored_triple = entry[3]                     # [3] long
            # TRIPLE-VERIFY (mandatory): a hash HIT is only honoured when
            # the stored triple equals the live query triple.
            if torch.equal(stored_triple, triples[i]):
                hit_positions.append(i)
                hit_keys.append(k)
                continue
        miss_positions.append(i)

    dev = queries.device
    if miss_positions:
        midx = torch.tensor(miss_positions, dtype=torch.long, device=dev)
        miss_queries = queries.index_select(0, midx)
        miss_mask = query_mask.index_select(0, midx)
    else:
        miss_queries = queries[:0]
        miss_mask = query_mask[:0]
    return (miss_queries, miss_mask, keys_t,
            miss_positions, hit_positions, hit_keys)


def accumulator_marker(grounder) -> int:
    """Return the current length of the per-step accumulator lists.

    Recorded before a chunk's depth loop so :func:`writeback_and_replay`
    only processes the rows THIS chunk appended (the chunked path keeps
    accumulating across chunks into the same lists).
    """
    return len(grounder._considered_acc_rule)


def _concat_accumulator(
    grounder, marker: int, M: int, dev,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Concatenate the per-step accumulator lists from ``marker`` onward.

    Returns ``(rule_idx[T], head[T, 3], body[T, M, 3], bidx[T])`` for the
    firings appended since ``marker``.
    """
    accr = grounder._considered_acc_rule[marker:]
    if accr:
        rule_idx = torch.cat(accr, 0)
        head = torch.cat(grounder._considered_acc_head[marker:], 0)
        body = torch.cat(grounder._considered_acc_body[marker:], 0)
        bidx = torch.cat(grounder._considered_acc_bidx[marker:], 0)
    else:
        rule_idx = torch.zeros(0, dtype=torch.long, device=dev)
        head = torch.zeros(0, 3, dtype=torch.long, device=dev)
        body = torch.zeros(0, M, 3, dtype=torch.long, device=dev)
        bidx = torch.zeros(0, dtype=torch.long, device=dev)
    return rule_idx, head, body, bidx


def writeback_and_replay(
    grounder,
    queries: Tensor,
    keys: Tensor,
    miss_positions: List[int],
    hit_positions: List[int],
    hit_keys: List[int],
    marker: int = 0,
) -> None:
    """Store MISS firings into the cache, then replay HIT firings.

    Called AFTER the depth loop (which ran on the MISS subset only) and
    BEFORE the considered finalize. On entry, the considered accumulator's
    rows ``[marker:]`` hold ONLY this chunk's MISS firings, with ``bidx`` =
    miss-local global index (``_chunk_query_offset + miss_local_idx``).
    This:

    1. WRITE-BACK: partitions the MISS accumulator rows by the original
       query they belong to and stores the RAW (pre-dedup, pre-prune)
       ``(rule_idx, head, body, query_triple)`` for each MISS query under
       its ``atom_hash`` key.
    2. Rewrites the accumulator's ``[marker:]`` rows so every MISS firing's
       ``bidx`` is its ORIGINAL query index (not miss-local), and
    3. REPLAY: appends each HIT query's cached rows with the HIT query's
       ORIGINAL index as ``bidx``.

    After this, the accumulator's ``bidx`` for every firing is the
    original query index, so the downstream ``query_pool_idx`` is correct,
    and the UNCHANGED ``finalize`` + ``prune`` runs over the full
    (miss + hit) considered set.

    ``marker`` is the accumulator list length captured before this chunk's
    depth loop (see :func:`accumulator_marker`); rows before it belong to
    earlier chunks and are left untouched.
    """
    M = int(grounder.kb.M)
    dev = queries.device
    triples = queries[:, :3].long()
    offset = grounder._chunk_query_offset

    rule_idx, head, body, bidx = _concat_accumulator(grounder, marker, M, dev)
    # Keep prior chunks' rows intact; rebuild only the suffix.
    prefix_rule = grounder._considered_acc_rule[:marker]
    prefix_head = grounder._considered_acc_head[:marker]
    prefix_body = grounder._considered_acc_body[:marker]
    prefix_bidx = grounder._considered_acc_bidx[:marker]

    # Map miss-local global bidx (offset + miss_local) back to the original
    # query index. ``miss_positions[miss_local] = original_index``.
    n_miss = len(miss_positions)
    miss_local_to_orig = torch.full(
        (n_miss,), -1, dtype=torch.long, device=dev)
    if n_miss:
        miss_local_to_orig[:] = torch.tensor(
            miss_positions, dtype=torch.long, device=dev)

    if bidx.numel():
        miss_local = bidx - offset                       # [T]
        # All accumulated rows belong to the miss subset by construction.
        orig_bidx = miss_local_to_orig[miss_local]       # [T] original idx
    else:
        miss_local = bidx
        orig_bidx = bidx

    # ── WRITE-BACK ── group the MISS rows by their original query index.
    cache = grounder._grounding_cache
    cap = grounder._tabling_max_entries
    if n_miss and bidx.numel():
        # Stable sort by miss-local index so each query's rows are
        # contiguous; bincount → offsets gives the per-query slice.
        order = torch.argsort(miss_local, stable=True)
        ml_sorted = miss_local[order]
        ridx_s = rule_idx[order]
        head_s = head[order]
        body_s = body[order]
        counts = torch.bincount(ml_sorted, minlength=n_miss)
        cum = torch.zeros(n_miss + 1, dtype=torch.long, device=dev)
        cum[1:] = torch.cumsum(counts, 0)
        cum_list = cum.tolist()
    else:
        order = None
        cum_list = [0] * (n_miss + 1)

    for ml in range(n_miss):
        orig = miss_positions[ml]
        # Skip inactive / padded queries — they never produced firings and
        # must not pollute the cache.
        if order is not None:
            lo, hi = cum_list[ml], cum_list[ml + 1]
            r_rows = ridx_s[lo:hi]
            h_rows = head_s[lo:hi]
            b_rows = body_s[lo:hi]
        else:
            r_rows = rule_idx[:0]
            h_rows = head[:0]
            b_rows = body[:0]
        k = int(keys[orig].item())
        if k in cache:
            # Already present from a TRIPLE-VERIFY miss (hash collision):
            # the live triple didn't match the stored one. Do NOT overwrite
            # — keep serving the original key's owner; this query stays a
            # passthrough MISS (computed fresh, never cached under a colliding
            # key). This keeps the cache single-owner per key.
            continue
        if len(cache) >= cap:
            if not grounder._tabling_full_warned:
                warnings.warn(
                    "grounder tabling cache reached "
                    f"_tabling_max_entries={cap}; new query groundings will "
                    "be computed fresh (passthrough) and not cached. "
                    "Existing entries continue to serve hits.",
                    stacklevel=2,
                )
                grounder._tabling_full_warned = True
            continue
        cache[k] = (
            r_rows.clone(), h_rows.clone(), b_rows.clone(),
            triples[orig].clone(),
        )

    # ── Rewrite the accumulator's suffix: MISS bidx → ORIGINAL query index ──
    # Preserve earlier chunks' rows (``[:marker]``); replace this chunk's
    # rows with a single (remapped) block so finalize / query_pool_idx see
    # original indices for the MISS firings.
    grounder._considered_acc_rule = list(prefix_rule)
    grounder._considered_acc_head = list(prefix_head)
    grounder._considered_acc_body = list(prefix_body)
    grounder._considered_acc_bidx = list(prefix_bidx)
    if bidx.numel():
        grounder._considered_acc_rule.append(rule_idx)
        grounder._considered_acc_head.append(head)
        grounder._considered_acc_body.append(body)
        grounder._considered_acc_bidx.append(orig_bidx)

    # ── REPLAY ── append each HIT query's cached rows with its ORIGINAL
    # index as bidx.
    for orig, k in zip(hit_positions, hit_keys):
        r_rows, h_rows, b_rows, _triple = cache[k]
        F = int(r_rows.size(0))
        if F == 0:
            continue
        grounder._considered_acc_rule.append(r_rows.to(dev))
        grounder._considered_acc_head.append(h_rows.to(dev))
        grounder._considered_acc_body.append(b_rows.to(dev))
        grounder._considered_acc_bidx.append(
            torch.full((F,), orig, dtype=torch.long, device=dev))


__all__ = [
    "tabling_active",
    "split_queries",
    "writeback_and_replay",
]
