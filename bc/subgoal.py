"""Per-subgoal grounding memo (task #48, Design #2 — subgoal granularity).

Where the per-query cache (:mod:`grounder.bc.tabling`, task #47) memoizes a
whole query atom's considered firing set, this memoizes the considered
firings of each *selected subgoal* during the depth loop. A subgoal's
considered rows are a pure function of ``(selected_goal_atom, is_last)``:

* ``selected_goal_atom`` — the (pred, subj, obj) atom currently being
  resolved at this proof step;
* ``is_last`` — whether this is the terminal step (``d == depth-1``). This
  is the ONLY depth component the flat resolve depends on (enum.py:1336-1341
  — ``width_d`` and ``head_pred_mask_d`` are the sole ``d``-dependence; the
  candidate enumeration reads only the goal args + fixed rule/fact buffers).

So subgoals shared ACROSS queries (e.g. a negative ``r(h, t')`` and the
positive ``r(h, t)`` both expanding the same intermediate ``(h, r)`` goal)
reuse one stored firing set, and recurring subgoals across epochs are
served from the memo.

Phase-1 (this module): byte-identical validation layer. The full resolve
+ extraction still runs over the whole step (no row-subsetting in the
resolve kernel yet — that is Phase-2, deferred). We then partition the
extracted valid rows by their selected-goal key, store MISS-goal rows into
the memo, and append the canonical rows (cached on HIT, fresh on MISS),
re-tagged to each live query occurrence, into ``_considered_acc_*``. Since
the canonical rows equal the freshly-extracted rows for the same goal
(grounding is a pure function of the goal), and the considered finalize is
order-/multiplicity-insensitive (dedups to a set), the output is
byte-identical. TRIPLE-VERIFY guards ``atom_hash`` collisions.

Guardrails mirror task #47: enum + flat + eager + not compiled, the
KB-immutability stamp, and overflow passthrough.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from grounder.groundings import atom_hash


def subgoal_active(grounder) -> bool:
    """Return True when the per-subgoal memo should run.

    Same hard guardrails as the per-query cache: eager enum FLAT considered
    path only, KB immutable. Asserts rather than silently degrading.
    """
    if not getattr(grounder, "_subgoal_enabled", False):
        return False
    assert grounder.resolution == "enum", (
        "subgoal memo requires resolution=='enum', "
        f"got {grounder.resolution!r}")
    assert getattr(grounder, "_flat_intermediate", False) \
        or getattr(grounder, "_flat_intermediate_flag", False), (
        "subgoal memo requires the enum FLAT path (flat_intermediate=True)")
    assert not torch.compiler.is_compiling(), (
        "subgoal memo must not run inside torch.compile tracing")
    assert not getattr(grounder, "_compiled", False), (
        "subgoal memo is incompatible with the compiled/dense step path")
    assert grounder._collect_rule_groundings, (
        "subgoal memo requires collect_rule_groundings=True")
    stamp = (id(grounder.kb.fact_index), int(grounder.kb.num_rules))
    assert stamp == grounder._tabling_kb_stamp, (
        "subgoal memo KB-immutability stamp changed "
        f"(was {grounder._tabling_kb_stamp}, now {stamp}); the fact index "
        "or rule count was mutated, invalidating every memo entry")
    return True


def route_rows(
    grounder,
    r_rule: Tensor,      # [V]    valid candidate rule_idx (post variant->orig)
    r_head: Tensor,      # [V, 3] selected-goal atom per row (== head)
    r_body: Tensor,      # [V, M, 3] canonical-order body atoms
    r_bidx: Tensor,      # [V]    batch-local query index
    d: Optional[int],    # current depth (for is_last)
) -> None:
    """Partition the step's valid rows by selected goal, memoize MISS-goal
    row sets, and append canonical rows (cached on HIT) re-tagged to each
    live query occurrence into ``_considered_acc_*``.

    Byte-identical to the direct append: per goal the canonical rows equal
    the freshly-extracted rows, and finalize dedups to a set, so neither
    ordering nor the within-(goal,bidx) multiplicity collapse changes the
    output. The per-(goal, bidx) attribution is preserved so the per-query
    cache (#47) write-back still gets each query's complete row set.
    """
    V = int(r_rule.size(0))
    if V == 0:
        # Append empty tensors so the per-step list lengths stay aligned
        # with the chunked-path accumulator marker bookkeeping.
        _append(grounder, r_rule, r_head, r_body,
                r_bidx + grounder._chunk_query_offset)
        return

    dev = r_rule.device
    is_last = bool(d == grounder.depth - 1) if d is not None else False
    is_last_tag = 1 if is_last else 0

    # ``r_head`` IS the selected goal atom. Group rows by the EXACT goal
    # triple (collision-safe — the hash only buckets, so two distinct
    # triples sharing a hash must not be merged). Stable-sort by goal hash
    # gives contiguous candidate groups; within a hash group we still split
    # by exact triple via a small python dict keyed on the triple tuple.
    goal_h = atom_hash(r_head)                                   # [V] long
    head_list = r_head.tolist()                                  # [V][3]
    bidx_list = r_bidx.tolist()

    memo = grounder._subgoal_memo
    cap = grounder._subgoal_max_entries

    # group exact-triple -> list of fresh-row positions.
    groups: Dict[tuple, List[int]] = {}
    for i in range(V):
        t = (head_list[i][0], head_list[i][1], head_list[i][2])
        groups.setdefault(t, []).append(i)

    # ── Pass 1: per distinct goal triple, fix canonical (rule, body) rows ──
    # MISS triples store fresh rows; HIT triples serve memo rows (after
    # TRIPLE-VERIFY). Keyed by ``(goal_hash, is_last)``; triple-verify on
    # every HIT resolves hash collisions (single owner per key, like #47).
    canon: Dict[tuple, Tuple[Tensor, Tensor]] = {}
    n_hit = 0
    n_miss = 0
    for t, rows_pos in groups.items():
        rep = rows_pos[0]
        goal_triple = r_head[rep]                     # [3]
        gh = int(goal_h[rep].item())
        key = (gh, is_last_tag)
        entry = memo.get(key)
        served = False
        if entry is not None:
            stored_rule, stored_body, stored_triple = entry
            # TRIPLE-VERIFY (mandatory).
            if torch.equal(stored_triple, goal_triple):
                canon[t] = (stored_rule, stored_body)
                n_hit += 1
                served = True
        if not served:
            # MISS: canonical = this triple's fresh rows (RAW, pre-prune).
            idx = torch.tensor(rows_pos, dtype=torch.long, device=dev)
            g_rule = r_rule.index_select(0, idx)
            g_body = r_body.index_select(0, idx)
            canon[t] = (g_rule, g_body)
            n_miss += 1
            if key in memo:
                # Hash collision with a different-triple owner — leave it;
                # this triple is served fresh (single owner per key).
                pass
            elif len(memo) >= cap:
                if not grounder._subgoal_full_warned:
                    warnings.warn(
                        "grounder subgoal memo reached "
                        f"_subgoal_max_entries={cap}; new subgoal groundings "
                        "will be computed fresh (passthrough) and not "
                        "memoized. Existing entries continue to serve hits.",
                        stacklevel=2,
                    )
                    grounder._subgoal_full_warned = True
            else:
                memo[key] = (g_rule.clone(), g_body.clone(),
                             goal_triple.clone())

    grounder._subgoal_last_hits = (
        getattr(grounder, "_subgoal_last_hits", 0) + n_hit)
    grounder._subgoal_last_misses = (
        getattr(grounder, "_subgoal_last_misses", 0) + n_miss)

    # ── Pass 2: append canonical rows per distinct (goal triple, bidx) ──
    # Preserves per-query attribution (so #47's per-query write-back gets
    # each query's complete set) while collapsing duplicate occurrences (the
    # within-batch dedup). Output byte-identity holds because finalize
    # dedups (rule, head, body) to a set.
    seen: set = set()
    offset = grounder._chunk_query_offset
    out_rule: List[Tensor] = []
    out_head: List[Tensor] = []
    out_body: List[Tensor] = []
    out_bidx: List[Tensor] = []
    for i in range(V):
        t = (head_list[i][0], head_list[i][1], head_list[i][2])
        b = bidx_list[i]
        occ = (t, b)
        if occ in seen:
            continue
        seen.add(occ)
        g_rule, g_body = canon[t]
        F = int(g_rule.size(0))
        if F == 0:
            continue
        out_rule.append(g_rule)
        out_body.append(g_body)
        # head = the live goal triple for this occurrence (== stored triple
        # by TRIPLE-VERIFY); use this row's head to stay on the live atom.
        out_head.append(r_head[i].unsqueeze(0).expand(F, 3))
        out_bidx.append(torch.full((F,), b + offset, dtype=torch.long,
                                   device=dev))

    if out_rule:
        _append(grounder,
                torch.cat(out_rule, 0),
                torch.cat(out_head, 0),
                torch.cat(out_body, 0),
                torch.cat(out_bidx, 0))
    else:
        _append(grounder,
                r_rule[:0], r_head[:0], r_body[:0],
                (r_bidx[:0] + offset))


def _append(grounder, rule: Tensor, head: Tensor, body: Tensor,
            bidx: Tensor) -> None:
    grounder._considered_acc_rule.append(rule)
    grounder._considered_acc_head.append(head)
    grounder._considered_acc_body.append(body)
    grounder._considered_acc_bidx.append(bidx)


__all__ = ["subgoal_active", "route_rows"]
