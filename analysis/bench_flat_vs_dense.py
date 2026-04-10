#!/usr/bin/env python3
"""Benchmark: Dense padded tensors vs Flat+Offsets (CSR) representation.

Compares memory and speed for core grounder operations at real-world scales
(fb15k237, countries_s3 parameters).

Operations benchmarked:
  1. fact_index.enumerate equivalent
  2. Conjunction (min over body atoms per grounding)
  3. Disjunction (max over groundings per query)
  4. Pack/compact (select valid children)
  5. Scatter (write children to parent positions)
  6. Full mini-pipeline: enumerate -> fill_body -> exists -> filter -> pack

Usage:
  python -u grounder/analysis/bench_flat_vs_dense.py
"""

from __future__ import annotations

import gc
import itertools
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")
WARMUP_ITERS = 10
MEASURE_ITERS = 100
DTYPE_IDX = torch.long
DTYPE_SCORE = torch.float32

# fb15k237 parameters
FB15K_E = 14541       # entities
FB15K_P = 237         # predicates
FB15K_F = 310116      # total facts

# Sweep ranges (from user specification)
B_VALS = [1, 32, 128]
K_F_VALS = [18, 200, 3612]       # avg, p95, max for fb15k237
K_R_VALS = [5, 15, 30]
S_VALS = [64, 256, 1024, 4096]
M_VALS = [1, 2, 3]
G_VALS = [3, 7, 13]


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    params: str
    memory_mb: float
    time_ms: float
    throughput: float  # elements per second


def reset_memory():
    """Reset CUDA memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)


def measure_memory_delta(fn: Callable) -> float:
    """Return peak memory delta in MB for running fn()."""
    reset_memory()
    before = torch.cuda.max_memory_allocated(DEVICE)
    fn()
    torch.cuda.synchronize(DEVICE)
    after = torch.cuda.max_memory_allocated(DEVICE)
    return (after - before) / (1024 * 1024)


def measure_time(fn: Callable, warmup: int = WARMUP_ITERS,
                 iters: int = MEASURE_ITERS) -> float:
    """Return median iteration time in ms using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(DEVICE)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize(DEVICE)
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    times.sort()
    # Return median
    return times[len(times) // 2]


def bench(name: str, params: str, dense_fn: Callable, flat_fn: Callable,
          elements: int, compile_dense: bool = True,
          compile_flat: bool = True) -> Tuple[BenchResult, BenchResult,
                                               Optional[BenchResult],
                                               Optional[BenchResult]]:
    """Run a complete benchmark comparison for one operation config.

    Returns (dense_result, flat_result, dense_compiled, flat_compiled).
    """
    # --- Eager ---
    dense_mem = measure_memory_delta(dense_fn)
    dense_time = measure_time(dense_fn)
    dense_tp = elements / (dense_time / 1000) if dense_time > 0 else float('inf')
    dr = BenchResult(f"{name} [Dense]", params, dense_mem, dense_time, dense_tp)

    flat_mem = measure_memory_delta(flat_fn)
    flat_time = measure_time(flat_fn)
    flat_tp = elements / (flat_time / 1000) if flat_time > 0 else float('inf')
    fr = BenchResult(f"{name} [Flat]", params, flat_mem, flat_time, flat_tp)

    # --- Compiled ---
    dc, fc = None, None
    if compile_dense:
        try:
            dense_c = torch.compile(dense_fn, mode="default")
            # Warmup compile
            for _ in range(3):
                dense_c()
            torch.cuda.synchronize(DEVICE)
            dc_mem = measure_memory_delta(dense_c)
            dc_time = measure_time(dense_c)
            dc_tp = elements / (dc_time / 1000) if dc_time > 0 else float('inf')
            dc = BenchResult(f"{name} [Dense compiled]", params,
                             dc_mem, dc_time, dc_tp)
        except Exception as e:
            print(f"  [WARN] Dense compile failed: {e}")

    if compile_flat:
        try:
            flat_c = torch.compile(flat_fn, mode="default")
            for _ in range(3):
                flat_c()
            torch.cuda.synchronize(DEVICE)
            fc_mem = measure_memory_delta(flat_c)
            fc_time = measure_time(flat_c)
            fc_tp = elements / (fc_time / 1000) if fc_time > 0 else float('inf')
            fc = BenchResult(f"{name} [Flat compiled]", params,
                             fc_mem, fc_time, fc_tp)
        except Exception as e:
            print(f"  [WARN] Flat compile failed: {e}")

    return dr, fr, dc, fc


def print_result_table(name: str, params: str,
                       dr: BenchResult, fr: BenchResult,
                       dc: Optional[BenchResult] = None,
                       fc: Optional[BenchResult] = None):
    """Print a formatted comparison table for one operation."""
    mem_ratio = dr.memory_mb / fr.memory_mb if fr.memory_mb > 0.001 else float('inf')
    speed_ratio = dr.time_ms / fr.time_ms if fr.time_ms > 0 else float('inf')

    print(f"\nOperation: {name}")
    print(f"  {params}")
    print(f"  {'Mode':<25} {'Memory (MB)':>12} {'Time (ms)':>12} {'Throughput':>16}")
    print(f"  {'-'*65}")
    print(f"  {'Dense (eager)':<25} {dr.memory_mb:>12.2f} {dr.time_ms:>12.4f} {dr.throughput:>16.0f}")
    print(f"  {'Flat (eager)':<25} {fr.memory_mb:>12.2f} {fr.time_ms:>12.4f} {fr.throughput:>16.0f}")
    if dc:
        print(f"  {'Dense (compiled)':<25} {dc.memory_mb:>12.2f} {dc.time_ms:>12.4f} {dc.throughput:>16.0f}")
    if fc:
        print(f"  {'Flat (compiled)':<25} {fc.memory_mb:>12.2f} {fc.time_ms:>12.4f} {fc.throughput:>16.0f}")
    print(f"  ---")
    print(f"  Ratio (dense/flat): memory={mem_ratio:.2f}x, "
          f"time={'%.2f' % speed_ratio}x "
          f"({'flat faster' if speed_ratio > 1 else 'dense faster'})")
    if dc and fc:
        cm = dc.memory_mb / fc.memory_mb if fc.memory_mb > 0.001 else float('inf')
        ct = dc.time_ms / fc.time_ms if fc.time_ms > 0 else float('inf')
        print(f"  Ratio (compiled):   memory={cm:.2f}x, "
              f"time={'%.2f' % ct}x "
              f"({'flat faster' if ct > 1 else 'dense faster'})")


# ---------------------------------------------------------------------------
# Accumulator for all results (for markdown report)
# ---------------------------------------------------------------------------

all_results: List[Dict[str, Any]] = []

def record(name, params, dr, fr, dc=None, fc=None):
    """Record and print result."""
    print_result_table(name, params, dr, fr, dc, fc)
    all_results.append({
        "name": name, "params": params,
        "dense": dr, "flat": fr,
        "dense_compiled": dc, "flat_compiled": fc,
    })


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 1: Enumerate (fact lookup)
# ═══════════════════════════════════════════════════════════════════════

def bench_enumerate():
    print("\n" + "=" * 72)
    print("BENCHMARK 1: Enumerate (fact_index equivalent)")
    print("=" * 72)

    # Build a realistic fact database
    # For the offset table we need: P * E entries
    P, E = FB15K_P, min(FB15K_E, 5000)  # cap E for memory feasibility
    F = min(FB15K_F, 200000)

    # Generate random facts
    preds = torch.randint(0, P, (F,), device=DEVICE, dtype=DTYPE_IDX)
    subjs = torch.randint(0, E, (F,), device=DEVICE, dtype=DTYPE_IDX)
    objs = torch.randint(0, E, (F,), device=DEVICE, dtype=DTYPE_IDX)

    # Build CSR offset table: for each (pred, subj) -> list of objects
    num_slots = P * E
    keys = preds * E + subjs
    sort_idx = keys.argsort()
    sorted_keys = keys[sort_idx]
    sorted_values = objs[sort_idx]
    offsets = torch.zeros(num_slots + 1, dtype=DTYPE_IDX, device=DEVICE)
    ones = torch.ones(F, dtype=DTYPE_IDX, device=DEVICE)
    offsets.scatter_add_(0, sorted_keys + 1, ones)
    offsets = torch.cumsum(offsets, dim=0)

    # Also build dense blocks [P*E, K_max] for comparison
    for K_f in K_F_VALS:
        for N in [1024, 4096, 16384]:
            params = f"N={N}, K_f={K_f}, P={P}, E={E}"

            # Random query keys
            query_preds = torch.randint(0, P, (N,), device=DEVICE, dtype=DTYPE_IDX)
            query_bound = torch.randint(0, E, (N,), device=DEVICE, dtype=DTYPE_IDX)
            query_keys = query_preds * E + query_bound

            # --- Dense enumerate ---
            def dense_enumerate():
                starts = offsets[query_keys]
                counts = (offsets[query_keys + 1] - starts).clamp(0, K_f)
                pos = torch.arange(K_f, device=DEVICE).unsqueeze(0).expand(N, -1)
                valid = pos < counts.unsqueeze(1)
                gi = (starts.unsqueeze(1) + pos).clamp(0, sorted_values.size(0) - 1)
                result = sorted_values[gi]  # [N, K_f]
                result = result.masked_fill(~valid, 0)
                return result, valid

            # --- Flat enumerate (CSR gather) ---
            def flat_enumerate():
                starts = offsets[query_keys]
                counts = (offsets[query_keys + 1] - starts).clamp(0, K_f)
                # Total valid elements
                total = counts.sum()
                # Build flat index using repeat_interleave
                query_ids = torch.arange(N, device=DEVICE).repeat_interleave(counts)
                # Within-group positions
                pos_in_group = torch.arange(total, device=DEVICE, dtype=DTYPE_IDX)
                group_starts = torch.zeros(N + 1, dtype=DTYPE_IDX, device=DEVICE)
                group_starts[1:] = counts.cumsum(0)
                pos_in_group = pos_in_group - group_starts[query_ids]
                # Global indices
                gi = starts[query_ids] + pos_in_group
                result = sorted_values[gi.clamp(0, sorted_values.size(0) - 1)]
                return result, group_starts

            elements = N * K_f
            dr, fr, dc, fc = bench("Enumerate", params,
                                   dense_enumerate, flat_enumerate,
                                   elements)
            record("Enumerate", params, dr, fr, dc, fc)

    # Cleanup
    del preds, subjs, objs, sorted_keys, sorted_values, offsets, ones
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 2: Conjunction (min over body atoms)
# ═══════════════════════════════════════════════════════════════════════

def bench_conjunction():
    print("\n" + "=" * 72)
    print("BENCHMARK 2: Conjunction (min over body atoms per grounding)")
    print("=" * 72)

    for B in B_VALS:
        for tG in [64, 256, 1024]:
            for M in M_VALS:
                params = f"B={B}, tG={tG}, M={M}"

                # Dense: [B, tG, M] scores with mask
                scores_dense = torch.rand(B, tG, M, device=DEVICE, dtype=DTYPE_SCORE)
                mask_dense = torch.rand(B, tG, M, device=DEVICE) > 0.1  # 90% valid

                def dense_conjunction():
                    masked = scores_dense.masked_fill(~mask_dense, 1.0)
                    return masked.min(dim=-1).values  # [B, tG]

                # Flat: body_offsets define variable-length groups
                # Simulate ragged: some groundings have fewer body atoms
                actual_M = torch.randint(1, M + 1, (B * tG,), device=DEVICE)
                total_atoms = actual_M.sum().item()
                scores_flat = torch.rand(total_atoms, device=DEVICE, dtype=DTYPE_SCORE)
                body_offsets = torch.zeros(B * tG + 1, dtype=DTYPE_IDX, device=DEVICE)
                body_offsets[1:] = actual_M.cumsum(0)

                def flat_conjunction():
                    # scatter_reduce: min per segment
                    seg_ids = torch.arange(B * tG, device=DEVICE).repeat_interleave(actual_M)
                    out = torch.ones(B * tG, device=DEVICE, dtype=DTYPE_SCORE)
                    out.scatter_reduce_(0, seg_ids, scores_flat, reduce="amin",
                                        include_self=False)
                    return out.reshape(B, tG)

                elements = B * tG * M
                dr, fr, dc, fc = bench("Conjunction", params,
                                       dense_conjunction, flat_conjunction,
                                       elements)
                record("Conjunction", params, dr, fr, dc, fc)

                # Cleanup intermediates
                del scores_dense, mask_dense, scores_flat, body_offsets, actual_M
                torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 3: Disjunction (max over groundings per query)
# ═══════════════════════════════════════════════════════════════════════

def bench_disjunction():
    print("\n" + "=" * 72)
    print("BENCHMARK 3: Disjunction (max over groundings per query)")
    print("=" * 72)

    for B in B_VALS:
        for tG in [64, 256, 1024]:
            params = f"B={B}, tG={tG}"

            # Dense: [B, tG] scores with mask
            scores_dense = torch.rand(B, tG, device=DEVICE, dtype=DTYPE_SCORE)
            mask_dense = torch.rand(B, tG, device=DEVICE) > 0.2  # 80% valid

            def dense_disjunction():
                masked = scores_dense.masked_fill(~mask_dense, float('-inf'))
                return masked.max(dim=-1).values  # [B]

            # Flat: variable number of groundings per query
            counts = torch.randint(1, tG + 1, (B,), device=DEVICE)
            total = counts.sum().item()
            scores_flat = torch.rand(total, device=DEVICE, dtype=DTYPE_SCORE)
            query_offsets = torch.zeros(B + 1, dtype=DTYPE_IDX, device=DEVICE)
            query_offsets[1:] = counts.cumsum(0)

            def flat_disjunction():
                seg_ids = torch.arange(B, device=DEVICE).repeat_interleave(counts)
                out = torch.full((B,), float('-inf'), device=DEVICE, dtype=DTYPE_SCORE)
                out.scatter_reduce_(0, seg_ids, scores_flat, reduce="amax",
                                    include_self=False)
                return out

            elements = B * tG
            dr, fr, dc, fc = bench("Disjunction", params,
                                   dense_disjunction, flat_disjunction,
                                   elements)
            record("Disjunction", params, dr, fr, dc, fc)

            del scores_dense, mask_dense, scores_flat, query_offsets, counts
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 4: Pack/Compact (select valid children)
# ═══════════════════════════════════════════════════════════════════════

def bench_pack():
    print("\n" + "=" * 72)
    print("BENCHMARK 4: Pack/Compact (select valid children)")
    print("=" * 72)

    for B in B_VALS:
        for S in S_VALS:
            for K in [18, 200]:
                S_out = S  # output budget = input budget
                params = f"B={B}, S={S}, K={K}"

                # Dense: [B, S*K] -> topk -> [B, S_out]
                total_in = S * K
                scores_in = torch.rand(B, total_in, device=DEVICE, dtype=DTYPE_SCORE)
                valid_in = torch.rand(B, total_in, device=DEVICE) > 0.3

                # Associated data: e.g. goals [B, S*K, G, 3] with G=7
                G = 7
                goals_in = torch.randint(0, 100, (B, total_in, G, 3),
                                         device=DEVICE, dtype=DTYPE_IDX)

                def dense_pack():
                    # Zero out invalid
                    s = scores_in.masked_fill(~valid_in, float('-inf'))
                    # topk
                    _, topk_idx = s.topk(min(S_out, total_in), dim=1)  # [B, S_out]
                    # Gather goals
                    idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S_out, G, 3)
                    packed_goals = torch.gather(goals_in, 1, idx_exp)
                    return packed_goals

                # Flat: masked_select + rebuild offsets
                # Each batch has variable number of valid items
                valid_counts = valid_in.sum(dim=1)  # [B]
                total_valid = valid_counts.sum().item()

                def flat_pack():
                    # Get flat valid indices
                    flat_valid = valid_in.reshape(-1)
                    flat_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
                    # Batch ids from flat indices
                    batch_ids = flat_indices // total_in
                    within_batch = flat_indices % total_in
                    # Gather: for each valid item, get its goals
                    flat_goals = goals_in.reshape(B * total_in, G, 3)
                    selected = flat_goals[flat_indices]
                    # Rebuild offsets
                    new_offsets = torch.zeros(B + 1, dtype=DTYPE_IDX, device=DEVICE)
                    new_offsets[1:] = valid_counts.cumsum(0)
                    # Cap to S_out per batch
                    capped_counts = valid_counts.clamp(max=S_out)
                    return selected, new_offsets, capped_counts

                elements = B * S * K
                dr, fr, dc, fc = bench("Pack/Compact", params,
                                       dense_pack, flat_pack,
                                       elements)
                record("Pack/Compact", params, dr, fr, dc, fc)

                del scores_in, valid_in, goals_in
                torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 5: Scatter (write children to parent positions)
# ═══════════════════════════════════════════════════════════════════════

def bench_scatter():
    print("\n" + "=" * 72)
    print("BENCHMARK 5: Scatter (write children to parent positions)")
    print("=" * 72)

    for B in B_VALS:
        for S_out in S_VALS:
            for K in [18, 200]:
                S_src = min(S_out, 512)  # source states
                params = f"B={B}, S_src={S_src}, S_out={S_out}, K={K}"

                # Dense: scatter_(1, indices, source) on [B, S_out, G, 3]
                G = 7
                target = torch.zeros(B, S_out, G, 3, device=DEVICE, dtype=DTYPE_IDX)
                source = torch.randint(0, 100, (B, S_src, G, 3),
                                       device=DEVICE, dtype=DTYPE_IDX)
                # Random target positions for each batch
                indices = torch.stack([
                    torch.randperm(S_out, device=DEVICE)[:S_src]
                    for _ in range(B)
                ])  # [B, S_src]

                def dense_scatter():
                    out = target.clone()
                    idx = indices.unsqueeze(-1).unsqueeze(-1).expand(B, S_src, G, 3)
                    out.scatter_(1, idx, source)
                    return out

                # Flat: index_copy_ with flat offsets
                flat_source = source.reshape(B * S_src, G, 3)

                def flat_scatter():
                    out_flat = torch.zeros(B * S_out, G, 3,
                                           device=DEVICE, dtype=DTYPE_IDX)
                    # Compute flat destination indices
                    batch_offset = torch.arange(B, device=DEVICE).unsqueeze(1) * S_out
                    flat_dst = (batch_offset + indices).reshape(-1)
                    out_flat[flat_dst] = flat_source
                    return out_flat.reshape(B, S_out, G, 3)

                elements = B * S_src * G * 3
                dr, fr, dc, fc = bench("Scatter", params,
                                       dense_scatter, flat_scatter,
                                       elements)
                record("Scatter", params, dr, fr, dc, fc)

                del target, source, indices, flat_source
                torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 6: Full mini-pipeline
# ═══════════════════════════════════════════════════════════════════════

def bench_pipeline():
    print("\n" + "=" * 72)
    print("BENCHMARK 6: Full mini-pipeline (enumerate -> fill_body -> exists -> filter -> pack)")
    print("=" * 72)

    P, E = FB15K_P, min(FB15K_E, 5000)
    F = min(FB15K_F, 200000)

    # Build fact database (CSR)
    preds_db = torch.randint(0, P, (F,), device=DEVICE, dtype=DTYPE_IDX)
    subjs_db = torch.randint(0, E, (F,), device=DEVICE, dtype=DTYPE_IDX)
    objs_db = torch.randint(0, E, (F,), device=DEVICE, dtype=DTYPE_IDX)

    # Hash for exists check
    base = max(E, P) + 2
    hashes = ((preds_db * base + subjs_db) * base + objs_db).long()
    sort_order = hashes.argsort()
    sorted_hashes = hashes[sort_order]

    # CSR offset table
    num_slots = P * E
    keys = preds_db * E + subjs_db
    sort_idx = keys.argsort()
    sorted_keys_db = keys[sort_idx]
    sorted_objs = objs_db[sort_idx]
    offsets = torch.zeros(num_slots + 1, dtype=DTYPE_IDX, device=DEVICE)
    ones = torch.ones(F, dtype=DTYPE_IDX, device=DEVICE)
    offsets.scatter_add_(0, sorted_keys_db + 1, ones)
    offsets = torch.cumsum(offsets, dim=0)

    for B in B_VALS:
        for S in [64, 256, 1024]:
            for K_f in [18, 200]:
                M = 2
                G = 7
                N = B * S
                params = f"B={B}, S={S}, K_f={K_f}, M={M}, G={G}"

                # Query atoms: [B, S, 3] (pred, subj, obj)
                query_preds = torch.randint(0, P, (B, S), device=DEVICE, dtype=DTYPE_IDX)
                query_subjs = torch.randint(0, E, (B, S), device=DEVICE, dtype=DTYPE_IDX)
                state_valid = torch.rand(B, S, device=DEVICE) > 0.1
                R_eff = 5  # rules matching

                # --- Dense pipeline ---
                def dense_pipeline():
                    # 1. Enumerate: [N, K_f]
                    flat_preds = query_preds.reshape(N)
                    flat_subjs = query_subjs.reshape(N)
                    qkeys = flat_preds * E + flat_subjs
                    starts = offsets[qkeys]
                    counts = (offsets[qkeys + 1] - starts).clamp(0, K_f)
                    pos = torch.arange(K_f, device=DEVICE).unsqueeze(0).expand(N, -1)
                    valid = pos < counts.unsqueeze(1)
                    gi = (starts.unsqueeze(1) + pos).clamp(0, sorted_objs.size(0) - 1)
                    candidates = sorted_objs[gi]  # [N, K_f]
                    candidates = candidates.masked_fill(~valid, 0)

                    # 2. Fill body: [B, S, R_eff*K_f, M, 3] (simplified)
                    total_children = R_eff * K_f
                    body = torch.zeros(B, S, total_children, M, 3,
                                       device=DEVICE, dtype=DTYPE_IDX)
                    # Just use candidates as obj for body atom 0
                    cands_bs = candidates.reshape(B, S, K_f)
                    body[:, :, :K_f, 0, 0] = query_preds.unsqueeze(-1).expand_as(cands_bs)
                    body[:, :, :K_f, 0, 1] = query_subjs.unsqueeze(-1).expand_as(cands_bs)
                    body[:, :, :K_f, 0, 2] = cands_bs

                    # 3. Exists check on body atoms
                    body_flat = body.reshape(-1, 3)
                    # Vectorized hash check
                    bh = ((body_flat[:, 0] * base + body_flat[:, 1]) * base + body_flat[:, 2]).long()
                    Fh = sorted_hashes.shape[0]
                    idx = torch.searchsorted(sorted_hashes, bh)
                    exists = (idx < Fh) & (sorted_hashes[idx.clamp(max=Fh - 1)] == bh)
                    exists = exists.reshape(B, S, total_children, M)

                    # 4. Conjunction: min over body (using exists as 0/1 scores)
                    scores = exists.float()
                    conj = scores.min(dim=-1).values  # [B, S, total_children]

                    # 5. Pack: topk over total_children
                    S_out = S
                    _, topk_idx = conj.reshape(B, S * total_children).topk(
                        min(S_out, S * total_children), dim=1)

                    return topk_idx

                # --- Flat pipeline ---
                def flat_pipeline():
                    # 1. Enumerate (CSR gather)
                    flat_preds = query_preds.reshape(N)
                    flat_subjs = query_subjs.reshape(N)
                    flat_valid = state_valid.reshape(N)
                    qkeys = flat_preds * E + flat_subjs
                    starts = offsets[qkeys]
                    counts = (offsets[qkeys + 1] - starts).clamp(0, K_f)
                    # Zero out invalid queries
                    counts = counts * flat_valid.long()
                    total = counts.sum()

                    # Build flat candidates
                    query_ids = torch.arange(N, device=DEVICE).repeat_interleave(counts)
                    cand_offsets = torch.zeros(N + 1, dtype=DTYPE_IDX, device=DEVICE)
                    cand_offsets[1:] = counts.cumsum(0)
                    pos_in = torch.arange(total, device=DEVICE, dtype=DTYPE_IDX)
                    pos_in = pos_in - cand_offsets[query_ids]
                    gi = starts[query_ids] + pos_in
                    candidates_flat = sorted_objs[gi.clamp(0, sorted_objs.size(0) - 1)]

                    # 2. Fill body for flat candidates
                    # Each candidate becomes R_eff children -> total * R_eff body atoms
                    # Simplified: just one body atom per candidate
                    body_p = flat_preds[query_ids]
                    body_s = flat_subjs[query_ids]
                    body_o = candidates_flat

                    # 3. Exists (vectorized hash, same as dense but on flat)
                    bh = ((body_p * base + body_s) * base + body_o).long()
                    Fh = sorted_hashes.shape[0]
                    idx = torch.searchsorted(sorted_hashes, bh)
                    exists = (idx < Fh) & (sorted_hashes[idx.clamp(max=Fh - 1)] == bh)

                    # 4. Conjunction: for M=1, score = exists itself
                    scores_flat = exists.float()

                    # 5. Filter + pack: segment max per query
                    out = torch.full((N,), float('-inf'), device=DEVICE, dtype=DTYPE_SCORE)
                    out.scatter_reduce_(0, query_ids, scores_flat, reduce="amax",
                                        include_self=False)

                    return out.reshape(B, S)

                elements = N * K_f * R_eff
                dr, fr, dc, fc = bench("Pipeline", params,
                                       dense_pipeline, flat_pipeline,
                                       elements)
                record("Pipeline", params, dr, fr, dc, fc)

                del query_preds, query_subjs, state_valid
                torch.cuda.empty_cache()

    del preds_db, subjs_db, objs_db, hashes, sorted_hashes
    del sorted_objs, offsets
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════

def generate_markdown_report(filepath: str):
    """Write full results as a markdown report."""
    lines = []
    lines.append("# Dense vs Flat+Offsets (CSR) Benchmark Report")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **GPU**: {torch.cuda.get_device_name(0)}")
    lines.append(f"- **PyTorch**: {torch.__version__}")
    lines.append(f"- **CUDA**: {torch.version.cuda}")
    lines.append(f"- **Warmup iters**: {WARMUP_ITERS}")
    lines.append(f"- **Measure iters**: {MEASURE_ITERS}")
    lines.append(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Parameters tested")
    lines.append("")
    lines.append(f"- B (batch): {B_VALS}")
    lines.append(f"- K_f (fact candidates): {K_F_VALS}")
    lines.append(f"- K_r (rule candidates): {K_R_VALS}")
    lines.append(f"- S (state counts): {S_VALS}")
    lines.append(f"- M (body atoms): {M_VALS}")
    lines.append(f"- G (goals): {G_VALS}")
    lines.append(f"- fb15k237 scale: E={FB15K_E}, P={FB15K_P}, F={FB15K_F}")
    lines.append("")

    # Group results by operation
    ops = {}
    for r in all_results:
        op = r["name"]
        if op not in ops:
            ops[op] = []
        ops[op].append(r)

    for op_name, results in ops.items():
        lines.append(f"## {op_name}")
        lines.append("")
        lines.append("| Parameters | Dense mem (MB) | Flat mem (MB) | Mem ratio | Dense time (ms) | Flat time (ms) | Speed ratio | Dense compiled (ms) | Flat compiled (ms) | Compiled ratio |")
        lines.append("|:-----------|---------------:|--------------:|----------:|----------------:|---------------:|------------:|--------------------:|-------------------:|---------------:|")

        for r in results:
            dr, fr = r["dense"], r["flat"]
            dc, fc = r.get("dense_compiled"), r.get("flat_compiled")
            mem_ratio = dr.memory_mb / fr.memory_mb if fr.memory_mb > 0.001 else float('inf')
            speed_ratio = dr.time_ms / fr.time_ms if fr.time_ms > 0 else float('inf')

            dc_str = f"{dc.time_ms:.4f}" if dc else "N/A"
            fc_str = f"{fc.time_ms:.4f}" if fc else "N/A"
            if dc and fc and fc.time_ms > 0:
                cr = f"{dc.time_ms / fc.time_ms:.2f}x"
            else:
                cr = "N/A"

            lines.append(
                f"| {r['params']} | {dr.memory_mb:.2f} | {fr.memory_mb:.2f} | "
                f"{mem_ratio:.2f}x | {dr.time_ms:.4f} | {fr.time_ms:.4f} | "
                f"{speed_ratio:.2f}x | {dc_str} | {fc_str} | {cr} |"
            )
        lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")

    if all_results:
        mem_ratios = []
        speed_ratios = []
        compiled_speed_ratios = []
        for r in all_results:
            dr, fr = r["dense"], r["flat"]
            if fr.memory_mb > 0.001:
                mem_ratios.append(dr.memory_mb / fr.memory_mb)
            if fr.time_ms > 0:
                speed_ratios.append(dr.time_ms / fr.time_ms)
            dc, fc = r.get("dense_compiled"), r.get("flat_compiled")
            if dc and fc and fc.time_ms > 0:
                compiled_speed_ratios.append(dc.time_ms / fc.time_ms)

        if mem_ratios:
            lines.append(f"### Memory (dense/flat ratio)")
            lines.append(f"- Mean: {sum(mem_ratios)/len(mem_ratios):.2f}x")
            lines.append(f"- Median: {sorted(mem_ratios)[len(mem_ratios)//2]:.2f}x")
            lines.append(f"- Min: {min(mem_ratios):.2f}x")
            lines.append(f"- Max: {max(mem_ratios):.2f}x")
            lines.append(f"- Configurations where flat uses less memory: {sum(1 for r in mem_ratios if r > 1.0)}/{len(mem_ratios)}")
            lines.append("")

        if speed_ratios:
            lines.append(f"### Speed (eager, dense/flat ratio: >1 means flat is faster)")
            lines.append(f"- Mean: {sum(speed_ratios)/len(speed_ratios):.2f}x")
            lines.append(f"- Median: {sorted(speed_ratios)[len(speed_ratios)//2]:.2f}x")
            lines.append(f"- Min: {min(speed_ratios):.2f}x")
            lines.append(f"- Max: {max(speed_ratios):.2f}x")
            lines.append(f"- Configurations where flat is faster: {sum(1 for r in speed_ratios if r > 1.0)}/{len(speed_ratios)}")
            lines.append("")

        if compiled_speed_ratios:
            lines.append(f"### Speed (compiled, dense/flat ratio: >1 means flat is faster)")
            lines.append(f"- Mean: {sum(compiled_speed_ratios)/len(compiled_speed_ratios):.2f}x")
            lines.append(f"- Median: {sorted(compiled_speed_ratios)[len(compiled_speed_ratios)//2]:.2f}x")
            lines.append(f"- Min: {min(compiled_speed_ratios):.2f}x")
            lines.append(f"- Max: {max(compiled_speed_ratios):.2f}x")
            lines.append(f"- Configurations where flat is faster: {sum(1 for r in compiled_speed_ratios if r > 1.0)}/{len(compiled_speed_ratios)}")
            lines.append("")

        # Per-operation breakdown
        lines.append("### Per-operation summary")
        lines.append("")
        for op_name, results in ops.items():
            op_mem = []
            op_speed = []
            for r in results:
                dr, fr = r["dense"], r["flat"]
                if fr.memory_mb > 0.001:
                    op_mem.append(dr.memory_mb / fr.memory_mb)
                if fr.time_ms > 0:
                    op_speed.append(dr.time_ms / fr.time_ms)
            if op_mem and op_speed:
                lines.append(f"**{op_name}**: mem {sum(op_mem)/len(op_mem):.2f}x avg, "
                             f"speed {sum(op_speed)/len(op_speed):.2f}x avg "
                             f"(flat {'faster' if sum(op_speed)/len(op_speed) > 1 else 'slower'} on average)")
            lines.append("")

    # Go/No-Go recommendation
    lines.append("## Recommendation")
    lines.append("")
    lines.append("*This section is auto-generated from the data above. "
                 "See the per-operation analysis for details.*")
    lines.append("")

    if all_results:
        avg_mem = sum(mem_ratios) / len(mem_ratios) if mem_ratios else 1.0
        avg_speed = sum(speed_ratios) / len(speed_ratios) if speed_ratios else 1.0
        avg_compiled = (sum(compiled_speed_ratios) / len(compiled_speed_ratios)
                        if compiled_speed_ratios else 1.0)

        # Pipeline-specific results
        pipeline_results = ops.get("Pipeline", [])
        pipe_speed = []
        pipe_mem = []
        for r in pipeline_results:
            dr, fr = r["dense"], r["flat"]
            if fr.time_ms > 0:
                pipe_speed.append(dr.time_ms / fr.time_ms)
            if fr.memory_mb > 0.001:
                pipe_mem.append(dr.memory_mb / fr.memory_mb)

        avg_pipe_speed = sum(pipe_speed) / len(pipe_speed) if pipe_speed else 1.0
        avg_pipe_mem = sum(pipe_mem) / len(pipe_mem) if pipe_mem else 1.0

        lines.append("### Key findings")
        lines.append("")
        lines.append(f"1. **Overall memory**: Dense uses {avg_mem:.1f}x more memory than flat on average")
        lines.append(f"2. **Overall eager speed**: Dense/flat ratio = {avg_speed:.2f}x "
                     f"({'flat is faster' if avg_speed > 1 else 'dense is faster'})")
        lines.append(f"3. **Overall compiled speed**: Dense/flat ratio = {avg_compiled:.2f}x "
                     f"({'flat is faster' if avg_compiled > 1 else 'dense is faster'})")
        lines.append(f"4. **Full pipeline**: memory {avg_pipe_mem:.1f}x savings, "
                     f"speed {avg_pipe_speed:.2f}x "
                     f"({'flat is faster' if avg_pipe_speed > 1 else 'dense is faster'})")
        lines.append("")

        # Decision logic
        go = False
        reasons = []
        if avg_pipe_mem > 2.0:
            reasons.append(f"Pipeline memory savings are significant ({avg_pipe_mem:.1f}x)")
            go = True
        if avg_pipe_speed > 1.5:
            reasons.append(f"Pipeline speed improvement is significant ({avg_pipe_speed:.2f}x)")
            go = True
        if avg_pipe_speed < 0.8:
            reasons.append(f"Pipeline is slower with flat ({avg_pipe_speed:.2f}x) -- major concern")
            go = False
        if avg_compiled < 0.8:
            reasons.append(f"Compiled path regresses with flat ({avg_compiled:.2f}x) -- torch.compile compatibility concern")

        lines.append("### Verdict")
        lines.append("")
        if go and avg_pipe_speed >= 0.8:
            lines.append("**GO** -- The data supports a rewrite to flat+offsets (CSR) representation.")
        elif avg_pipe_speed >= 0.95 and avg_pipe_mem > 1.5:
            lines.append("**CONDITIONAL GO** -- Memory benefits are clear but speed gains are marginal. "
                         "Consider a targeted migration for memory-critical paths only.")
        else:
            lines.append("**NO-GO** -- The data does not support a full library rewrite. "
                         "Dense padded tensors are competitive or faster in most configurations.")

        lines.append("")
        lines.append("**Reasons:**")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to: {filepath}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    try:
        bench_enumerate()
    except Exception as e:
        print(f"\n[ERROR] Enumerate benchmark failed: {e}")
        import traceback; traceback.print_exc()

    try:
        bench_conjunction()
    except Exception as e:
        print(f"\n[ERROR] Conjunction benchmark failed: {e}")
        import traceback; traceback.print_exc()

    try:
        bench_disjunction()
    except Exception as e:
        print(f"\n[ERROR] Disjunction benchmark failed: {e}")
        import traceback; traceback.print_exc()

    try:
        bench_pack()
    except Exception as e:
        print(f"\n[ERROR] Pack benchmark failed: {e}")
        import traceback; traceback.print_exc()

    try:
        bench_scatter()
    except Exception as e:
        print(f"\n[ERROR] Scatter benchmark failed: {e}")
        import traceback; traceback.print_exc()

    try:
        bench_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Pipeline benchmark failed: {e}")
        import traceback; traceback.print_exc()

    # Generate report into this repo's docs/exp_analysis/ directory.
    from pathlib import Path
    report_path = str(
        Path(__file__).resolve().parents[1] / "docs" / "exp_analysis" / "benchmark_flat_vs_dense.md"
    )
    generate_markdown_report(report_path)

    print("\n" + "=" * 72)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
