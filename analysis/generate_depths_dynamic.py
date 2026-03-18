"""Dynamic depth generation with external BFS loop and inter-depth compaction.

Uses BCGrounder._step_impl() directly with a flat frontier representation
([N, G, 3] + owner IDs). Between depths, performs deduplication, per-query
capping, and frontier compaction in Python. The step + prune + compact pass
is compiled as a single CUDA graph.

Usage:
    python -m grounder.analysis.generate_depths_dynamic \
        --data_dir kge_experiments/data/family \
        --splits test --max_depth 7
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import torch
from torch import Tensor

# Suppress CUDA graph dynamic shape warnings (expected with variable frontier)
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None  # type: ignore[attr-defined]

from grounder.data.loader import KGDataset
from grounder.bc.bc import BCGrounder
from grounder.bc.common import compact_atoms, prune_ground_facts
from grounder.analysis._dedup import (
    dedup_within_depth,
    dedup_cross_depth,
    cap_frontier_per_query,
)
from grounder.analysis._report import DepthStats, write_report


def _build_compiled_pass(grounder):
    """Build a compiled function that chains _step_impl + prune + compact.

    Returns a callable (proof_goals, dummy_gbody, dummy_ridx, state_valid,
    next_vars) -> (new_goals, new_valid, new_next_var).
    """
    def _one_pass(proof_goals, dummy_gbody, dummy_ridx, state_valid, next_vars):
        new_gbody, new_goals, new_ridx, new_valid, new_next_var = grounder._step_impl(
            dummy_gbody, proof_goals, dummy_ridx, state_valid, next_vars)
        new_goals, _, _ = prune_ground_facts(
            new_goals, new_valid, grounder.fact_hashes, grounder.pack_base,
            grounder.constant_no, grounder.padding_idx)
        new_goals = compact_atoms(new_goals, grounder.padding_idx)
        return new_goals, new_valid, new_next_var

    if grounder._device.type == "cuda":
        return torch.compile(_one_pass, fullgraph=True, mode="reduce-overhead")
    return _one_pass


def _process_chunk(
    chunk_states: Tensor,   # [C, G_in, 3]
    chunk_vars: Tensor,     # [C]
    chunk_owner: Tensor,    # [C]
    grounder,
    compiled_pass,
    depths: Tensor,         # [N] current depths
    G_out: int,             # target G for output states
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor]:
    """Process one chunk through the compiled pass. Returns (states, owners, vars, proof_owners)."""
    C = chunk_states.shape[0]
    pad = grounder.padding_idx
    dev = chunk_states.device
    M = grounder.M

    # Reshape flat [C, G_in, 3] -> [C, 1, G_in, 3] for _step_impl
    proof_goals = chunk_states.unsqueeze(1)
    dummy_gbody = torch.full((C, 1, M, 3), pad, dtype=torch.long, device=dev)
    dummy_ridx = torch.full((C, 1), -1, dtype=torch.long, device=dev)
    state_valid = torch.ones(C, 1, dtype=torch.bool, device=dev)

    torch.compiler.cudagraph_mark_step_begin()

    # Compiled CUDA graph: step + prune + compact
    new_goals, new_valid, new_next_var = compiled_pass(
        proof_goals, dummy_gbody, dummy_ridx, state_valid, chunk_vars)
    # new_goals: [C, S_out, G_pack, 3], new_valid: [C, S_out]
    # G_pack may differ from G_in when rule bodies widen the goal tensor

    S_out = new_goals.shape[1]
    G_pack = new_goals.shape[2]

    # Detect proofs: all goals are padding AND state is valid
    all_padding = (new_goals[:, :, :, 0] == pad).all(dim=2)  # [C, S_out]
    is_proof = all_padding & new_valid  # [C, S_out]

    # Map proof states back to owners
    batch_idx = torch.arange(C, device=dev).unsqueeze(1).expand(C, S_out)
    proof_mask = is_proof.reshape(-1)
    proof_batch = batch_idx.reshape(-1)[proof_mask]
    proof_owners = chunk_owner[proof_batch] if proof_mask.any() else torch.empty(0, dtype=torch.long, device=dev)

    # Valid next states: valid AND not all-padding AND owner not yet proven
    valid_next = new_valid & ~all_padding
    state_owners = chunk_owner[batch_idx.reshape(-1)]
    owner_unproven = (depths[state_owners] < 0)
    valid_flat = valid_next.reshape(-1) & owner_unproven

    if not valid_flat.any():
        return None, None, None, proof_owners

    valid_idx = valid_flat.nonzero(as_tuple=True)[0]
    valid_states = new_goals.reshape(C * S_out, G_pack, 3)[valid_idx]

    # Pad/truncate to G_out atoms for consistent frontier shape
    if G_pack < G_out:
        padded = torch.full((valid_states.shape[0], G_out, 3), pad, dtype=torch.long, device=dev)
        padded[:, :G_pack, :] = valid_states
        valid_states = padded
    elif G_pack > G_out:
        valid_states = valid_states[:, :G_out, :].contiguous()

    return (
        valid_states.clone(),
        state_owners[valid_idx].clone(),
        new_next_var[batch_idx.reshape(-1)[valid_idx]].clone(),
        proof_owners,
    )


@torch.no_grad()
def generate_depths_dynamic(
    dataset: KGDataset,
    resolution: str,
    split: str,
    max_depth: int,
    max_goals: int,
    batch_size: int = 512,
    max_frontier: int = 2_000_000,
    max_per_query: int = 5000,
    query_batch_size: int = 0,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Tensor:
    """Generate depth annotations using dynamic BFS with inter-depth compaction.

    Args:
        dataset: Loaded KGDataset
        resolution: 'sld' or 'rtf'
        split: 'train', 'valid', or 'test'
        max_depth: Maximum BFS depth
        max_goals: G dimension (max atoms per state)
        batch_size: Chunk size for compiled _step_impl calls
        max_frontier: Global cap on frontier size
        max_per_query: Per-query frontier cap
        query_batch_size: Queries to BFS simultaneously (0 = all)
        device: Target device
        output_dir: Output directory (default: dataset directory)

    Returns:
        [N] int tensor of minimum proof depths (-1 if not provable)
    """
    queries_idx = dataset.get_queries(split)
    query_strings = dataset.get_query_strings(split)
    N_total = queries_idx.shape[0]
    if N_total == 0:
        print(f"No queries for split '{split}', skipping.")
        return torch.empty(0, dtype=torch.long)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    queries_idx = queries_idx.to(dev)

    # Create grounder with depth=1 (external loop), no internal compilation
    kb = dataset.make_kb()
    grounder = BCGrounder(
        kb,
        resolution=resolution,
        filter="fp_batch",
        depth=1,
        max_goals=max_goals,
        compile_mode=None,  # we compile externally
        collect_evidence=False,  # skip proof body tracking
    )
    grounder = grounder.to(dev)
    grounder_type = resolution

    K = grounder.K
    G = max_goals
    M = grounder.M
    pad = grounder.padding_idx
    pack_base = grounder.pack_base

    print(f"\n{'='*60}")
    print(f"Dynamic depth generation: {dataset.data_dir.name} / {split}")
    print(f"Grounder: {grounder_type}, K={K}, S={grounder.S}, G={G}, M={M}")
    print(f"Max depth: {max_depth}, Batch size: {batch_size}")
    print(f"Max frontier: {max_frontier}, Max/query: {max_per_query}")
    print(f"Device: {dev}")
    print(f"{'='*60}\n")

    # Build compiled pass
    compiled_pass = _build_compiled_pass(grounder)

    # Memory budget
    bytes_per_state = K * G * 3 * 8
    safety_factor = 6
    max_gpu_bytes = 4 * 1024**3
    safe_batch = max(64, min(batch_size, int(max_gpu_bytes / (bytes_per_state * safety_factor))))
    print(f"Safe batch size: {safe_batch} (requested: {batch_size})")

    # Process in query batches
    qbs = query_batch_size if query_batch_size > 0 else N_total
    all_depths = torch.full((N_total,), -1, dtype=torch.long, device=dev)
    all_depth_stats: list[DepthStats] = []
    t_total = time.time()

    for qstart in range(0, N_total, qbs):
        qend = min(qstart + qbs, N_total)
        N = qend - qstart
        queries = queries_idx[qstart:qend]

        if qbs < N_total:
            print(f"\nQuery batch {qstart}-{qend} / {N_total}")

        depths = torch.full((N,), -1, dtype=torch.long, device=dev)

        # Initialize frontier
        frontier_states = torch.full((N, G, 3), pad, dtype=torch.long, device=dev)
        frontier_states[:, 0, :] = queries
        frontier_owner = torch.arange(N, device=dev)
        frontier_vars = torch.full((N,), dataset.constant_no + 1, dtype=torch.long, device=dev)

        visited_hashes = torch.empty(0, dtype=torch.int64, device=dev)
        flush_threshold = max_frontier

        for depth in range(1, max_depth + 1):
            F = frontier_states.shape[0]
            if F == 0:
                break

            t0 = time.time()
            effective_batch = safe_batch

            accum_states: list[Tensor] = []
            accum_owners: list[Tensor] = []
            accum_vars: list[Tensor] = []
            accum_count = 0

            for start in range(0, F, effective_batch):
                end = min(start + effective_batch, F)

                ns, no, nv, proof_owners = _process_chunk(
                    frontier_states[start:end],
                    frontier_vars[start:end],
                    frontier_owner[start:end],
                    grounder, compiled_pass, depths, G,
                )

                # Update depths for proven queries
                if proof_owners.numel() > 0:
                    not_yet = (depths[proof_owners] < 0)
                    depths[proof_owners] = torch.where(
                        not_yet, torch.tensor(depth, device=dev), depths[proof_owners])

                if ns is not None:
                    accum_states.append(ns)
                    accum_owners.append(no)
                    accum_vars.append(nv)
                    accum_count += ns.shape[0]

                # Intermediate flush
                if accum_count > flush_threshold and accum_states:
                    flush_s = torch.cat(accum_states, dim=0)
                    flush_o = torch.cat(accum_owners, dim=0)
                    flush_v = torch.cat(accum_vars, dim=0)
                    accum_states.clear()
                    accum_owners.clear()
                    accum_vars.clear()

                    flush_s, flush_o, flush_v, _ = dedup_within_depth(
                        flush_s, flush_o, flush_v, pack_base, pad)
                    still_unproven = (depths[flush_o] < 0)
                    flush_s = flush_s[still_unproven]
                    flush_o = flush_o[still_unproven]
                    flush_v = flush_v[still_unproven]
                    if flush_s.shape[0] > max_frontier:
                        flush_s = flush_s[:max_frontier]
                        flush_o = flush_o[:max_frontier]
                        flush_v = flush_v[:max_frontier]

                    accum_states.append(flush_s)
                    accum_owners.append(flush_o)
                    accum_vars.append(flush_v)
                    accum_count = flush_s.shape[0]
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

            # Assemble next frontier
            if not accum_states:
                elapsed = time.time() - t0
                n_proven = (depths >= 0).sum().item()
                print(f"  Depth {depth}: frontier=0 (was {F}), proven={n_proven}/{N}, {elapsed:.2f}s")
                break

            frontier_states = torch.cat(accum_states, dim=0)
            frontier_owner = torch.cat(accum_owners, dim=0)
            frontier_vars = torch.cat(accum_vars, dim=0)
            del accum_states, accum_owners, accum_vars

            pre_dedup = frontier_states.shape[0]

            # Within-depth dedup
            frontier_states, frontier_owner, frontier_vars, new_hashes = dedup_within_depth(
                frontier_states, frontier_owner, frontier_vars, pack_base, pad)

            # Cross-depth dedup
            frontier_states, frontier_owner, frontier_vars, new_hashes = dedup_cross_depth(
                frontier_states, frontier_owner, frontier_vars, new_hashes, visited_hashes)

            # Update visited set
            if new_hashes.shape[0] > 0:
                visited_hashes = torch.cat([visited_hashes, new_hashes]).sort()[0]

            # Remove proven
            still_unproven = (depths[frontier_owner] < 0)
            frontier_states = frontier_states[still_unproven]
            frontier_owner = frontier_owner[still_unproven]
            frontier_vars = frontier_vars[still_unproven]

            # Per-query cap
            before_caps = frontier_states.shape[0]
            frontier_states, frontier_owner, frontier_vars = cap_frontier_per_query(
                frontier_states, frontier_owner, frontier_vars, max_per_query, N)
            after_per_query_cap = frontier_states.shape[0]

            # Global cap
            if frontier_states.shape[0] > max_frontier:
                frontier_states = frontier_states[:max_frontier]
                frontier_owner = frontier_owner[:max_frontier]
                frontier_vars = frontier_vars[:max_frontier]

            elapsed = time.time() - t0
            n_proven = (depths >= 0).sum().item()
            post_dedup = frontier_states.shape[0]
            n_deduped = pre_dedup - post_dedup
            capped_pq = before_caps - after_per_query_cap
            capped_g = after_per_query_cap - post_dedup

            peak_mem = torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0.0

            cap_info = ""
            if capped_pq > 0 or capped_g > 0:
                cap_info = f" (capped: per-query={capped_pq}, global={capped_g})"
            print(f"  Depth {depth}: frontier {pre_dedup}->{post_dedup}, "
                  f"proven={n_proven}/{N}, {elapsed:.2f}s{cap_info}")

            all_depth_stats.append(DepthStats(
                depth=depth,
                frontier_size=post_dedup,
                n_proven_at_depth=(depths == depth).sum().item(),
                n_proven_cumulative=n_proven,
                n_deduped=n_deduped,
                n_capped_per_query=capped_pq,
                n_capped_global=capped_g,
                elapsed_sec=elapsed,
                peak_mem_mb=peak_mem,
            ))

            if dev.type == "cuda":
                torch.cuda.empty_cache()

        all_depths[qstart:qend] = depths

    # Write output
    depths_cpu = all_depths.cpu().tolist()
    depth_dist: dict[int, int] = {}
    for d in depths_cpu:
        depth_dist[d] = depth_dist.get(d, 0) + 1
    total_proven = sum(1 for d in depths_cpu if d >= 0)

    out_dir = output_dir or str(dataset.data_dir)
    os.makedirs(out_dir, exist_ok=True)
    depth_file = os.path.join(out_dir, f"{split}_depths_dynamic_{grounder_type}.txt")
    with open(depth_file, "w") as f:
        for i, qs in enumerate(query_strings):
            f.write(f"{qs} {depths_cpu[i]}\n")
    print(f"\nDepth file saved to: {depth_file}")

    # Write report
    report_file = os.path.join(out_dir, f"{split}_depths_dynamic_{grounder_type}_report.json")
    elapsed_total = time.time() - t_total
    write_report(
        output_path=report_file,
        mode="dynamic",
        dataset_name=dataset.data_dir.name,
        split=split,
        grounder_type=grounder_type,
        total_queries=N_total,
        total_proven=total_proven,
        depth_stats=all_depth_stats,
        config={
            "max_depth": max_depth,
            "max_goals": max_goals,
            "batch_size": batch_size,
            "max_frontier": max_frontier,
            "max_per_query": max_per_query,
            "query_batch_size": query_batch_size,
            "device": str(dev),
            "K": K,
            "M": M,
        },
        depth_distribution=depth_dist,
    )

    print(f"\nSummary:")
    print(f"  Total: {N_total}, Proven: {total_proven} ({total_proven/N_total:.1%})")
    print(f"  Distribution: {dict(sorted(depth_dist.items()))}")
    print(f"  Time: {elapsed_total:.1f}s ({N_total/elapsed_total:.1f} queries/sec)")

    return all_depths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate depth files using dynamic BFS with inter-depth compaction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Splits to process")
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--max_goals", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_frontier", type=int, default=2_000_000)
    parser.add_argument("--max_per_query", type=int, default=5000)
    parser.add_argument("--query_batch_size", type=int, default=0,
                        help="Queries to BFS simultaneously (0 = all)")
    parser.add_argument("--grounder", type=str, default="sld",
                        choices=["sld", "rtf"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--facts_file", type=str, default="facts.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dataset = KGDataset(args.data_dir, facts_file=args.facts_file, device=args.device)
    print(f"Loaded: {dataset}")

    for split in args.splits:
        generate_depths_dynamic(
            dataset=dataset,
            resolution=args.grounder,
            split=split,
            max_depth=args.max_depth,
            max_goals=args.max_goals,
            batch_size=args.batch_size,
            max_frontier=args.max_frontier,
            max_per_query=args.max_per_query,
            query_batch_size=args.query_batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
