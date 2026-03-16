"""Static depth generation using BCGrounder.forward() with fixed-shape tensors.

Uses BCGrounder's compiled forward() with an on_depth_complete callback to track
which queries are first proved at each depth. States that overflow S are lost,
so this approach may miss proofs that the dynamic approach finds.

Usage:
    python -m grounder.analysis.generate_depths_static \
        --data_dir kge_experiments/data/family \
        --splits test --max_depth 5
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import torch
from torch import Tensor

from grounder.data_loader import KGDataset
from grounder.bc.bc import BCGrounder
from grounder.analysis._report import DepthStats, write_report


@torch.no_grad()
def generate_depths_static(
    dataset: KGDataset,
    resolution: str,
    split: str,
    max_depth: int,
    max_goals: int,
    max_states: Optional[int] = None,
    hard_cap: int = 4096,
    batch_size: int = 256,
    compile_mode: str = "reduce-overhead",
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Tensor:
    """Generate depth annotations using BCGrounder.forward() with callback.

    Args:
        dataset: Loaded KGDataset
        resolution: 'sld' or 'rtf'
        split: 'train', 'valid', or 'test'
        max_depth: Maximum proof depth
        max_goals: G dimension (max atoms per state)
        max_states: S cap (None = auto: min(K^depth, hard_cap))
        hard_cap: Upper bound for auto-S calculation
        batch_size: Queries per batch (B dimension)
        compile_mode: torch.compile mode
        device: Target device
        output_dir: Output directory (default: dataset directory)

    Returns:
        [N] int tensor of minimum proof depths (-1 if not provable)
    """
    queries_idx = dataset.get_queries(split)
    query_strings = dataset.get_query_strings(split)
    N = queries_idx.shape[0]
    if N == 0:
        print(f"No queries for split '{split}', skipping.")
        return torch.empty(0, dtype=torch.long)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    queries_idx = queries_idx.to(dev)

    # Create grounder with full depth
    kb = dataset.make_kb()
    grounder = BCGrounder(
        kb,
        resolution=resolution,
        filter="prune",
        depth=max_depth,
        max_goals=max_goals,
        max_states=max_states if max_states is not None else hard_cap,
        compile_mode=compile_mode if dev.type == "cuda" else None,
        track_grounding_body=False,
    )

    # Auto-calculate S if not provided
    K = grounder.K
    if max_states is None:
        auto_S = min(K ** max_depth, hard_cap)
        if K ** max_depth > hard_cap:
            print(f"WARNING: K^depth = {K}^{max_depth} = {K**max_depth} > hard_cap={hard_cap}. "
                  f"Some proofs may be missed due to state overflow.")
        grounder.S = auto_S
        grounder._build_compiled_fns()

    grounder = grounder.to(dev)
    grounder_type = resolution

    print(f"\n{'='*60}")
    print(f"Static depth generation: {dataset.data_dir.name} / {split}")
    print(f"Grounder: {grounder_type}, K={K}, S={grounder.S}, G={max_goals}")
    print(f"Max depth: {max_depth}, Batch size: {batch_size}")
    print(f"Device: {dev}, Compile: {compile_mode}")
    print(f"{'='*60}\n")

    # Result: -1 means not provable
    depths = torch.full((N,), -1, dtype=torch.long, device=dev)
    depth_stats: list[DepthStats] = []
    t_total = time.time()

    # Process in batches
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_queries = queries_idx[batch_start:batch_end]
        B = batch_queries.shape[0]
        query_mask = torch.ones(B, dtype=torch.bool, device=dev)

        # Track per-depth proofs via callback
        batch_depths = torch.full((B,), -1, dtype=torch.long, device=dev)

        def _on_depth(d: int, newly_proved: Tensor) -> None:
            not_yet = (batch_depths[newly_proved] < 0)
            if not_yet.any():
                indices = newly_proved.nonzero(as_tuple=True)[0]
                indices = indices[not_yet]
                batch_depths[indices] = d

        grounder.forward(batch_queries, query_mask, on_depth_complete=_on_depth)
        depths[batch_start:batch_end] = batch_depths

        if batch_start % (batch_size * 10) == 0 or batch_end == N:
            n_proven = (depths[:batch_end] >= 0).sum().item()
            print(f"  Processed {batch_end}/{N} queries, proven={n_proven}")

    # Compute statistics
    depths_cpu = depths.cpu().tolist()
    depth_dist: dict[int, int] = {}
    for d in depths_cpu:
        depth_dist[d] = depth_dist.get(d, 0) + 1

    total_proven = sum(1 for d in depths_cpu if d >= 0)
    elapsed = time.time() - t_total

    # Collect depth stats
    cumulative = 0
    for d in range(1, max_depth + 1):
        count = depth_dist.get(d, 0)
        cumulative += count
        depth_stats.append(DepthStats(
            depth=d,
            frontier_size=0,  # not tracked in static mode
            n_proven_at_depth=count,
            n_proven_cumulative=cumulative,
            elapsed_sec=elapsed / max_depth,
        ))

    # Write output
    out_dir = output_dir or str(dataset.data_dir)
    os.makedirs(out_dir, exist_ok=True)
    depth_file = os.path.join(out_dir, f"{split}_depths_static_{grounder_type}.txt")
    with open(depth_file, "w") as f:
        for i, qs in enumerate(query_strings):
            f.write(f"{qs} {depths_cpu[i]}\n")
    print(f"\nDepth file saved to: {depth_file}")

    # Write report
    report_file = os.path.join(out_dir, f"{split}_depths_static_{grounder_type}_report.json")
    write_report(
        output_path=report_file,
        mode="static",
        dataset_name=dataset.data_dir.name,
        split=split,
        grounder_type=grounder_type,
        total_queries=N,
        total_proven=total_proven,
        depth_stats=depth_stats,
        config={
            "max_depth": max_depth,
            "max_goals": max_goals,
            "max_states": grounder.S,
            "hard_cap": hard_cap,
            "batch_size": batch_size,
            "compile_mode": compile_mode,
            "device": str(dev),
            "K": K,
        },
        depth_distribution=depth_dist,
    )

    print(f"\nSummary:")
    print(f"  Total: {N}, Proven: {total_proven} ({total_proven/N:.1%})")
    print(f"  Distribution: {dict(sorted(depth_dist.items()))}")
    print(f"  Time: {elapsed:.1f}s ({N/elapsed:.1f} queries/sec)")

    return depths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate depth files using static BCGrounder.forward()")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Splits to process")
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--max_goals", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_states_cap", type=int, default=4096,
                        help="Hard cap for auto-S calculation")
    parser.add_argument("--grounder", type=str, default="sld",
                        choices=["sld", "rtf"])
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--facts_file", type=str, default="facts.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dataset = KGDataset(args.data_dir, facts_file=args.facts_file, device=args.device)
    print(f"Loaded: {dataset}")

    for split in args.splits:
        generate_depths_static(
            dataset=dataset,
            resolution=args.grounder,
            split=split,
            max_depth=args.max_depth,
            max_goals=args.max_goals,
            hard_cap=args.max_states_cap,
            batch_size=args.batch_size,
            compile_mode=args.compile_mode,
            device=args.device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
