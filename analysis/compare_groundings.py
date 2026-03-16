"""Compare provability and depth distributions across grounders.

Supports:
  - swi: SWI-Prolog gold standard
  - sld: BCGrounder with SLD resolution (static depth gen)
  - rtf: BCGrounder with RTF resolution (static depth gen)
  - sld_dynamic: BCGrounder SLD with dynamic depth gen
  - rtf_dynamic: BCGrounder RTF with dynamic depth gen

Usage:
    python -m grounder.analysis.compare_groundings --data_dir kge_experiments/data/family --grounders swi,sld,rtf
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from grounder.data_loader import KGDataset


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

def run_swi_prolog(
    dataset: KGDataset,
    queries: List[Tuple[str, str, str]],
    max_depth: int,
    inference_limit: int = 100_000_000,
    depth_semantics: str = "sld",
    exclude_self: bool = True,
) -> Tuple[Set[int], float]:
    """Run SWI-Prolog gold standard prover.

    Returns:
        (set of provable query indices, elapsed seconds)
    """
    from grounder.analysis.gold_standard import PrologProver

    prover = PrologProver(
        dataset=dataset,
        max_depth=max_depth,
        inference_limit=inference_limit,
        depth_semantics=depth_semantics,
    )
    prover.verify(n_samples=min(5, len(dataset._facts_raw)))
    return prover.prove(queries, exclude_self=exclude_self)


def run_gpu_static(
    dataset: KGDataset,
    resolution: str,
    split: str,
    max_depth: int,
    max_goals: int = 20,
    max_states: Optional[int] = None,
    hard_cap: int = 4096,
    batch_size: int = 256,
    compile_mode: str = "reduce-overhead",
    device: str = "cuda",
) -> Tuple[Set[int], float]:
    """Run static depth generation via BCGrounder.forward().

    Returns:
        (set of provable query indices, elapsed seconds)
    """
    from grounder.analysis.generate_depths_static import generate_depths_static

    t0 = time.perf_counter()
    depths = generate_depths_static(
        dataset=dataset,
        resolution=resolution,
        split=split,
        max_depth=max_depth,
        max_goals=max_goals,
        max_states=max_states,
        hard_cap=hard_cap,
        batch_size=batch_size,
        compile_mode=compile_mode,
        device=device,
        output_dir=None,
    )
    elapsed = time.perf_counter() - t0
    provable = set((depths >= 0).nonzero(as_tuple=True)[0].cpu().tolist())
    return provable, elapsed


def run_gpu_dynamic(
    dataset: KGDataset,
    resolution: str,
    split: str,
    max_depth: int,
    max_goals: int = 20,
    batch_size: int = 512,
    max_frontier: int = 2_000_000,
    max_per_query: int = 5000,
    device: str = "cuda",
) -> Tuple[Set[int], float]:
    """Run dynamic depth generation via BCGrounder BFS loop.

    Returns:
        (set of provable query indices, elapsed seconds)
    """
    from grounder.analysis.generate_depths_dynamic import generate_depths_dynamic

    t0 = time.perf_counter()
    depths = generate_depths_dynamic(
        dataset=dataset,
        resolution=resolution,
        split=split,
        max_depth=max_depth,
        max_goals=max_goals,
        batch_size=batch_size,
        max_frontier=max_frontier,
        max_per_query=max_per_query,
        device=device,
        output_dir=None,
    )
    elapsed = time.perf_counter() - t0
    provable = set((depths >= 0).nonzero(as_tuple=True)[0].cpu().tolist())
    return provable, elapsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(
    results: Dict[str, Tuple[Set[int], float]],
    max_depth: int,
    N: int,
) -> None:
    """Print provability comparison table."""
    names = list(results.keys())
    col_w = max(12, max(len(n) for n in names) + 2)

    header = f"{'Grounder':<{col_w}} {'Proven':>8} {'Rate':>8} {'Time (s)':>10}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for name in names:
        provable, elapsed = results[name]
        n_proven = len(provable)
        rate = n_proven / N if N > 0 else 0.0
        print(f"{name:<{col_w}} {n_proven:>8} {rate:>8.1%} {elapsed:>10.2f}")

    print(f"{'='*len(header)}")
    print(f"Total queries: {N}")


def print_pairwise(
    results: Dict[str, Tuple[Set[int], float]],
    N: int,
) -> None:
    """Print pairwise agreement between grounders."""
    names = list(results.keys())
    if len(names) < 2:
        return

    print(f"\nPairwise agreement (intersection / union):")
    col_w = max(12, max(len(n) for n in names) + 2)

    # Header
    header = f"{'':>{col_w}}"
    for n in names:
        header += f" {n:>{col_w}}"
    print(header)
    print("-" * len(header))

    for i, n1 in enumerate(names):
        s1 = results[n1][0]
        row = f"{n1:>{col_w}}"
        for j, n2 in enumerate(names):
            s2 = results[n2][0]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            if union == 0:
                row += f" {'N/A':>{col_w}}"
            else:
                jaccard = inter / union
                row += f" {jaccard:>{col_w}.3f}"
        print(row)

    # Detailed pairwise diffs
    print(f"\nPairwise differences:")
    for i, n1 in enumerate(names):
        s1 = results[n1][0]
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            s2 = results[n2][0]
            only_1 = s1 - s2
            only_2 = s2 - s1
            both = s1 & s2
            print(f"  {n1} vs {n2}: both={len(both)}, "
                  f"only-{n1}={len(only_1)}, only-{n2}={len(only_2)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GROUNDER_CHOICES = ["swi", "sld", "rtf", "sld_dynamic", "rtf_dynamic"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare provability and depth distributions across grounders.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to evaluate (default: test)")
    parser.add_argument("--grounders", type=str, default="swi,sld",
                        help=f"Comma-separated grounders: {','.join(GROUNDER_CHOICES)}")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum proof depth (default: 3)")
    parser.add_argument("--max_goals", type=int, default=20,
                        help="G dimension for GPU grounders (default: 20)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for GPU grounders (default: 256)")
    parser.add_argument("--hard_cap", type=int, default=4096,
                        help="Hard cap for static S (default: 4096)")
    parser.add_argument("--max_frontier", type=int, default=2_000_000,
                        help="Max frontier for dynamic (default: 2M)")
    parser.add_argument("--max_per_query", type=int, default=5000,
                        help="Per-query frontier cap for dynamic (default: 5000)")
    parser.add_argument("--inference_limit", type=int, default=100_000_000,
                        help="SWI-Prolog inference limit (default: 100M)")
    parser.add_argument("--depth_semantics", type=str, default="sld",
                        choices=["sld", "rule_only"],
                        help="Depth semantics for SWI-Prolog (default: sld)")
    parser.add_argument("--exclude_self", action="store_true", default=True,
                        help="Exclude self-proofs (default: True)")
    parser.add_argument("--no_exclude_self", dest="exclude_self", action="store_false")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        help="torch.compile mode (default: reduce-overhead)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for GPU grounders (default: cuda)")
    parser.add_argument("--facts_file", type=str, default="facts.txt",
                        help="Facts file name (default: facts.txt)")
    args = parser.parse_args()

    grounder_names = [g.strip() for g in args.grounders.split(",")]
    for g in grounder_names:
        if g not in GROUNDER_CHOICES:
            parser.error(f"Unknown grounder '{g}'. Choose from: {', '.join(GROUNDER_CHOICES)}")

    # Determine if we need GPU
    needs_gpu = any(g in grounder_names for g in ("sld", "rtf", "sld_dynamic", "rtf_dynamic"))
    device = args.device if needs_gpu else "cpu"

    dataset = KGDataset(args.data_dir, facts_file=args.facts_file, device=device)
    print(f"Loaded: {dataset}")

    # Get queries as string tuples
    if args.split not in dataset._splits_raw:
        print(f"Split '{args.split}' not found. Available: {list(dataset._splits_raw.keys())}")
        return
    queries = dataset._splits_raw[args.split]
    N = len(queries)
    print(f"Split: {args.split}, Queries: {N}")

    results: Dict[str, Tuple[Set[int], float]] = {}

    # Map grounder names to resolution strings
    _RESOLUTION = {"sld": "sld", "rtf": "rtf", "sld_dynamic": "sld", "rtf_dynamic": "rtf"}

    for name in grounder_names:
        print(f"\n--- Running: {name} ---")
        if name == "swi":
            provable, elapsed = run_swi_prolog(
                dataset=dataset,
                queries=queries,
                max_depth=args.max_depth,
                inference_limit=args.inference_limit,
                depth_semantics=args.depth_semantics,
                exclude_self=args.exclude_self,
            )
        elif name in ("sld", "rtf"):
            provable, elapsed = run_gpu_static(
                dataset=dataset,
                resolution=_RESOLUTION[name],
                split=args.split,
                max_depth=args.max_depth,
                max_goals=args.max_goals,
                hard_cap=args.hard_cap,
                batch_size=args.batch_size,
                compile_mode=args.compile_mode,
                device=args.device,
            )
        elif name in ("sld_dynamic", "rtf_dynamic"):
            provable, elapsed = run_gpu_dynamic(
                dataset=dataset,
                resolution=_RESOLUTION[name],
                split=args.split,
                max_depth=args.max_depth,
                max_goals=args.max_goals,
                batch_size=args.batch_size,
                max_frontier=args.max_frontier,
                max_per_query=args.max_per_query,
                device=args.device,
            )
        else:
            raise ValueError(f"Unknown grounder: {name}")

        results[name] = (provable, elapsed)
        print(f"  {name}: {len(provable)}/{N} proven in {elapsed:.2f}s")

    # Print comparison
    print_table(results, args.max_depth, N)
    print_pairwise(results, N)


if __name__ == "__main__":
    main()
