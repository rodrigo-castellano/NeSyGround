"""JSON summary report generation for depth analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class DepthStats:
    """Statistics for a single BFS depth level."""
    depth: int
    frontier_size: int
    n_proven_at_depth: int
    n_proven_cumulative: int
    n_deduped: int = 0
    n_capped_per_query: int = 0
    n_capped_global: int = 0
    elapsed_sec: float = 0.0
    peak_mem_mb: float = 0.0


def write_report(
    output_path: str,
    mode: str,
    dataset_name: str,
    split: str,
    grounder_type: str,
    total_queries: int,
    total_proven: int,
    depth_stats: List[DepthStats],
    config: dict,
    depth_distribution: Dict[int, int],
) -> None:
    """Write JSON summary report.

    Args:
        output_path: Path to write JSON file
        mode: 'dynamic' or 'static'
        dataset_name: Name of the dataset
        split: Data split ('train', 'valid', 'test')
        grounder_type: 'PrologGrounder' or 'RTFGrounder'
        total_queries: Total number of queries processed
        total_proven: Number of queries proven
        depth_stats: Per-depth statistics
        config: CLI args / config used
        depth_distribution: {depth: count} distribution
    """
    report = {
        "mode": mode,
        "dataset": dataset_name,
        "split": split,
        "grounder_type": grounder_type,
        "total_queries": total_queries,
        "total_proven": total_proven,
        "prove_rate": total_proven / total_queries if total_queries > 0 else 0.0,
        "depth_distribution": {str(k): v for k, v in sorted(depth_distribution.items())},
        "per_depth": [asdict(s) for s in depth_stats],
        "config": config,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {output_path}")
