"""Grounding count regression test: per-query exact match vs baseline.

Default: family, sld resolution, depth=4, 100 test queries.
Uses the grounder's own KGDataset and BCGrounder — no torch-ns/DpRL deps.

Override: pytest tests/test_groundings.py --dataset wn18rr --depth 2
Generate: pytest tests/test_groundings.py --generate-baseline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

from baseline_utils import (
    canonicalize_dict,
    fingerprint_paths,
    git_info,
    json_dump,
    load_json,
    runtime_env,
    utc_now_iso,
)

MAX_QUERIES = 100
BASELINE_FILE = "groundings.json"
DEVICE = torch.device("cpu")


def _stats_from_counts(counts: torch.Tensor, n_queries: int) -> dict[str, Any]:
    c = counts.float()
    n = int(n_queries)
    proved = int((c > 0).sum().item())
    return {
        "n_queries": n,
        "mean": round(float(c.mean()), 3),
        "std": round(float(c.std(unbiased=False)), 3),
        "pct_proved": round(100.0 * proved / max(n, 1), 1),
        "max": int(c.max().item()),
        "total": int(c.sum().item()),
        "p50": round(float(c.quantile(0.50).item()), 1),
        "p95": round(float(c.quantile(0.95).item()), 1),
        "p99": round(float(c.quantile(0.99).item()), 1),
    }


def _run_groundings(
    dataset: str, grounder_type: str, depth: int, data_root: Path,
) -> dict[str, Any]:
    """Run grounder on test queries, return metrics with per_query_counts."""
    from grounder.data_loader import KGDataset
    from grounder import BCGrounder

    ds = KGDataset(str(data_root / dataset), device=str(DEVICE))
    kb = ds.make_kb()

    # Build queries from test split
    queries_all = ds.get_queries("test")
    if queries_all.shape[0] == 0:
        pytest.skip(f"No test queries for {dataset}")

    # Subsample to MAX_QUERIES (deterministic via sorted dataset)
    n_queries = min(MAX_QUERIES, queries_all.shape[0])
    queries = queries_all[:n_queries]
    query_mask = torch.ones(n_queries, dtype=torch.bool, device=DEVICE)

    # Replace second argument with variable for open queries
    var_idx = kb.constant_no + 1
    queries = queries.clone()
    queries[:, 2] = var_idx

    # Build grounder
    max_body = kb.M
    max_goals = max_body + (max_body - 1) * depth + 1
    grounder = BCGrounder(
        kb,
        resolution=grounder_type,
        filter='fp_batch',
        max_goals=max_goals,
        depth=depth,
        max_total_groundings=64,
        K_MAX=50,
    )

    result = grounder(queries, query_mask)
    per_query = result.count.cpu().long().tolist()
    metrics = _stats_from_counts(result.count, n_queries)
    metrics["per_query_counts"] = per_query
    return metrics


def _dataset_files(data_root: Path, dataset: str) -> list[Path]:
    ds = data_root / dataset
    return [ds / f for f in ("train.txt", "test.txt", "rules.txt")]


def test_grounding_counts(dataset, grounder_type, depth, baseline_dir,
                          data_root, generate_baseline):
    data_root = Path(data_root)
    ds_path = data_root / dataset
    if not ds_path.exists():
        pytest.skip(f"Dataset not found: {ds_path}")

    print(f"\n[groundings] dataset={dataset} grounder={grounder_type} depth={depth}",
          flush=True)

    metrics = _run_groundings(dataset, grounder_type, depth, data_root)
    print(f"  mean={metrics['mean']:.1f} pct_proved={metrics['pct_proved']:.1f}% "
          f"total={metrics['total']}", flush=True)

    bp = baseline_dir / BASELINE_FILE
    config_dict = canonicalize_dict({
        "dataset": dataset,
        "grounder_type": grounder_type,
        "depth": depth,
        "max_queries": MAX_QUERIES,
    })

    if generate_baseline:
        payload = {
            "schema_version": 1,
            "kind": "groundings",
            "generated_at_utc": utc_now_iso(),
            "config": config_dict,
            "runtime_env": runtime_env(),
            "data_fingerprint": fingerprint_paths(_dataset_files(data_root, dataset)),
            "code_fingerprint": git_info(),
            "result": metrics,
        }
        json_dump(bp, payload)
        print(f"\n[baseline] wrote: {bp}")
        pytest.skip("Baseline generated")
        return

    assert bp.exists(), (
        f"FAILED: Baseline not found at {bp}\n"
        f"Run: pytest tests/test_groundings.py --generate-baseline"
    )

    bl = load_json(bp)
    base_pq = bl["result"]["per_query_counts"]
    cur_pq = metrics["per_query_counts"]

    assert len(base_pq) == len(cur_pq), (
        f"per_query_counts length mismatch: baseline={len(base_pq)} current={len(cur_pq)}"
    )

    mismatches = [
        i for i, (b, c) in enumerate(zip(base_pq, cur_pq)) if b != c
    ]
    if mismatches:
        n_mis = len(mismatches)
        examples = mismatches[:5]
        details = ", ".join(f"q{i}:{base_pq[i]}→{cur_pq[i]}" for i in examples)
        suffix = f" (+{n_mis - 5} more)" if n_mis > 5 else ""
        pytest.fail(
            f"Grounding counts differ: {n_mis}/{len(base_pq)} queries: {details}{suffix}"
        )
