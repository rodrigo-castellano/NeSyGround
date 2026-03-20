"""Speed profile: default enum vs cartesian+all_anchors enum.

Runs queries on the family dataset with both configurations and
asserts the new options complete in < 2x the wall time of the default.

Usage:
    cd grounder && PYTHONUNBUFFERED=1 python tests/profile_speed_enum.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Ensure grounder package is importable when run from tests/ dir
_grounder_root = Path(__file__).resolve().parent.parent.parent
if str(_grounder_root) not in sys.path:
    sys.path.insert(0, str(_grounder_root))

from grounder.data.loader import KGDataset
from grounder.bc.bc import BCGrounder

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "family"
DEVICE = "cpu"
N_QUERIES = 20
SLOWDOWN_LIMIT = 2.0


def _build_grounder(
    kb, *, cartesian_product: bool = False, all_anchors: bool = False,
) -> BCGrounder:
    # Keep budgets small for cartesian to avoid OOM at step-1
    if cartesian_product:
        tG, gpq, ms = 1024, 1024, 256
    else:
        tG, gpq, ms = 4096, 4096, None
    return BCGrounder(
        kb,
        resolution="enum",
        filter="fp_batch",
        depth=2,
        width=1,
        max_total_groundings=tG,
        max_groundings_per_query=gpq,
        max_states=ms,
        cartesian_product=cartesian_product,
        all_anchors=all_anchors,
    )


def _run_queries(grounder: BCGrounder, queries: torch.Tensor) -> float:
    """Run queries one at a time and return total wall time in seconds."""
    N = queries.shape[0]
    start = time.perf_counter()
    for i in range(N):
        q = queries[i : i + 1]
        mask = torch.ones(1, dtype=torch.bool)
        grounder(q, mask)
    elapsed = time.perf_counter() - start
    return elapsed


def main() -> None:
    assert DATA_DIR.exists(), f"Family dataset not found: {DATA_DIR}"

    ds = KGDataset(str(DATA_DIR), device=DEVICE)
    kb = ds.make_kb(fact_index_type="block_sparse")

    # Use test queries (ground triples)
    queries_all = ds.get_queries("test")
    n = min(N_QUERIES, queries_all.shape[0])
    queries = queries_all[:n]

    print(f"Dataset: {ds}")
    print(f"Queries: {n}")

    # ── Default enum ──
    g_default = _build_grounder(kb)
    t_default = _run_queries(g_default, queries)
    print(f"Default enum:        {t_default:.3f}s ({n/t_default:.0f} q/s)")

    # ── Cartesian + all_anchors ──
    g_new = _build_grounder(kb, cartesian_product=True, all_anchors=True)
    t_new = _run_queries(g_new, queries)
    print(f"Cartesian+allAnch:   {t_new:.3f}s ({n/t_new:.0f} q/s)")

    ratio = t_new / t_default
    print(f"Slowdown ratio:      {ratio:.2f}x")

    if ratio >= SLOWDOWN_LIMIT:
        raise AssertionError(
            f"New options are {ratio:.2f}x slower than default "
            f"(limit: {SLOWDOWN_LIMIT}x)"
        )
    print("PASSED: slowdown within acceptable bounds.")


if __name__ == "__main__":
    main()
