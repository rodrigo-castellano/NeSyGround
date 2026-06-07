"""FC closure-fingerprint oracle — byte-identity gate for every FC-touching step.

Builds the ``ForwardGrounder`` per cell (CPU, real predicate range), runs the
closure, and freezes ``(sha over sorted closure hashes, n_provable)`` per
(dataset, method) against ``tests/baselines/fc_fingerprint.json``. Mirrors
``fingerprint_new.py`` for the backward side.

OOM caveat (documented): FC on wn18rr at full depth is a transitive-closure
blow-up (60GB runaway). This oracle uses ONLY small closures — family +
countries_s2 — on spmm AND staged. NEVER add a wn18rr full-depth FC cell.

Usage:
    python tests/fc_fingerprint.py            # check all cells
    python tests/fc_fingerprint.py --cell family|spmm|d10
    python tests/fc_fingerprint.py --write     # (re)write the baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from grounder.data.dataset import KGDataset
from grounder.forward.grounder import ForwardGrounder

_HERE = Path(__file__).resolve().parent
_BASELINE = (_HERE / "baselines" / "fc_fingerprint.json")
_RULES_FILE = {"family": "rules_old.txt"}
_GATE_FIELDS = ("sha", "n_provable", "numel")

# SMALL closures only (family + countries_s2), spmm + staged, plus one
# bounded-depth ablation. NEVER a wn18rr full-depth cell (60GB runaway).
_MATRIX = [
    ("family", "spmm", {"depth": 10}),
    ("family", "staged", {"depth": 10}),
    ("countries_s2", "spmm", {"depth": 10}),
    ("countries_s2", "staged", {"depth": 10}),
    ("countries_s2", "spmm", {"depth": 3}),
    ("countries_s2", "staged", {"depth": 3}),
]


def _canonical_fingerprint(hashes, n_provable) -> dict:
    """SHA over the sorted closure hashes (the FC closure is set-valued)."""
    h = hashes.to("cpu", torch.int64)
    if h.numel() == 0:
        return {"sha": "EMPTY", "n_provable": int(n_provable), "numel": 0}
    sha = hashlib.sha256(h.numpy().tobytes()).hexdigest()[:16]
    return {"sha": sha, "n_provable": int(n_provable), "numel": int(h.numel())}


def _cell_key(dataset, method, cfg) -> str:
    return f"{dataset}|{method}|d{cfg['depth']}"


def _old_closure_facts(hashes: torch.Tensor, n_provable: int, E: int) -> torch.Tensor:
    """The VERBATIM pre-Step-2 decode (frozen reference for the Closure.facts() A/B)."""
    if n_provable == 0:
        return torch.empty(0, 3, dtype=torch.long, device=hashes.device)
    E2 = E * E
    pred = hashes // E2
    rem = hashes % E2
    return torch.stack([pred, rem // E, rem % E], dim=1)


def compute_fingerprint(dataset, method, cfg, *, data_root) -> dict:
    ds_path = Path(data_root).expanduser() / dataset
    rules_file = _RULES_FILE.get(dataset, "rules.txt")
    ds = KGDataset(str(ds_path), device="cpu", rules_file=rules_file)
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")

    g = ForwardGrounder(kb, method=method, depth=cfg["depth"])
    with torch.no_grad():
        hashes, n_provable = g.closure_hashes()
        closure = g.ground()                     # Step 2: ground() -> Closure

    # Closure.facts() must decode EXACTLY as the old closure_facts() did.
    new_facts = closure.facts()
    old_facts = _old_closure_facts(hashes, n_provable, int(g._num_entities))
    assert new_facts.shape == old_facts.shape and torch.equal(new_facts, old_facts), (
        f"{dataset}|{method}|d{cfg['depth']}: Closure.facts() != old closure_facts()")
    assert int(closure.n_provable) == int(n_provable)

    fp = _canonical_fingerprint(hashes, n_provable)
    fp.update(dataset=dataset, method=method, cfg=cfg, rules=rules_file,
              num_entities=int(g._num_entities),
              num_predicates=int(g._num_predicates))
    return fp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path.home() / "repos/data-swarm/main"))
    p.add_argument("--cell", default=None, help="run a single cell key, e.g. 'family|spmm|d10'")
    p.add_argument("--write", action="store_true", help="(re)write the frozen baseline")
    args = p.parse_args()

    matrix = _MATRIX
    if args.cell:
        matrix = [(d, m, c) for (d, m, c) in _MATRIX if _cell_key(d, m, c) == args.cell]
        if not matrix:
            print(f"unknown cell {args.cell!r}"); sys.exit(2)

    computed = {}
    for dataset, method, cfg in matrix:
        key = _cell_key(dataset, method, cfg)
        computed[key] = compute_fingerprint(dataset, method, cfg, data_root=args.data_root)

    if args.write:
        _BASELINE.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE.write_text(json.dumps({"cells": computed}, indent=2, sort_keys=True) + "\n")
        print(f"wrote baseline -> {_BASELINE} ({len(computed)} cells)")
        return

    frozen = json.loads(_BASELINE.read_text())["cells"]
    drift = []
    for key, fp in computed.items():
        exp = frozen.get(key, {})
        status = "OK"
        for field in _GATE_FIELDS:
            if exp.get(field) != fp.get(field):
                status = "DRIFT"
                drift.append((key, f"{field}: {exp.get(field)!r} -> {fp.get(field)!r}"))
        print(f"  [{status:5s}] {key:26s} sha={fp['sha']} "
              f"n_provable={fp['n_provable']:>8} numel={fp['numel']:>8}")

    if drift:
        print(f"\nFAIL — {len(drift)} drift line(s):")
        for key, msg in drift:
            print(f"  DRIFT {key}: {msg}")
        sys.exit(1)
    print(f"\nPASS — all {len(computed)} cells match the frozen FC fingerprint.")


if __name__ == "__main__":
    main()
