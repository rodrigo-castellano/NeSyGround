"""Grounding-fingerprint harness — gates the engine against the frozen oracle
baseline (``tests/baselines/ground_fingerprint.json``).

For every cell it builds the ``BackwardGrounder`` (CPU/eager, 50 queries) and
computes the canonical SHA over ``out.rule_groundings`` +
``out.completed_tree_firings.count``, then checks TWO things:

  1. byte-identity vs the frozen baseline, and
  2. batch-size invariance: the cell is re-run at chunk sizes {None, 8, 4, 1} and
     every run must produce the IDENTICAL SHA (the chunk driver must not change
     what is grounded).

Matrix: 6 datasets {ablation_d2, ablation_d3, countries_s2, countries_s3, family,
wn18rr} x bc01/bc12/bc13 x {enum-flat, enum-dense}, plus SLD/RTF on family d3 and
countries_s2 d2 = 40 cells.

Usage:
    python tests/fingerprint_new.py                 # check all 40, all batch sizes
    python tests/fingerprint_new.py --cell family|SLD|d3
    python tests/fingerprint_new.py --no-batch-invariance   # full-batch only (fast)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from grounder.api.config import Backward, PBC, RTF, SLD
from grounder.data.dataset import KGDataset
from grounder.api.backward import BackwardGrounder
from grounder.core import GroundRequest, OutputSpec, Tier

_ALL_TIERS = OutputSpec(frozenset({Tier.PROOF_STATE, Tier.FIRINGS, Tier.TREES}))
_HERE = Path(__file__).resolve().parent
_BASELINE = (_HERE / "baselines" / "ground_fingerprint.json")
_RULES_FILE = {"family": "rules_old.txt"}
_GATE_FIELDS = ("sha", "n_firings", "ground_proofs")
_DEFAULT_MAX_QUERIES = 50
_BATCH_SIZES = (None, 8, 4, 1)   # None = all queries in one chunk

_ENUM_DATASETS = ("ablation_d2", "ablation_d3", "countries_s2", "countries_s3",
                  "family", "wn18rr")
_ENUM_WD = ((0, 1), (1, 2), (1, 3))   # bc01, bc12, bc13


def _build_matrix():
    cells = []
    for ds in _ENUM_DATASETS:
        for (w, d) in _ENUM_WD:
            for kind in ("enum-flat", "enum-dense"):
                cells.append((ds, kind, {"w": w, "d": d}))
    for ds, dep in (("family", 3), ("countries_s2", 2)):
        cells.append((ds, "SLD", {"depth": dep}))
        cells.append((ds, "RTF", {"depth": dep}))
    return cells


_MATRIX = _build_matrix()


def _build_grounder(kind: str, kb, cfg: dict, chunk_size=None) -> BackwardGrounder:
    """Mirror ``tests/_runners.build_torch_grounder`` config exactly."""
    if kind in ("SLD", "RTF"):
        res = SLD(depth=cfg["depth"]) if kind == "SLD" else RTF(depth=cfg["depth"])
        config = Backward(res, filter="none", max_groundings_per_query=4096,
                          max_children=64, max_states=256, prune_facts=True)
        return BackwardGrounder(kb, config, chunk_size=chunk_size)
    if kind in ("enum-flat", "enum-dense"):
        flat = (kind == "enum-flat")
        config = Backward(
            PBC(depth=cfg["d"], width=cfg["w"], u=0, flat_intermediate=flat,
                max_groundings_per_rule=64),
            max_groundings_per_query=4096, max_states=256, prune_facts=True,
            bump_s_to_k=False, init_state_shape="minimal")
        return BackwardGrounder(kb, config, chunk_size=chunk_size)
    raise ValueError(f"unsupported kind: {kind!r}")


def _ground_proofs(out) -> int:
    ev = getattr(out, "completed_tree_firings", None)
    if ev is not None and getattr(ev, "count", None) is not None:
        return int(ev.count.sum().item())
    return 0


def _canonical_fingerprint(out) -> dict:
    rg = getattr(out, "rule_groundings", None)
    if rg is None or getattr(rg, "head_pool_idx", None) is None \
            or rg.head_pool_idx.numel() == 0:
        return {"sha": "EMPTY", "n_firings": 0, "num_atoms": 0,
                "per_rule": {}, "ground_proofs": _ground_proofs(out)}

    at = rg.atom_table.to("cpu", torch.int64)
    head_idx = rg.head_pool_idx.to("cpu", torch.int64)
    body_idx = rg.body_pool_idx.to("cpu", torch.int64)
    bvalid = rg.body_atom_valid.to("cpu", torch.bool)
    rule_idx = rg.rule_idx.to("cpu", torch.int64)
    fvalid = (rg.firing_valid.to("cpu", torch.bool)
              if getattr(rg, "firing_valid", None) is not None
              else torch.ones(head_idx.shape[0], dtype=torch.bool))

    rows = []
    per_rule: dict = {}
    F = head_idx.shape[0]
    for f in range(F):
        if not bool(fvalid[f]):
            continue
        r = int(rule_idx[f])
        head = tuple(at[head_idx[f]].tolist())
        body = sorted(tuple(at[body_idx[f, j]].tolist())
                      for j in range(body_idx.shape[1]) if bool(bvalid[f, j]))
        rows.append((r, head, tuple(body)))
        per_rule[r] = per_rule.get(r, 0) + 1

    rows.sort()
    h = hashlib.sha256()
    for row in rows:
        h.update(repr(row).encode())
    return {"sha": h.hexdigest()[:16], "n_firings": len(rows),
            "num_atoms": int(at.shape[0]),
            "per_rule": {int(k): int(v) for k, v in sorted(per_rule.items())},
            "ground_proofs": _ground_proofs(out)}


def _cell_key(dataset, kind, cfg) -> str:
    if kind in ("SLD", "RTF"):
        return f"{dataset}|{kind}|d{cfg['depth']}"
    return f"{dataset}|{kind}|w{cfg['w']}d{cfg['d']}"


def compute_fingerprint(dataset, kind, cfg, *, data_root,
                        max_queries=_DEFAULT_MAX_QUERIES, chunk_size=None) -> dict:
    ds_path = Path(data_root).expanduser() / dataset
    rules_file = _RULES_FILE.get(dataset, "rules.txt")
    ds = KGDataset(str(ds_path), device="cpu", rules_file=rules_file)
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    queries = ds.get_queries("test")
    if max_queries > 0:
        queries = queries[:max_queries]
    qmask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    g = _build_grounder(kind, kb, cfg, chunk_size=chunk_size)
    with torch.no_grad():
        out = g.ground(GroundRequest(queries=queries, query_mask=qmask, output_spec=_ALL_TIERS))

    fp = _canonical_fingerprint(out)
    fp.update(dataset=dataset, kind=kind, cfg=cfg, rules=rules_file,
              n_queries=int(queries.shape[0]))
    return fp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path.home() / "repos/data-swarm/main"))
    p.add_argument("--max-queries", type=int, default=_DEFAULT_MAX_QUERIES)
    p.add_argument("--cell", default=None, help="run a single cell key, e.g. 'family|SLD|d3'")
    p.add_argument("--no-batch-invariance", action="store_true",
                   help="full-batch only; skip the chunk-size invariance sweep")
    args = p.parse_args()

    frozen = json.loads(_BASELINE.read_text())["cells"]
    batch_sizes = (None,) if args.no_batch_invariance else _BATCH_SIZES

    matrix = _MATRIX
    if args.cell:
        matrix = [(d, k, c) for (d, k, c) in _MATRIX if _cell_key(d, k, c) == args.cell]
        if not matrix:
            print(f"unknown cell {args.cell!r}"); sys.exit(2)

    drift = []
    for dataset, kind, cfg in matrix:
        key = _cell_key(dataset, kind, cfg)
        # full-batch fingerprint vs the frozen oracle
        fp = compute_fingerprint(dataset, kind, cfg, data_root=args.data_root,
                                 max_queries=args.max_queries, chunk_size=None)
        exp = frozen.get(key, {})
        status = "OK"
        if not exp:
            status = "DRIFT"; drift.append((key, "NOT in baseline"))
        for field in _GATE_FIELDS:
            if exp.get(field) != fp.get(field):
                status = "DRIFT"
                drift.append((key, f"{field}: {exp.get(field)!r} -> {fp.get(field)!r}"))
                if field == "sha":
                    drift.append((key, f"  per_rule exp={exp.get('per_rule')}"))
                    drift.append((key, f"  per_rule new={fp.get('per_rule')}"))
        # batch-size invariance: every chunk size must reproduce the full-batch SHA
        inv = "INVARIANT"
        for bs in batch_sizes:
            if bs is None:
                continue
            fp_bs = compute_fingerprint(dataset, kind, cfg, data_root=args.data_root,
                                        max_queries=args.max_queries, chunk_size=bs)
            if fp_bs["sha"] != fp["sha"]:
                status = "DRIFT"; inv = "BATCH-VARYING"
                drift.append((key, f"chunk={bs}: sha {fp['sha']} -> {fp_bs['sha']}"))
        print(f"  [{status:5s}] {key:30s} sha={fp['sha']:>16} "
              f"n_firings={fp['n_firings']:>6} ground_proofs={fp['ground_proofs']:>6} "
              f"[{inv}]")

    if drift:
        print(f"\nFAIL — {len(drift)} drift line(s):")
        for key, msg in drift:
            print(f"  DRIFT {key}: {msg}")
        sys.exit(1)
    n_bs = len(batch_sizes)
    print(f"\nPASS — all {len(matrix)} cells match the frozen oracle "
          f"AND are batch-invariant across {n_bs} chunk size(s).")


if __name__ == "__main__":
    main()
