"""NEW grounding-fingerprint harness — gates the rebuilt engine against the frozen
14-cell baseline (``tests/baselines/ground_fingerprint.json``).

Builds the NEW ``BackwardGrounder`` per cell (CPU/eager, 50 queries), computes the
SAME canonical SHA the OLD harness uses over ``out.rule_groundings`` +
``out.evidence.count``, and diffs against the frozen JSON.

Usage:
    python tests/fingerprint_new.py            # check all 14
    python tests/fingerprint_new.py --cell family|SLD|d3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from grounder.data.dataset import KGDataset
from grounder.grounder.backward import BackwardGrounder

_HERE = Path(__file__).resolve().parent
_BASELINE = (_HERE / "baselines" / "ground_fingerprint.json")
_RULES_FILE = {"family": "rules_old.txt"}
_GATE_FIELDS = ("sha", "n_firings", "ground_proofs")
_DEFAULT_MAX_QUERIES = 50

_MATRIX = [
    ("family", "enum-flat",  {"w": 1, "d": 2}),
    ("family", "enum-flat",  {"w": 1, "d": 3}),
    ("family", "enum-dense", {"w": 1, "d": 2}),
    ("family", "enum-dense", {"w": 1, "d": 3}),
    ("family", "SLD", {"depth": 3}),
    ("family", "RTF", {"depth": 3}),
    ("countries_s2", "enum-flat",  {"w": 1, "d": 2}),
    ("countries_s2", "enum-dense", {"w": 1, "d": 2}),
    ("countries_s2", "SLD", {"depth": 2}),
    ("countries_s2", "RTF", {"depth": 2}),
    ("ablation_d2", "enum-flat",  {"w": 1, "d": 2}),
    ("ablation_d2", "enum-dense", {"w": 1, "d": 2}),
    ("wn18rr", "enum-flat",  {"w": 1, "d": 2}),
    ("wn18rr", "enum-dense", {"w": 1, "d": 2}),
]


def _build_grounder(kind: str, kb, cfg: dict) -> BackwardGrounder:
    """Mirror ``tests/_runners.build_torch_grounder`` config exactly."""
    if kind in ("SLD", "RTF"):
        return BackwardGrounder(
            kb, resolution="sld" if kind == "SLD" else "rtf", filter="none",
            depth=cfg["depth"], max_total_groundings=4096,
            max_derived_per_state=64, max_states=256, prune_facts=True,
            collect_evidence=True, collect_rule_groundings=True)
    if kind in ("enum-flat", "enum-dense"):
        flat = (kind == "enum-flat")
        return BackwardGrounder(
            kb, resolution="pbc", w=cfg["w"], depth=cfg["d"], u=0,
            flat_intermediate=flat, max_groundings_per_query=64,
            max_total_groundings=4096, max_states=256, prune_facts=True,
            bump_s_to_k=False, init_state_shape="minimal",
            collect_evidence=True, collect_rule_groundings=True)
    raise ValueError(f"unsupported kind: {kind!r}")


def _ground_proofs(out) -> int:
    ev = getattr(out, "evidence", None)
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
    ridx = rg.rule_idx.to("cpu", torch.int64)
    fvalid = (rg.firing_valid.to("cpu", torch.bool)
              if getattr(rg, "firing_valid", None) is not None
              else torch.ones(head_idx.shape[0], dtype=torch.bool))

    rows = []
    per_rule: dict = {}
    F = head_idx.shape[0]
    for f in range(F):
        if not bool(fvalid[f]):
            continue
        r = int(ridx[f])
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


def compute_fingerprint(dataset, kind, cfg, *, data_root, max_queries=_DEFAULT_MAX_QUERIES) -> dict:
    ds_path = Path(data_root).expanduser() / dataset
    rules_file = _RULES_FILE.get(dataset, "rules.txt")
    ds = KGDataset(str(ds_path), device="cpu", rules_file=rules_file)
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    queries = ds.get_queries("test")
    if max_queries > 0:
        queries = queries[:max_queries]
    qmask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    g = _build_grounder(kind, kb, cfg)
    with torch.no_grad():
        out = g(queries, qmask)

    fp = _canonical_fingerprint(out)
    fp.update(dataset=dataset, kind=kind, cfg=cfg, rules=rules_file,
              n_queries=int(queries.shape[0]))
    return fp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path.home() / "repos/data-swarm/main"))
    p.add_argument("--max-queries", type=int, default=_DEFAULT_MAX_QUERIES)
    p.add_argument("--cell", default=None, help="run a single cell key, e.g. 'family|SLD|d3'")
    args = p.parse_args()

    frozen = json.loads(_BASELINE.read_text())["cells"]

    matrix = _MATRIX
    if args.cell:
        matrix = [(d, k, c) for (d, k, c) in _MATRIX if _cell_key(d, k, c) == args.cell]
        if not matrix:
            print(f"unknown cell {args.cell!r}"); sys.exit(2)

    drift = []
    for dataset, kind, cfg in matrix:
        key = _cell_key(dataset, kind, cfg)
        fp = compute_fingerprint(dataset, kind, cfg, data_root=args.data_root,
                                 max_queries=args.max_queries)
        exp = frozen.get(key, {})
        status = "OK"
        for field in _GATE_FIELDS:
            if exp.get(field) != fp.get(field):
                status = "DRIFT"
                drift.append((key, f"{field}: {exp.get(field)!r} -> {fp.get(field)!r}"))
                if field == "sha":
                    drift.append((key, f"  per_rule exp={exp.get('per_rule')}"))
                    drift.append((key, f"  per_rule new={fp.get('per_rule')}"))
        print(f"  [{status:5s}] {key:30s} sha={fp['sha']} "
              f"n_firings={fp['n_firings']:>6} ground_proofs={fp['ground_proofs']:>6} "
              f"num_atoms={fp['num_atoms']:>6}")

    if drift:
        print(f"\nFAIL — {len(drift)} drift line(s):")
        for key, msg in drift:
            print(f"  DRIFT {key}: {msg}")
        sys.exit(1)
    print(f"\nPASS — all {len(matrix)} cells match the frozen fingerprint.")


if __name__ == "__main__":
    main()
