"""Deterministic grounding-behavior fingerprint — the fast (<1 s) unit
for verifying the grounder produces IDENTICAL output across configs,
code versions (e.g. a refactor), or devices.

The fingerprint is a canonical SHA over the *set of ground rule
applications* the grounder emits — `(rule_idx, head_atom, sorted_body_
atoms)` mapped through `atom_table` to real `(pred, head, tail)` ids —
so two runs that produce the same logical groundings hash identically
regardless of internal tensor layout / ordering / padding. Plus scalar
counts for quick human diffing.

Runs EAGER on CPU by default (counts/groundings are device- and
compile-independent), so a fleet of agents can fingerprint the whole
config matrix in parallel with ZERO GPU contention. Use `--device cuda
--compile <mode>` to fingerprint the compiled path and confirm it
matches the eager-CPU canonical.

Usage:
    python tests/ground_fingerprint.py --dataset family --kind enum-flat --w 1 --d 2 --max-queries 50
    python tests/ground_fingerprint.py --dataset family --kind SLD --depth 4 --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grounder.data.loader import KGDataset                      # noqa: E402
from _runners import build_torch_grounder, DEFAULT_COMPILE_MODES  # noqa: E402

# Family ships a paper ruleset; others use rules.txt (mirror test_groundings).
_RULES_FILE = {"family": "rules_old.txt"}


def _canonical_fingerprint(out, kb) -> dict:
    """Canonical, layout-independent fingerprint of the grounding output."""
    rg = getattr(out, "rule_groundings", None)
    if rg is None or getattr(rg, "head_pool_idx", None) is None \
            or rg.head_pool_idx.numel() == 0:
        return {"sha": "EMPTY", "n_firings": 0, "num_atoms": 0,
                "per_rule": {}, "ground_proofs": _ground_proofs(out)}

    at = rg.atom_table.to("cpu", torch.int64)          # [num_atoms, 3]
    head_idx = rg.head_pool_idx.to("cpu", torch.int64)  # [F]
    body_idx = rg.body_pool_idx.to("cpu", torch.int64)  # [F, M_max]
    bvalid = rg.body_atom_valid.to("cpu", torch.bool)   # [F, M_max]
    ridx = rg.rule_idx.to("cpu", torch.int64)           # [F]
    fvalid = (rg.firing_valid.to("cpu", torch.bool)
              if getattr(rg, "firing_valid", None) is not None
              else torch.ones(head_idx.shape[0], dtype=torch.bool))

    rows = []
    per_rule: dict[int, int] = {}
    F = head_idx.shape[0]
    for f in range(F):
        if not bool(fvalid[f]):
            continue
        r = int(ridx[f])
        head = tuple(at[head_idx[f]].tolist())
        body = sorted(
            tuple(at[body_idx[f, j]].tolist())
            for j in range(body_idx.shape[1]) if bool(bvalid[f, j])
        )
        rows.append((r, head, tuple(body)))
        per_rule[r] = per_rule.get(r, 0) + 1

    rows.sort()
    h = hashlib.sha256()
    for row in rows:
        h.update(repr(row).encode())
    return {
        "sha": h.hexdigest()[:16],
        "n_firings": len(rows),
        "num_atoms": int(at.shape[0]),
        "per_rule": {int(k): int(v) for k, v in sorted(per_rule.items())},
        "ground_proofs": _ground_proofs(out),
    }


def _ground_proofs(out) -> int:
    ev = getattr(out, "evidence", None)
    if ev is not None and getattr(ev, "count", None) is not None:
        return int(ev.count.sum().item())
    return 0


# ══════════════════════════════════════════════════════════════════════
# Result-neutrality MATRIX — the refactor's primary per-step gate.
#
# Each cell is fingerprinted CPU-eager (deterministic, ~1 s, GPU-free).
# ``enum-flat`` is the EXACT paper/consumer path; ``enum-dense`` the
# alternate enum path; ``SLD``/``RTF`` the other two BC resolutions the
# refactor edits. FC / closure stay covered by the keras count oracle
# (``tests/baselines/comparison.json`` via ``test_groundings.py``), which
# THE GATE already runs — no need to duplicate that fragile path here.
#
# ``freeze`` writes the canonical SHA per cell to
# ``tests/baselines/ground_fingerprint.json``; ``check`` recomputes and
# diffs. A refactor step is result-neutral iff ``check`` is clean.
# ══════════════════════════════════════════════════════════════════════

_MATRIX: list[tuple[str, str, dict]] = [
    # family — 47-rule paper set (rules_old.txt); eager rule tail (>10 rules)
    ("family", "enum-flat",  {"w": 1, "d": 2}),
    ("family", "enum-flat",  {"w": 1, "d": 3}),
    ("family", "enum-dense", {"w": 1, "d": 2}),
    ("family", "enum-dense", {"w": 1, "d": 3}),
    ("family", "SLD", {"depth": 3}),
    ("family", "RTF", {"depth": 3}),
    # countries_s2 — small, all four BC paths
    ("countries_s2", "enum-flat",  {"w": 1, "d": 2}),
    ("countries_s2", "enum-dense", {"w": 1, "d": 2}),
    ("countries_s2", "SLD", {"depth": 2}),
    ("countries_s2", "RTF", {"depth": 2}),
    # ablation_d2
    ("ablation_d2", "enum-flat",  {"w": 1, "d": 2}),
    ("ablation_d2", "enum-dense", {"w": 1, "d": 2}),
    # wn18rr — the 1-body symmetry quirk dataset
    ("wn18rr", "enum-flat",  {"w": 1, "d": 2}),
    ("wn18rr", "enum-dense", {"w": 1, "d": 2}),
]

# Fields compared by ``check`` (the canonical SHA is primary; the two
# scalar counts are sanity belts). ``num_atoms`` / ``per_rule`` are kept
# in the frozen JSON for human diffing but are NOT gate fields.
_GATE_FIELDS = ("sha", "n_firings", "ground_proofs")
_BASELINE = _HERE / "baselines" / "ground_fingerprint.json"
_DEFAULT_MAX_QUERIES = 50


def _cell_key(dataset: str, kind: str, cfg: dict) -> str:
    if kind in ("SLD", "RTF"):
        return f"{dataset}|{kind}|d{cfg['depth']}"
    return f"{dataset}|{kind}|w{cfg['w']}d{cfg['d']}"


def compute_fingerprint(dataset: str, kind: str, cfg: dict, *,
                        data_root: str, device: str = "cpu",
                        max_queries: int = _DEFAULT_MAX_QUERIES,
                        compile_mode=None) -> dict:
    """Build the grounder for one cell and return its canonical fingerprint."""
    ds_path = Path(data_root).expanduser() / dataset
    rules_file = _RULES_FILE.get(dataset, "rules.txt")
    ds = KGDataset(str(ds_path), device=device, rules_file=rules_file)
    kb = ds.make_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    queries = ds.get_queries("test")
    if max_queries > 0:
        queries = queries[:max_queries]
    qmask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    build_cfg = (dict(cfg) if kind in ("SLD", "RTF")
                 else {"w": cfg["w"], "d": cfg["d"], "flat": kind == "enum-flat"})
    g = build_torch_grounder(kind, kb, build_cfg, compile_mode=compile_mode)
    with torch.no_grad():
        out = g(queries, qmask)

    fp = _canonical_fingerprint(out, kb)
    fp.update(dataset=dataset, kind=kind, cfg=cfg, rules=rules_file,
              n_queries=int(queries.shape[0]))
    return fp


def _run_matrix(data_root: str, device: str, max_queries: int) -> dict:
    results: dict[str, dict] = {}
    for dataset, kind, cfg in _MATRIX:
        key = _cell_key(dataset, kind, cfg)
        fp = compute_fingerprint(dataset, kind, cfg, data_root=data_root,
                                 device=device, max_queries=max_queries)
        results[key] = fp
        print(f"  {key:30s}  sha={fp['sha']}  n_firings={fp['n_firings']:>6}  "
              f"ground_proofs={fp['ground_proofs']:>6}  num_atoms={fp['num_atoms']:>6}")
    return results


def _freeze(data_root: str, device: str, max_queries: int) -> None:
    print(f"FREEZE — {len(_MATRIX)} cells, device={device}, "
          f"max_queries={max_queries}")
    results = _run_matrix(data_root, device, max_queries)
    payload = {
        "_meta": {
            "n_cells": len(results),
            "gate_fields": list(_GATE_FIELDS),
            "max_queries": max_queries,
            "device": device,
        },
        "cells": results,
    }
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nFROZE {len(results)} cells -> {_BASELINE}")


def _check(data_root: str, device: str, max_queries: int) -> None:
    if not _BASELINE.exists():
        print(f"ERROR: no frozen baseline at {_BASELINE}; run --freeze first.")
        sys.exit(2)
    frozen_payload = json.loads(_BASELINE.read_text())
    frozen = frozen_payload["cells"]
    mq = frozen_payload.get("_meta", {}).get("max_queries", max_queries)
    print(f"CHECK — {len(_MATRIX)} cells vs frozen baseline "
          f"({len(frozen)} cells), device={device}, max_queries={mq}")
    results = _run_matrix(data_root, device, mq)

    drift: list[tuple[str, str]] = []
    for key, fp in results.items():
        if key not in frozen:
            drift.append((key, "NEW cell not in baseline"))
            continue
        for field in _GATE_FIELDS:
            old, new = frozen[key].get(field), fp.get(field)
            if old != new:
                drift.append((key, f"{field}: {old!r} -> {new!r}"))
                if field == "sha":
                    drift.append((key, f"  per_rule old={frozen[key].get('per_rule')}"))
                    drift.append((key, f"  per_rule new={fp.get('per_rule')}"))
    for key in frozen:
        if key not in results:
            drift.append((key, "MISSING cell (in baseline, not computed)"))

    if drift:
        print(f"\nFAIL — {len(drift)} drift line(s):")
        for key, msg in drift:
            print(f"  DRIFT {key}: {msg}")
        sys.exit(1)
    print(f"\nPASS — all {len(results)} cells match the frozen fingerprint.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", action="store_true",
                   help="run the full MATRIX and write the frozen baseline JSON")
    p.add_argument("--check", action="store_true",
                   help="run the full MATRIX and diff against the frozen baseline "
                        "(exit 1 on any drift) — the refactor's per-step gate")
    p.add_argument("--dataset", help="single-cell mode: dataset name")
    p.add_argument("--data-root", default=str(Path.home() / "repos/data-swarm/main"))
    p.add_argument("--kind", default="enum-flat",
                   choices=["SLD", "RTF", "enum-flat", "enum-dense"])
    p.add_argument("--w", type=int, default=1)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--depth", type=int, default=4, help="SLD/RTF depth")
    p.add_argument("--max-queries", type=int, default=_DEFAULT_MAX_QUERIES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", default=None,
                   help="compile_mode; default = eager (None). Use the "
                        "DEFAULT_COMPILE_MODES value to test the compiled path.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.freeze:
        _freeze(args.data_root, args.device, args.max_queries)
        return
    if args.check:
        _check(args.data_root, args.device, args.max_queries)
        return

    if not args.dataset:
        p.error("single-cell mode needs --dataset (or use --freeze / --check)")

    cfg = ({"depth": args.depth} if args.kind in ("SLD", "RTF")
           else {"w": args.w, "d": args.d})
    compile_mode = args.compile if args.compile not in (None, "none") else None
    fp = compute_fingerprint(args.dataset, args.kind, cfg,
                             data_root=args.data_root, device=args.device,
                             max_queries=args.max_queries, compile_mode=compile_mode)
    fp.update(device=args.device, compile=compile_mode)
    if args.json:
        print(json.dumps(fp, sort_keys=True))
    else:
        print(f"{args.dataset}|{args.kind}|{cfg}  device={args.device} "
              f"compile={compile_mode}")
        print(f"  sha={fp['sha']}  n_firings={fp['n_firings']}  "
              f"num_atoms={fp['num_atoms']}  ground_proofs={fp['ground_proofs']}")
        print(f"  per_rule={fp['per_rule']}")


if __name__ == "__main__":
    main()
