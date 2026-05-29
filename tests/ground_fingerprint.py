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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", default=str(Path.home() / "repos/data-swarm/main"))
    p.add_argument("--kind", default="enum-flat",
                   choices=["SLD", "enum-flat", "enum-dense"])
    p.add_argument("--w", type=int, default=1)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--depth", type=int, default=4, help="SLD depth")
    p.add_argument("--max-queries", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", default=None,
                   help="compile_mode; default = eager (None). Use the "
                        "DEFAULT_COMPILE_MODES value to test the compiled path.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    ds_path = Path(args.data_root).expanduser() / args.dataset
    rules_file = _RULES_FILE.get(args.dataset, "rules.txt")
    ds = KGDataset(str(ds_path), device=args.device, rules_file=rules_file)
    kb = ds.make_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    queries = ds.get_queries("test")
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    qmask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)

    cfg = ({"depth": args.depth} if args.kind == "SLD"
           else {"w": args.w, "d": args.d, "flat": args.kind == "enum-flat"})
    compile_mode = args.compile if args.compile not in (None, "none") else None
    g = build_torch_grounder(args.kind, kb, cfg, compile_mode=compile_mode)
    with torch.no_grad():
        out = g(queries, qmask)

    fp = _canonical_fingerprint(out, kb)
    fp.update(dataset=args.dataset, kind=args.kind,
              cfg=cfg, n_queries=int(queries.shape[0]),
              device=args.device, compile=compile_mode, rules=rules_file)
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
