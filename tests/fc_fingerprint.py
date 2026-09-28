"""FC closure-fingerprint oracle — byte-identity gate for every FC-touching step.

Runs ``Forward`` per cell (CPU) and checks ``(sha over the sorted closure
hashes p * E**2 + s * E + o, n_provable)`` per (dataset, method) against
``tests/baselines/fc_fingerprint.json``.

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

from grounder.forward import Forward
from grounder.kb import KB, is_variable, parse_rules, parse_triples

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
    ja = cfg.get("join_algo")
    suffix = f"|{ja}" if ja and ja != "staged" else ""
    return f"{dataset}|{method}|d{cfg['depth']}{suffix}"


def _dataset(path: Path, rules_file: str):
    """``(facts [F, 3], heads, bodies, lens, E, pad)`` with the ids the recorded fingerprints use: every name sorted,
    1-based; variables after the entities; ``pad = entities + variables + 10``; ``E = entities + 1``."""
    read = lambda f: sorted(parse_triples(path / f)) if (path / f).is_file() else []  # noqa: E731
    facts = read("facts.txt") or read("train.txt")
    rules = sorted(parse_rules(path / rules_file))
    atoms = facts + [t for s in ("train", "valid", "test") for t in read(f"{s}.txt")]
    atoms += [a for head, body in rules for a in [head] + body]
    preds = {p: i + 1 for i, p in enumerate(sorted({a[0] for a in atoms}))}
    ents = sorted({x for a in atoms for x in a[1:] if not is_variable(x)} | {x for t in facts for x in t[1:]})
    ent = {e: i + 1 for i, e in enumerate(ents)}
    var = {v: len(ent) + 1 + i for i, v in enumerate(sorted({x for a in atoms for x in a[1:] if is_variable(x)}))}
    pad = len(ent) + len(var) + 10
    idx = lambda x: var.get(x, ent.get(x, pad))  # noqa: E731
    M = max(len(b) for _, b in rules)
    heads = torch.tensor([[preds[h[0]], idx(h[1]), idx(h[2])] for h, _ in rules])
    bodies = torch.tensor([[[preds[a[0]], idx(a[1]), idx(a[2])] for a in b] + [[pad] * 3] * (M - len(b))
                           for _, b in rules])
    lens = torch.tensor([len(b) for _, b in rules])
    facts_t = torch.tensor([[preds[p], ent[a], ent[b]] for p, a, b in facts])
    return facts_t, heads, bodies, lens, len(ent) + 1, pad


def compute_fingerprint(dataset, method, cfg, *, data_root) -> dict:
    rules_file = _RULES_FILE.get(dataset, "rules.txt")
    facts, heads, bodies, lens, E, pad = _dataset(Path(data_root).expanduser() / dataset, rules_file)
    kb = KB(facts, heads, bodies, lens, E=E, pad=pad)
    atoms = Forward(kb, depth=cfg["depth"], method="spmm" if method == "spmm" else "join").closure().atoms
    # the recorded hashes: p * E**2 + s * E + o, sorted
    hashes = (atoms[:, 0] * E * E + atoms[:, 1] * E + atoms[:, 2]).sort()[0]
    fp = _canonical_fingerprint(hashes, len(atoms))
    fp.update(dataset=dataset, method=method, cfg=cfg, rules=rules_file, num_entities=E, num_predicates=kb.P)
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
