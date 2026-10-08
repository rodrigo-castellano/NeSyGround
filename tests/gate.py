"""The grounder's gate (GPU, < 90 s): what is grounded, exact, and how fast, against a ratchet.

    python tests/gate.py             # exactness + speed against tests/baselines/gate.json
    python tests/gate.py --update    # record this run's times as the reference (after an intended speed change)

Exact: every ``counts.py`` cell (PBC on Family / WN18RR / Countries S3, fp_batch and keras, depths 1-3: each step's
goals, the kept firings and atoms, firings per rule, an output hash) and every ``fc_fingerprint.py`` cell (forward
chaining's closures, spmm and join) equal their references. Speed: each cell's best of two runs; the PBC cells' and
the forward-chaining cells' totals no slower than the reference's by more than ``SLACK``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import counts  # noqa: E402
import fc_fingerprint as fc  # noqa: E402

REFERENCE = Path(__file__).with_name("baselines") / "gate.json"
SLACK = 1.25                     # times a total may exceed the reference's (shared GPU, noise)


def _pbc() -> tuple:
    """``(drift, seconds)`` over counts.py's cells."""
    ref = json.loads(counts.REFERENCE.read_text())
    drift, total, kbs = [], 0.0, {}
    for dataset, split, n, types in counts.CELLS:
        kb = kbs.setdefault(dataset, counts.KB(dataset, "cuda"))
        for grounder_type in types:
            name = f"{dataset}:{split}{'' if n is None else f'[:{n}]'}:{grounder_type}"
            runs = [counts.run_cell(kb, dataset, split, n, grounder_type) for _ in range(2)]
            got, want = runs[-1], ref[name]
            bad = [k for k in got if k not in counts.INFO and got[k] != want.get(k)]
            if bad:
                drift.append(f"{name}: {', '.join(bad)}")
            total += min(r["seconds"] for r in runs)
    return drift, total


def _forward() -> tuple:
    """``(drift, seconds)`` over fc_fingerprint.py's cells."""
    frozen = json.loads(fc._BASELINE.read_text())["cells"]
    root = Path.home() / "repos/data-swarm/main"
    drift, total = [], 0.0
    for dataset, method, cfg in fc._MATRIX:
        key = fc._cell_key(dataset, method, cfg)
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fp = fc.compute_fingerprint(dataset, method, cfg, data_root=root)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        if any(frozen[key].get(f) != fp.get(f) for f in fc._GATE_FIELDS):
            drift.append(key)
        total += min(times)
    return drift, total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update", action="store_true", help="record this run's times as the reference")
    a = p.parse_args()
    if not torch.cuda.is_available():
        print("gate: needs a GPU (PBC's kernels are Triton)")
        return 2
    wall = time.perf_counter()
    pbc_drift, pbc_s = _pbc()
    fc_drift, fc_s = _forward()
    got = {"pbc_seconds": round(pbc_s, 2), "forward_seconds": round(fc_s, 2)}
    ref = json.loads(REFERENCE.read_text()) if REFERENCE.is_file() else None
    slow = [] if ref is None else [k for k in got if got[k] > ref[k] * SLACK]
    for k in got:
        print(f"{k:16s} {got[k]:7.2f}" + ("" if ref is None else f"   reference {ref[k]:7.2f}"
                                          + ("  SLOWER" if k in slow else "")))
    for line in pbc_drift + fc_drift:
        print("DRIFT", line)
    print(f"wall {time.perf_counter() - wall:.1f} s")
    if a.update:
        REFERENCE.write_text(json.dumps(got, indent=1) + "\n")
        print(f"recorded {REFERENCE}")
        return 1 if pbc_drift or fc_drift else 0
    ok = not (pbc_drift or fc_drift or slow or ref is None)
    print("PASS" if ok else "FAIL" + ("" if ref is not None else " (no reference: --update)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
