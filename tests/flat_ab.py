"""Flat-vs-dense A/B oracle — gates the flat SLD/RTF path against dense.

For {(family, rules_old, d3), (countries_s2, rules.txt, d2)} x {sld, rtf} builds
TWO BackwardGrounders identical except the layout knob (dense vs flat) and asserts
they produce byte-identical groundings: same canonical rule_groundings SHA +
n_firings and torch.equal(evidence.count). Also re-asserts the 4 dense baseline
cells still match the frozen fingerprint (regression guard).

Usage:
    python tests/flat_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from grounder.api.config import Backward, RTF, SLD
from grounder.data.dataset import KGDataset
from grounder.api.backward import BackwardGrounder
from grounder.tests.fingerprint_new import _ALL_TIERS, _canonical_fingerprint
from grounder.core import GroundRequest

_DATA_ROOT = Path.home() / "repos/data-swarm/main"
_RULES_FILE = {"family": "rules_old.txt"}
_MAX_QUERIES = 50

# (dataset, resolution, depth, frozen dense baseline: sha, n_firings, ground_proofs)
_MATRIX = [
    ("family",       "sld", 3, "865bd52a40954f82", 8671, 195),
    ("family",       "rtf", 3, "EMPTY",                0, 480),
    ("countries_s2", "sld", 2, "d89d732d98097181",   48,  67),
    ("countries_s2", "rtf", 2, "EMPTY",                0, 145),
]


def _build(kb, resolution, depth, layout):
    res = SLD(depth=depth) if resolution == "sld" else RTF(depth=depth)
    config = Backward(res, filter="none", max_groundings_per_query=4096,
                      max_children=64, max_goals=256, prune_facts=True)
    return BackwardGrounder(kb, config, layout=layout, compile="off")


def _run(dataset, resolution, depth, layout):
    ds_path = _DATA_ROOT / dataset
    rules_file = _RULES_FILE.get(dataset, "rules.txt")
    ds = KGDataset(str(ds_path), device="cpu", rules_file=rules_file)
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    queries = ds.get_queries("test")[:_MAX_QUERIES]
    qmask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)
    g = _build(kb, resolution, depth, layout)
    with torch.no_grad():
        out = g.ground(GroundRequest(queries=queries, query_mask=qmask, output_spec=_ALL_TIERS))
    fp = _canonical_fingerprint(out)
    ev = getattr(out, "completed_tree_firings", None)
    count = ev.count.clone() if (ev is not None and ev.count is not None) else None
    return fp, count


def main() -> None:
    fails = []
    for dataset, resolution, depth, exp_sha, exp_n, exp_gp in _MATRIX:
        key = f"{dataset}|{resolution.upper()}|d{depth}"
        fp_d, cnt_d = _run(dataset, resolution, depth, "dense")
        fp_f, cnt_f = _run(dataset, resolution, depth, "flat")

        # regression guard: dense must still match the frozen baseline
        if fp_d["sha"] != exp_sha or fp_d["n_firings"] != exp_n \
                or fp_d["ground_proofs"] != exp_gp:
            fails.append((key, f"dense drifted from frozen: sha={fp_d['sha']} "
                          f"n={fp_d['n_firings']} gp={fp_d['ground_proofs']}"))

        # flat == dense (canonical SHA + firings)
        if fp_f["sha"] != fp_d["sha"] or fp_f["n_firings"] != fp_d["n_firings"]:
            fails.append((key, f"flat!=dense rule_groundings: "
                          f"dense sha={fp_d['sha']} n={fp_d['n_firings']} | "
                          f"flat sha={fp_f['sha']} n={fp_f['n_firings']}"))

        # flat == dense evidence.count (caps unhit on these 4 cells)
        ev_ok = (cnt_d is not None and cnt_f is not None
                 and cnt_d.shape == cnt_f.shape and torch.equal(cnt_d, cnt_f))
        if not ev_ok:
            d_gp = int(cnt_d.sum().item()) if cnt_d is not None else None
            f_gp = int(cnt_f.sum().item()) if cnt_f is not None else None
            fails.append((key, f"evidence.count flat!=dense: "
                          f"dense gp={d_gp} flat gp={f_gp}"))

        status = "OK" if all(k != key for k, _ in fails) else "FAIL"
        print(f"  [{status:4s}] {key:22s} dense(sha={fp_d['sha']} n={fp_d['n_firings']} "
              f"gp={fp_d['ground_proofs']})  flat(sha={fp_f['sha']} n={fp_f['n_firings']} "
              f"gp={fp_f['ground_proofs']})")

    if fails:
        print(f"\nFAIL — {len(fails)} mismatch(es):")
        for key, msg in fails:
            print(f"  {key}: {msg}")
        sys.exit(1)
    print("\nPASS — sld/rtf flat==dense on 4 cells")


if __name__ == "__main__":
    main()
