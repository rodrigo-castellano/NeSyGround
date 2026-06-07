"""Join-vs-cartesian SET-equality oracle — gates the L3 JoinResolver against pbc.

For (dataset in {family, countries_s2, ablation_d2} + countries_s3) x (w,d,u)
covering the fingerprint configs, builds TWO BackwardGrounders identical except the
pbc layout knob (materialization="cartesian" vs "join") and asserts the EMITTED
GROUNDING SET is EXACTLY equal — canonical ``(orig_rule_idx, head, sorted_body)``
keys read off ``out.rule_groundings``. This is set-equality, NOT byte-identity:
the join pushes the width predicate into the join as a branch pruner, so the row
ORDER differs but the survivor SET must be identical.

Plus a peak-memory check: on a higher-fanout cell (countries_s3 w1d3) the join's
largest intermediate (rows reaching the fill/exists stage) must be NO WORSE than
the cartesian-flat path's — the whole point of pushing width into the join.

Usage:
    python tests/join_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from grounder.data.dataset import KGDataset
from grounder.grounder.backward import BackwardGrounder

_DATA_ROOT = Path.home() / "repos/data-swarm/main"
_RULES_FILE = {"family": "rules_old.txt"}
_MAX_QUERIES = 50

# (dataset, w, d, u) — the fingerprint configs + extra depths/widths for coverage.
_MATRIX = [
    ("family",       1, 2, 0),
    ("family",       1, 3, 0),
    ("family",       2, 2, 0),
    ("countries_s2", 1, 2, 0),
    ("countries_s2", 1, 3, 0),
    ("ablation_d2",  1, 2, 0),
    ("ablation_d2",  1, 3, 0),
    ("countries_s3", 1, 2, 0),
    ("countries_s3", 1, 3, 0),
]

# higher-fanout cell for the peak-memory regression.
_PEAK_CELL = ("countries_s3", 1, 3, 0)


def _build(kb, materialization, w, d, u):
    return BackwardGrounder(
        kb, resolution="pbc", materialization=materialization, w=w, depth=d, u=u,
        flat_intermediate=True, max_groundings_per_query=64,
        max_total_groundings=4096, max_states=256, prune_facts=True,
        bump_s_to_k=False, init_state_shape="minimal",
        collect_evidence=True, collect_rule_groundings=True)


def _grounding_set(out) -> set:
    """Canonical (orig_rule_idx, head, sorted_body) key SET from rule_groundings."""
    rg = getattr(out, "rule_groundings", None)
    if rg is None or getattr(rg, "head_pool_idx", None) is None \
            or rg.head_pool_idx.numel() == 0:
        return set()
    at = rg.atom_table.to("cpu", torch.int64)
    head_idx = rg.head_pool_idx.to("cpu", torch.int64)
    body_idx = rg.body_pool_idx.to("cpu", torch.int64)
    bvalid = rg.body_atom_valid.to("cpu", torch.bool)
    ridx = rg.rule_idx.to("cpu", torch.int64)
    fvalid = (rg.firing_valid.to("cpu", torch.bool)
              if getattr(rg, "firing_valid", None) is not None
              else torch.ones(head_idx.shape[0], dtype=torch.bool))
    keys = set()
    for f in range(head_idx.shape[0]):
        if not bool(fvalid[f]):
            continue
        r = int(ridx[f])
        head = tuple(at[head_idx[f]].tolist())
        body = tuple(sorted(tuple(at[body_idx[f, j]].tolist())
                            for j in range(body_idx.shape[1]) if bool(bvalid[f, j])))
        keys.add((r, head, body))
    return keys


def _load(dataset):
    rf = _RULES_FILE.get(dataset, "rules.txt")
    ds = KGDataset(str(_DATA_ROOT / dataset), device="cpu", rules_file=rf)
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    q = ds.get_queries("test")[:_MAX_QUERIES]
    m = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
    return kb, q, m


def _run_set(kb, q, m, materialization, w, d, u) -> set:
    g = _build(kb, materialization, w, d, u)
    with torch.no_grad():
        out = g(q, m)
    return _grounding_set(out)


def _peak_rows(kb, q, m, materialization, w, d, u) -> int:
    """Max rows reaching the fill/exists stage (the largest flat intermediate)."""
    import grounder.resolution.pbc.resolve as R
    peak = {"v": 0}
    orig = R.FlatMaterializer.fill

    def fillwrap(self, cand, cl, plan, fact_index, work):
        peak["v"] = max(peak["v"], int(cand.flat_source.size(0)))
        return orig(self, cand, cl, plan, fact_index, work)

    R.FlatMaterializer.fill = fillwrap
    try:
        g = _build(kb, materialization, w, d, u)
        with torch.no_grad():
            g(q, m)
    finally:
        R.FlatMaterializer.fill = orig
    return peak["v"]


def main() -> None:
    fails = []
    n_cells = 0
    for dataset, w, d, u in _MATRIX:
        kb, q, m = _load(dataset)
        s_cart = _run_set(kb, q, m, "cartesian", w, d, u)
        s_join = _run_set(kb, q, m, "join", w, d, u)
        key = f"{dataset}|w{w}d{d}u{u}"
        n_cells += 1
        if s_cart != s_join:
            only_c = len(s_cart - s_join)
            only_j = len(s_join - s_cart)
            fails.append((key, f"survivor set differs: |cart|={len(s_cart)} "
                          f"|join|={len(s_join)} cart-only={only_c} join-only={only_j}"))
            status = "FAIL"
        else:
            status = "OK"
        print(f"  [{status:4s}] {key:24s} |set|={len(s_cart):>6}")

    # peak-memory regression: join not worse than cartesian on a high-fanout cell.
    pds, pw, pd, pu = _PEAK_CELL
    kb, q, m = _load(pds)
    peak_cart = _peak_rows(kb, q, m, "cartesian", pw, pd, pu)
    peak_join = _peak_rows(kb, q, m, "join", pw, pd, pu)
    pkey = f"{pds}|w{pw}d{pd}u{pu}"
    if peak_join > peak_cart:
        fails.append((pkey, f"join peak rows {peak_join} > cartesian {peak_cart}"))
    print(f"\n  peak-mem [{pkey}]: cartesian-flat max-fill-rows={peak_cart}  "
          f"join max-fill-rows={peak_join}  "
          f"({peak_cart / max(peak_join, 1):.1f}x smaller)")

    if fails:
        print(f"\nFAIL — {len(fails)} mismatch(es):")
        for key, msg in fails:
            print(f"  {key}: {msg}")
        sys.exit(1)
    print(f"\nPASS — join survivor set == pbc on {n_cells} cells")


if __name__ == "__main__":
    main()
