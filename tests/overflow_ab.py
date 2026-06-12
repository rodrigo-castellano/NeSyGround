"""Atom-cap overflow oracle — overflowing children are DROPPED, never truncated.

When a goal's atoms exceed the capacity ``G`` (= ``max_atoms``), the grounder must
INVALIDATE the child rather than silently truncate its remaining tail — a truncated
child would "prove" a head with a missing body atom (a soundness error). This gates the
drop-not-truncate guard on ALL four assembly paths (pbc flat, pbc dense, sld, rtf).

Method: on family|d3 (rules_old, M=2 → natural goal-length bound L=5), a deliberately
SMALL ``max_atoms`` forces depth≥2 goals to overflow. The emitted firing SET with the
small cap MUST be a SUBSET of the set with a large (no-overflow) cap: a truncated child
has an incomplete body, so its canonical ``(rule, head, sorted_body)`` key matches NO
complete-body firing in the large set. Thus  small ⊆ large  ⟺  drop-not-truncate.
We also require small ⊊ large on at least one path, so the test actually exercises overflow.

Usage:
    python tests/overflow_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from grounder.api.config import Backward, PBC, SLD, RTF
from grounder.data.dataset import KGDataset
from grounder.api.backward import BackwardGrounder
from grounder.core import GroundRequest, OutputSpec, Tier

_ALL = OutputSpec(frozenset({Tier.PROOF_STATE, Tier.FIRINGS, Tier.TREES}))
_DATA = Path.home() / "repos/data-swarm/main"
_MAXQ = 30
_SMALL, _LARGE = 2, 12     # family M=2: small cap forces depth≥2 overflow; large fits everything


def _firing_set(out) -> set:
    """Canonical (rule_idx, head, sorted_body) key SET from rule_groundings."""
    rg = getattr(out, "rule_groundings", None)
    if rg is None or getattr(rg, "head_pool_idx", None) is None or rg.head_pool_idx.numel() == 0:
        return set()
    at = rg.atom_table.to("cpu", torch.int64)
    head_idx = rg.head_pool_idx.to("cpu", torch.int64)
    body_idx = rg.body_pool_idx.to("cpu", torch.int64)
    bvalid = rg.body_atom_valid.to("cpu", torch.bool)
    rule_idx = rg.rule_idx.to("cpu", torch.int64)
    fvalid = (rg.firing_valid.to("cpu", torch.bool)
              if getattr(rg, "firing_valid", None) is not None
              else torch.ones(head_idx.shape[0], dtype=torch.bool))
    keys = set()
    for f in range(head_idx.shape[0]):
        if not bool(fvalid[f]):
            continue
        head = tuple(at[head_idx[f]].tolist())
        body = tuple(sorted(tuple(at[body_idx[f, j]].tolist())
                            for j in range(body_idx.shape[1]) if bool(bvalid[f, j])))
        keys.add((int(rule_idx[f]), head, body))
    return keys


def _run(kb, q, m, *, res, layout, max_atoms) -> set:
    cfg = Backward(res, filter=("fp_batch" if isinstance(res, PBC) else "none"),
                   max_groundings_per_query=4096, max_goals=256, max_atoms=max_atoms,
                   prune_facts=True, bump_s_to_k=False)
    g = BackwardGrounder(kb, cfg, layout=layout)
    with torch.no_grad():
        out = g.ground(GroundRequest(queries=q, query_mask=m, output_spec=_ALL))
    return _firing_set(out)


def main() -> None:
    ds = KGDataset(str(_DATA / "family"), device="cpu", rules_file="rules_old.txt")
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    q = ds.get_queries("test")[:_MAXQ]
    m = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)

    cases = [
        ("pbc-flat",  lambda: PBC(depth=3, width=1, u=0, max_groundings_per_rule=64), "flat"),
        ("pbc-dense", lambda: PBC(depth=3, width=1, u=0, max_groundings_per_rule=64), "dense"),
        ("sld",       lambda: SLD(depth=3), "dense"),
        ("rtf",       lambda: RTF(depth=3), "dense"),
    ]
    fails, exercised = [], False
    for name, mk, layout in cases:
        small = _run(kb, q, m, res=mk(), layout=layout, max_atoms=_SMALL)
        large = _run(kb, q, m, res=mk(), layout=layout, max_atoms=_LARGE)
        spurious = small - large           # truncated (incomplete-body) firings, if any
        exercised = exercised or (len(small) < len(large))
        status = "FAIL" if spurious else "OK"
        if spurious:
            fails.append((name, len(spurious)))
        print(f"  [{status:4s}] {name:9s} small(maxA={_SMALL})={len(small):>5}  "
              f"large={len(large):>5}  spurious(small-large)={len(spurious)}")

    if fails:
        print(f"\nFAIL — truncated/spurious firings (overflow children were CUT, not dropped): {fails}")
        sys.exit(1)
    if not exercised:
        print("\nFAIL — small cap never reduced the firing set: overflow was not exercised "
              "(lower _SMALL or deepen the cell)")
        sys.exit(1)
    print(f"\nPASS — overflow children DROPPED not truncated on {len(cases)} paths "
          f"(small ⊆ large, and small ⊊ large somewhere)")


if __name__ == "__main__":
    main()
