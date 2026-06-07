"""Step-5 oracle — KB.with_program + index cache (STRONG A/B; do not weaken).

Proves the three Step-5 invariants:

  (A) SOUNDNESS EQUIVALENCE — with_program(facts=X) produces a KB whose facts AND
      whose pbc grounding output are BYTE-IDENTICAL to the pre-Step-5 with_closure
      semantics (a fresh KB rebuilt from the merged facts, NO cache). This proves
      the REUSED rule index answers exactly like a freshly rebuilt one.
  (B) FACTS-ONLY CACHE HIT — a facts-only rewrite REUSES the rule-derived index
      (rule_index identity preserved; binding tables shared) and rebuilds only the
      fact index. The magic-set skeleton+seeds shape hits the cache per call.
  (C) RULE-CHANGING RE-DERIVES — a with_program(heads,bodies,lens) rewrite builds
      a NEW rule index reflecting the new program; the memo keys on the rule
      signature so a repeat rule-change hits it, and a facts-only/rule-changing
      mix never cross-contaminates.

Usage:
    python _build/tests/with_program_ab.py
    python _build/tests/with_program_ab.py --data-root <path>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from grounder._build.data import kb as kb_mod
from grounder._build.data.dataset import KGDataset
from grounder._build.data.kb import KB
from grounder._build.grounder.backward import BackwardGrounder

_RULES_FILE = {"family": "rules_old.txt"}


def _fp(out) -> str:
    """Canonical SHA over the pbc firings (same recipe as fingerprint_new)."""
    rg = getattr(out, "rule_groundings", None)
    if rg is None or getattr(rg, "head_pool_idx", None) is None \
            or rg.head_pool_idx.numel() == 0:
        return "EMPTY"
    at = rg.atom_table.to("cpu", torch.int64)
    head_idx = rg.head_pool_idx.to("cpu", torch.int64)
    body_idx = rg.body_pool_idx.to("cpu", torch.int64)
    bvalid = rg.body_atom_valid.to("cpu", torch.bool)
    ridx = rg.rule_idx.to("cpu", torch.int64)
    fvalid = (rg.firing_valid.to("cpu", torch.bool)
              if getattr(rg, "firing_valid", None) is not None
              else torch.ones(head_idx.shape[0], dtype=torch.bool))
    rows = []
    for f in range(head_idx.shape[0]):
        if not bool(fvalid[f]):
            continue
        head = tuple(at[head_idx[f]].tolist())
        body = sorted(tuple(at[body_idx[f, j]].tolist())
                      for j in range(body_idx.shape[1]) if bool(bvalid[f, j]))
        rows.append((int(ridx[f]), head, tuple(body)))
    rows.sort()
    h = hashlib.sha256()
    for row in rows:
        h.update(repr(row).encode())
    return h.hexdigest()[:16]


def _ground(kb, queries) -> str:
    g = BackwardGrounder(kb, resolution="pbc", w=1, depth=2, u=0,
                         flat_intermediate=True, max_groundings_per_query=64,
                         max_total_groundings=4096, max_states=256, prune_facts=True,
                         bump_s_to_k=False, init_state_shape="minimal",
                         collect_evidence=True, collect_rule_groundings=True)
    qmask = torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device)
    with torch.no_grad():
        return _fp(g(queries, qmask))


def _facts_set(kb) -> set:
    return {tuple(r) for r in kb.facts_idx.tolist()}


def _rebuild_like_old_with_closure(kb: KB, extra: torch.Tensor) -> KB:
    """The VERBATIM pre-Step-5 with_closure: merge facts, REBUILD the whole KB
    (no cache). The reference the reused-index path must reproduce."""
    extra = extra.to(device=kb.device_, dtype=torch.long)
    merged = torch.unique(torch.cat([kb.fact_index.facts_idx, extra], 0), dim=0)
    return KB(merged, kb.rules_heads_idx, kb.rules_bodies_idx, kb.rule_lens,
              constant_no=kb.constant_no, predicate_no=kb.predicate_no,
              padding_idx=kb.padding_idx, device=kb.device_, **kb._index_cfg)


# ── (A) soundness equivalence on real datasets ──────────────────────────────
def check_soundness_equiv(data_root) -> None:
    for dataset in ("family", "countries_s2"):
        ds = KGDataset(str(Path(data_root).expanduser() / dataset), device="cpu",
                       rules_file=_RULES_FILE.get(dataset, "rules.txt"))
        kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
        q = ds.get_queries("test")[:50]
        # A closure-like seed set: a handful of NEW fact triples not already in kb.
        seeds = q[:10].to(torch.long)

        new = kb.with_program(facts=seeds)              # Step-5 (reuse index)
        ref = _rebuild_like_old_with_closure(kb, seeds)  # old (fresh rebuild)
        closure = kb.with_closure(seeds)                 # delegates to with_program

        assert _facts_set(new) == _facts_set(ref), f"{dataset}: facts differ vs old"
        assert _facts_set(closure) == _facts_set(ref), f"{dataset}: with_closure facts differ"
        assert new.num_facts >= kb.num_facts, f"{dataset}: facts shrank"

        # The reused index must answer byte-identically to a fresh rebuild.
        fp_new, fp_ref = _ground(new, q), _ground(ref, q)
        assert fp_new == fp_ref, (
            f"{dataset}: reused-index grounding {fp_new} != fresh-rebuild {fp_ref}")
        # And the rewrite must NOT corrupt the base KB's own grounding.
        assert _ground(kb, q) == _ground(_rebuild_like_old_with_closure(
            kb, torch.empty(0, 3, dtype=torch.long)), q), f"{dataset}: base drift"
        print(f"  [OK  ] (A) soundness-equiv {dataset:14s} "
              f"facts {kb.num_facts}->{new.num_facts} fp={fp_new}")


# ── (B) facts-only cache hit ────────────────────────────────────────────────
def check_facts_only_cache_hit(data_root) -> None:
    ds = KGDataset(str(Path(data_root).expanduser() / "family"), device="cpu",
                   rules_file="rules_old.txt")
    kb = ds.build_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    kb.binding_tables(kb.M, kb.padding_idx)   # force the lazy KB binding tables

    seeds = ds.get_queries("test")[:5].to(torch.long)
    new = kb.with_program(facts=seeds)
    # Rule index REUSED (no per-rule re-tensorization); fact index REBUILT.
    assert new.rule_index is kb.rule_index, "facts-only did NOT reuse rule_index"
    assert new.fact_index is not kb.fact_index, "fact index was not rebuilt"
    # KB binding tables shared (the considered.finalize guard reads them).
    assert getattr(new, "_binding_tables_cache", None) is kb._binding_tables_cache, \
        "binding tables not shared on facts-only rewrite"

    # Magic-set shape: skeleton built once, per-call seeds reuse the skeleton index.
    skel = kb.with_program(heads=kb.rules_heads_idx, bodies=kb.rules_bodies_idx,
                           lens=kb.rule_lens, facts=kb.facts_idx)
    s1 = skel.with_program(facts=seeds)
    s2 = skel.with_program(facts=seeds[:2])
    assert s1.rule_index is skel.rule_index and s2.rule_index is skel.rule_index, \
        "per-call seeds re-tensorized the skeleton rule index"
    print(f"  [OK  ] (B) facts-only cache hit   rule_index reused, fact_index rebuilt")


# ── (C) rule-changing re-derives + memo keyed on rule signature ─────────────
def check_rule_change_rederives() -> None:
    facts = torch.tensor([[1, 1, 2], [1, 2, 3]])            # p(1,2), p(2,3)
    H = torch.tensor([[2, 4, 5]])                           # q(x,y)
    B = torch.tensor([[[1, 4, 6], [1, 6, 5]]])             # :- p(x,z), p(z,y)
    L = torch.tensor([2])
    kb = KB(facts, H, B, L, constant_no=3, predicate_no=12,
            padding_idx=9, device=torch.device("cpu"))

    # A DIFFERENT rule program: TWO rules, distinct heads.
    H2 = torch.tensor([[2, 4, 5], [3, 4, 5]])
    B2 = torch.tensor([[[1, 4, 6], [1, 6, 5]], [[1, 4, 5], [9, 9, 9]]])
    L2 = torch.tensor([2, 1])

    new = kb.with_program(heads=H2, bodies=B2, lens=L2)
    assert new.rule_index is not kb.rule_index, "rule-change reused the OLD index"
    assert new.num_rules == 2 and kb.num_rules == 1, "rule program not re-derived"
    # The new index reflects the NEW head predicates (sorted storage).
    new_heads = set(new.rule_index.rules_heads[:, 0].tolist())
    assert new_heads == {2, 3}, f"new rule index head preds wrong: {new_heads}"
    # Facts carried through unchanged (no facts arg -> base facts).
    assert new.num_facts == kb.num_facts

    # Memo keyed on rule signature: a SECOND identical rule-change hits the memo.
    sig2 = kb._rule_signature(H2.to(torch.long), B2.to(torch.long), L2.to(torch.long))
    assert sig2 in kb_mod._RULE_INDEX_CACHE, "rule-change did not populate the memo"
    again = kb.with_program(heads=H2, bodies=B2, lens=L2)
    assert again.rule_index is new.rule_index, "repeat rule-change missed the memo"

    # Distinct programs -> distinct signatures (no false cache hit).
    sig1 = kb._rule_signature(H.to(torch.long), B.to(torch.long), L.to(torch.long))
    assert sig1 != sig2, "distinct rule programs collided in the cache key"
    print(f"  [OK  ] (C) rule-change re-derives  new index (2 rules) + memo keyed/hit")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(Path.home() / "repos/data-swarm/main"))
    args = p.parse_args()
    print("Step-5 KB.with_program + index cache oracle:")
    check_soundness_equiv(args.data_root)
    check_facts_only_cache_hit(args.data_root)
    check_rule_change_rederives()
    print("\nPASS — with_program facts-only cache hit + rule-change re-derive + "
          "with_closure soundness-equivalent.")


if __name__ == "__main__":
    main()
