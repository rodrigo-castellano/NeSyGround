"""Guided-instantiation oracle — gates ``PBC.guided_topk`` against the join path.

Five contracts, per dataset cell:

  1. CENSUS PARITY  — ``guided_topk=None`` + attached GuidedStats produces the
     EXACT exhaustive-join grounding set (counting must never select), with
     counters populated and in == out.
  2. NEUTRAL BEAM   — ``k=10^9`` (every speculative row fits the beam) equals
     the exhaustive set.
  3. SUBSET         — any finite k produces a SUBSET of the exhaustive set,
     and counters show bindings_out ≤ bindings_in / states_out ≤ states_in.
  4. MONOTONE PROOFS— for k ∈ {1, 3} AND exhaustive, the grounding set is
     NESTED in depth (d1 ⊆ d2 ⊆ d3): fact-perfect rows ride outside the
     budget, so a proof found shallow is never lost deeper. This is the
     property the 2026-06-10 v1 beam violated (proofs FELL with depth).
  5. SCORER STEERING— two different scorers at k=1 may keep different beams,
     but BOTH stay subsets of exhaustive (sanity that scores reach selection).

Usage:
    python tests/guided_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from join_ab import _grounding_set, _load  # noqa: E402

from grounder.api.config import Backward, PBC  # noqa: E402
from grounder.api.backward import BackwardGrounder  # noqa: E402
from grounder.core import GroundRequest, OutputSpec, Tier  # noqa: E402
from grounder.resolution.pbc.guided import GuidedStats  # noqa: E402

_ALL = OutputSpec(frozenset({Tier.PROOF_STATE, Tier.FIRINGS, Tier.TREES}))
_CELLS = ["family", "countries_s2", "ablation_d2"]


class HashScorer:
    """Deterministic pseudo-KGE prior: hash-based prob in [0.1, 0.9]."""

    def __init__(self, salt: int = 0):
        self._salt = salt

    def score_atoms(self, atoms):
        h = (atoms[:, 0] * 31 + atoms[:, 1] * 17 + atoms[:, 2] * 13
             + self._salt) % 97
        return (h.float() / 97.0) * 0.8 + 0.1


def _build(kb, *, d, k=None, scorer=None, stats=None):
    cfg = Backward(
        PBC(depth=d, width=1, u=0,
            max_groundings_per_rule=64,
            guided_topk=k),
        max_groundings_per_query=4096, max_goals=256, prune_facts=True,
        bump_s_to_k=False, guided_scorer=scorer)
    g = BackwardGrounder(kb, cfg)
    if stats is not None:
        g.guided_stats = stats
    return g


def _set(g, q, m):
    with torch.no_grad():
        out = g.ground(GroundRequest(queries=q, query_mask=m, output_spec=_ALL))
    return _grounding_set(out)


def main() -> None:
    fails = []

    def check(key, ok, msg=""):
        print(f"  [{'OK  ' if ok else 'FAIL'}] {key}" + (f"  {msg}" if msg else ""))
        if not ok:
            fails.append((key, msg))

    for ds in _CELLS:
        kb, q, m = _load(ds)
        q, m = q[:30], m[:30]
        s_exh = _set(_build(kb, d=2), q, m)

        st = GuidedStats()
        s_census = _set(_build(kb, d=2, stats=st), q, m)
        t = st.totals()
        check(f"{ds}|census-parity", s_census == s_exh,
              f"|exh|={len(s_exh)} |census|={len(s_census)}")
        check(f"{ds}|census-counters",
              sum(t["bindings_in"].values()) > 0
              and t["bindings_in"] == t["bindings_out"]
              and t["states_in"] == t["states_out"], str(t))

        s_huge = _set(_build(kb, d=2, k=10**9, scorer=HashScorer()), q, m)
        check(f"{ds}|k=1e9==exhaustive", s_huge == s_exh,
              f"{len(s_huge)} vs {len(s_exh)}")

        st1 = GuidedStats()
        s_k1 = _set(_build(kb, d=2, k=1, scorer=HashScorer(), stats=st1), q, m)
        t1 = st1.totals()
        check(f"{ds}|k=1-subset", s_k1 <= s_exh,
              f"|k1|={len(s_k1)} |exh|={len(s_exh)}")
        check(f"{ds}|k=1-counters",
              sum(t1["bindings_out"].values()) <= sum(t1["bindings_in"].values())
              and sum(t1["states_out"].values()) <= sum(t1["states_in"].values()),
              str(t1))

        for label, kk, sc in (("exh", None, None),
                              ("k1", 1, HashScorer()), ("k3", 3, HashScorer())):
            sets = {d: _set(_build(kb, d=d, k=kk, scorer=sc), q, m)
                    for d in (1, 2, 3)}
            check(f"{ds}|monotone-proofs[{label}]",
                  sets[1] <= sets[2] <= sets[3],
                  f"d1/d2/d3 = {len(sets[1])}/{len(sets[2])}/{len(sets[3])}")

        s_alt = _set(_build(kb, d=2, k=1, scorer=HashScorer(salt=7)), q, m)
        check(f"{ds}|alt-scorer-subset", s_alt <= s_exh,
              f"|alt|={len(s_alt)} (Δ vs k1 = {len(s_alt ^ s_k1)})")

    if fails:
        print(f"\nFAIL — {len(fails)} contract violation(s)")
        sys.exit(1)
    print(f"\nPASS — guided contracts hold on {len(_CELLS)} cells")


if __name__ == "__main__":
    main()
