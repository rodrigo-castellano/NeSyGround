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
  6. PER-QUERY ≡ SCALAR — ``guided_query_topk = full(B, k)`` reproduces the
     scalar-k set EXACTLY (the tensor path is the same selection).
  7. MIXED BUDGETS  — heterogeneous per-query k stays a subset of exhaustive,
     and the all-zero budget (fact-perfect rows only) is a subset of scalar-k.
  8. CAPTURE NEUTRAL— attaching ``guided_capture`` must not select (set equals
     scalar-k), records carry in-range query indices, and every exempt row is
     kept in the recorded mask.
  9. CHUNK ALIGNMENT— per-query budgets + capture are batch-invariant under
     grounder-internal chunking (budgets sliced per chunk; b_idx lifted by the
     chunk offset spans the full batch).
 10. SAMPLED SUBSET — ``guided_sample_tau`` (stochastic PL selection at BOTH
     the binding and state level) stays a subset of exhaustive; exempt
     (fact-perfect) rows always survive; binding + state records captured.
 11. TAU→0 ≡ DETERMINISTIC — at tau=1e-6 the Gumbel noise cannot reorder
     DISTINCT scores, so within EVERY capture record the kept speculative
     rows carry exactly the top-k clean scores of their group (multiset
     comparison — exact ties may swap rows: min t-norm makes ties inherent,
     two rows sharing their min atom tie for any scorer, and Gumbel breaks
     them uniformly where the deterministic path is stable-by-index).
 12. SAMPLED ORDER CONSISTENT — in every capture record the kept speculative
     rows are EXACTLY the top-k rows per group under ``order_score`` (the
     perturbed ranking a PL trainer needs), and two seeds at tau=1 differ
     (noise actually reaches selection) while both remain subsets.
 13. DEPTH GATE ≡ PER-QUERY DEPTH — ``guided_query_depth`` (per-query depth
     gate; kills EVERY row past the gate, fact-perfect included; the
     query's last gated step applies native last-step strictness) is
     batch-mix invariant: each query's grounding set in a mixed-depth batch
     equals a plain single-query run at that query's own global depth
     (depth 0 = no groundings), both with and without a beam budget. The
     whole-batch union check runs under ``filter='none'`` — the fp_batch
     closure is batch-level BY DESIGN (native batched runs already close
     across queries), which is orthogonal to the gate.

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
from grounder.api.factory import make_grounder  # noqa: E402
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


def _build(kb, *, d, k=None, scorer=None, stats=None, filter=None):
    cfg = Backward(
        PBC(depth=d, width=1, u=0,
            max_groundings_per_rule=64,
            guided_topk=k),
        max_groundings_per_query=4096, max_goals=256, prune_facts=True,
        bump_s_to_k=False, guided_scorer=scorer, filter=filter)
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

        # ── per-query budgets (guided_query_topk) + training capture ──
        B, dev = q.shape[0], q.device
        s_k3 = _set(_build(kb, d=2, k=3, scorer=HashScorer()), q, m)

        g_pq = _build(kb, d=2, k=3, scorer=HashScorer())
        g_pq.guided_query_topk = torch.full((B,), 3, dtype=torch.long, device=dev)
        s_pq = _set(g_pq, q, m)
        check(f"{ds}|query-topk==scalar", s_pq == s_k3,
              f"|pq|={len(s_pq)} |k3|={len(s_k3)}")

        kvec = torch.tensor([0, 1, 3], device=dev).repeat(B // 3 + 1)[:B]
        g_mix = _build(kb, d=2, k=3, scorer=HashScorer())
        g_mix.guided_query_topk = kvec
        s_mix = _set(g_mix, q, m)
        check(f"{ds}|mixed-budget-subset", s_mix <= s_exh,
              f"|mix|={len(s_mix)} |exh|={len(s_exh)}")

        g_zero = _build(kb, d=2, k=3, scorer=HashScorer())
        g_zero.guided_query_topk = torch.zeros(B, dtype=torch.long, device=dev)
        s_zero = _set(g_zero, q, m)
        check(f"{ds}|zero-budget-subset", s_zero <= s_k3,
              f"|zero|={len(s_zero)} |k3|={len(s_k3)}")

        cap = []
        g_cap = _build(kb, d=2, k=3, scorer=HashScorer())
        g_cap.guided_capture = cap
        s_cap = _set(g_cap, q, m)
        b_ok = all(int(r["b_idx"].max()) < B for r in cap if r["b_idx"].numel())
        ex_ok = all(bool(r["kept"][r["exempt"]].all()) for r in cap)
        check(f"{ds}|capture-neutral", s_cap == s_k3 and cap and b_ok and ex_ok,
              f"records={len(cap)} b_ok={b_ok} exempt_kept={ex_ok}")

        cap_ch = []
        cfg_ch = Backward(
            PBC(depth=2, width=1, u=0, max_groundings_per_rule=64, guided_topk=3),
            max_groundings_per_query=4096, max_goals=256, prune_facts=True,
            bump_s_to_k=False, guided_scorer=HashScorer())
        g_ch = make_grounder(kb, cfg_ch, layout="flat", compile="off",
                             chunk_size=max(B // 2, 1))
        g_ch.guided_query_topk = kvec
        g_ch.guided_capture = cap_ch
        s_ch = _set(g_ch, q, m)
        seen_b = (torch.cat([r["b_idx"] for r in cap_ch]).unique()
                  if cap_ch else torch.empty(0, dtype=torch.long))
        spans = seen_b.numel() > 0 and int(seen_b.max()) >= B // 2
        check(f"{ds}|chunk-alignment", s_ch == s_mix and spans,
              f"|ch|={len(s_ch)} |mix|={len(s_mix)} b_span={spans}")

        # ── stochastic PL selection (guided_sample_tau; binding + state) ──
        def _sampled(tau, seed, cap=None):
            g_s = _build(kb, d=2, k=1, scorer=HashScorer())
            g_s.guided_sample_tau = tau
            if cap is not None:
                g_s.guided_capture = cap
            torch.manual_seed(seed)
            return _set(g_s, q, m)

        cap_s = []
        s_s1 = _sampled(1.0, 0, cap_s)
        ex_s = all(bool(r["kept"][r["exempt"]].all()) for r in cap_s)
        levels = {r["level"] for r in cap_s}
        check(f"{ds}|sampled-subset",
              s_s1 <= s_exh and ex_s and levels == {"binding", "state"},
              f"|s|={len(s_s1)} |exh|={len(s_exh)} exempt_kept={ex_s} "
              f"levels={sorted(levels)}")

        cap_cold = []
        s_cold = _sampled(1e-6, 0, cap_cold)

        def _cold_topk(records):
            for r in records:
                spec, kept, grp = ~r["exempt"], r["kept"], r["groups"]
                sc = (r["score"] * 1e6).round()
                for gb in grp[spec].unique().tolist():
                    rows = torch.nonzero(spec & (grp == gb)).squeeze(1)
                    kk = int(kept[rows].sum())
                    if kk == 0:
                        continue
                    top = sorted(sc[rows].tolist(), reverse=True)[:kk]
                    got = sorted(sc[rows[kept[rows]]].tolist(), reverse=True)
                    if got != top:
                        return False
            return True
        check(f"{ds}|tau0==deterministic",
              s_cold <= s_exh and _cold_topk(cap_cold),
              f"|tau0|={len(s_cold)} |k1|={len(s_k1)} Δset={len(s_cold ^ s_k1)}")

        def _order_ok(records):
            for r in records:
                spec = ~r["exempt"]
                if not spec.any():
                    continue
                os_, kept, grp = r["order_score"], r["kept"], r["groups"]
                for gb in grp[spec].unique().tolist():
                    rows = torch.nonzero(spec & (grp == gb)).squeeze(1)
                    kk = int(kept[rows].sum())
                    if kk == 0 or kk == rows.numel():
                        continue
                    if os_[rows[kept[rows]]].min() < os_[rows[~kept[rows]]].max():
                        return False
            return True
        check(f"{ds}|sampled-order-consistent", _order_ok(cap_s),
              f"records={len(cap_s)}")

        s_s2 = _sampled(1.0, 1)
        check(f"{ds}|sampled-varies", s_s2 <= s_exh and s_s2 != s_s1,
              f"Δ(seed0,seed1)={len(s_s2 ^ s_s1)}")

        # ── per-query depth gate (guided_query_depth) ──
        nq = min(9, B)
        dvec = torch.tensor([0, 1, 2, 3] * (nq // 4 + 1), device=dev)[:nq]
        for tag, kk, sc in (("gate-only", None, None),
                            ("gate+k3", 3, HashScorer())):
            ok, msg = True, ""
            for b in range(nq):
                D = int(dvec[b])
                # reference: plain single-query run at global depth D
                ref = set() if D == 0 else _set(
                    _build(kb, d=D, k=kk, scorer=sc), q[b:b + 1], m[b:b + 1])
                # gated: same query inside a d=3 grounder, gate = D
                g_g = _build(kb, d=3, k=kk, scorer=sc)
                g_g.guided_query_depth = dvec[b:b + 1]
                got = _set(g_g, q[b:b + 1], m[b:b + 1])
                if got != ref:
                    ok, msg = False, f"q{b} D={D} |got|={len(got)} |ref|={len(ref)}"
                    break
            # batch-mix union under filter='none' (the fp_batch closure is
            # batch-level by design and would add cross-query closures).
            if ok:
                g_mix2 = _build(kb, d=3, k=kk, scorer=sc, filter="none")
                g_mix2.guided_query_depth = dvec
                s_gmix = _set(g_mix2, q[:nq], m[:nq])
                refs = set()
                for b in range(nq):
                    D = int(dvec[b])
                    if D:
                        refs |= _set(_build(kb, d=D, k=kk, scorer=sc,
                                            filter="none"),
                                     q[b:b + 1], m[b:b + 1])
                ok, msg = s_gmix == refs, f"|mix|={len(s_gmix)} |refs|={len(refs)}"
            check(f"{ds}|depth-gate[{tag}]", ok, msg)

    if fails:
        print(f"\nFAIL — {len(fails)} contract violation(s)")
        sys.exit(1)
    print(f"\nPASS — guided contracts hold on {len(_CELLS)} cells")


if __name__ == "__main__":
    main()
