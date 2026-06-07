"""magic_ab — the MagicSetTransform oracle (AXIS 4, magic-sets over ForwardGrounder).

Magic-sets is a BOTTOM-UP demand transformation; its home is the ForwardGrounder
(T_P fixpoint ACCUMULATES the closure, so a magic guard genuinely RESTRICTS it —
unlike backward chaining, which is already demand-restricted and never accumulates).

Toy KB: recursive ``ancestor`` over a single ``parent`` relation with TWO DISJOINT
components — A around entity 1 (a 1->2->3->4 chain) and B around entity 8 (an
8->9->10->11 chain). Query ``ancestor(1, ?)`` is relevant to component A ONLY; the
plain FORWARD closure derives BOTH components (query-independent), so its closure is
full of query-IRRELEVANT atoms (all of component B).

Two assertions (do NOT weaken):
1. CORRECTNESS — the magic-set query-relevant answers (the adorned-head atoms,
   decoded back to ``ancestor``) EQUAL the plain full-closure ``ancestor`` atoms
   filtered to the query-relevant set (subjects in the magic demand set reachable
   from the query). Magic-set loses NO relevant answer.
2. RESTRICTION — the magic-set's total derived-atom count (``Closure.n_provable``)
   is STRICTLY LESS than the plain full-closure ``n_provable``: it derives component
   A + its magic bookkeeping but skips ALL of component B. The "fewer groundings"
   proof, achievable precisely because FC accumulates the closure.

The transform is exercised through the FULL seam: ``make_grounder(kb, FCConfig,
transforms=[MagicSetTransform(...)])`` -> a ``core.Pipeline`` that per-call seeds
the magic demand facts and re-snapshots the base ForwardGrounder.

Usage:
    python _build/tests/magic_ab.py
"""
from __future__ import annotations

import sys

import torch

from grounder._build.config import FCConfig
from grounder._build.core.request import GroundRequest
from grounder._build.data.kb import KB
from grounder._build.factory import make_grounder
from grounder._build.forward.grounder import ForwardGrounder
from grounder._build.transform.magic_set import MagicSetTransform

# Toy KB encoding (entities 1..11; predicate ids parent=0, ancestor=1; vars > C).
_PAD = 12
_C = 11
_P_PARENT = 0
_P_ANC = 1


def _build_toy_kb() -> KB:
    """parent edges over two disjoint chains; recursive ancestor rules."""
    facts = torch.tensor([
        # component A (around entity 1): 1 -> 2 -> 3 -> 4
        [_P_PARENT, 1, 2], [_P_PARENT, 2, 3], [_P_PARENT, 3, 4],
        # component B (around entity 8): 8 -> 9 -> 10 -> 11
        [_P_PARENT, 8, 9], [_P_PARENT, 9, 10], [_P_PARENT, 10, 11],
    ], dtype=torch.long)
    X, Y, V = 13, 14, 15
    # r1: ancestor(X,Y) :- parent(X,Y)
    # r2: ancestor(X,Y) :- parent(X,V), ancestor(V,Y)
    heads = torch.tensor([[_P_ANC, X, Y], [_P_ANC, X, Y]], dtype=torch.long)
    bodies = torch.tensor([
        [[_P_PARENT, X, Y], [_PAD, _PAD, _PAD]],
        [[_P_PARENT, X, V], [_P_ANC, V, Y]],
    ], dtype=torch.long)
    lens = torch.tensor([1, 2], dtype=torch.long)
    return KB(facts, heads, bodies, lens, constant_no=_C, predicate_no=2,
              padding_idx=_PAD, device=torch.device("cpu"))


def _anc_atoms(facts: torch.Tensor, pred: int) -> set:
    """Set of (subj, obj) for atoms with predicate ``pred``."""
    rows = facts[facts[:, 0] == pred]
    return {(int(s), int(o)) for _, s, o in rows.tolist()}


def main() -> None:
    kb = _build_toy_kb()
    query_subj = 1
    queries = torch.tensor([[_P_ANC, query_subj, 0]], dtype=torch.long)  # ancestor(1, ?)

    # ── plain ForwardGrounder: the FULL closure (query-independent) ──
    plain = ForwardGrounder(kb, method="spmm", depth=10)
    with torch.no_grad():
        full = plain.ground()
    n_full = int(full.n_provable)
    full_anc = _anc_atoms(full.facts(), _P_ANC)

    # ── magic-set via the full Pipeline seam ──
    mt = MagicSetTransform(kb, query_pred=_P_ANC)
    g = make_grounder(kb, FCConfig(depth=10), transforms=[mt])
    with torch.no_grad():
        magic = g.ground(GroundRequest(queries=queries))
    n_magic = int(magic.n_provable)
    mfacts = magic.facts()
    # adorned-head atoms -> decode to the original ``ancestor`` answers.
    magic_answers = _anc_atoms(mfacts, mt.adorned_pred)
    # the magic DEMAND set = query subject(s) (the SEED, a base fact) UNION the
    # recursively-DERIVED magic atoms (closure-only). These are exactly the
    # query-relevant subjects.
    demand = {query_subj} | {s for (s, o) in _anc_atoms(mfacts, mt.magic_pred)}

    # query-relevant slice of the plain full closure: answers whose subj is demanded.
    relevant_full = {(s, o) for (s, o) in full_anc if s in demand}

    # ── assertions ──
    ok_correct = magic_answers == relevant_full
    ok_restrict = n_magic < n_full
    # sanity: the full closure genuinely contains IRRELEVANT (component-B) atoms,
    # so the restriction is non-trivial (the test KB is doing its job).
    ok_nontrivial = len(full_anc) > len(relevant_full)

    print(f"  plain FC  : n_provable={n_full:>4}  ancestor_atoms={len(full_anc)}")
    print(f"  magic-set : n_provable={n_magic:>4}  relevant_answers={len(magic_answers)}")
    print(f"  demand set (relevant subjects): {sorted(demand)}")
    print(f"  relevant slice of full closure : {len(relevant_full)} atoms")
    print(f"  correctness (magic == relevant-full)     : {ok_correct}")
    print(f"  restriction (n_magic < n_full)           : {ok_restrict}  ({n_magic} < {n_full})")
    print(f"  non-trivial (full has irrelevant atoms)  : {ok_nontrivial}")

    if not (ok_correct and ok_restrict and ok_nontrivial):
        if not ok_correct:
            print(f"\n  magic-only : {sorted(magic_answers - relevant_full)}")
            print(f"  full-only  : {sorted(relevant_full - magic_answers)}")
        print("\nFAIL — magic-set does NOT both preserve relevant answers and restrict "
              f"the FC closure (correct={ok_correct} restrict={ok_restrict} "
              f"nontrivial={ok_nontrivial})")
        sys.exit(1)

    print(f"\nPASS — magic-set restricts FC closure, same relevant answers, fewer "
          f"derived (N_magic={n_magic} < N_full={n_full})")


if __name__ == "__main__":
    main()
