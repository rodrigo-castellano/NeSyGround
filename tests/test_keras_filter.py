"""The keras filter (keras-ns's BC_{w,d}): its cycle rule and its proof-walk rounds.

    python -m pytest tests/test_keras_filter.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from grounder.api.rule_grounder import TensorFactIndex, create_grounder

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PREDS = ["loc", "nb"]
A, B, Z, C = range(4)
FACTS = [(1, A, B), (1, B, A), (0, A, Z), (1, B, C)]                    # nb(a, b), nb(b, a), loc(a, z), nb(b, c)
# Countries S3's r2: a chain that can come back to the head's own atom (K = X)
RULES = [(("loc", "X", "Z"), [("nb", "X", "Y"), ("nb", "Y", "K"), ("loc", "K", "Z")])]


def _groundings(grounder_type: str, queries, facts=FACTS, rules=RULES):
    vocab = SimpleNamespace(relation2id={p: i for i, p in enumerate(PREDS)}, entity2id={f"e{i}": i for i in range(4)},
                            rule_names=[f"r{i}" for i in range(len(rules))])
    fi = TensorFactIndex(facts, num_predicates=len(PREDS), num_entities=4, device=DEV, max_facts_per_query=8)
    rg = create_grounder(grounder_type, fact_index=fi, rules=rules, kb=vocab, max_groundings=32,
                         max_total_groundings=64, provable_set_method="spmm", device=DEV)
    q = torch.tensor(queries, device=DEV)
    out = rg.run_bc(q, torch.ones(len(q), dtype=torch.bool, device=DEV))
    at = out.atom_table.tolist()
    return rg, {(tuple(at[h]), tuple(tuple(at[b]) for b in body))
                for h, body, ok in zip(out.head_pool_idx.tolist(), out.body_pool_idx.tolist(),
                                       out.firing_valid.tolist()) if ok}


def test_keras_keeps_a_body_holding_its_head_only_as_a_fact():
    """keras-ns draws one body atom from the facts and rejects the query among the others: a grounding whose body
    holds its own head survives when that head is a fact (it can be the drawn atom), never otherwise."""
    fact_head, unknown_head = (0, A, Z), (0, B, Z)
    cycle = ((fact_head, ((1, A, B), (1, B, A), fact_head)))
    _, keras = _groundings("enum.keras.w0.d1.flat", [fact_head])
    assert cycle in keras
    _, default = _groundings("enum.fp_batch.w0.d1.flat", [fact_head])
    assert cycle not in default                                        # the default drops every such grounding
    _, keras_w1 = _groundings("enum.keras.w1.d1.flat", [unknown_head])
    assert all(unknown_head not in body for head, body in keras_w1 if head == unknown_head)


def test_keras_drops_a_body_holding_its_head_twice():
    """Only one body atom is drawn from the facts: a second atom equal to the query is checked like any other and
    rejected, so a grounding whose body holds its (fact) head twice never survives (FB15k-237's self-loop queries)."""
    loop = (1, A, A)
    rules = [(("nb", "X", "Y"), [("nb", "X", "K"), ("nb", "K", "Y")])]
    _, keras = _groundings("enum.keras.w0.d1.flat", [loop], facts=FACTS + [loop], rules=rules)
    assert keras == {(loop, ((1, A, B), (1, B, A)))}                   # not nb(a, a), nb(a, a)


def test_keras_head_only_body_atoms_need_every_atom_a_fact_at_depth_one():
    """A body atom bound by the head alone is known to keras-ns without a lookup; with no proof rounds (IJCAI-25's
    depth 1) its pruning still keeps only all-fact bodies. Deeper, the grounder refuses such rules."""
    rules = [(("nb", "X", "Y"), [("nb", "Y", "X"), ("loc", "X", "Y")])]
    facts = FACTS + [(0, B, A)]                                        # loc(b, a): nb(a, b)'s second atom not a fact
    _, keras = _groundings("enum.keras.w0.d1.flat", [(1, A, B), (1, B, A)], facts=facts, rules=rules)
    assert keras == {((1, B, A), ((1, A, B), (0, B, A)))}
    with pytest.raises(NotImplementedError):
        _groundings("enum.keras.w1.d2.flat", [(1, A, B)], facts=facts, rules=rules)


def test_keras_one_round_at_depth_one_keeps_a_one_body_atom_the_batch_proves():
    """keras-ns adds a one-body rule's grounding untested; the later keras-ns (XAI-25, NeSy-25) walks one proof round
    at depth 1 (``.r1``), so the grounding survives when a grounding of the same batch proves its non-fact atom; with
    no round (IJCAI-25) only all-fact bodies do."""
    rules = [(("nb", "X", "Y"), [("nb", "Y", "X")]), (("nb", "X", "Y"), [("nb", "X", "K"), ("nb", "K", "Y")])]
    chain = ((1, A, C), ((1, A, B), (1, B, C)))                    # nb(a, c): proved by nb(a, b), nb(b, c)
    flip = ((1, C, A), ((1, A, C),))                               # nb(c, a) :- nb(a, c), not a fact
    unpadded = lambda out: {(h, tuple(a for a in b if a[0] < len(PREDS))) for h, b in out[1]}      # noqa: E731
    assert unpadded(_groundings("enum.keras.w0.d1.r1.flat", [(1, A, C), (1, C, A)], rules=rules)) == {chain, flip}
    assert unpadded(_groundings("enum.keras.w0.d1.flat", [(1, A, C), (1, C, A)], rules=rules)) == {chain}


RANDOM_PREDS = ["p0", "p1", "p2", "p3", "p4"]
RANDOM_RULES = [
    (("p2", "X", "Y"), [("p0", "X", "Z"), ("p1", "Z", "Y")]),                       # a chain: one free variable
    (("p3", "X", "Y"), [("p2", "Y", "X")]),                                        # one body atom, no free variable
    (("p4", "X", "Y"), [("p0", "X", "Z"), ("p0", "Z", "W"), ("p1", "W", "Y")]),    # two free variables
    (("p1", "X", "Y"), [("p3", "X", "Y"), ("p0", "Y", "Z")]),                       # a free variable only in the body
    (("p2", "X", "Y"), [("p2", "Y", "X")]),                                        # recursive
]


def _setup(seed: int = 0, E: int = 30, F: int = 160):
    """A random KB over ``RANDOM_PREDS``: ``(fact index, vocabulary, queries)`` (facts and random atoms)."""
    g = torch.Generator().manual_seed(seed)
    facts = torch.unique(torch.stack([torch.randint(0, len(RANDOM_PREDS), (F,), generator=g),
                                      torch.randint(0, E, (F,), generator=g),
                                      torch.randint(0, E, (F,), generator=g)], 1), dim=0)
    vocab = SimpleNamespace(relation2id={p: i for i, p in enumerate(RANDOM_PREDS)},
                            entity2id={f"e{i}": i for i in range(E)})
    fi = TensorFactIndex([tuple(f) for f in facts.tolist()], num_predicates=len(RANDOM_PREDS), num_entities=E,
                         device=DEV)
    queries = torch.cat([facts[:40], torch.stack([torch.randint(0, len(RANDOM_PREDS), (60,), generator=g),
                                                  torch.randint(0, E, (60,), generator=g),
                                                  torch.randint(0, E, (60,), generator=g)], 1)]).to(DEV)
    return fi, vocab, queries


@pytest.mark.parametrize("grounder_type, rounds", [("enum.keras.w1.d2.flat", None), ("enum.keras.w1.d2.r2.flat", 2)])
def test_keras_walk_rounds_parse(grounder_type, rounds):
    """``.r<N>``: the proof walk's rounds (the later keras-ns walks depth rounds, the IJCAI-25 code depth - 1)."""
    rg, _ = _groundings(grounder_type, [(0, A, Z)])
    assert rg.pbc.rounds == rounds


@pytest.mark.parametrize("grounder_type", ["enum.keras.w1.d2.flat", "enum.keras.w1.d3.flat", "enum.keras.w1.d3.r3.flat"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_keras_prefilter_keeps_the_walks_output(grounder_type, seed, monkeypatch):
    """Dropping the groundings no chain from the all-fact ones can prove (``engine._keras_provable``) before keras-ns's
    proof walk changes nothing it outputs: recursive and one-body rules, several batches at once."""
    from grounder.pbc import engine as fast
    rules = [(h, b) for h, b in RANDOM_RULES if len(b) == 1 or all(set(a[1:]) - set(h[1:]) for a in b)]   # keras's
    fi, vocab, queries = _setup(seed)
    vocab.rule_names = [f"r{i}" for i in range(len(rules))]
    rg = create_grounder(grounder_type, fact_index=fi, rules=rules, kb=vocab, max_groundings=32,
                         max_total_groundings=64, provable_set_method="spmm", device=DEV)
    qs = torch.stack([queries, queries.flip(0), queries.roll(17, 0)])
    mask = torch.rand(qs.shape[:2], generator=torch.Generator().manual_seed(seed)).to(DEV) < 0.9
    outs, sizes, provable = {}, [], fast._keras_provable

    def counted(g, facts, rule, *args):
        out = provable(g, facts, rule, *args)
        sizes.append((int(rule.shape[0]), int(out[0].shape[0])))
        return out
    monkeypatch.setattr(fast, "_keras_provable", counted)
    for rows in (0, 1 << 62):                          # the prefilter on every call / never
        monkeypatch.setattr(fast, "KERAS_PREFILTER_ROWS", rows)
        outs[rows] = rg.run_bc_many(qs, mask)
    assert sizes and all(after < before for before, after in sizes)      # it drops groundings
    for a, b in zip(*outs.values()):
        for f in ("atom_table", "body_pool_idx", "body_atom_valid", "head_pool_idx", "rule_idx", "rule_offsets",
                  "query_pool_idx"):
            assert torch.equal(getattr(a, f), getattr(b, f)), f
    assert sum(int(o.rule_idx.shape[0]) for o in outs[0]) > 0
