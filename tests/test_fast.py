"""backward.fast: the width <= 1 pbc fast path grounds the same firings as the engine.

    python -m pytest tests/test_fast.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from grounder.api.rule_grounder import TensorFactIndex, create_grounder
from grounder.backward import fast
from grounder.core import GroundRequest, OutputSpec, Tier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PREDS = ["p0", "p1", "p2", "p3", "p4"]
RULES = [
    (("p2", "X", "Y"), [("p0", "X", "Z"), ("p1", "Z", "Y")]),                       # a chain: one free variable
    (("p3", "X", "Y"), [("p2", "Y", "X")]),                                        # one body atom, no free variable
    (("p4", "X", "Y"), [("p0", "X", "Z"), ("p0", "Z", "W"), ("p1", "W", "Y")]),    # two free variables
    (("p1", "X", "Y"), [("p3", "X", "Y"), ("p0", "Y", "Z")]),                       # a free variable only in the body
    (("p2", "X", "Y"), [("p2", "Y", "X")]),                                        # recursive
]


def _setup(seed: int = 0, E: int = 30, F: int = 160, max_facts_per_query: int = 64):
    g = torch.Generator().manual_seed(seed)
    facts = torch.unique(torch.stack([torch.randint(0, len(PREDS), (F,), generator=g),
                                      torch.randint(0, E, (F,), generator=g),
                                      torch.randint(0, E, (F,), generator=g)], 1), dim=0)
    vocab = SimpleNamespace(relation2id={p: i for i, p in enumerate(PREDS)},
                            entity2id={f"e{i}": i for i in range(E)}, rule_names=[f"r{i}" for i in range(len(RULES))])
    fi = TensorFactIndex([tuple(f) for f in facts.tolist()], num_predicates=len(PREDS), num_entities=E, device=DEV,
                         max_facts_per_query=max_facts_per_query)
    queries = torch.cat([facts[:40], torch.stack([torch.randint(0, len(PREDS), (60,), generator=g),
                                                  torch.randint(0, E, (60,), generator=g),
                                                  torch.randint(0, E, (60,), generator=g)], 1)]).to(DEV)
    return fi, vocab, queries


def _content(rg):
    at = rg.atom_table
    rows = torch.cat([rg.rule_idx.unsqueeze(1), at[rg.head_pool_idx],
                      at[rg.body_pool_idx].flatten(1)], 1)
    return rows, rg.body_atom_valid, rg.rule_offsets, at[rg.query_pool_idx]


@pytest.mark.parametrize("grounder_type", ["enum.fp_batch.w0.d1.flat", "enum.fp_batch.w1.d2.flat",
                                           "enum.fp_batch.w1.d3.flat"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("index", ["dense", "offsets", "offsets_capped"])
def test_fast_path_grounds_the_engines_firings(grounder_type, seed, index):
    """Dense [P, E, K] blocks or offset tables (the large-KG fallback), with a slot cap (the first
    ``max_facts_per_query`` facts of a (predicate, argument) slot are enumerated)."""
    fi, vocab, queries = _setup(seed, max_facts_per_query=2 if index == "offsets_capped" else 64)
    rg = create_grounder(grounder_type, fact_index=fi, rules=RULES, kb=vocab, max_groundings=32,
                         max_total_groundings=64, provable_set_method="spmm", device=DEV)
    if index != "dense":
        rg._inner.kb.fact_index._use_dense = False
    assert fast.supports(rg._inner)
    mask = torch.ones(len(queries), dtype=torch.bool, device=DEV)
    mask[::7] = False
    engine = rg._inner.ground(GroundRequest(queries=queries, query_mask=mask,
                                            output_spec=OutputSpec(frozenset({Tier.FIRINGS})))).rule_groundings
    got = rg.run_bc(queries, mask)
    for a, b in zip(_content(engine), _content(got)):
        assert torch.equal(a, b)
    assert torch.equal(_content(got)[3], queries)
    assert len(got.atom_table) <= len(engine.atom_table)          # compacted to the firings' atoms + the queries


@pytest.mark.parametrize("grounder_type", ["enum.fp_batch.w0.d1.flat", "enum.fp_batch.w1.d2.flat"])
@pytest.mark.parametrize("chunk", [None, 37])
def test_many_batches_ground_each_batch_alone(grounder_type, chunk):
    fi, vocab, queries = _setup(3)
    rg = create_grounder(grounder_type, fact_index=fi, rules=RULES, kb=vocab, max_groundings=32,
                         max_total_groundings=64, provable_set_method="spmm", device=DEV)
    qs = torch.stack([queries, queries.flip(0), queries.roll(17, 0), queries[:, [0, 2, 1]]])      # [4, Q, 3]
    mask = torch.rand(qs.shape[:2], generator=torch.Generator().manual_seed(0)).to(DEV) < 0.8
    mask[2] = False                                                                           # an empty batch
    for got, s in zip(fast.ground_many(rg._inner, qs, mask, chunk), range(4)):
        alone = fast.ground(rg._inner, qs[s], mask[s])
        for a, b in zip(_content(alone), _content(got)):
            assert torch.equal(a, b)
        assert torch.equal(got.atom_table, alone.atom_table) and got.num_atoms == alone.num_atoms
        assert torch.equal(got.head_pool_idx, alone.head_pool_idx)
