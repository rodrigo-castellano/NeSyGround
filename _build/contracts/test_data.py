"""CONTRACTS for the data layer (pure / tiny-tensor; no dataset or old-index dep).

Byte-identity vs the OLD index is proven by build-time A/B scripts + the
fingerprint (the old code is deleted at promotion, so it can't be a permanent
test). These contracts pin the design invariants that DON'T depend on old code:

- the fact-index registry is exactly {arg_key, inverted, block_sparse};
- ``create`` rejects an unknown type;
- ``pack_triples_64`` uses the contracted key order and int64 dtype;
- ``pack_base`` defaults to max(constant_no, padding_idx) + 2;
- ``exists`` membership is correct on a hand-built KB (facts in, non-facts out,
  padding/variables out).
"""
from __future__ import annotations

import pytest
import torch

import grounder._build.data.fact_index as FI
import grounder._build.data.rule_index as RI
from grounder._build.data.encoding import Encoding
from grounder._build.data.fact_index import create, pack_triples_64
from grounder._build.data.kb import KB
from grounder._build.data.rule_index import PbcRuleIndex, SldRuleIndex


def test_fact_index_registry_is_closed() -> None:
    assert set(FI._REGISTRY) == {"arg_key", "inverted", "block_sparse"}


def test_create_rejects_unknown_type() -> None:
    facts = torch.tensor([[1, 1, 2]])
    with pytest.raises(ValueError):
        create(facts, type="nope", constant_no=5, predicate_no=3,
               padding_idx=9, device=torch.device("cpu"))


def test_pack_triples_key_order() -> None:
    atoms = torch.tensor([[2, 3, 4]])
    base = 10
    assert pack_triples_64(atoms, base).tolist() == [((2 * 10) + 3) * 10 + 4]
    assert pack_triples_64(atoms, base).dtype == torch.int64
    assert pack_triples_64(torch.empty(0, 3, dtype=torch.long), base).numel() == 0


def test_pack_base_default_recipe() -> None:
    facts = torch.tensor([[1, 1, 2], [1, 2, 3]])
    idx = create(facts, type="arg_key", constant_no=5, predicate_no=3,
                 padding_idx=9, device=torch.device("cpu"))
    assert idx.pack_base == max(5, 9) + 2


def test_exists_membership() -> None:
    # KB: p(1,2), p(2,3), q(1,3). constants 1..3, padding=9.
    facts = torch.tensor([[1, 1, 2], [1, 2, 3], [2, 1, 3]])
    idx = create(facts, type="arg_key", constant_no=3, predicate_no=3,
                 padding_idx=9, device=torch.device("cpu"))
    probe = torch.tensor([[1, 1, 2],   # fact   -> True
                          [1, 2, 3],   # fact   -> True
                          [1, 1, 3],   # absent -> False
                          [2, 2, 3],   # absent -> False
                          [9, 9, 9]])  # padding-> False
    assert idx.exists(probe).tolist() == [True, True, False, False, False]


def test_encoding_boundary() -> None:
    # constants 1..3, vars 4.., pad=9 -> E=4
    enc = Encoding.from_constant_no(constant_no=3, pad=9)
    assert enc.E == 4 and enc.var_base() == 4
    ids = torch.tensor([1, 3, 4, 5, 9])
    assert enc.is_const(ids).tolist() == [True, True, False, False, False]
    assert enc.is_var(ids).tolist() == [False, False, True, True, False]  # pad is neither


def test_rule_index_factory_by_resolution() -> None:
    H = torch.tensor([[1, 4, 5]])                  # q(x,y)
    B = torch.tensor([[[2, 4, 6], [2, 6, 5]]])     # :- p(x,z), p(z,y)
    L = torch.tensor([2])
    dev = torch.device("cpu")
    assert isinstance(RI.create(H, B, L, resolution="sld", device=dev), SldRuleIndex)
    assert isinstance(RI.create(H, B, L, resolution="rtf", device=dev), SldRuleIndex)
    pbc = RI.create(H, B, L, resolution="pbc", constant_no=3, num_predicates=12, device=dev)
    assert isinstance(pbc, PbcRuleIndex) and hasattr(pbc, "variant_to_orig")
    with pytest.raises(ValueError):
        RI.create(H, B, L, resolution="nope", device=dev)


def _tiny_kb() -> KB:
    facts = torch.tensor([[1, 1, 2], [1, 2, 3]])             # p(1,2), p(2,3)
    H = torch.tensor([[2, 4, 5]])                            # q(x,y)
    B = torch.tensor([[[1, 4, 6], [1, 6, 5]]])              # :- p(x,z), p(z,y)
    L = torch.tensor([2])
    return KB(facts, H, B, L, constant_no=3, predicate_no=12,
              padding_idx=9, device=torch.device("cpu"))


def test_with_closure_returns_new_kb_without_mutating() -> None:
    kb = _tiny_kb()
    before = kb.num_facts
    new = kb.with_closure(torch.tensor([[2, 1, 3]]))         # add q(1,3)
    assert new is not kb
    assert kb.num_facts == before              # original untouched
    assert new.num_facts == before + 1         # closure has the extra fact
    # Facts-only rewrite REUSES the rule-derived index (the Step-5 cache), but
    # rebuilds the fact index (facts changed). with_closure now delegates here.
    assert kb.rule_index is new.rule_index
    assert kb.fact_index is not new.fact_index

