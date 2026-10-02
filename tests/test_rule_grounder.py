"""grounder.api.rule_grounder (torch-ns's entry point): string rules + fact ids -> each pool's proving groundings (GPU).

    python -m pytest tests/test_rule_grounder.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from grounder.api.rule_grounder import TensorFactIndex, create_grounder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="PBC's kernels need the GPU")
ENT = {n: i for i, n in enumerate(["ann", "bob", "cid", "dan"])}
REL = {"parent": 0, "grandparent": 1}
RULES = [(("grandparent", "X", "Y"), [("parent", "X", "Z"), ("parent", "Z", "Y")])]
FACTS = [(0, ENT["ann"], ENT["bob"]), (0, ENT["bob"], ENT["cid"]), (0, ENT["cid"], ENT["dan"])]


def _grounder(grounder_type="enum.fp_batch.w1.d2.flat"):
    vocab = SimpleNamespace(relation2id=REL, entity2id=ENT)
    facts = TensorFactIndex(FACTS, num_predicates=len(REL), num_entities=len(ENT), device="cuda")
    return facts, create_grounder(grounder_type, fact_index=facts, rules=RULES, kb=vocab, device="cuda")


def test_fact_membership():
    facts, _ = _grounder()
    atoms = torch.tensor([[0, ENT["ann"], ENT["bob"]], [0, ENT["ann"], ENT["cid"]]], device="cuda")
    assert facts.contains_atoms(atoms).tolist() == [True, False]


@pytest.mark.parametrize("grounder_type", ["enum.fp_batch.w1.d2.flat", "enum.keras.w1.d2.flat"])
def test_run_bc_proves_a_two_hop_query_and_not_a_false_one(grounder_type):
    _, g = _grounder(grounder_type)
    queries = torch.tensor([[1, ENT["ann"], ENT["cid"]], [1, ENT["ann"], ENT["dan"]]], device="cuda")
    rg = g.run_bc(queries, torch.ones(2, dtype=torch.bool, device="cuda"))
    heads = rg.atom_table[rg.head_pool_idx].tolist()
    assert [1, ENT["ann"], ENT["cid"]] in heads          # parent(ann, bob), parent(bob, cid)
    assert [1, ENT["ann"], ENT["dan"]] not in heads      # no parent path of length two
    assert rg.atom_table[rg.query_pool_idx].tolist() == queries.tolist()
    assert g.input_rule == [0] and g.input_body == [[0, 1]] and g._inner.depth == 2
