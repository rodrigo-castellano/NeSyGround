"""grounder.api.rule_grounder: string rules + fact ids -> a grounder whose run_bc returns the proving firings.

    python -m pytest tests/test_rule_grounder.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from grounder.api.rule_grounder import CompiledRule, TensorFactIndex, create_grounder, parse_grounder_type

ENT = {n: i for i, n in enumerate(["ann", "bob", "cid", "dan"])}
REL = {"parent": 0, "grandparent": 1}
RULES = [(("grandparent", "X", "Y"), [("parent", "X", "Z"), ("parent", "Z", "Y")])]
FACTS = [(0, ENT["ann"], ENT["bob"]), (0, ENT["bob"], ENT["cid"]), (0, ENT["cid"], ENT["dan"])]


def _grounder(grounder_type="enum.fp_batch.w1.d2.flat"):
    vocab = SimpleNamespace(relation2id=REL, entity2id=ENT, rule_names=["r0"])
    facts = TensorFactIndex(FACTS, num_predicates=len(REL), num_entities=len(ENT), device="cpu")
    return facts, create_grounder(grounder_type, fact_index=facts, rules=RULES, kb=vocab, max_groundings=8,
                                  max_total_groundings=16, provable_set_method="join", device="cpu")


def test_compiled_rule_orders_body_so_each_atom_has_a_bound_argument():
    cr = CompiledRule(RULES[0], REL, name="r0")
    assert cr.num_body == 2 and cr.num_free == 1 and cr.body_order == [0, 1]


def test_parse_grounder_type():
    assert parse_grounder_type("enum.fp_batch.w1.d2.flat") == (1, 2)
    assert parse_grounder_type("sld.prune.d1") == (1, 1)


def test_fact_membership():
    facts, _ = _grounder()
    atoms = torch.tensor([[0, ENT["ann"], ENT["bob"]], [0, ENT["ann"], ENT["cid"]]])
    assert facts.contains_atoms(atoms).tolist() == [True, False]


def test_run_bc_proves_a_two_hop_query_and_not_a_false_one():
    _, g = _grounder()
    queries = torch.tensor([[1, ENT["ann"], ENT["cid"]], [1, ENT["ann"], ENT["dan"]]])
    rg = g.run_bc(queries, torch.ones(2, dtype=torch.bool))
    valid = rg.firing_valid.bool()
    heads = rg.atom_table[rg.head_pool_idx[valid]].tolist()
    assert [1, ENT["ann"], ENT["cid"]] in heads          # parent(ann, bob), parent(bob, cid)
    assert [1, ENT["ann"], ENT["dan"]] not in heads      # no parent path of length two
