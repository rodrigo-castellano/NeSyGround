"""SLD: proofs on a toy program, and ``derive`` against LOGIC2RL's engine (unmodified, when installed)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from grounder.kb import KB
from grounder.sld import SLD

DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "repos/data-swarm/main"))


def _toy() -> KB:
    """parent(a,b) parent(b,c) parent(c,d); grandparent(X,Y) :- parent(X,Z), parent(Z,Y);
    ancestor(X,Y) :- parent(X,Y); ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y). Constants a..d = 0..3,
    predicates parent 0, grandparent 1, ancestor 2; variables X, Y, Z = 4, 5, 6; pad 9."""
    facts = torch.tensor([[0, 0, 1], [0, 1, 2], [0, 2, 3]])
    heads = torch.tensor([[1, 4, 5], [2, 4, 5], [2, 4, 5]])
    bodies = torch.tensor([[[0, 4, 6], [0, 6, 5]], [[0, 4, 5], [9, 9, 9]], [[0, 4, 6], [2, 6, 5]]])
    return KB(facts, heads, bodies, torch.tensor([2, 1, 2]), E=4, pad=9)


def test_toy_proofs():
    kb = _toy()
    q = torch.tensor([[1, 0, 2], [1, 0, 3], [2, 0, 3], [2, 3, 0]])        # gp(a,c), gp(a,d), anc(a,d), anc(d,a)
    p = SLD(kb, depth=6).prove(q)
    assert p.mask.sum(1).tolist() == [1, 0, 1, 0]
    b, slot = 2, int(p.mask[2].nonzero()[0])    # anc(a,d): rule 2, a fact, rule 2, a fact, rule 1 (-1: a fact step)
    assert p.rule[b, slot].tolist() == [2, -1, 2, -1, 1, -1]
    assert p.head[b, slot, ::2].tolist() == [[2, 0, 3], [2, 1, 3], [2, 2, 3]]
    assert p.body[b, slot, 0].tolist() == [[0, 0, 1], [2, 1, 3]]            # under every later binding


def test_derive_one_step():
    kb = _toy()
    children, count, next_var, rule = SLD(kb, depth=1).derive(torch.tensor([[[1, 0, 2]]]), torch.tensor([4]))
    assert count.tolist() == [1] and rule[0, 0].item() == 0                 # gp(a,c) against rule 0's head
    assert children[0, 0].tolist() == [[0, 0, 4], [0, 4, 2], [9, 9, 9]]       # parent(a,Z), parent(Z,c): Z renamed to 4
    assert next_var.tolist() == [5]


@pytest.mark.skipif(not (DATA / "countries_s3").is_dir(), reason="needs data-swarm")
def test_derive_matches_logic2rl():
    L2RL = pytest.importorskip("logic2rl.unification").SLD
    from grounder.kb import parse_rules, parse_triples
    facts = parse_triples(DATA / "countries_s3" / "train.txt")
    test = parse_triples(DATA / "countries_s3" / "test.txt")
    rel = {r: i for i, r in enumerate(sorted({t[0] for t in facts + test}))}
    ent = {e: i for i, e in enumerate(sorted({x for t in facts + test for x in t[1:]}))}
    rules = [r for r in parse_rules(DATA / "countries_s3" / "rules.txt") if all(a[0] in rel for a in (r[0], *r[1]))]
    base = KB.from_strings([(rel[r], ent[h], ent[t]) for r, h, t in facts], rules, ent, rel)
    up = lambda t: t + 1                                                      # noqa: E731  LOGIC2RL: 0 is padding
    on = (torch.arange(base.rules.M) < base.rules.lens.unsqueeze(1)).unsqueeze(-1)
    facts_t, heads = up(base.facts.atoms), up(base.rules.heads)
    bodies = torch.where(on, up(base.rules.bodies), 0)
    ref = L2RL(facts_t, torch.cat([heads.unsqueeze(1), bodies], 1), padding_idx=0, constant_no=base.E,
               n_runtime_vars=4096, device=torch.device("cpu"))
    sld = SLD(KB(facts_t, heads, bodies, base.rules.lens, E=base.E + 1, pad=0), depth=1, max_atoms=ref.max_atoms,
              max_states=ref.G)
    states = up(torch.tensor([[rel[r], ent[h], ent[t]] for r, h, t in test])).unsqueeze(1)
    nv, excl = torch.full((len(test),), base.E + 1), states.clone()
    for _ in range(3):
        a, b = ref.derive(states, nv, excl), sld.derive(states, nv, excl)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        open_ = (torch.arange(a[0].shape[1]) < a[1].unsqueeze(1)) & (a[0][..., 0] != 0).any(-1)
        n, s = open_.nonzero(as_tuple=True)
        states, nv, excl = a[0][n, s], a[2][n], excl[n]
