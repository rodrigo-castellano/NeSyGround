"""Forward.ground: each derived query's witness — the first rule whose body grounds, its least binding."""
from __future__ import annotations

import torch

from grounder.forward import Forward
from grounder.kb import KB

# entities a..e = 0..4; predicates: parent 0, sibling 1, grandparent 2, ancestor 3
ENT = {e: i for i, e in enumerate("abcde")}
REL = {"parent": 0, "sibling": 1, "grandparent": 2, "ancestor": 3}
FACTS = [("parent", "a", "b"), ("parent", "a", "c"), ("parent", "b", "d"), ("parent", "c", "d"), ("parent", "d", "e"),
         ("sibling", "b", "c")]
RULES = [(("grandparent", "X", "Y"), [("parent", "X", "Z"), ("parent", "Z", "Y")]),
         (("ancestor", "X", "Y"), [("parent", "X", "Y")]),
         (("ancestor", "X", "Y"), [("parent", "X", "Z"), ("ancestor", "Z", "Y")])]


def _forward():
    facts = torch.tensor([[REL[r], ENT[h], ENT[t]] for r, h, t in FACTS])
    return Forward(KB.from_strings(facts, RULES, ENT, REL, device="cpu"))


def _rows(g):
    """Per pool, the (rule, head, body atoms) of its groundings."""
    out = []
    for s in range(g.offsets.shape[0]):
        p = g.pool(s)
        out.append({(int(r), tuple(p.atoms[h].tolist()), tuple(tuple(p.atoms[b].tolist()) for b, ok in zip(bs, m) if ok))
                    for r, h, bs, m in zip(p.rule, p.head, p.body, p.body_mask)})
    return out


def test_witnesses():
    fw = _forward()
    q = lambda r, h, t: [REL[r], ENT[h], ENT[t]]  # noqa: E731
    queries = torch.tensor([[q("grandparent", "a", "d"), q("ancestor", "a", "e"), q("parent", "a", "b"),
                             q("grandparent", "a", "e")],
                            [q("ancestor", "b", "d"), q("ancestor", "e", "a"), q("grandparent", "a", "d"),
                             q("grandparent", "a", "d")]])
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    rule_of = {i: fw.kb.rules.order.tolist().index(i) for i in range(len(RULES))}     # input rule -> KB rule
    g = _forward().ground(queries, mask)
    a, b, c, d, e = range(5)
    P, GP, AN = REL["parent"], REL["grandparent"], REL["ancestor"]
    assert _rows(g) == [
        {(rule_of[0], (GP, a, d), ((P, a, b), (P, b, d))),                  # Z = b, the least of b and c
         (rule_of[2], (AN, a, e), ((P, a, b), (AN, b, e)))},                # not a parent fact: the recursive rule
        {(rule_of[1], (AN, b, d), ((P, b, d),)),                            # the first rule that grounds
         (rule_of[0], (GP, a, d), ((P, a, b), (P, b, d)))}]                 # its own pool's grounding
    assert g.atoms[g.query[0, 2]].tolist() == q("parent", "a", "b")        # a fact no rule derives: a query, no row
