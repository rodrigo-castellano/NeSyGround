"""PBC (fp_batch) against a brute-force reference on random KBs (GPU).

The reference enumerates every binding of a rule's free variables over all entities and keeps the ones PBC defines:
some anchor variant generates it (the atoms that variant looks up are facts), at most ``width`` body atoms are not
facts (0 at the last step), none is the goal itself, and an unknown atom's predicate is some rule's head (it can be
proved later). The unknown atoms are the next step's goals; at the end a grounding is kept when its body is proved
within ``depth`` rounds of propagation from the facts, over the pool's groundings.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from grounder.kb import KB
from grounder.pbc import PBC

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="PBC's kernels need the GPU")
E, P = 9, 4
X, Y, Z = E, E + 1, E + 2                                 # variable ids
PAD = E + 3
RULES = [                                                 # (head pred, body [(pred, arg0, arg1)])
    (2, [(0, X, Z), (1, Z, Y)]),                          # a chain
    (3, [(2, Y, X)]),                                     # one body atom, no free variable
    (1, [(3, X, Y), (0, Y, Z)]),                          # a free variable in one atom only
    (2, [(2, Y, X)]),                                     # recursive
    (3, [(0, X, Y), (2, X, Y)]),                          # both atoms bound by the head
]


def _kb(seed: int):
    g = torch.Generator().manual_seed(seed)
    facts = torch.unique(torch.stack([torch.randint(0, 2, (45,), generator=g), torch.randint(0, E, (45,), generator=g),
                                      torch.randint(0, E, (45,), generator=g)], 1), dim=0)
    heads = torch.tensor([[h, X, Y] for h, _ in RULES])
    bodies = torch.tensor([b + [(PAD, PAD, PAD)] * (2 - len(b)) for _, b in RULES])
    lens = torch.tensor([len(b) for _, b in RULES])
    kb = KB(facts, heads, bodies, lens, E=E, pad=PAD, P=P, device="cuda")
    queries = torch.cat([facts[:10], torch.stack([torch.randint(0, P, (40,), generator=g),
                                                 torch.randint(0, E, (40,), generator=g),
                                                 torch.randint(0, E, (40,), generator=g)], 1)])
    return kb, facts, queries


def _variants(kb):
    """Per rule, each anchor variant's looked-up body positions (given order)."""
    out = {}
    for p in kb.patterns():
        out[p.rule_idx] = []
        for j in range(p.num_body):
            bps, _, meta = p.reorder_with_anchor(j)
            given = [p._orig_body_patterns.index(bp) for bp in bps]
            out[p.rule_idx].append({given[i] for i, m in enumerate(meta) if m["introduces_fv"] >= 0})
    return out


def _reference(kb, facts, queries, depth: int, width: int):
    F = {tuple(f) for f in facts.tolist()}
    heads = {h for h, _ in RULES}
    variants = _variants(kb)
    out, goals = [], sorted({tuple(q) for q in queries.tolist()})
    for k in range(depth):
        last, w, nxt = k == depth - 1, (0 if k == depth - 1 else width), set()
        for goal in goals:
            for r, (hp, body) in enumerate(RULES):
                if hp != goal[0]:
                    continue
                free = sorted({a for atom in body for a in atom[1:] if a >= E} - {X, Y})
                for vals in itertools.product(range(E), repeat=len(free)):
                    env = {X: goal[1], Y: goal[2], **dict(zip(free, vals))}
                    atoms = [(q, env.get(a, a), env.get(b, b)) for q, a, b in body]
                    unknown = [i for i, a in enumerate(atoms) if a not in F]
                    if not any(all(atoms[i] in F for i in v) for v in variants[r]) or len(unknown) > w \
                            or goal in atoms or any(atoms[i][0] not in heads for i in unknown):
                        continue
                    out.append((r, goal, tuple(atoms)))
                    if not last:
                        nxt |= {atoms[i] for i in unknown}
        goals = sorted(nxt)
    proved = set()
    for _ in range(max(1, depth)):
        proved = {h for _, h, b in out if all(a in F or a in proved for a in b)}
    return {(r, h, b) for r, h, b in out if all(a in F or a in proved for a in b)}


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("depth,width", [(1, 0), (2, 1), (3, 1)])
def test_pbc_matches_the_reference(seed, depth, width):
    kb, facts, queries = _kb(seed)
    g = PBC(kb, depth=depth, width=width).ground(queries.cuda())
    atoms = g.atoms.tolist()
    got = {(r, tuple(atoms[h]), tuple(tuple(atoms[b]) for b, ok in zip(body, mask) if ok))
           for r, h, body, mask in zip(g.rule.tolist(), g.head.tolist(), g.body.tolist(), g.body_mask.tolist())}
    assert got == _reference(kb, facts, queries, depth, width)
    assert torch.equal(g.atoms[g.query[0]].cpu(), queries)              # every query has a row
