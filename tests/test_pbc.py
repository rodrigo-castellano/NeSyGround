"""PBC on Countries S3 and Family: the Full grounder and the guide, by their properties (GPU)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from grounder.kb import KB, parse_rules, parse_triples
from grounder.pbc import PBC, Guide

DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "repos/data-swarm/main"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="PBC's kernels need the GPU")


def _kb(name: str):
    split = {s: parse_triples(DATA / name / f"{s}.txt") for s in ("train", "valid", "test")}
    every = [t for ts in split.values() for t in ts]
    rel = {r: i for i, r in enumerate(sorted({t[0] for t in every}))}
    ent = {e: i for i, e in enumerate(sorted({x for t in every for x in t[1:]}))}
    rules = [r for r in parse_rules(DATA / name / "rules.txt") if all(a[0] in rel for a in (r[0], *r[1]))]
    ids = lambda ts: torch.tensor([[rel[r], ent[h], ent[t]] for r, h, t in ts], device="cuda")  # noqa: E731
    return KB.from_strings(ids(split["train"]), rules, ent, rel, device="cuda"), ids(split["test"])


def _rows(g):
    """The groundings as a set of (pool, rule, head atom, body atoms) tuples."""
    pool = torch.repeat_interleave(torch.arange(g.offsets.shape[0], device=g.rule.device),
                                   g.offsets[:, -1] - g.offsets[:, 0])
    t = torch.cat([pool[:, None], g.rule[:, None], g.atoms[g.head], g.atoms[g.body].flatten(1)], 1)
    return set(map(tuple, t.tolist()))


class _Score:
    """A deterministic prior: an atom's score from its ids."""
    def score(self, atoms):
        return ((atoms * torch.tensor([7, 13, 31], device=atoms.device)).sum(1) % 97 + 1).float() / 98


@pytest.mark.parametrize("name", ["countries_s3", "family"])
def test_full_contains_facts_and_equals_them_at_width_0(name):
    kb, q = _kb(name)
    q = q[:64]
    for depth, width in ((1, 0), (2, 1)):
        facts = _rows(PBC(kb, depth=depth, width=width).ground(q))
        full = _rows(PBC(kb, depth=depth, width=width, bind="all").ground(q))
        assert facts <= full
        if width == 0:
            assert facts == full


@pytest.mark.parametrize("name", ["countries_s3", "family"])
def test_guide(name):
    kb, q = _kb(name)
    q = q[:128]
    for prune in ("fp_batch", "keras"):
        plain = _rows(PBC(kb, depth=3, width=1, prune=prune).ground(q))
        assert _rows(PBC(kb, depth=3, width=1, prune=prune, guide=Guide(_Score(), k=10 ** 9)).ground(q)) == plain
        capture = []
        few = PBC(kb, depth=3, width=1, prune=prune, guide=Guide(_Score(), k=1, capture=capture)).ground(q)
        assert _rows(few) <= plain and capture and all(bool((c["kept"] | ~c["exempt"]).all()) for c in capture)
        again = PBC(kb, depth=3, width=1, prune=prune, guide=Guide(_Score(), k=1)).ground(q)
        assert _rows(again) == _rows(few)                       # deterministic


@pytest.mark.parametrize("name", ["countries_s3", "family"])
@pytest.mark.parametrize("prune", ["fp_batch", "keras"])
def test_chunked_steps_change_nothing(name, prune, monkeypatch):
    """A step walks its candidate rows in chunks (``engine.ROWS``): tiny chunks give the same groundings."""
    from grounder.pbc import engine
    kb, q = _kb(name)
    whole = PBC(kb, depth=3, width=1, prune=prune).ground(q[:256])
    monkeypatch.setattr(engine, "ROWS", 37)
    chunked = PBC(kb, depth=3, width=1, prune=prune).ground(q[:256])
    for a, b in zip(whole, chunked):
        assert torch.equal(a, b)
