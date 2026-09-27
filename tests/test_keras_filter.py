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


def _groundings(grounder_type: str, queries):
    vocab = SimpleNamespace(relation2id={p: i for i, p in enumerate(PREDS)}, entity2id={f"e{i}": i for i in range(4)},
                            rule_names=["r0"])
    fi = TensorFactIndex(FACTS, num_predicates=len(PREDS), num_entities=4, device=DEV, max_facts_per_query=8)
    rg = create_grounder(grounder_type, fact_index=fi, rules=RULES, kb=vocab, max_groundings=32,
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


@pytest.mark.parametrize("grounder_type, rounds", [("enum.keras.w1.d2.flat", None), ("enum.keras.w1.d2.r2.flat", 2)])
def test_keras_walk_rounds_parse(grounder_type, rounds):
    """``.r<N>``: the proof walk's rounds (the later keras-ns walks depth rounds, the IJCAI-25 code depth - 1)."""
    rg, _ = _groundings(grounder_type, [(0, A, Z)])
    assert rg._inner.keras_rounds == rounds


@pytest.mark.parametrize("grounder_type", ["enum.keras.w1.d2.flat", "enum.keras.w1.d3.flat", "enum.keras.w1.d3.r3.flat"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_keras_prefilter_keeps_the_walks_output(grounder_type, seed, monkeypatch):
    """Dropping the groundings no chain from the all-fact ones can prove (``fast._keras_provable``) before keras-ns's
    proof walk changes nothing it outputs: recursive and one-body rules, several batches at once."""
    from grounder.backward import fast
    from test_fast import RULES, _setup
    rules = [(h, b) for h, b in RULES if len(b) == 1 or all(set(a[1:]) - set(h[1:]) for a in b)]   # keras's
    fi, vocab, queries = _setup(seed)
    vocab.rule_names = [f"r{i}" for i in range(len(rules))]
    rg = create_grounder(grounder_type, fact_index=fi, rules=rules, kb=vocab, max_groundings=32,
                         max_total_groundings=64, provable_set_method="spmm", device=DEV)
    qs = torch.stack([queries, queries.flip(0), queries.roll(17, 0)])
    mask = torch.rand(qs.shape[:2], generator=torch.Generator().manual_seed(seed)).to(DEV) < 0.9
    outs, sizes, provable = {}, [], fast._keras_provable

    def counted(g, rule, *args):
        out = provable(g, rule, *args)
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
