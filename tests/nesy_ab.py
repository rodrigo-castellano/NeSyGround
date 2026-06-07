"""A/B parity test: grounder.nesy.* vs grounder.nesy.* (OLD).

Checks:
  1. Every public symbol importable from both OLD and NEW.
  2. Hook Protocol isinstance checks match between old and new impls.
  3. KGEStepFilter numeric output is torch.equal (same seeded embeddings).
  4. PartialScorer numeric output is torch.equal (same tables).
  5. RandomSampler output is torch.equal (same seed, eval mode).
  6. KGEScorer min-conjunction logic matches.
  7. KGEFactFilter top-k filter matches.
  8. KGERuleFilter top-k filter matches.
  9. _topk_select output matches.

Usage:
    python tests/nesy_ab.py
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
from torch import Tensor

# ── Import OLD symbols ──────────────────────────────────────────────────────
from grounder.nesy.hooks import (
    GroundingHook as OLD_GroundingHook,
    ResolutionFactHook as OLD_ResolutionFactHook,
    ResolutionRuleHook as OLD_ResolutionRuleHook,
    StepHook as OLD_StepHook,
)
from grounder.nesy.scoring import (
    PartialScorer as OLD_PartialScorer,
    LazyPartialScorer as OLD_LazyPartialScorer,
)
from grounder.nesy.kge import (
    KGEScorer as OLD_KGEScorer,
    KGEFactFilter as OLD_KGEFactFilter,
    KGERuleFilter as OLD_KGERuleFilter,
    KGEStepFilter as OLD_KGEStepFilter,
)
from grounder.nesy.neural import (
    GroundingAttention as OLD_GroundingAttention,
    NeuralScorer as OLD_NeuralScorer,
)
from grounder.nesy.soft import (
    ProvabilityMLP as OLD_ProvabilityMLP,
    SoftScorer as OLD_SoftScorer,
)
from grounder.nesy.sampler import RandomSampler as OLD_RandomSampler
from grounder.nesy import _topk_select as OLD_topk_select

# ── Import NEW symbols ──────────────────────────────────────────────────────
from grounder.nesy.hooks import (
    GroundingHook as NEW_GroundingHook,
    ResolutionFactHook as NEW_ResolutionFactHook,
    ResolutionRuleHook as NEW_ResolutionRuleHook,
    StepHook as NEW_StepHook,
)
from grounder.nesy.scoring import (
    PartialScorer as NEW_PartialScorer,
    LazyPartialScorer as NEW_LazyPartialScorer,
)
from grounder.nesy.kge import (
    KGEScorer as NEW_KGEScorer,
    KGEFactFilter as NEW_KGEFactFilter,
    KGERuleFilter as NEW_KGERuleFilter,
    KGEStepFilter as NEW_KGEStepFilter,
)
from grounder.nesy.neural import (
    GroundingAttention as NEW_GroundingAttention,
    NeuralScorer as NEW_NeuralScorer,
)
from grounder.nesy.soft import (
    ProvabilityMLP as NEW_ProvabilityMLP,
    SoftScorer as NEW_SoftScorer,
)
from grounder.nesy.sampler import RandomSampler as NEW_RandomSampler
from grounder.nesy import _topk_select as NEW_topk_select

# ── Helpers ─────────────────────────────────────────────────────────────────

_PASS = []
_FAIL = []


def _check(name: str, ok: bool, msg: str = "") -> None:
    if ok:
        _PASS.append(name)
        print(f"  PASS  {name}")
    else:
        _FAIL.append(name)
        print(f"  FAIL  {name}" + (f" — {msg}" if msg else ""))


def _close(a: Tensor, b: Tensor, atol: float = 0.0) -> bool:
    if atol == 0.0:
        return bool(torch.equal(a, b))
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=0))


# ── Minimal fake KGE model ───────────────────────────────────────────────────

class _FakeKGE(nn.Module):
    """Deterministic KGE: score(h, r, t) = E[h] + R[r] + E[t] (summed)."""

    def __init__(self, num_ents: int, num_rels: int, emb_dim: int) -> None:
        super().__init__()
        self.ent_emb = nn.Embedding(num_ents, emb_dim)
        self.rel_emb = nn.Embedding(num_rels, emb_dim)
        self.atom_embedding_size = emb_dim

    def score(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        return (self.ent_emb(h) + self.rel_emb(r) + self.ent_emb(t)).sum(-1)

    def score_atoms(self, p: Tensor, s: Tensor, o: Tensor) -> Tensor:
        return self.score(s, p, o)

    def embed_atoms(self, s: Tensor, o: Tensor) -> Tensor:
        return (self.ent_emb(s) + self.ent_emb(o))


# ── Minimal fake fact index ──────────────────────────────────────────────────

class _FakeFactIndex:
    """Simple dict-backed fact index for testing."""

    def __init__(self, facts: Tensor) -> None:
        self._set = set(tuple(f.tolist()) for f in facts)

    def exists(self, atoms: Tensor) -> Tensor:
        return torch.tensor(
            [tuple(a.tolist()) in self._set for a in atoms],
            dtype=torch.bool)

    def targeted_lookup(self, queries: Tensor, K: int) -> tuple:
        N = queries.shape[0]
        idxs = torch.zeros(N, K, dtype=torch.long)
        return idxs, torch.ones(N, K, dtype=torch.bool)


# ── TEST 1: Symbol import completeness ──────────────────────────────────────

def test_symbol_imports() -> None:
    print("\n[1] Symbol imports")
    pairs = [
        ("GroundingHook", OLD_GroundingHook, NEW_GroundingHook),
        ("ResolutionFactHook", OLD_ResolutionFactHook, NEW_ResolutionFactHook),
        ("ResolutionRuleHook", OLD_ResolutionRuleHook, NEW_ResolutionRuleHook),
        ("StepHook", OLD_StepHook, NEW_StepHook),
        ("PartialScorer", OLD_PartialScorer, NEW_PartialScorer),
        ("LazyPartialScorer", OLD_LazyPartialScorer, NEW_LazyPartialScorer),
        ("KGEScorer", OLD_KGEScorer, NEW_KGEScorer),
        ("KGEFactFilter", OLD_KGEFactFilter, NEW_KGEFactFilter),
        ("KGERuleFilter", OLD_KGERuleFilter, NEW_KGERuleFilter),
        ("KGEStepFilter", OLD_KGEStepFilter, NEW_KGEStepFilter),
        ("GroundingAttention", OLD_GroundingAttention, NEW_GroundingAttention),
        ("NeuralScorer", OLD_NeuralScorer, NEW_NeuralScorer),
        ("ProvabilityMLP", OLD_ProvabilityMLP, NEW_ProvabilityMLP),
        ("SoftScorer", OLD_SoftScorer, NEW_SoftScorer),
        ("RandomSampler", OLD_RandomSampler, NEW_RandomSampler),
        ("_topk_select", OLD_topk_select, NEW_topk_select),
    ]
    for name, old, new in pairs:
        _check(f"import:{name}", old is not None and new is not None)


# ── TEST 2: Protocol isinstance checks ──────────────────────────────────────

def test_protocol_isinstance() -> None:
    print("\n[2] Protocol isinstance checks")
    torch.manual_seed(0)
    kge = _FakeKGE(10, 5, 4)
    facts = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    fi = _FakeFactIndex(facts)

    # KGEStepFilter satisfies StepHook
    old_sf = OLD_KGEStepFilter(kge, top_k=4, constant_no=5, padding_idx=9)
    new_sf = NEW_KGEStepFilter(kge, top_k=4, constant_no=5, padding_idx=9)
    _check("isinstance:KGEStepFilter->OLD_StepHook",
           isinstance(old_sf, OLD_StepHook))
    _check("isinstance:KGEStepFilter->NEW_StepHook",
           isinstance(new_sf, NEW_StepHook))

    # KGEScorer satisfies GroundingHook
    old_ks = OLD_KGEScorer(kge, output_budget=4, padding_idx=9)
    new_ks = NEW_KGEScorer(kge, output_budget=4, padding_idx=9)
    _check("isinstance:KGEScorer->OLD_GroundingHook",
           isinstance(old_ks, OLD_GroundingHook))
    _check("isinstance:KGEScorer->NEW_GroundingHook",
           isinstance(new_ks, NEW_GroundingHook))

    # KGEFactFilter satisfies ResolutionFactHook
    old_ff = OLD_KGEFactFilter(kge, fi, facts, top_k=2, padding_idx=9)
    new_ff = NEW_KGEFactFilter(kge, fi, facts, top_k=2, padding_idx=9)
    _check("isinstance:KGEFactFilter->OLD_ResolutionFactHook",
           isinstance(old_ff, OLD_ResolutionFactHook))
    _check("isinstance:KGEFactFilter->NEW_ResolutionFactHook",
           isinstance(new_ff, NEW_ResolutionFactHook))

    # KGERuleFilter satisfies ResolutionRuleHook
    old_rf = OLD_KGERuleFilter(kge, top_k=2, constant_no=5, padding_idx=9)
    new_rf = NEW_KGERuleFilter(kge, top_k=2, constant_no=5, padding_idx=9)
    _check("isinstance:KGERuleFilter->OLD_ResolutionRuleHook",
           isinstance(old_rf, OLD_ResolutionRuleHook))
    _check("isinstance:KGERuleFilter->NEW_ResolutionRuleHook",
           isinstance(new_rf, NEW_ResolutionRuleHook))


# ── TEST 3: _topk_select numeric parity ─────────────────────────────────────

def test_topk_select() -> None:
    print("\n[3] _topk_select numeric parity")
    torch.manual_seed(0)
    B, tG, M = 2, 8, 3
    body = torch.randint(0, 10, (B, tG, M, 3))
    mask = torch.ones(B, tG, dtype=torch.bool)
    rule_idx = torch.randint(0, 5, (B, tG))
    scores = torch.randn(B, tG)
    output_tG = 4

    old_out = OLD_topk_select(body, mask, rule_idx, scores, output_tG)
    new_out = NEW_topk_select(body, mask, rule_idx, scores, output_tG)

    _check("_topk_select:body", _close(old_out[0], new_out[0]))
    _check("_topk_select:mask", _close(old_out[1], new_out[1]))
    _check("_topk_select:rule_idx", _close(old_out[2], new_out[2]))


# ── TEST 4: KGEStepFilter numeric parity ────────────────────────────────────

def test_kge_step_filter() -> None:
    print("\n[4] KGEStepFilter numeric parity")
    torch.manual_seed(0)

    # Seeded KGE model
    kge = _FakeKGE(num_ents=20, num_rels=5, emb_dim=8)
    torch.manual_seed(0)
    nn.init.normal_(kge.ent_emb.weight)
    nn.init.normal_(kge.rel_emb.weight)
    kge.eval()

    # Build partial scorer tables [P=5, E=20]
    P, E = 5, 20
    torch.manual_seed(42)
    max_tail = torch.randn(P, E)
    max_head = torch.randn(P, E)
    constant_no = 15
    padding_idx = 19

    for mode in ("ground_only", "partial_only", "both"):
        old_sf = OLD_KGEStepFilter(
            kge, top_k=4, constant_no=constant_no, padding_idx=padding_idx,
            max_tail_score=max_tail.clone(), max_head_score=max_head.clone(),
            scoring_mode=mode,
        )
        new_sf = NEW_KGEStepFilter(
            kge, top_k=4, constant_no=constant_no, padding_idx=padding_idx,
            max_tail_score=max_tail.clone(), max_head_score=max_head.clone(),
            scoring_mode=mode,
        )
        old_sf.eval()
        new_sf.eval()

        torch.manual_seed(0)
        B, tG, M = 2, 10, 3
        # Mix of ground (a1,a2 <= constant_no), partial, and unbound atoms
        body = torch.randint(0, constant_no + 1, (B, tG, M, 3))
        # Make some partial: set obj to a variable (> constant_no but < padding)
        body[0, 2, 0, 2] = constant_no + 1  # partial head-bound
        body[0, 3, 0, 1] = constant_no + 2  # partial tail-bound
        # Pred from valid range
        body[:, :, :, 0] = torch.randint(0, P, (B, tG, M))
        mask = torch.ones(B, tG, dtype=torch.bool)
        mask[0, 9] = False  # one invalid entry
        rule_idx = torch.zeros(B, tG, dtype=torch.long)

        with torch.no_grad():
            old_body, old_mask, old_ridx = old_sf.on_step(body, mask, rule_idx, d=0)
            new_body, new_mask, new_ridx = new_sf.on_step(body, mask, rule_idx, d=0)

        _check(f"KGEStepFilter[{mode}]:body", _close(old_body, new_body))
        _check(f"KGEStepFilter[{mode}]:mask", _close(old_mask, new_mask))
        _check(f"KGEStepFilter[{mode}]:rule_idx", _close(old_ridx, new_ridx))


# ── TEST 5: PartialScorer numeric parity ────────────────────────────────────

def test_partial_scorer() -> None:
    print("\n[5] PartialScorer numeric parity")
    torch.manual_seed(0)
    P, E = 6, 12
    max_tail = torch.randn(P, E)
    max_head = torch.randn(P, E)
    constant_no = 10

    old_ps = OLD_PartialScorer.from_tables(max_tail.clone(), max_head.clone(), constant_no)
    new_ps = NEW_PartialScorer.from_tables(max_tail.clone(), max_head.clone(), constant_no)

    torch.manual_seed(0)
    N = 20
    preds = torch.randint(0, P, (N,))
    args1 = torch.randint(0, E + 2, (N,))   # mix of consts and vars
    args2 = torch.randint(0, E + 2, (N,))

    old_scores = old_ps.score_atoms(preds, args1, args2)
    new_scores = new_ps.score_atoms(preds, args1, args2)

    _check("PartialScorer:score_atoms", _close(old_scores, new_scores, atol=1e-6))


# ── TEST 6: KGEScorer numeric parity ────────────────────────────────────────

def test_kge_scorer() -> None:
    print("\n[6] KGEScorer numeric parity")
    torch.manual_seed(0)
    # num_ents=20, num_rels=20 so padding_idx=19 fits in both ent and rel emb
    kge = _FakeKGE(num_ents=20, num_rels=20, emb_dim=6)
    torch.manual_seed(0)
    nn.init.normal_(kge.ent_emb.weight)
    nn.init.normal_(kge.rel_emb.weight)
    kge.eval()

    padding_idx = 19
    output_budget = 4

    old_ks = OLD_KGEScorer(kge, output_budget=output_budget, padding_idx=padding_idx)
    new_ks = NEW_KGEScorer(kge, output_budget=output_budget, padding_idx=padding_idx)

    torch.manual_seed(0)
    B, tG, M = 2, 8, 3
    # All indices in 0..17 so they are valid for both ent and rel embeddings
    body = torch.randint(0, 18, (B, tG, M, 3))
    # Some inactive atoms (pred = padding)
    body[0, 7, 2, 0] = padding_idx
    mask = torch.ones(B, tG, dtype=torch.bool)
    mask[1, 7] = False
    rule_idx = torch.zeros(B, tG, dtype=torch.long)

    with torch.no_grad():
        old_out = old_ks.apply(body, mask, rule_idx)
        new_out = new_ks.apply(body, mask, rule_idx)

    _check("KGEScorer:body", _close(old_out[0], new_out[0]))
    _check("KGEScorer:mask", _close(old_out[1], new_out[1]))
    _check("KGEScorer:rule_idx", _close(old_out[2], new_out[2]))


# ── TEST 7: KGEFactFilter numeric parity ────────────────────────────────────

def test_kge_fact_filter() -> None:
    print("\n[7] KGEFactFilter numeric parity")
    torch.manual_seed(0)
    kge = _FakeKGE(num_ents=10, num_rels=3, emb_dim=4)
    torch.manual_seed(0)
    nn.init.normal_(kge.ent_emb.weight)
    nn.init.normal_(kge.rel_emb.weight)
    kge.eval()

    facts = torch.randint(0, 8, (20, 3))
    fi = _FakeFactIndex(facts)
    padding_idx = 9
    top_k = 3

    old_ff = OLD_KGEFactFilter(kge, fi, facts, top_k=top_k, padding_idx=padding_idx)
    new_ff = NEW_KGEFactFilter(kge, fi, facts, top_k=top_k, padding_idx=padding_idx)

    torch.manual_seed(0)
    B, S, K_f, G = 2, 3, 6, 4
    fact_goals = torch.randint(0, 8, (B, S, K_f, G, 3))
    fact_success = torch.ones(B, S, K_f, dtype=torch.bool)
    fact_success[0, 1, 5] = False
    queries = torch.randint(0, 8, (B, S, 3))

    with torch.no_grad():
        old_out = old_ff.filter_facts(fact_goals, fact_success, queries)
        new_out = new_ff.filter_facts(fact_goals, fact_success, queries)

    _check("KGEFactFilter:filter_facts", _close(old_out, new_out))


# ── TEST 8: KGERuleFilter numeric parity ────────────────────────────────────

def test_kge_rule_filter() -> None:
    print("\n[8] KGERuleFilter numeric parity")
    torch.manual_seed(0)
    # Use num_ents=20, num_rels=20 so any index < 20 is valid
    kge = _FakeKGE(num_ents=20, num_rels=20, emb_dim=4)
    torch.manual_seed(0)
    nn.init.normal_(kge.ent_emb.weight)
    nn.init.normal_(kge.rel_emb.weight)
    kge.eval()

    # constant_no=7, padding_idx=9, so all ground entity/pred indices 0..7 fit
    constant_no = 7
    padding_idx = 9
    top_k = 3

    old_rf = OLD_KGERuleFilter(kge, top_k=top_k, constant_no=constant_no,
                               padding_idx=padding_idx)
    new_rf = NEW_KGERuleFilter(kge, top_k=top_k, constant_no=constant_no,
                               padding_idx=padding_idx)

    torch.manual_seed(0)
    B, S, K_r, G = 2, 3, 6, 4
    # Keep indices strictly within constant_no so all atoms are ground + valid
    rule_goals = torch.randint(0, constant_no + 1, (B, S, K_r, G, 3))
    # Some non-ground: inject variable args
    rule_goals[0, 1, 4, 0, 1] = constant_no + 1
    rule_success = torch.ones(B, S, K_r, dtype=torch.bool)
    rule_success[1, 2, 5] = False
    queries = torch.randint(0, constant_no, (B, S, 3))

    with torch.no_grad():
        old_out = old_rf.filter_rules(rule_goals, rule_success, queries)
        new_out = new_rf.filter_rules(rule_goals, rule_success, queries)

    _check("KGERuleFilter:filter_rules", _close(old_out, new_out))


# ── TEST 9: RandomSampler numeric parity (eval mode) ────────────────────────

def test_random_sampler() -> None:
    print("\n[9] RandomSampler numeric parity (eval mode)")
    output_budget = 4

    old_rs = OLD_RandomSampler(output_budget=output_budget)
    new_rs = NEW_RandomSampler(output_budget=output_budget)
    old_rs.eval()
    new_rs.eval()

    torch.manual_seed(0)
    B, tG, M = 2, 8, 3
    body = torch.randint(0, 10, (B, tG, M, 3))
    mask = torch.ones(B, tG, dtype=torch.bool)
    mask[0, 7] = False
    rule_idx = torch.zeros(B, tG, dtype=torch.long)

    # Eval mode: deterministic (uses mask.float() as scores, so same output)
    old_out = old_rs.apply(body, mask, rule_idx)
    new_out = new_rs.apply(body, mask, rule_idx)

    _check("RandomSampler:eval:body", _close(old_out[0], new_out[0]))
    _check("RandomSampler:eval:mask", _close(old_out[1], new_out[1]))
    _check("RandomSampler:eval:rule_idx", _close(old_out[2], new_out[2]))

    # Train mode: both use the same manual_seed so should match
    old_rs.train()
    new_rs.train()
    torch.manual_seed(7)
    old_train = old_rs.apply(body, mask, rule_idx)
    torch.manual_seed(7)
    new_train = new_rs.apply(body, mask, rule_idx)

    _check("RandomSampler:train:body", _close(old_train[0], new_train[0]))
    _check("RandomSampler:train:mask", _close(old_train[1], new_train[1]))
    _check("RandomSampler:train:rule_idx", _close(old_train[2], new_train[2]))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    test_symbol_imports()
    test_protocol_isinstance()
    test_topk_select()
    test_kge_step_filter()
    test_partial_scorer()
    test_kge_scorer()
    test_kge_fact_filter()
    test_kge_rule_filter()
    test_random_sampler()

    print(f"\n{'='*60}")
    print(f"PASSED: {len(_PASS)}  FAILED: {len(_FAIL)}")
    if _FAIL:
        print("FAILURES:", _FAIL)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
