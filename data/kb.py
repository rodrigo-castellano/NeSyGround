"""KB — the immutable knowledge base: facts + rules + indices.

Read-only and resolution-agnostic: one KB is shared across grounders (sld/rtf/pbc
each build their own resolver on top). It holds the generic ``SldRuleIndex``
(segment lookup); the pbc resolver builds its own ``PbcRuleIndex`` from the same
rule tensors. ``with_closure`` returns a NEW KB with extra (forward-derived)
facts — the KB itself is never mutated.

    kb = KB(facts, heads, bodies, lens, constant_no=C, predicate_no=P, padding_idx=pad, device=dev)
    g1 = BackwardGrounder(kb, ...);  g2 = BackwardGrounder(kb, resolution="rtf", ...)
"""
from __future__ import annotations

import hashlib
from typing import Hashable, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.data import fact_index as fact_index_mod
from grounder.data import rule_index as rule_index_mod
from grounder.data.encoding import Encoding

FactIndexType = Literal["arg_key", "inverted", "block_sparse"]

# Module-level index memo: a facts-only with_program reuses the rule-derived
# SldRuleIndex (+ the KB binding-tables) keyed on the rule-program signature, so
# the magic-set hot path (seeds-only union) does NOT re-tensorize per-rule tables.
_RULE_INDEX_CACHE: dict = {}


def _rule_program_signature(heads: Tensor, bodies: Tensor, lens: Tensor,
                            *, predicate_no: int, padding_idx: int,
                            rule_order: str, order_seed: int,
                            pack_base: Optional[int], device) -> str:
    """Content hash over everything SldRuleIndex/binding-tables depend on.

    Facts are EXCLUDED (facts-only rewrites must hit the same key); a rule-changing
    rewrite alters heads/bodies/lens and therefore the signature -> cache miss.
    """
    h = hashlib.sha256()
    for t in (heads, bodies, lens):
        c = t.detach().to("cpu", torch.long).contiguous()
        h.update(str(tuple(c.shape)).encode())
        h.update(c.numpy().tobytes())
    h.update(repr((int(predicate_no), int(padding_idx), str(rule_order),
                   int(order_seed), pack_base, str(device))).encode())
    return h.hexdigest()


class KB(nn.Module):
    """Immutable facts + rules + fact/rule indices."""

    def __init__(
        self,
        facts_idx: Tensor,
        rules_heads_idx: Tensor,
        rules_bodies_idx: Tensor,
        rule_lens: Tensor,
        *,
        constant_no: int,
        predicate_no: int,
        padding_idx: int,
        device: torch.device,
        fact_index_type: FactIndexType = "arg_key",
        max_facts_per_query: int = 64,
        max_memory_mb: int = 256,
        fact_order: Literal["original", "shuffle"] = "original",
        rule_order: Literal["original", "shuffle"] = "original",
        order_seed: int = 42,
        pack_base: Optional[int] = None,
        _cached_rule_index=None,
        _cached_binding_tables=None,
    ) -> None:
        super().__init__()
        if facts_idx.numel() == 0:
            raise ValueError("facts_idx is empty — a KB must have at least one fact")
        if rules_heads_idx.shape[0] == 0:
            raise ValueError("rules_heads_idx is empty — a KB must have at least one rule")

        self.constant_no = int(constant_no)
        self.predicate_no = int(predicate_no)
        self.padding_idx = int(padding_idx)
        self.device_ = device
        self.encoding = Encoding.from_constant_no(self.constant_no, self.padding_idx)

        # Index tables are sized predicate_no + 1; padding appears in predicate
        # slots of inactive states, so predicate_no must cover padding_idx.
        index_predicate_no = max(predicate_no, padding_idx)

        facts_idx = facts_idx.to(device=device, dtype=torch.long)
        rules_heads_idx = rules_heads_idx.to(device=device, dtype=torch.long)
        rules_bodies_idx = rules_bodies_idx.to(device=device, dtype=torch.long)
        rule_lens = rule_lens.to(device=device, dtype=torch.long)
        self.M = int(rule_lens.max().item())

        # Remember construction config so with_closure rebuilds identically.
        self._index_cfg = dict(
            fact_index_type=fact_index_type, max_facts_per_query=max_facts_per_query,
            max_memory_mb=max_memory_mb, fact_order=fact_order, rule_order=rule_order,
            order_seed=order_seed, pack_base=pack_base)
        self.register_buffer("rules_heads_idx", rules_heads_idx)
        self.register_buffer("rules_bodies_idx", rules_bodies_idx)
        self.register_buffer("rule_lens", rule_lens)

        self.fact_index = fact_index_mod.create(
            facts_idx, type=fact_index_type,
            constant_no=constant_no, predicate_no=index_predicate_no,
            padding_idx=padding_idx, device=device, pack_base=pack_base,
            max_facts_per_query=max_facts_per_query, max_memory_mb=max_memory_mb,
            order=fact_order, order_seed=order_seed)
        # Rule index: reuse the cached one on a facts-only rewrite (the rule
        # program is unchanged), else build it (the byte-identity default path).
        if _cached_rule_index is not None:
            self.rule_index = _cached_rule_index
            if _cached_binding_tables is not None:
                self._binding_tables_cache = _cached_binding_tables
        else:
            self.rule_index = rule_index_mod.create(
                rules_heads_idx, rules_bodies_idx, rule_lens, resolution="sld",
                predicate_no=index_predicate_no, padding_idx=padding_idx, device=device,
                order=rule_order, order_seed=order_seed)

        self.num_rules = rules_heads_idx.shape[0]
        self.K_r = self.rule_index.max_rule_pairs
        self.K_f = self.fact_index.max_fact_pairs

    def binding_tables(self, M: int, pad: int) -> dict:
        """Lazy per-rule variable-binding constraints (memoized; pure fn of the KB)."""
        cache = getattr(self, "_binding_tables_cache", None)
        num_rules = int(self.num_rules)
        if cache is not None and cache["num_rules"] == num_rules and cache["M"] == M:
            return cache
        device = self.device_
        ri = self.rule_index
        heads = ri.rules_heads.to("cpu")
        bodies = ri.rules_bodies.to("cpu")
        lens = ri.rule_lens.to("cpu")
        Nslot = 2 + 2 * M
        head_pred = torch.full((num_rules,), pad, dtype=torch.long)
        body_pred = torch.full((num_rules, M), pad, dtype=torch.long)
        slot_active = torch.zeros((num_rules, Nslot), dtype=torch.bool)
        canon_src = torch.arange(Nslot).unsqueeze(0).repeat(num_rules, 1).long()
        for r in range(num_rules):
            L = int(lens[r])
            head_pred[r] = int(heads[r, 0])
            var = [-(s + 1) for s in range(Nslot)]
            var[0] = int(heads[r, 1]); var[1] = int(heads[r, 2])
            slot_active[r, 0] = slot_active[r, 1] = True
            for m in range(M):
                body_pred[r, m] = int(bodies[r, m, 0]) if m < L else pad
                if m < L:
                    var[2 + 2 * m] = int(bodies[r, m, 1])
                    var[3 + 2 * m] = int(bodies[r, m, 2])
                    slot_active[r, 2 + 2 * m] = slot_active[r, 3 + 2 * m] = True
            first = {}
            for s in range(Nslot):
                if not bool(slot_active[r, s]):
                    continue
                v = var[s]
                if v in first:
                    canon_src[r, s] = first[v]
                else:
                    first[v] = s
                    canon_src[r, s] = s
        cache = {
            "num_rules": num_rules, "M": M,
            "head_pred": head_pred.to(device), "body_pred": body_pred.to(device),
            "slot_active": slot_active.to(device), "canon_src": canon_src.to(device),
        }
        self._binding_tables_cache = cache
        return cache

    @property
    def num_facts(self) -> int:
        return self.fact_index.num_facts

    @property
    def facts_idx(self) -> Tensor:
        """The (sorted) fact triples, via the fact index."""
        return self.fact_index.facts_idx

    def real_num_predicates(self) -> int:
        """Max predicate id across facts + rule heads + (non-pad) rule bodies, +1 —
        the TRUE range, ignoring the loader's inflated ``predicate_no`` padding."""
        ids = [self.fact_index.facts_idx[:, 0], self.rules_heads_idx[:, 0]]
        body_preds = self.rules_bodies_idx[..., 0].reshape(-1)
        ids.append(body_preds[body_preds != self.padding_idx])
        allp = torch.cat([t for t in ids if t.numel() > 0])
        return int(allp.max().item()) + 1

    def _rule_signature(self, heads: Tensor, bodies: Tensor, lens: Tensor) -> str:
        index_predicate_no = max(self.predicate_no, self.padding_idx)
        return _rule_program_signature(
            heads, bodies, lens, predicate_no=index_predicate_no,
            padding_idx=self.padding_idx, rule_order=self._index_cfg["rule_order"],
            order_seed=self._index_cfg["order_seed"],
            pack_base=self._index_cfg["pack_base"], device=self.device_)

    def with_program(
        self,
        *,
        facts: Optional[Tensor] = None,
        heads: Optional[Tensor] = None,
        bodies: Optional[Tensor] = None,
        lens: Optional[Tensor] = None,
        index_cache_key: Optional[Hashable] = None,
    ) -> "KB":
        """Return a NEW KB rewriting facts and/or the rule program (never mutates).

        FACTS-ONLY (heads/bodies/lens all None): the rule program is unchanged, so
        the new KB REUSES this KB's rule-derived SldRuleIndex + binding tables via
        the module-level memo (no per-rule re-tensorization — the magic-set hot
        path). Facts union this KB's facts (== ``with_closure``).

        RULE-CHANGING (any of heads/bodies/lens given): the rule program changes;
        indices are re-derived in ``__init__`` and the memo is (re)populated for
        the new signature. Requires all three of heads/bodies/lens together.
        """
        rule_change = any(t is not None for t in (heads, bodies, lens))
        if rule_change and not all(t is not None for t in (heads, bodies, lens)):
            raise ValueError("with_program: heads, bodies, lens must be given together")

        new_heads = heads if heads is not None else self.rules_heads_idx
        new_bodies = bodies if bodies is not None else self.rules_bodies_idx
        new_lens = lens if lens is not None else self.rule_lens
        new_heads = new_heads.to(device=self.device_, dtype=torch.long)
        new_bodies = new_bodies.to(device=self.device_, dtype=torch.long)
        new_lens = new_lens.to(device=self.device_, dtype=torch.long)

        if facts is not None:
            extra = facts.to(device=self.device_, dtype=torch.long)
            merged = torch.unique(torch.cat([self.fact_index.facts_idx, extra], 0), dim=0)
        else:
            merged = self.fact_index.facts_idx

        # Cache key: explicit override, else the rule-program content signature.
        sig = (index_cache_key if index_cache_key is not None
               else self._rule_signature(new_heads, new_bodies, new_lens))

        cached_ri, cached_bt = None, None
        if not rule_change:
            # Facts-only: reuse THIS KB's indices directly (same rule program).
            cached_ri = self.rule_index
            cached_bt = getattr(self, "_binding_tables_cache", None)
            _RULE_INDEX_CACHE.setdefault(sig, (cached_ri, cached_bt))
        else:
            hit = _RULE_INDEX_CACHE.get(sig)
            if hit is not None:
                cached_ri, cached_bt = hit

        new_kb = KB(merged, new_heads, new_bodies, new_lens,
                    constant_no=self.constant_no, predicate_no=self.predicate_no,
                    padding_idx=self.padding_idx, device=self.device_,
                    _cached_rule_index=cached_ri, _cached_binding_tables=cached_bt,
                    **self._index_cfg)
        if cached_ri is None:
            # Rule-changing MISS: memoize the freshly-built index for later reuse.
            _RULE_INDEX_CACHE.setdefault(
                sig, (new_kb.rule_index, getattr(new_kb, "_binding_tables_cache", None)))
        return new_kb

    def with_closure(self, closure_facts: Tensor) -> "KB":
        """Return a NEW KB whose facts are this KB's ∪ ``closure_facts``.

        Delegates to ``with_program(facts=...)`` — behavior identical. Used by
        forward-chaining soundness (the closure set). The KB is never mutated.
        """
        return self.with_program(facts=closure_facts)

    def __repr__(self) -> str:
        return (f"KB(facts={self.num_facts}, rules={self.num_rules}, "
                f"entities={self.constant_no}, predicates={self.predicate_no})")


__all__ = ["KB"]
