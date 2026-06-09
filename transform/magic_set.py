"""MagicSetTransform — magic-sets (demand transformation) over ForwardGrounder.

Magic-sets is a BOTTOM-UP demand transformation: its home is the ForwardGrounder
(T_P fixpoint), which ACCUMULATES the closure, so a magic guard genuinely
RESTRICTS which atoms the fixpoint derives. (It cannot compose with backward
chaining — BC is already demand-restricted and does not accumulate derived facts.)

Pipeline:
  construction (query-pattern-specific, constant-independent — built ONCE):
    - adorn the rules deriving ``query_pred`` by left-to-right SIPS for the query's
      bound/free pattern (default ``bf``: subj bound, obj free);
    - generate ``magic_<pred>^<adn>`` rules (the demand set) + adorned guarded
      rules; encode each unary magic atom ``m(c)`` as the binary triple ``(mp,c,c)``;
    - build the adorned-rule SKELETON via ``KB.with_program(heads,bodies,lens)``.
  per call (apply):
    - seed the query's bound arg as a magic fact via ``KB.with_program(facts=...)``
      (facts-only → hits the step-5 rule-index cache; cheap).
  The base ForwardGrounder then computes the T_P closure of the magic-augmented
  program = the query-relevant subset only (component reachable from the seed).

Decode: the adorned head predicate ``query_pred^<adn>`` carries a FRESH id; the
relevant answers are its closure atoms, mapped back to ``query_pred`` via
``adorned_pred_of`` / ``base_pred_of``.

Scope (sufficient for the oracle, faithful to the seam): a single query predicate,
left-to-right SIPS, one binding pattern. Rules NOT deriving ``query_pred`` (or its
adorned descendants) are carried through unchanged — they stay query-irrelevant and
are never reached by the magic guard, so the fixpoint skips them.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from grounder.core import GroundRequest
from grounder.data.kb import KB
from grounder.data.rule_index import compile_rules


class MagicSetTransform:
    """Adorn + magic-guard the rules deriving ``query_pred`` (base = ForwardGrounder)."""

    name = "magic_set"

    def __init__(self, base_kb: KB, *, query_pred: int, bound: Tuple[bool, bool] = (True, False)) -> None:
        self.query_pred = int(query_pred)
        self.bound = bound
        self._pad = base_kb.padding_idx
        self._C = base_kb.constant_no
        self._adn = "".join("b" if b else "f" for b in bound)

        # Two fresh predicate ids beyond the base program's real range: the adorned
        # head ``query_pred^<adn>`` and its magic (demand) predicate ``magic^<adn>``.
        base_P = base_kb.real_num_predicates()
        self.adorned_pred = base_P              # query_pred^<adn>
        self.magic_pred = base_P + 1            # magic_query_pred^<adn>  (unary -> (mp,c,c))

        heads, bodies, lens = self._build_adorned_program(base_kb)
        # Build the adorned-rule SKELETON once (rule-changing with_program).
        self._skeleton_kb = base_kb.with_program(heads=heads, bodies=bodies, lens=lens)

    # ── construction: adorn + generate magic rules (query-pattern-specific) ──
    def _build_adorned_program(self, base_kb: KB) -> Tuple[Tensor, Tensor, Tensor]:
        """Original rules (carried through) + adorned guarded rules + magic rules.

        For each rule deriving ``query_pred`` with body left-to-right ordered:
          adorned:  query_pred^<adn>(head) :- magic(bound_head_arg), <body, with
                    recursive ``query_pred`` atoms replaced by ``query_pred^<adn>``>
          magic:    one ``magic(<demanded arg>)`` rule per recursive body atom, whose
                    body is ``magic(bound_head_arg)`` + the body prefix that binds the
                    recursive atom's bound argument (SIPS).
        """
        pad, C = self._pad, self._C
        patterns = compile_rules(base_kb.rules_heads_idx, base_kb.rules_bodies_idx,
                                 base_kb.rule_lens, C)
        heads_in = base_kb.rules_heads_idx
        bodies_in = base_kb.rules_bodies_idx
        lens_in = base_kb.rule_lens

        out_heads: List[List[int]] = []
        out_bodies: List[List[List[int]]] = []
        out_lens: List[int] = []

        # which head-arg position is bound (single-bound pattern: the first True).
        bound_pos = 0 if self.bound[0] else 1

        def is_var(v: int) -> bool:
            return v > C

        for r, pat in enumerate(patterns):
            head = heads_in[r]
            if int(head[0].item()) != self.query_pred:
                # Not a query-predicate rule: carry through unchanged (irrelevant).
                out_heads.append(head.tolist())
                out_bodies.append(bodies_in[r].tolist())
                out_lens.append(int(lens_in[r].item()))
                continue

            L = int(lens_in[r].item())
            body = [bodies_in[r, j].tolist() for j in range(L)]
            head_args = [int(head[1].item()), int(head[2].item())]
            bound_var = head_args[bound_pos]

            # adorned guarded rule: magic(bound_var) + body (recursive query atoms adorned)
            adn_body: List[List[int]] = [[self.magic_pred, bound_var, bound_var]]
            for atom in body:
                p = atom[0]
                if p == self.query_pred:
                    adn_body.append([self.adorned_pred, atom[1], atom[2]])
                else:
                    adn_body.append(list(atom))
            out_heads.append([self.adorned_pred, head_args[0], head_args[1]])
            out_bodies.append(adn_body)
            out_lens.append(len(adn_body))

            # magic rules: one per recursive (query_pred) body atom, demanding its
            # bound arg, guarded by magic(bound_var) + the SIPS-binding prefix.
            known = {bound_var}
            prefix: List[List[int]] = []
            for atom in body:
                p = atom[0]
                if p == self.query_pred:
                    rec_bound = atom[1 + bound_pos]   # demanded arg of the recursive call
                    mbody: List[List[int]] = [[self.magic_pred, bound_var, bound_var]]
                    mbody.extend(list(a) for a in prefix)
                    out_heads.append([self.magic_pred, rec_bound, rec_bound])
                    out_bodies.append(mbody)
                    out_lens.append(len(mbody))
                # grow the SIPS prefix / known-var set with this body atom
                prefix.append(list(atom))
                for k in (1, 2):
                    if is_var(atom[k]):
                        known.add(atom[k])

        M = max(len(b) for b in out_bodies)
        H = torch.tensor(out_heads, dtype=torch.long)
        B = torch.full((len(out_bodies), M, 3), pad, dtype=torch.long)
        for i, b in enumerate(out_bodies):
            for j, atom in enumerate(b):
                B[i, j] = torch.tensor(atom, dtype=torch.long)
        Ln = torch.tensor(out_lens, dtype=torch.long)
        return H, B, Ln

    # ── per-call: seed the query's bound args as magic facts (facts-only) ──
    def apply(self, kb: KB, req: GroundRequest) -> Tuple[KB, GroundRequest]:
        """Seed ``magic(bound_arg)`` per query (facts-only union -> cache hit)."""
        queries = req.queries
        if queries is None or queries.numel() == 0:
            return self._skeleton_kb, req
        bound_pos = 0 if self.bound[0] else 1
        consts = queries[:, 1 + bound_pos].to(torch.long)
        seeds = torch.stack([torch.full_like(consts, self.magic_pred), consts, consts], dim=1)
        kb2 = self._skeleton_kb.with_program(facts=seeds)
        # FC is data-directed; the magic seeds already encode the query demand.
        req2 = GroundRequest(queries=None, closure_depth=req.closure_depth)
        return kb2, req2

    # ── decode helpers ──
    def adorned_pred_of(self, base_pred: int) -> Optional[int]:
        return self.adorned_pred if int(base_pred) == self.query_pred else None

    def base_pred_of(self, pred: int) -> Optional[int]:
        return self.query_pred if int(pred) == self.adorned_pred else None


__all__ = ["MagicSetTransform"]
