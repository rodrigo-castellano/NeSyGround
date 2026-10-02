"""torch-ns's entry point until it calls ``grounder.pbc.PBC`` directly: ``create_grounder`` builds a PBC grounder from
string rules and a type string, and ``RuleGrounder.run_bc_many`` returns each pool's groundings as a ``RuleGroundings``
(the old field names).

    fact_index = TensorFactIndex(facts, num_predicates, num_entities, device=...)
    grounder = create_grounder("enum.fp_batch.w1.d2.flat", fact_index=fact_index, rules=rules, kb=vocab, device=...)
    per_pool = grounder.run_bc_many(queries [S, Q, 3], mask [S, Q])
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.base.types import RuleGroundings
from grounder.kb import KB
from grounder.pbc import PBC


class TensorFactIndex(nn.Module):
    """The facts ``(r, h, t)``, sorted, with ``contains_atoms``. (``max_facts_per_query`` is accepted and ignored:
    every lookup is enumerated in full.)"""

    def __init__(self, facts: List[Tuple[int, int, int]], num_predicates: int, num_entities: int,
                 max_facts_per_query: int = 1 << 30, device: str = "cuda") -> None:
        super().__init__()
        self.num_predicates, self.num_entities, self.device = num_predicates, num_entities, device
        f = torch.tensor(list(facts), dtype=torch.long, device=device).reshape(-1, 3)
        E = num_entities
        h = f[:, 0] * E * E + f[:, 1] * E + f[:, 2]
        order = torch.argsort(h)
        for name, t in (("fact_preds", f[order, 0]), ("fact_subjs", f[order, 1]), ("fact_objs", f[order, 2]),
                        ("fact_hashes", h[order])):
            self.register_buffer(name, t)
        self.register_buffer("num_facts", torch.tensor(len(f), device=device))

    def contains_atoms(self, atoms: Tensor) -> Tensor:
        """``[B, 3] -> [B]``: whether each atom is a fact."""
        E, a = self.num_entities, atoms.long()
        h = a[:, 0] * E * E + a[:, 1] * E + a[:, 2]
        pos = torch.searchsorted(self.fact_hashes, h).clamp(max=int(self.fact_hashes.shape[0]) - 1)
        return self.fact_hashes[pos] == h


class RuleGrounder(nn.Module):
    """A PBC grounder over string rules (``input_rule[i]``: rule ``i``'s index in ``rules``; ``input_body[i]``: the
    input position of each of its body columns)."""

    def __init__(self, grounder_type: str, fact_index: TensorFactIndex, rules: list, vocab, device) -> None:
        super().__init__()
        facts = torch.stack([fact_index.fact_preds, fact_index.fact_subjs, fact_index.fact_objs], 1)
        self.kb = KB.from_strings(facts, rules, vocab.entity2id, vocab.relation2id, device=device)
        self.pbc = PBC.parse(self.kb, grounder_type)
        self.fact_index = fact_index
        self.input_rule: List[int] = self.kb.rules.order.tolist()
        self.input_body: List[List[int]] = self.kb.rules.body_order
        self._inner = SimpleNamespace(depth=self.pbc.depth, kb=SimpleNamespace(predicate_no=len(vocab.relation2id)))

    def fast(self) -> bool:
        return self.kb.device.type == "cuda"

    def run_bc_many(self, queries: Tensor, query_mask: Tensor, *, depth: Optional[int] = None,
                    stats: Optional[list] = None) -> List[RuleGroundings]:
        """Each pool's groundings, for ``queries [S, Q, 3]``."""
        g = self.pbc.ground(queries, query_mask, depth=depth, stats=stats)
        S, R, M = queries.shape[0], len(self.kb.rules), g.body.shape[1]
        n_atoms, n_rows = g.atom_offsets.diff(), g.offsets[:, -1] - g.offsets[:, 0]
        row_pool = torch.repeat_interleave(torch.arange(S, device=g.rule.device), n_rows)
        at = g.atom_offsets[row_pool]
        head, body = g.head - at, g.body - at.unsqueeze(1)
        offsets = g.offsets - g.offsets[:, :1]
        query = g.query - g.atom_offsets[:-1].unsqueeze(1)
        na, nr = n_atoms.tolist(), n_rows.tolist()
        return [RuleGroundings(atom_table=a, body_pool_idx=b, body_atom_valid=v, head_pool_idx=h, rule_idx=r,
                               rule_offsets=o, num_atoms=n, num_rules=R, M_max=M, query_pool_idx=q)
                for a, b, v, h, r, o, n, q in zip(g.atoms.split(na), body.split(nr), g.body_mask.split(nr),
                                                  head.split(nr), g.rule.split(nr), offsets, na, query)]

    def run_bc(self, queries: Tensor, query_mask: Tensor) -> RuleGroundings:
        return self.run_bc_many(queries.unsqueeze(0), query_mask.unsqueeze(0))[0]


def create_grounder(grounder_type: str, *, fact_index: TensorFactIndex, rules, kb, device, **_) -> RuleGrounder:
    """A PBC grounder from a type string (``enum.fp_batch.w1.d2.flat``, ``enum.keras.w1.d2.r2.flat``, ...); the
    former sizing knobs (``max_groundings``, ``max_states``, ``chunk_size``, ...) are accepted and ignored."""
    return RuleGrounder(grounder_type, fact_index, rules, kb, device)


__all__ = ["TensorFactIndex", "RuleGrounder", "create_grounder"]
