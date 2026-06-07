"""KGDataset — load a dataset directory, build vocabularies, tensorize, → KB.

Owns the id ENCODING: parses facts/rules/splits (via ``loader``), assigns
predicate/entity/variable ids, and exposes the tensors + an ``Encoding``.
``build_kb`` hands those to ``KB`` (which builds the indices).

BIT-EXACT RECIPES (fingerprint-enforced — the id scheme is the grounding's
coordinate system; any drift changes every downstream tensor):
  * raw facts/rules/splits are ``sorted()`` before vocab assignment (determinism)
  * ids are 1-based: ``pred2idx``/``entity2idx`` = sorted-rank + 1
  * variables follow entities: ``var_id = #entities + 1 + sorted-rank``
  * ``constant_no = #entities``;  ``padding_idx = constant_no + #vars + 10``
  * ``predicate_no = max(#preds + 1, padding_idx + 1)`` (padding fits index tables)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor

from grounder._build.data import loader
from grounder._build.data.encoding import Encoding


class KGDataset:
    """Parsed + tensorized KG dataset with a fixed id encoding."""

    def __init__(
        self,
        data_dir: str,
        facts_file: str = "facts.txt",
        rules_file: str = "rules.txt",
        device: str = "cpu",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        assert self.data_dir.exists(), f"Dataset directory not found: {self.data_dir}"

        facts_path = self.data_dir / facts_file
        if not facts_path.exists():
            facts_path = self.data_dir / "train.txt"
        self._facts_file = facts_path.name
        self._rules_file = rules_file

        facts_raw = sorted(loader.parse_triples(facts_path))
        rules_raw = sorted(loader.parse_rules(self.data_dir / rules_file))
        self._facts_raw = facts_raw
        self._rules_raw = rules_raw

        self._splits_raw: Dict[str, List[loader.Triple]] = {}
        for split in ("train", "valid", "test"):
            p = self.data_dir / f"{split}.txt"
            if p.exists():
                self._splits_raw[split] = sorted(loader.parse_triples(p))

        # ── vocabularies from ALL data ──
        all_preds: set = set()
        all_entities: set = set()
        var_names: set = set()
        for pred, a0, a1 in facts_raw:
            all_preds.add(pred); all_entities.update((a0, a1))
        for triples in self._splits_raw.values():
            for pred, a0, a1 in triples:
                all_preds.add(pred); all_entities.update((a0, a1))
        for head, body in rules_raw:
            for atom in [head] + body:
                all_preds.add(atom[0])
                for arg in atom[1:]:
                    (var_names if loader.is_variable(arg) else all_entities).add(arg)

        # 1-based ids; variables follow entities; padding/predicate sizing.
        self.pred2idx: Dict[str, int] = {p: i + 1 for i, p in enumerate(sorted(all_preds))}
        self.entity2idx: Dict[str, int] = {e: i + 1 for i, e in enumerate(sorted(all_entities))}
        var2idx: Dict[str, int] = {v: len(self.entity2idx) + 1 + i
                                   for i, v in enumerate(sorted(var_names))}
        self.idx2pred = {v: k for k, v in self.pred2idx.items()}
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}

        self.constant_no = len(self.entity2idx)
        self.padding_idx = self.constant_no + len(var_names) + 10
        self.predicate_no = max(len(self.pred2idx) + 1, self.padding_idx + 1)
        self.encoding = Encoding.from_constant_no(self.constant_no, self.padding_idx)

        def _lookup(arg: str) -> int:
            if arg in var2idx:
                return var2idx[arg]
            if arg in self.entity2idx:
                return self.entity2idx[arg]
            return self.padding_idx

        # ── fact tensor (entity-only facts) ──
        fact_list = [[self.pred2idx[p], self.entity2idx[a0], self.entity2idx[a1]]
                     for p, a0, a1 in facts_raw
                     if p in self.pred2idx and a0 in self.entity2idx and a1 in self.entity2idx]
        self.facts_idx = (torch.tensor(fact_list, dtype=torch.long, device=self.device)
                          if fact_list else torch.empty(0, 3, dtype=torch.long, device=self.device))

        # ── rule tensors (padded bodies) ──
        max_body = max((len(body) for _, body in rules_raw), default=1)
        heads_list, bodies_list, lens_list = [], [], []
        for head, body in rules_raw:
            heads_list.append([self.pred2idx[head[0]], _lookup(head[1]), _lookup(head[2])])
            row = [[self.pred2idx[a[0]], _lookup(a[1]), _lookup(a[2])] for a in body]
            row += [[self.padding_idx] * 3] * (max_body - len(row))
            bodies_list.append(row)
            lens_list.append(len(body))
        if heads_list:
            self.rules_heads_idx = torch.tensor(heads_list, dtype=torch.long, device=self.device)
            self.rules_bodies_idx = torch.tensor(bodies_list, dtype=torch.long, device=self.device)
            self.rule_lens = torch.tensor(lens_list, dtype=torch.long, device=self.device)
        else:
            self.rules_heads_idx = torch.empty(0, 3, dtype=torch.long, device=self.device)
            self.rules_bodies_idx = torch.empty(0, 1, 3, dtype=torch.long, device=self.device)
            self.rule_lens = torch.empty(0, dtype=torch.long, device=self.device)

        # ── split query tensors (entity-only) ──
        self._splits_idx: Dict[str, Tensor] = {}
        for split, triples in self._splits_raw.items():
            rows = [[self.pred2idx[p], self.entity2idx[a0], self.entity2idx[a1]]
                    for p, a0, a1 in triples
                    if p in self.pred2idx and a0 in self.entity2idx and a1 in self.entity2idx]
            self._splits_idx[split] = (torch.tensor(rows, dtype=torch.long, device=self.device)
                                       if rows else torch.empty(0, 3, dtype=torch.long, device=self.device))

    def get_queries(self, split: str) -> Tensor:
        """``[N,3]`` query tensor for a split (empty if absent)."""
        return self._splits_idx.get(
            split, torch.empty(0, 3, dtype=torch.long, device=self.device))

    def get_query_strings(self, split: str) -> List[str]:
        return [f"{p}({a0},{a1})" for p, a0, a1 in self._splits_raw.get(split, [])]

    def build_kb(self, **kwargs):
        """Construct a KB from this dataset's tensors + encoding."""
        from grounder._build.data.kb import KB
        return KB(self.facts_idx, self.rules_heads_idx, self.rules_bodies_idx, self.rule_lens,
                  constant_no=self.constant_no, predicate_no=self.predicate_no,
                  padding_idx=self.padding_idx, device=self.device, **kwargs)

    def __repr__(self) -> str:
        return (f"KGDataset(dir={self.data_dir.name}, facts={self.facts_idx.shape[0]}, "
                f"rules={self.rules_heads_idx.shape[0]}, entities={self.constant_no}, "
                f"predicates={self.predicate_no - 1})")


__all__ = ["KGDataset"]
