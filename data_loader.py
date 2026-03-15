"""Lightweight KG dataset loader for the grounder package.

Loads a dataset directory containing triples, rules, and optional facts,
builds vocabularies and converts to tensors suitable for BCGrounder.

Usage:
    from grounder.data_loader import KGDataset
    ds = KGDataset('kge_experiments/data/family', device='cuda')
    grounder = ds.make_grounder(PrologGrounder, max_goals=20, depth=3)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor


# Regex for atoms: pred(arg0,arg1)
_ATOM_RE = re.compile(r"(\w+)\(([^,]+),([^)]+)\)")
# Variable detection: single lowercase letter OR uppercase-starting name
_VAR_PATTERN_LOWER = re.compile(r"^[a-z]$")
_VAR_PATTERN_UPPER = re.compile(r"^[A-Z]")


def _parse_triples(path: Path) -> List[Tuple[str, str, str]]:
    """Parse triples file: pred(arg0,arg1) or tab-separated."""
    triples: list[tuple[str, str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(".")
            if not line or line.startswith("#"):
                continue
            # Try Prolog format: pred(arg0,arg1)
            paren_idx = line.rfind("(")
            if paren_idx > 0 and line.endswith(")"):
                pred = line[:paren_idx]
                args = line[paren_idx + 1:-1]
                parts = args.split(",", 1)
                if len(parts) == 2:
                    triples.append((pred.strip(), parts[0].strip(), parts[1].strip()))
                    continue
            # Try tab-separated format
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def _parse_rules_arrow(
    path: Path,
) -> List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]]:
    """Parse arrow-format rules: rN:score:body1(a,h), body2(b,h) -> head(a,b)."""
    rules: list[tuple[tuple[str, str, str], list[tuple[str, str, str]]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            rest = ":".join(parts[2:])
            if "->" not in rest:
                continue
            body_str, head_str = rest.rsplit("->", 1)
            head_m = _ATOM_RE.search(head_str.strip())
            if not head_m:
                continue
            head = (head_m.group(1).strip(), head_m.group(2).strip(),
                    head_m.group(3).strip())
            body_atoms: list[tuple[str, str, str]] = []
            for b in body_str.split("),"):
                b = b.strip()
                if not b.endswith(")"):
                    b += ")"
                m = _ATOM_RE.search(b)
                if m:
                    body_atoms.append((m.group(1).strip(), m.group(2).strip(),
                                       m.group(3).strip()))
            if body_atoms:
                rules.append((head, body_atoms))
    return rules


def _parse_rules_prolog(
    path: Path,
) -> List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]]:
    """Parse Prolog-format rules: head(X,Y) :- body1(X,Z), body2(Z,Y)."""
    rules: list[tuple[tuple[str, str, str], list[tuple[str, str, str]]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(".")
            if not line or line.startswith("#"):
                continue
            if ":-" not in line:
                continue
            head_str, body_str = line.split(":-", 1)
            head_m = _ATOM_RE.search(head_str.strip())
            if not head_m:
                continue
            head = (head_m.group(1).strip(), head_m.group(2).strip(),
                    head_m.group(3).strip())
            body_atoms: list[tuple[str, str, str]] = []
            for b in body_str.split("),"):
                b = b.strip()
                if not b.endswith(")"):
                    b += ")"
                m = _ATOM_RE.search(b)
                if m:
                    body_atoms.append((m.group(1).strip(), m.group(2).strip(),
                                       m.group(3).strip()))
            if body_atoms:
                rules.append((head, body_atoms))
    return rules


def _parse_rules(
    path: Path,
) -> List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]]:
    """Auto-detect and parse rules file (arrow or Prolog format)."""
    with open(path) as f:
        first_line = ""
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                first_line = line
                break
    if ":-" in first_line:
        return _parse_rules_prolog(path)
    return _parse_rules_arrow(path)


def _is_variable(name: str) -> bool:
    """Check if a name is a logical variable (single lowercase or uppercase-starting)."""
    return bool(_VAR_PATTERN_LOWER.match(name) or _VAR_PATTERN_UPPER.match(name))


class KGDataset:
    """Load a KG dataset from a directory.

    Args:
        data_dir: Path to dataset directory containing train.txt, rules.txt, etc.
        facts_file: File for KB facts (default: 'facts.txt', fallback to 'train.txt')
        device: Target device for tensors
    """

    def __init__(
        self,
        data_dir: str,
        facts_file: str = "facts.txt",
        device: str = "cpu",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        assert self.data_dir.exists(), f"Dataset directory not found: {self.data_dir}"

        # Determine facts file (fallback to train.txt)
        facts_path = self.data_dir / facts_file
        if not facts_path.exists():
            facts_path = self.data_dir / "train.txt"
        self._facts_file = facts_path.name

        # Parse raw data (kept for analysis tools)
        facts_raw = _parse_triples(facts_path)
        rules_raw = _parse_rules(self.data_dir / "rules.txt")
        self._facts_raw: List[Tuple[str, str, str]] = facts_raw
        self._rules_raw: List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]] = rules_raw

        # Parse all splits
        self._splits_raw: Dict[str, List[Tuple[str, str, str]]] = {}
        for split in ("train", "valid", "test"):
            split_path = self.data_dir / f"{split}.txt"
            if split_path.exists():
                self._splits_raw[split] = _parse_triples(split_path)

        # Build vocabularies from ALL data
        all_preds: set[str] = set()
        all_entities: set[str] = set()
        var_names: set[str] = set()

        for pred, a0, a1 in facts_raw:
            all_preds.add(pred)
            all_entities.add(a0)
            all_entities.add(a1)

        for split_triples in self._splits_raw.values():
            for pred, a0, a1 in split_triples:
                all_preds.add(pred)
                all_entities.add(a0)
                all_entities.add(a1)

        for head, body_atoms in rules_raw:
            for atom in [head] + body_atoms:
                all_preds.add(atom[0])
                for arg in atom[1:]:
                    if _is_variable(arg):
                        var_names.add(arg)
                    else:
                        all_entities.add(arg)

        # 1-based indexing
        self.pred2idx: Dict[str, int] = {p: i + 1 for i, p in enumerate(sorted(all_preds))}
        self.entity2idx: Dict[str, int] = {e: i + 1 for i, e in enumerate(sorted(all_entities))}
        var2idx: Dict[str, int] = {v: len(self.entity2idx) + 1 + i
                                   for i, v in enumerate(sorted(var_names))}

        self.idx2pred: Dict[int, str] = {v: k for k, v in self.pred2idx.items()}
        self.idx2entity: Dict[int, str] = {v: k for k, v in self.entity2idx.items()}

        self.constant_no = len(self.entity2idx)
        self.padding_idx = self.constant_no + len(var_names) + 10
        # predicate_no must be >= padding_idx so RuleIndex segment table
        # can handle padding values that appear as predicate indices in
        # inactive proof states
        self.predicate_no = max(len(self.pred2idx) + 1, self.padding_idx)

        def _lookup(arg: str) -> int:
            if arg in var2idx:
                return var2idx[arg]
            if arg in self.entity2idx:
                return self.entity2idx[arg]
            return self.padding_idx

        # Build fact tensors
        fact_list = []
        for pred, a0, a1 in facts_raw:
            if pred in self.pred2idx and a0 in self.entity2idx and a1 in self.entity2idx:
                fact_list.append([self.pred2idx[pred], self.entity2idx[a0], self.entity2idx[a1]])
        self.facts_idx = (torch.tensor(fact_list, dtype=torch.long, device=self.device)
                          if fact_list else torch.empty(0, 3, dtype=torch.long, device=self.device))

        # Build rule tensors
        max_body = max((len(body) for _, body in rules_raw), default=1)
        heads_list: list[list[int]] = []
        bodies_list: list[list[list[int]]] = []
        lens_list: list[int] = []

        for head, body_atoms in rules_raw:
            h = [self.pred2idx[head[0]], _lookup(head[1]), _lookup(head[2])]
            heads_list.append(h)

            body_row: list[list[int]] = []
            for atom in body_atoms:
                b = [self.pred2idx[atom[0]], _lookup(atom[1]), _lookup(atom[2])]
                body_row.append(b)
            while len(body_row) < max_body:
                body_row.append([self.padding_idx, self.padding_idx, self.padding_idx])
            bodies_list.append(body_row)
            lens_list.append(len(body_atoms))

        if heads_list:
            self.rules_heads_idx = torch.tensor(heads_list, dtype=torch.long, device=self.device)
            self.rules_bodies_idx = torch.tensor(bodies_list, dtype=torch.long, device=self.device)
            self.rule_lens = torch.tensor(lens_list, dtype=torch.long, device=self.device)
        else:
            self.rules_heads_idx = torch.empty(0, 3, dtype=torch.long, device=self.device)
            self.rules_bodies_idx = torch.empty(0, 1, 3, dtype=torch.long, device=self.device)
            self.rule_lens = torch.empty(0, dtype=torch.long, device=self.device)

        # Build split tensors
        self._splits_idx: Dict[str, Tensor] = {}
        for split, triples in self._splits_raw.items():
            rows = []
            for pred, a0, a1 in triples:
                if pred in self.pred2idx and a0 in self.entity2idx and a1 in self.entity2idx:
                    rows.append([self.pred2idx[pred], self.entity2idx[a0], self.entity2idx[a1]])
            if rows:
                self._splits_idx[split] = torch.tensor(rows, dtype=torch.long, device=self.device)
            else:
                self._splits_idx[split] = torch.empty(0, 3, dtype=torch.long, device=self.device)

    def get_queries(self, split: str) -> Tensor:
        """Get query tensor for a split. [N, 3] (pred, arg0, arg1)."""
        if split not in self._splits_idx:
            return torch.empty(0, 3, dtype=torch.long, device=self.device)
        return self._splits_idx[split]

    def get_query_strings(self, split: str) -> List[str]:
        """Get query strings in 'pred(arg0,arg1)' format."""
        if split not in self._splits_raw:
            return []
        return [f"{p}({a0},{a1})" for p, a0, a1 in self._splits_raw[split]]

    def make_grounder(
        self,
        grounder_cls: Type,
        max_goals: int,
        depth: int = 1,
        max_states: Optional[int] = None,
        compile_mode: Optional[str] = None,
        track_grounding_body: bool = True,
        **kwargs,
    ):
        """Create a BCGrounder from this dataset's tensors.

        Args:
            grounder_cls: PrologGrounder or RTFGrounder class
            max_goals: G dimension
            depth: number of proof steps
            max_states: S cap (None = K)
            compile_mode: torch.compile mode (None, 'reduce-overhead', etc.)
            track_grounding_body: whether to track grounding body atoms
            **kwargs: additional args to grounder constructor
        """
        return grounder_cls(
            facts_idx=self.facts_idx,
            rules_heads_idx=self.rules_heads_idx,
            rules_bodies_idx=self.rules_bodies_idx,
            rule_lens=self.rule_lens,
            constant_no=self.constant_no,
            padding_idx=self.padding_idx,
            device=self.device,
            predicate_no=self.predicate_no,
            max_goals=max_goals,
            depth=depth,
            max_states=max_states,
            compile_mode=compile_mode,
            track_grounding_body=track_grounding_body,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (f"KGDataset(dir={self.data_dir.name}, "
                f"facts={self.facts_idx.shape[0]}, "
                f"rules={self.rules_heads_idx.shape[0]}, "
                f"entities={self.constant_no}, "
                f"predicates={self.predicate_no - 1})")
