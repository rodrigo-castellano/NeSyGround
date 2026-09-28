"""The program: facts and rules, indexed for every technique.

    kb = KB(facts, heads, bodies, lens, E=num_entities, pad=pad)          # tensors, variables are ids >= E
    kb = KB.from_strings(facts, rules, entity2id, relation2id)            # (r, h, t) ids and string rules

An atom is ``(p, a1, ..., a_n)`` (width ``W = n + 1``): constants are ids ``< E``, variables ids ``>= E`` other than
``pad``. ``Facts`` sorts the facts by key and indexes them by predicate and by each argument (CSR, stable: a lookup
lists its facts in the facts' order); ``Rules`` indexes the rules by head predicate; a rule's id — what every grounder reports — is
its position in the KB. ``RulePattern`` is the
binding analysis of one rule (its head and free variables, each body argument's source, a dependency order of its
body), shared by PBC and forward chaining.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from grounder.ops import key

Triple = Tuple[str, str, str]
Rule = Tuple[Triple, List[Triple]]


# ── parsing ──

_ATOM = re.compile(r"([\w/.]+)\(([^,]+),([^)]+)\)")        # predicates may hold / and . (FB15k-237 relations)


def is_variable(name: str) -> bool:
    """A logical variable: one lowercase letter, or a name starting upper case."""
    return bool(re.match(r"^[a-z]$", name) or re.match(r"^[A-Z]", name))


def parse_triples(path: Path) -> List[Triple]:
    """A triples file: ``pred(a,b)`` lines, or tab-separated ``pred a b``."""
    triples: List[Triple] = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(".")
            if not line or line.startswith("#"):
                continue
            paren = line.rfind("(")
            if paren > 0 and line.endswith(")"):
                parts = line[paren + 1:-1].split(",", 1)
                if len(parts) == 2:
                    triples.append((line[:paren].strip(), parts[0].strip(), parts[1].strip()))
                    continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def _atom(s: str) -> Optional[Triple]:
    m = _ATOM.search(s.strip())
    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else None


def _body(s: str) -> List[Triple]:
    atoms = []
    for frag in s.split("),"):
        frag = frag.strip()
        atom = _atom(frag if frag.endswith(")") else frag + ")")
        if atom is not None:
            atoms.append(atom)
    return atoms


def parse_rules(path: Path) -> List[Rule]:
    """A rules file: ``rN:score:body1(a,b), body2(b,c) -> head(a,c)`` lines, or Prolog ``head(X,Y) :- body``."""
    rules: List[Rule] = []
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    prolog = bool(lines) and ":-" in lines[0]
    for line in lines:
        if prolog:
            if ":-" not in line:
                continue
            head_str, body_str = line.rstrip(".").split(":-", 1)
        else:
            parts = line.split(":")
            rest = ":".join(parts[2:])
            if len(parts) < 3 or "->" not in rest:
                continue
            body_str, head_str = rest.rsplit("->", 1)
        head, body = _atom(head_str), _body(body_str)
        if head is not None and body:
            rules.append((head, body))
    return rules


# ── rule binding analysis ──

HEAD0, HEAD1, FREE = 0, 1, 2          # an argument's source: head variable 0 or 1, or free variable FREE + k


def atom_enum_meta(body_pattern: dict, known: set) -> dict:
    """How a body atom is reached given the ``known`` sources (grown in place): its enumeration predicate, bound source,
    direction (0: subject bound), and the free variable it introduces (-1: none)."""
    b0, b1 = body_pattern["arg0_binding"], body_pattern["arg1_binding"]
    pred = body_pattern["pred_idx"]
    if b0 in known and b1 in known:
        return {"introduces_fv": -1, "enum_bound_src": 0, "enum_direction": 0, "enum_pred": pred}
    if b0 in known:
        known.add(b1)
        return {"introduces_fv": b1 - FREE, "enum_bound_src": b0, "enum_direction": 0, "enum_pred": pred}
    if b1 in known:
        known.add(b0)
        return {"introduces_fv": b0 - FREE, "enum_bound_src": b1, "enum_direction": 1, "enum_pred": pred}
    return {"introduces_fv": -1, "enum_bound_src": 0, "enum_direction": 0, "enum_pred": pred}


def greedy_body_order(body_patterns: List[dict], remaining: List[int], known: set, order: List[int],
                      meta: List[dict]) -> None:
    """The dependency order of the ``remaining`` body atoms (appended to ``order``, their meta to ``meta``): repeatedly
    the first atom with a known argument; the unreachable rest last."""
    while remaining:
        for idx in remaining:
            bp = body_patterns[idx]
            if bp["arg0_binding"] in known or bp["arg1_binding"] in known:
                order.append(idx)
                meta.append(atom_enum_meta(bp, known))
                remaining.remove(idx)
                break
        else:
            for idx in remaining:
                order.append(idx)
                meta.append(atom_enum_meta(body_patterns[idx], known))
            break


class RulePattern:
    """One rule's binding analysis. Free variables are the body's variables (ids ``> constant_no``) that are not the
    head's, numbered in id order; ``body_patterns`` follow the dependency order (``enum_meta`` per atom), and
    ``_orig_body_patterns`` the given order."""

    def __init__(self, rule_idx: int, head: Tensor, body: Tensor, body_len: int, constant_no: int) -> None:
        self.rule_idx = rule_idx
        self.constant_no = constant_no
        self.head_pred_idx: int = head[0].item()
        self.head_var0: int = head[1].item()
        self.head_var1: int = head[2].item()
        self.num_body: int = body_len
        self.body_pred_indices: List[int] = [body[j, 0].item() for j in range(body_len)]
        head_vars = {v for v in (self.head_var0, self.head_var1) if v > constant_no}
        body_vars = {body[j, k].item() for j in range(body_len) for k in (1, 2) if body[j, k].item() > constant_no}
        self.free_vars_list = sorted(body_vars - head_vars)
        self.num_free = len(self.free_vars_list)
        self._fv_idx = {v: i for i, v in enumerate(self.free_vars_list)}
        self.body_patterns = [
            {"pred_idx": body[j, 0].item(), "arg0_binding": self._binding(body[j, 1].item()),
             "arg1_binding": self._binding(body[j, 2].item()), "arg0_var": body[j, 1].item(),
             "arg1_var": body[j, 2].item()}
            for j in range(body_len)]
        self._orig_body_patterns = list(self.body_patterns)
        self._orig_body_pred_indices = list(self.body_pred_indices)
        known, order, meta = {HEAD0, HEAD1}, [], []
        greedy_body_order(self.body_patterns, list(range(self.num_body)), known, order, meta)
        self.enum_meta = meta
        self.body_order = order
        self.body_patterns = [self.body_patterns[i] for i in order]
        self.body_pred_indices = [self.body_pred_indices[i] for i in order]

    def _binding(self, val: int) -> int:
        if val == self.head_var0:
            return HEAD0
        if val == self.head_var1:
            return HEAD1
        if val in self._fv_idx:
            return FREE + self._fv_idx[val]
        return HEAD0                       # a constant: an inert source

    def reorder_with_anchor(self, first: int) -> Tuple[List, List, List]:
        """The dependency order with body atom ``first`` (given order) forced first: ``(body_patterns,
        body_pred_indices, enum_meta)``."""
        known = {HEAD0, HEAD1}
        order, meta = [first], [atom_enum_meta(self._orig_body_patterns[first], known)]
        greedy_body_order(self._orig_body_patterns, [i for i in range(self.num_body) if i != first], known, order,
                          meta)
        return ([self._orig_body_patterns[i] for i in order], [self._orig_body_pred_indices[i] for i in order], meta)


class PatternVariant:
    """An anchor variant of a rule: its body in the order the anchor forces, and the rule's given order (so every
    variant of a rule writes its groundings' body columns alike)."""

    __slots__ = ("rule_idx", "head_pred_idx", "num_body", "num_free", "body_patterns", "body_pred_indices",
                 "enum_meta", "_orig_body_patterns", "_orig_body_pred_indices")

    def __init__(self, base: RulePattern, body_patterns: List, body_pred_indices: List, enum_meta: List) -> None:
        self.rule_idx, self.head_pred_idx = base.rule_idx, base.head_pred_idx
        self.num_body, self.num_free = base.num_body, base.num_free
        self.body_patterns, self.body_pred_indices, self.enum_meta = body_patterns, body_pred_indices, enum_meta
        self._orig_body_patterns = base._orig_body_patterns
        self._orig_body_pred_indices = base._orig_body_pred_indices


def compile_rules(rule_heads: Tensor, rule_bodies: Tensor, rule_lens: Tensor, constant_no: int) -> List[RulePattern]:
    """Rule tensors → their ``RulePattern`` s."""
    return [RulePattern(i, rule_heads[i], rule_bodies[i], int(rule_lens[i].item()), constant_no)
            for i in range(rule_heads.size(0))]


# ── the indexed program ──

def _csr(keys: Tensor, size: int) -> Tuple[Tensor, Tensor]:
    """``(order, offsets)``: rows by ``keys`` (stable), and ``offsets [size + 1]``: key ``k``'s rows are
    ``order[offsets[k]:offsets[k + 1]]``."""
    order = torch.argsort(keys, stable=True)
    offsets = torch.zeros(size + 1, dtype=torch.long, device=keys.device)
    offsets[1:] = torch.bincount(keys, minlength=size).cumsum(0)
    return order, offsets


class Facts:
    """The facts ``atoms [F, W]``, sorted by key (lexicographic), indexed by predicate (``pred_offsets``) and by each
    argument ``j``: ``order[j]`` lists the facts by (predicate, argument j) — stable, so a lookup's facts keep the
    facts' order — and ``offsets[j]`` the slice of each ``pred * base + value``."""

    def __init__(self, atoms: Tensor, *, P: int, base: int) -> None:
        atoms = atoms.long()
        self.base, self.P, self.W = base, P, atoms.shape[1]
        keys = key(atoms, base)
        order = torch.argsort(keys, stable=True)
        self.atoms, self.keys = atoms[order], keys[order]
        self.order, self.offsets = [], []
        for j in range(1, self.W):
            o, off = _csr(self.atoms[:, 0] * base + self.atoms[:, j], P * base)
            self.order.append(o)
            self.offsets.append(off)
        self.pred_offsets = _csr(self.atoms[:, 0], P)[1]
        spans = [off.diff().max() for off in self.offsets]
        self.max_lookup = max(int(torch.stack(spans).max()) if spans else 0, 1)

    def __len__(self) -> int:
        return self.atoms.shape[0]

    def contains(self, atoms: Tensor) -> Tensor:
        """``[..., W] -> [...]``: whether each atom is a fact (atoms with an id past the base are not)."""
        k = key(atoms.long(), self.base)
        ok = (atoms >= 0).all(-1) & (atoms < self.base).all(-1)
        pos = torch.searchsorted(self.keys, k).clamp(max=max(len(self) - 1, 0))
        return ok & (self.keys[pos] == k) if len(self) else torch.zeros_like(ok)

    def lookup(self, pred: Tensor, arg: Tensor, col: int) -> Tuple[Tensor, Tensor]:
        """``(start, count)``: the facts of predicate ``pred`` with argument ``col`` (1-based) equal to ``arg`` are
        ``order[col - 1][start : start + count]`` (none for ids out of range)."""
        ok = (pred >= 0) & (pred < self.P) & (arg >= 0) & (arg < self.base)
        slot = torch.where(ok, pred * self.base + arg, 0)
        off = self.offsets[col - 1]
        start = off[slot]
        return start, torch.where(ok, off[slot + 1] - start, 0)


class Rules:
    """The rules, ``heads [R, W]``, ``bodies [R, M, W]`` (padded with ``pad``), ``lens [R]``: a rule's id is its
    position. Indexed by head predicate: ``by_pred`` lists the rules sorted by it (stable) and ``pred_offsets`` the
    slice of each; ``order [R]`` the index each rule had in the input (``KB.from_strings`` sorts them)."""

    def __init__(self, heads: Tensor, bodies: Tensor, lens: Tensor, *, P: int, order: Optional[Tensor] = None) -> None:
        self.heads, self.bodies, self.lens = heads, bodies, lens
        self.order = order if order is not None else torch.arange(len(heads), device=heads.device)
        self.by_pred, self.pred_offsets = _csr(heads[:, 0], P)
        self.max_per_pred = max(int(self.pred_offsets.diff().max()), 1)

    def __len__(self) -> int:
        return self.heads.shape[0]

    @property
    def M(self) -> int:
        return self.bodies.shape[1]

    def lookup(self, pred: Tensor, K: int) -> Tuple[Tensor, Tensor]:
        """``(rule [N, K], valid [N, K])``: the rules with head predicate ``pred``, in id order (none out of range)."""
        P = self.pred_offsets.shape[0] - 1
        ok = (pred >= 0) & (pred < P)
        p = torch.where(ok, pred, 0)
        start = self.pred_offsets[p]
        n = torch.where(ok, self.pred_offsets[p + 1] - start, 0)
        pos = torch.arange(K, device=pred.device)
        return self.by_pred[(start.unsqueeze(-1) + pos).clamp(max=len(self) - 1)], pos < n.unsqueeze(-1)


class KB:
    """Facts and rules over ``E`` constants (ids ``< E``; variables above them, ``pad`` any id): ``P`` predicates, at
    least the largest predicate id of the facts and rules + 1. Immutable."""

    def __init__(self, facts: Tensor, heads: Tensor, bodies: Tensor, lens: Tensor, *, E: int, pad: int,
                 P: Optional[int] = None, device=None, order: Optional[Tensor] = None) -> None:
        device = torch.device(device) if device is not None else facts.device
        facts, heads, bodies, lens = (t.to(device=device, dtype=torch.long) for t in (facts, heads, bodies, lens))
        if len(facts) == 0 or len(heads) == 0:
            raise ValueError("a KB needs at least one fact and one rule")
        preds = torch.cat([facts[:, 0], heads[:, 0], bodies[..., 0][bodies[..., 0] != pad]])
        self.E, self.pad, self.device = int(E), int(pad), device
        self.P = max(int(P or 0), int(preds.max()) + 1)       # an atom of a predicate >= P matches nothing
        ids = torch.cat([facts.reshape(-1), heads.reshape(-1), bodies.reshape(-1)])
        self.base = max(self.E, self.P, self.pad + 1, int(ids.max()) + 1) + 1        # above every id
        self.facts = Facts(facts, P=self.P, base=self.base)
        self.rules = Rules(heads, bodies, lens, P=self.P, order=None if order is None else order.to(device))
        self.W, self.M = facts.shape[1], int(lens.max())

    @classmethod
    def from_strings(cls, facts: Sequence[Tuple[int, int, int]], rules: Sequence[Rule], entity2id: Dict[str, int],
                     relation2id: Dict[str, int], device=None) -> "KB":
        """A KB of ``(r, h, t)`` fact ids and ``(head, body)`` string rules. The rules are sorted by head predicate
        (stable; ``rules.order``: each one's input index); each rule's body atoms take the dependency order from its head
        variables; its head variables are ids ``E`` and ``E + 1``, its free variables ``E + 2 + i`` in name order; a
        constant argument is its entity id; ``pad = E + 2 + the most free variables``."""
        E = len(entity2id)
        order = sorted(range(len(rules)), key=lambda i: relation2id[rules[i][0][0]])
        heads, bodies, n_free = [], [], 0
        for head, body in (rules[i] for i in order):
            hv = (head[1], head[2])
            free = sorted({a for atom in body for a in atom[1:]} - set(hv))
            ids = {hv[0]: E, hv[1]: E + 1, **{v: E + 2 + i for i, v in enumerate(free)}}
            known, rest, order = set(hv), list(range(len(body))), []
            while rest:                                    # the dependency order
                nxt = next((i for i in rest if body[i][1] in known or body[i][2] in known), None)
                for i in (rest if nxt is None else [nxt]):
                    order.append(i)
                    known |= set(body[i][1:])
                rest = [] if nxt is None else [i for i in rest if i != nxt]

            def arg(a):
                return entity2id[a] if not is_variable(a) and a in entity2id else ids[a]
            heads.append([relation2id[head[0]], E, E + 1])
            bodies.append([[relation2id[body[i][0]], arg(body[i][1]), arg(body[i][2])] for i in order])
            n_free = max(n_free, len(free))
        pad = E + 2 + n_free
        M = max(len(b) for b in bodies)
        body_t = torch.full((len(bodies), M, 3), 0, dtype=torch.long)
        for r, b in enumerate(bodies):
            body_t[r, :len(b)] = torch.tensor(b)
        lens = torch.tensor([len(b) for b in bodies])
        fact_t = torch.tensor(list(facts), dtype=torch.long).reshape(-1, 3)
        return cls(fact_t, torch.tensor(heads), body_t, lens, E=E, pad=pad, P=len(relation2id), device=device,
                   order=torch.tensor(order))

    def patterns(self) -> List[RulePattern]:
        """The rules' binding analysis."""
        return compile_rules(self.rules.heads, self.rules.bodies, self.rules.lens, self.E - 1)

    def __repr__(self) -> str:
        return f"KB(facts={len(self.facts)}, rules={len(self.rules)}, E={self.E}, P={self.P})"


__all__ = ["KB", "Facts", "Rules", "RulePattern", "PatternVariant", "compile_rules", "parse_rules",
           "parse_triples", "is_variable", "HEAD0", "HEAD1", "FREE"]
