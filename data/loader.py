"""Text parsing for KG datasets — facts, rules (arrow + Prolog), variables.

Pure string→tuple parsing, no tensors and no vocabulary (``dataset.py`` owns
those). Split out so the dataset module stays about encoding, and FC/analysis
callers can parse rule text without constructing a full dataset.

BIT-EXACT RECIPES (fingerprint-enforced — these fix the id assignment):
  * the atom regex accepts ``/`` and ``.`` in predicate names (fb15k relations);
    ``\\w+`` would truncate at the last dotted segment and desync vocabularies
  * triple parsing prefers ``pred(a,b)`` via ``rfind("(")``, then tab-separated
  * a variable is a single lowercase letter OR an uppercase-initial token
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

Triple = Tuple[str, str, str]
Rule = Tuple[Triple, List[Triple]]

# pred(arg0,arg1) — predicate may contain / and . (e.g. /people/person/place_of_birth)
_ATOM_RE = re.compile(r"([\w/.]+)\(([^,]+),([^)]+)\)")
_VAR_LOWER = re.compile(r"^[a-z]$")
_VAR_UPPER = re.compile(r"^[A-Z]")


def is_variable(name: str) -> bool:
    """A logical variable: single lowercase letter or uppercase-initial token."""
    return bool(_VAR_LOWER.match(name) or _VAR_UPPER.match(name))


def parse_triples(path: Path) -> List[Triple]:
    """Parse a triples file: ``pred(a,b)`` (preferred) or tab-separated."""
    triples: List[Triple] = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(".")
            if not line or line.startswith("#"):
                continue
            paren = line.rfind("(")
            if paren > 0 and line.endswith(")"):
                pred, args = line[:paren], line[paren + 1:-1]
                parts = args.split(",", 1)
                if len(parts) == 2:
                    triples.append((pred.strip(), parts[0].strip(), parts[1].strip()))
                    continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def parse_atom(s: str) -> Optional[Triple]:
    """Parse one ``pred(a, b)`` atom into ``(pred, a, b)`` (or None)."""
    m = _ATOM_RE.search(s.strip())
    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else None


def parse_body_atoms(body_str: str) -> List[Triple]:
    """Parse a comma-separated rule body. Splits on ``),`` (re-adding ``)``)."""
    atoms: List[Triple] = []
    for frag in body_str.split("),"):
        frag = frag.strip()
        if not frag.endswith(")"):
            frag += ")"
        atom = parse_atom(frag)
        if atom is not None:
            atoms.append(atom)
    return atoms


def _parse_rules_arrow(path: Path) -> List[Rule]:
    """Arrow format: ``rN:score:body1(a,h), body2(b,h) -> head(a,b)``."""
    rules: List[Rule] = []
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
            head = parse_atom(head_str)
            if head is None:
                continue
            body = parse_body_atoms(body_str)
            if body:
                rules.append((head, body))
    return rules


def _parse_rules_prolog(path: Path) -> List[Rule]:
    """Prolog format: ``head(X,Y) :- body1(X,Z), body2(Z,Y)``."""
    rules: List[Rule] = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(".")
            if not line or line.startswith("#") or ":-" not in line:
                continue
            head_str, body_str = line.split(":-", 1)
            head = parse_atom(head_str)
            if head is None:
                continue
            body = parse_body_atoms(body_str)
            if body:
                rules.append((head, body))
    return rules


def parse_rules(path: Path) -> List[Rule]:
    """Auto-detect arrow vs Prolog by the first non-comment line."""
    first = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                first = line
                break
    return _parse_rules_prolog(path) if ":-" in first else _parse_rules_arrow(path)


__all__ = [
    "Triple", "Rule", "is_variable",
    "parse_triples", "parse_atom", "parse_body_atoms", "parse_rules",
]
