"""CONTRACTS for the registry seams (AXIS 1 Resolver + AXIS 3 ForwardMethod +
AXIS 4 ProgramTransform).

- RESOLVERS is keyed exactly {sld, rtf, pbc, join}; every entry structurally
  implements the Resolver Protocol and its ``.name`` matches its registry key.
  ``join`` is the L3 semantics-preserving sibling of pbc (flat-eager cell).
- the resolve() dispatch in backward/step.py is a pure registry lookup: NO if/elif
  branching on a ``.resolution`` attribute survives in that module (AST scan).
- declared_cells() is a non-empty FrozenSet[Cell] for every resolver.
- FORWARD_METHODS is keyed exactly {spmm, staged}; every entry impls the
  ForwardMethod Protocol with matching ``.name``; run_forward_chaining (the
  per-rule router) dispatches via FORWARD_METHODS, not a direct engine call.
- TRANSFORMS is keyed exactly {identity, magic_set}; every entry is a TYPE whose
  instances implement the ProgramTransform Protocol with matching ``.name``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from grounder.core.request import GroundRequest
from grounder.data.kb import KB
from grounder.execution.capability import Cell
from grounder.forward.methods import ForwardMethod
from grounder.grounder.registry import FORWARD_METHODS, RESOLVERS, TRANSFORMS
from grounder.resolution.api import Resolver
from grounder.transform.api import IdentityTransform, ProgramTransform
from grounder.transform.magic_set import MagicSetTransform

_STEP_PY = Path(__file__).parent.parent / "backward" / "step.py"
_FC_PY = Path(__file__).parent.parent / "forward" / "fc.py"


def test_resolvers_keyed_exactly() -> None:
    assert set(RESOLVERS) == {"sld", "rtf", "pbc", "join"}


def test_every_resolver_impls_protocol() -> None:
    for key, r in RESOLVERS.items():
        assert isinstance(r, Resolver), f"{key} does not implement Resolver"
        assert r.name == key, f"{key}: name {r.name!r} != registry key"


def test_resolver_declared_cells_nonempty_frozenset_of_cell() -> None:
    for key, r in RESOLVERS.items():
        cells = r.declared_cells()
        assert isinstance(cells, frozenset) and cells, f"{key}: declared_cells empty/not frozenset"
        assert all(isinstance(c, Cell) for c in cells), f"{key}: non-Cell in declared_cells"


def _references_resolution_attr(node: ast.AST) -> bool:
    """True if the test contains an attribute access ending in `.resolution`."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "resolution":
            return True
    return False


def test_step_resolve_has_no_resolution_branching() -> None:
    """backward/step.py must dispatch via RESOLVERS, not an if/elif on .resolution."""
    tree = ast.parse(_STEP_PY.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _references_resolution_attr(node.test):
            offenders.append(node.lineno)
    assert not offenders, f"if/elif on .resolution survives in step.py at lines {offenders}"


def test_step_resolve_uses_registry() -> None:
    """resolve() must reference the RESOLVERS table (the dispatch is a lookup)."""
    src = _STEP_PY.read_text()
    assert "RESOLVERS[" in src, "backward/step.py does not dispatch via RESOLVERS[...]"


# ── AXIS 3 — ForwardMethod registry ──

def test_forward_methods_keyed_exactly() -> None:
    assert set(FORWARD_METHODS) == {"spmm", "staged"}


def test_every_forward_method_impls_protocol() -> None:
    for key, m in FORWARD_METHODS.items():
        assert isinstance(m, ForwardMethod), f"{key} does not implement ForwardMethod"
        assert m.name == key, f"{key}: name {m.name!r} != registry key"


def test_run_forward_chaining_uses_registry() -> None:
    """run_forward_chaining is the per-rule router: it dispatches via
    FORWARD_METHODS, not by directly constructing FCDynamic / calling spmm."""
    src = _FC_PY.read_text()
    assert "FORWARD_METHODS[" in src, "fc.py router does not dispatch via FORWARD_METHODS[...]"


# ── AXIS 4 — ProgramTransform registry ──

def test_transforms_keyed_exactly() -> None:
    assert set(TRANSFORMS) == {"identity", "magic_set"}


def test_transforms_are_types_with_matching_name() -> None:
    for key, t in TRANSFORMS.items():
        assert isinstance(t, type), f"{key}: TRANSFORMS entry is not a type"
        assert t.name == key, f"{key}: type.name {t.name!r} != registry key"


def _toy_kb() -> KB:
    pad, C = 12, 11
    facts = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    X, Y = 13, 14
    heads = torch.tensor([[1, X, Y]], dtype=torch.long)
    bodies = torch.tensor([[[0, X, Y]]], dtype=torch.long)
    lens = torch.tensor([1], dtype=torch.long)
    return KB(facts, heads, bodies, lens, constant_no=C, predicate_no=2,
              padding_idx=pad, device=torch.device("cpu"))


def test_identity_transform_impls_protocol_and_is_noop() -> None:
    t = IdentityTransform()
    assert isinstance(t, ProgramTransform)
    kb = _toy_kb()
    req = GroundRequest()
    kb2, req2 = t.apply(kb, req)
    assert kb2 is kb and req2 is req


def test_magic_set_instance_impls_protocol() -> None:
    inst = MagicSetTransform(_toy_kb(), query_pred=1)
    assert isinstance(inst, ProgramTransform)
    assert inst.name == "magic_set"
