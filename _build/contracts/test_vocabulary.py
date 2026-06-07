"""CONTRACT: every data-structure field name lives in ``glossary.FIELDS``.

Enforces the controlled vocabulary — one canonical name per concept, no random or
overlapping names. To add a field you must first register its name in
``glossary.FIELDS``; that single chokepoint is where you notice an existing
synonym. Scope: all dataclasses / NamedTuples in ``state.py`` and ``types.py``
(more modules are added to ``_MODULES`` as they are built).
"""
from __future__ import annotations

import ast
import collections
import dataclasses
from pathlib import Path

from grounder._build import state as state_mod
from grounder._build import types as types_mod
from grounder._build.glossary import FIELDS

_GLOSSARY_SRC = Path(__file__).parent.parent / "glossary.py"

_MODULES = (state_mod, types_mod)


def _collect_fields() -> dict[str, set[str]]:
    """field name -> {Module.Class that declares it}."""
    used: dict[str, set[str]] = {}
    for mod in _MODULES:
        short = mod.__name__.split(".")[-1]
        for cls_name in dir(mod):
            obj = getattr(mod, cls_name)
            if not isinstance(obj, type):
                continue
            if obj.__module__ != mod.__name__:
                continue  # skip imported classes (e.g. Shapes — its own symbol vocab)
            if dataclasses.is_dataclass(obj):
                names = [f.name for f in dataclasses.fields(obj)]
            elif issubclass(obj, tuple) and hasattr(obj, "_fields"):
                names = list(obj._fields)
            else:
                continue
            for fn in names:
                used.setdefault(fn, set()).add(f"{short}.{cls_name}")
    return used


def test_all_field_names_registered() -> None:
    used = _collect_fields()
    unregistered = {fn: sorted(w) for fn, w in used.items() if fn not in FIELDS}
    assert not unregistered, (
        "Unregistered field name(s) — add the canonical name to glossary.FIELDS, "
        f"or reuse an existing one: {unregistered}")


def test_no_dead_vocab_entries() -> None:
    """FIELDS shouldn't accumulate names nothing uses (keeps the vocab honest)."""
    used = set(_collect_fields())
    dead = sorted(set(FIELDS) - used)
    assert not dead, f"FIELDS entries used by no data structure (remove them): {dead}"


def _literal_keys(dict_name: str) -> list[str]:
    """Keys as written in the glossary SOURCE (a dict literal silently drops
    duplicate keys at runtime, so we must read the source to detect them)."""
    tree = ast.parse(_GLOSSARY_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tgt, val = node.target.id, node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            tgt, val = node.targets[0].id, node.value
        else:
            continue
        if tgt == dict_name and isinstance(val, ast.Dict):
            return [k.value for k in val.keys if isinstance(k, ast.Constant)]
    raise AssertionError(f"{dict_name} dict literal not found in glossary.py")


def test_no_duplicate_vocab_keys() -> None:
    for name in ("GLOSSARY", "FIELDS", "SHAPES", "LAYOUTS", "INTEGRATIONS", "COMPILE_BACKENDS"):
        keys = _literal_keys(name)
        dupes = sorted(k for k, c in collections.Counter(keys).items() if c > 1)
        assert not dupes, f"{name} has duplicate keys (a dict literal hides these!): {dupes}"


def test_shape_symbols_match_vocab() -> None:
    """The Shapes dataclass fields must EXACTLY equal the SHAPES registry —
    no rogue or missing shape symbol."""
    from grounder._build.shapes import Shapes
    from grounder._build.glossary import SHAPES
    fields = {f.name for f in dataclasses.fields(Shapes)}
    only_in_shapes = fields - set(SHAPES)
    only_in_vocab = set(SHAPES) - fields
    assert not only_in_shapes and not only_in_vocab, (
        f"Shapes <-> SHAPES vocab mismatch: only in Shapes={only_in_shapes}, "
        f"only in vocab={only_in_vocab}")


if __name__ == "__main__":
    used = _collect_fields()
    print(f"collected {len(used)} distinct field names across {len(_MODULES)} modules")
    bad = {fn: sorted(w) for fn, w in used.items() if fn not in FIELDS}
    print("unregistered:", bad or "none")
    print("dead FIELDS entries:", sorted(set(FIELDS) - set(used)) or "none")
    # demonstrate the contract catches a rogue overlapping name:
    rogue = "mask"  # the old name we removed; must NOT be accepted anymore
    print(f"rogue name {rogue!r} in vocab? {rogue in FIELDS} (expected False)")
