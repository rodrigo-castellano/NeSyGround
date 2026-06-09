"""CONTRACT: every data-structure field name lives in ``glossary.FIELDS``, every
config field + ctor param lives in ``glossary.KNOBS``, and there are NO aliases.

Enforces the controlled vocabulary — one canonical name per concept, no random or
overlapping names, ZERO aliases. To add a field/knob you must first
register its canonical name; that single chokepoint is where you notice an
existing synonym. Scope: EVERY package module (auto-discovered), not a fixed list.
"""
from __future__ import annotations

import ast
import collections
import dataclasses
import importlib
import inspect
import pkgutil
from pathlib import Path

import grounder
from grounder.vocab.glossary import FIELDS, KNOBS

_GLOSSARY_SRC = Path(__file__).parent.parent / "vocab" / "glossary.py"

# Tokens that legitimately span GLOSSARY (concept) and FIELDS (field) — the same
# concept named in both registries. The disjointness check (below) allows ONLY
# these and flags any other overlap.
_GLOSSARY_FIELDS_CROSSLINKS = frozenset({
    "layout", "cell", "frontier", "pbc", "resolution", "strategy", "width",
})

# OutputSpec's per-tier @property accessors over its ``tiers`` field. They are
# @property (not dataclass fields) but are part of the documented field surface,
# so they are NOT "dead" FIELDS entries.
_OUTPUTSPEC_BACKCOMPAT = frozenset({"groundings", "firings", "trees"})

# ``Shapes`` fields are owned by the SHAPES registry (enforced by
# ``test_shape_symbols_match_vocab``); the ``config`` module's dataclass fields are
# owned by KNOBS (enforced by ``test_config_fields_registered``). Both have their
# own vocabulary, so they are NOT scanned against FIELDS.
# Shapes owns the SHAPES vocab; config owns KNOBS; the analysis BFS diagnostics
# (BFSResult/BatchBFSStats) are a separate metric vocabulary (branching factors,
# frontier sizes) — diagnostic outputs, not grounding-pipeline fields.
_SKIP_FIELD_CLASSES = frozenset({"Shapes", "BFSResult", "BatchBFSStats"})


def _iter_package_modules():
    """Auto-discover every importable module in the grounder package (skip the
    test/contract/doc/script trees — they are not part of the shipped surface)."""
    for info in pkgutil.walk_packages(grounder.__path__, grounder.__name__ + "."):
        name = info.name
        if any(seg in name.split(".")
               for seg in ("contracts", "tests", "docs", "scripts")):
            continue
        try:
            yield importlib.import_module(name)
        except Exception:
            continue  # optional deps; the import contract tests own importability


def _collect_fields() -> dict[str, set[str]]:
    """field name -> {Module.Class that declares it}, over EVERY package module."""
    used: dict[str, set[str]] = {}
    for mod in _iter_package_modules():
        name = mod.__name__
        short = name.split(".")[-1]
        is_config = name == "grounder.api.config"
        for cls_name in dir(mod):
            obj = getattr(mod, cls_name)
            if not isinstance(obj, type) or getattr(obj, "__module__", None) != name:
                continue  # skip imported classes (declared elsewhere)
            if cls_name in _SKIP_FIELD_CLASSES or is_config:
                continue  # SHAPES / KNOBS own these vocabularies
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
    used = set(_collect_fields()) | _OUTPUTSPEC_BACKCOMPAT
    dead = sorted(set(FIELDS) - used)
    assert not dead, f"FIELDS entries used by no data structure (remove them): {dead}"


# ── KNOBS contracts (config fields + exec knobs) ──

def _config_field_owners() -> dict[str, set[str]]:
    """config dataclass field name -> {config class(es) declaring it}."""
    from grounder.api import config as config_mod
    used: dict[str, set[str]] = {}
    for name in dir(config_mod):
        obj = getattr(config_mod, name)
        if (isinstance(obj, type) and obj.__module__ == config_mod.__name__
                and dataclasses.is_dataclass(obj)):
            for f in dataclasses.fields(obj):
                used.setdefault(f.name, set()).add(name)
    return used


def test_config_fields_registered() -> None:
    """Every grounder config dataclass field must be registered in glossary.KNOBS."""
    used = _config_field_owners()
    bad = {f: sorted(w) for f, w in used.items() if f not in KNOBS}
    assert not bad, f"Config fields not in glossary.KNOBS: {bad}"


# make_grounder exec knobs + the KB construction knobs (genuine ctor knobs that are
# NOT config-dataclass fields). KNOBS == config_fields ∪ this set, exactly.
_EXEC_AND_KB_KNOBS = frozenset({
    "layout", "compile", "chunk_size", "transforms",        # make_grounder exec knobs
    "max_facts_per_query", "fact_index_type", "max_memory_mb",  # KB knobs
    "fact_order", "rule_order", "order_seed", "pack_base",
})


def test_no_dead_knobs() -> None:
    """KNOBS must not list params no config / entry-point carries (keep it honest)."""
    used = set(_config_field_owners())
    dead = sorted(set(KNOBS) - used - _EXEC_AND_KB_KNOBS)
    assert not dead, f"KNOBS lists unused params: {dead}"


# Structural / data params on entry points — the subject + raw data, NOT knobs.
_STRUCTURAL_PARAMS = frozenset({
    "self", "cls", "kb", "config",
    "facts_idx", "rules_heads_idx", "rules_bodies_idx", "rule_lens",
    "constant_no", "predicate_no", "padding_idx", "device",
    "_cached_rule_index", "_cached_binding_tables",
})


def _param_names(func) -> list[str]:
    return [p.name for p in inspect.signature(func).parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def test_ctor_params_registered() -> None:
    """Every parameter of the public entry points must be a known knob (in KNOBS)
    or a structural/data param. No unknown args sneak in."""
    from grounder.api.factory import make_grounder
    from grounder.api.backward import BackwardGrounder
    from grounder.api.forward import ForwardGrounder
    from grounder.data.kb import KB
    bad: dict[str, list[str]] = {}
    for fn, label in ((make_grounder, "make_grounder"),
                      (BackwardGrounder.__init__, "BackwardGrounder.__init__"),
                      (ForwardGrounder.__init__, "ForwardGrounder.__init__"),
                      (KB.__init__, "KB.__init__")):
        unknown = [p for p in _param_names(fn)
                   if p not in KNOBS and p not in _STRUCTURAL_PARAMS]
        if unknown:
            bad[label] = unknown
    assert not bad, f"Entry-point params not in glossary.KNOBS (nor structural): {bad}"


# ── alias guard ──

_PUBLIC_PKG_DIR = Path(__file__).parent.parent
_ALIAS_ALLOW: frozenset[str] = frozenset()   # empty: NO module-level Name=Name aliases


def test_no_aliases() -> None:
    """Forbid module-level ``Name = OtherName`` re-export aliases in package code.

    A bare ``ast.Name`` on the RHS of a module-level assignment is an alias (e.g.
    ``GrounderOutput = BackwardResult``). Subscripts/calls (``X = Union[...]``,
    ``X = make()``) are NOT aliases. Empty allow-list — every name is canonical,
    so no module-level re-export alias is permitted."""
    offenders: list[str] = []
    for py in _PUBLIC_PKG_DIR.rglob("*.py"):
        parts = py.relative_to(_PUBLIC_PKG_DIR).parts
        if any(seg in ("contracts", "tests", "docs", "scripts") for seg in parts):
            continue
        tree = ast.parse(py.read_text())
        for node in tree.body:  # module level ONLY
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Name)):
                lhs = node.targets[0].id
                if lhs not in _ALIAS_ALLOW:
                    offenders.append(f"{py.name}:{node.lineno} {lhs} = {node.value.id}")
    assert not offenders, f"Module-level aliases (one canonical name per concept!): {offenders}"


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
    for name in ("GLOSSARY", "FIELDS", "SHAPES", "KNOBS",
                 "LAYOUTS", "INTEGRATIONS", "COMPILE_BACKENDS"):
        keys = _literal_keys(name)
        dupes = sorted(k for k, c in collections.Counter(keys).items() if c > 1)
        assert not dupes, f"{name} has duplicate keys (a dict literal hides these!): {dupes}"


def test_glossary_fields_token_disjoint() -> None:
    """GLOSSARY (concepts) and FIELDS (field names) must be token-disjoint, EXCEPT
    where a token legitimately spans both as the same concept — those must be
    declared in ``_GLOSSARY_FIELDS_CROSSLINKS``."""
    from grounder.vocab.glossary import GLOSSARY
    overlap = set(GLOSSARY) & set(FIELDS)
    undeclared = sorted(overlap - _GLOSSARY_FIELDS_CROSSLINKS)
    assert not undeclared, (
        "GLOSSARY concept and FIELDS field share a token without a declared "
        f"cross-link: {undeclared}. Rename one, or add it to "
        "_GLOSSARY_FIELDS_CROSSLINKS if it is the SAME concept in both.")
    # keep the allow-list honest: every declared crosslink must actually overlap.
    stale = sorted(_GLOSSARY_FIELDS_CROSSLINKS - overlap)
    assert not stale, f"_GLOSSARY_FIELDS_CROSSLINKS lists non-overlapping tokens: {stale}"


def test_shape_symbols_match_vocab() -> None:
    """The Shapes dataclass fields must EXACTLY equal the SHAPES registry —
    no rogue or missing shape symbol."""
    from grounder.vocab.shapes import Shapes
    from grounder.vocab.glossary import SHAPES
    fields = {f.name for f in dataclasses.fields(Shapes)}
    only_in_shapes = fields - set(SHAPES)
    only_in_vocab = set(SHAPES) - fields
    assert not only_in_shapes and not only_in_vocab, (
        f"Shapes <-> SHAPES vocab mismatch: only in Shapes={only_in_shapes}, "
        f"only in vocab={only_in_vocab}")


# ── grounder instance-attribute contract ──
# Private impl attributes whose name is NOT a glossary concept. Everything else must
# resolve — after stripping a leading underscore — to a SHAPES/SCALARS/KNOBS/FIELDS/
# GLOSSARY name. One canonical name per concept; no second spelling of engine state.
_GROUNDER_PRIVATE: dict[str, str] = {
    "kb": "the bound KB (submodule)",
    "_enum_ri": "the PbcRuleIndex — pbc binding analysis (submodule)",
    "_build": "ctor inputs snapshot, replayed by rebound()",
    "_dispatch_resolution": "resolver registry key: sld | rtf | pbc | join",
    "_exec_layout": "resolved Layout for the exec cell",
    "_exec_compile": "resolved compile spec for the exec cell",
    "_knobs_set": "whether any exec knob was set explicitly",
    "_layout_knob": "raw layout knob string (auto | dense | flat)",
    "_compile_knob": "raw compile knob string (off | graph | dynamic)",
    "_chunk_size": "raw chunk-size knob (None = auto)",
    "_collect_mode": "groundings collection mode (terminal)",
    "_standardize_fn": "built variable-renaming fn, or None",
    "_num_entities": "FC: entity count (E)",
    "_num_predicates": "FC: real predicate range (P)",
    "_rules": "FC: compiled RulePattern list",
}


def _sample_grounders() -> list:
    """Representative grounders (pbc / sld / FC) for attribute introspection."""
    import torch
    from grounder.api.config import Backward, Forward, PBC, SLD
    from grounder.api.factory import make_grounder
    from grounder.data.kb import KB
    kb = KB(torch.tensor([[2, 1, 2], [2, 2, 3]]), torch.tensor([[1, 4, 5]]),
            torch.tensor([[[2, 4, 6], [2, 6, 5]]]), torch.tensor([2]),
            constant_no=3, predicate_no=17, padding_idx=16,
            device=torch.device("cpu"), fact_index_type="block_sparse")
    return [make_grounder(kb, Backward(PBC(depth=2, width=1))),
            make_grounder(kb, Backward(SLD(depth=2), max_children=64)),
            make_grounder(kb, Forward(depth=5))]


def test_grounder_attributes_registered() -> None:
    """Every grounder instance attribute is a glossary name (stripped of a leading
    underscore) or an explicitly-allowed private impl attr — no unregistered state,
    no second spelling, no dead allow-list entry."""
    import torch.nn as nn
    from grounder.vocab.glossary import FIELDS, GLOSSARY, KNOBS, SCALARS, SHAPES
    known = set(SHAPES) | set(SCALARS) | set(KNOBS) | set(FIELDS) | set(GLOSSARY)
    nn_base = set(nn.Module().__dict__)
    seen, bad = set(), {}
    for g in _sample_grounders():
        for a in (set(g.__dict__) - nn_base) | set(g._modules):
            seen.add(a)
            if a.lstrip("_") not in known and a not in _GROUNDER_PRIVATE:
                bad.setdefault(type(g).__name__, []).append(a)
    assert not bad, ("Grounder attrs not in glossary nor _GROUNDER_PRIVATE "
                     f"(register the canonical name, or document it private): {bad}")
    dead = sorted(set(_GROUNDER_PRIVATE) - seen)
    assert not dead, f"_GROUNDER_PRIVATE lists attrs no grounder has: {dead}"


if __name__ == "__main__":
    used = _collect_fields()
    print(f"collected {len(used)} distinct field names")
    bad = {fn: sorted(w) for fn, w in used.items() if fn not in FIELDS}
    print("unregistered:", bad or "none")
    print("dead FIELDS entries:", sorted(set(FIELDS) - set(used)) or "none")
