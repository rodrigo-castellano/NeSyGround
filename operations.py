"""Backward-compatibility shim — canonical location is resolution/mgu.py."""
from grounder.resolution.mgu import (  # noqa: F401
    FactIndex,
    RuleIndex,
    mgu_resolve_atom_facts,
    mgu_resolve_atom_rules,
    unify_with_facts,
    unify_with_rules,
)
