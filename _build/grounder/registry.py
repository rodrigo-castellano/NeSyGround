"""Seam registries — one keyed table per axis (each has >=2 real impls).

AXIS 1 (Resolver): ``RESOLVERS`` maps the backward resolution name
``{sld, rtf, pbc, join}`` to its ``Resolver`` impl, replacing backward/step.py's
if/elif. ``join`` is the L3 semantics-preserving sibling of ``pbc``.
AXIS 3 (ForwardMethod): ``FORWARD_METHODS`` maps ``{spmm, staged}`` to its
closure-engine impl (owned by forward/methods.py; re-exported here as the
canonical registry surface).
AXIS 4 (ProgramTransform): ``TRANSFORMS`` maps ``{identity, magic_set}`` to its
transform TYPE (constructed by ``make_grounder`` over a KB, composed by Pipeline).
"""
from __future__ import annotations

from typing import Dict, Type

from grounder._build.forward.methods import FORWARD_METHODS, ForwardMethod
from grounder._build.resolution.api import Resolver
from grounder._build.resolution.pbc.resolve import JoinResolver, PbcResolver
from grounder._build.resolution.rtf import RtfResolver
from grounder._build.resolution.sld import SldResolver
from grounder._build.transform.api import IdentityTransform, ProgramTransform
from grounder._build.transform.magic_set import MagicSetTransform

RESOLVERS: Dict[str, Resolver] = {
    "sld": SldResolver(),
    "rtf": RtfResolver(),
    "pbc": PbcResolver(),
    "join": JoinResolver(),
}

# Transforms are keyed to their TYPE (not an instance): they construct over a KB
# (and a query pattern), so make_grounder / callers instantiate them.
TRANSFORMS: Dict[str, Type[ProgramTransform]] = {
    "identity": IdentityTransform,
    "magic_set": MagicSetTransform,
}

__all__ = ["RESOLVERS", "FORWARD_METHODS", "ForwardMethod", "TRANSFORMS"]
