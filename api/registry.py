"""Seam registries — one keyed table per axis (each has >=2 real impls).

AXIS 1 (Resolver): ``RESOLVERS`` maps the backward resolution name
``{sld, pbc}`` to its ``Resolver`` impl, replacing backward/step.py's
if/elif. (pbc's flat path prunes by default — ``PBC.flat_prune`` — folding the
former "join" sibling into ``PbcResolver``.)
AXIS 3 (ForwardMethod): ``FORWARD_METHODS`` maps ``{spmm, staged}`` to its
closure-engine impl (owned by forward/methods.py; re-exported here as the
canonical registry surface).
"""
from __future__ import annotations

from typing import Dict

from grounder.forward.methods import FORWARD_METHODS, ForwardMethod
from grounder.resolution.api import Resolver
from grounder.resolution.pbc.resolve import PbcResolver
from grounder.resolution.sld import SldResolver

RESOLVERS: Dict[str, Resolver] = {
    "sld": SldResolver(),
    "pbc": PbcResolver(),
}

__all__ = ["RESOLVERS", "FORWARD_METHODS", "ForwardMethod"]
