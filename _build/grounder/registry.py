"""Seam registries — one keyed table per axis (each has >=2 real impls).

AXIS 1 (Resolver): ``RESOLVERS`` maps the backward resolution name
``{sld, rtf, pbc}`` to its ``Resolver`` impl, replacing engine/step.py's if/elif.
AXIS 3 (ForwardMethod): ``FORWARD_METHODS`` maps ``{spmm, staged}`` to its
closure-engine impl (owned by forward/methods.py; re-exported here as the
canonical registry surface). ``TRANSFORMS`` arrives in a later step.
"""
from __future__ import annotations

from typing import Dict

from grounder._build.forward.methods import FORWARD_METHODS, ForwardMethod
from grounder._build.resolution.api import Resolver
from grounder._build.resolution.pbc.resolve import PbcResolver
from grounder._build.resolution.rtf import RtfResolver
from grounder._build.resolution.sld import SldResolver

RESOLVERS: Dict[str, Resolver] = {
    "sld": SldResolver(),
    "rtf": RtfResolver(),
    "pbc": PbcResolver(),
}

__all__ = ["RESOLVERS", "FORWARD_METHODS", "ForwardMethod"]
