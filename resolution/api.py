"""AXIS 1 — the Resolver seam contract.

A ``Resolver`` is the backward RESOLVE-step strategy ``{sld, rtf, pbc}``: given one
proof step's request it produces the children (``ResolvedChildren`` dense or
``FlatResolvedChildren`` flat). ``ResolveRequest`` is the honest superset of the
per-family kwarg sets (and pbc's ``StepInputs``) carried to the seam.

The hook asymmetry is REAL: sld/rtf take ``fact_hook``/``rule_hook`` but pbc's
resolve_step takes none. The unified ``ResolveRequest`` carries the hooks Optional
+ INERT for pbc (the PbcResolver ignores them). The dense/flat materializer choice
stays pbc-internal (PbcResolver picks it).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, NamedTuple, Optional, Protocol, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    from grounder.execution.capability import Cell
    from grounder.execution.depth import DepthSelector


@runtime_checkable
class Resolver(Protocol):
    name: str                                   # "sld" | "rtf" | "pbc" | "join"

    def declared_cells(self) -> FrozenSet["Cell"]: ...   # contributes to CapabilityRow

    def resolve(self, req: "ResolveRequest"):            # -> ResolvedChildren | FlatResolvedChildren
        ...


class ResolveRequest(NamedTuple):
    """One backward RESOLVE step — the honest superset of the sld/rtf/pbc kwargs.

    Carries the per-step tensors, the run-level plan (binding tables, KB, knobs)
    and a ``DepthSelector`` (NOT a bare int, so the compiled path stays expressible).
    ``fact_hook``/``rule_hook`` are Optional: wired for sld/rtf, INERT for pbc.
    """
    plan: object                                # RunPlan (binding tables, KB, knobs)
    queries: Tensor                             # [B, G, 3]
    remaining: Tensor                           # [B, G, L, 3]
    goal_valid: Tensor                         # [B, G]
    active_mask: Tensor                         # [B, G]
    frontier: object                            # Frontier (next_var/top_rule_idx/grounding_body/...)
    depth_selector: "DepthSelector"             # d / is_last duality (compiled path expressible)
    excluded_queries: Optional[Tensor] = None
    fact_hook: object = None                    # Optional; wired for sld/rtf, INERT for pbc
    rule_hook: object = None                    # Optional; wired for sld/rtf, INERT for pbc


__all__ = ["Resolver", "ResolveRequest"]
