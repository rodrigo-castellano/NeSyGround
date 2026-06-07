"""Grounder factory — parse a type string and build a grounder.

Type grammar (dot-separated):  {resolution}[.{filter}][.pd][.full|.wW][.dD]
  sld.fp_batch.d2 · rtf.d4 · pbc.fp_batch.w1.d2 · pbc.full · closure
BC_{w,d}[u] shorthand: bc12, bc13, bc12u1 → pbc with the paper-aligned filter.

``enum`` is accepted as a legacy alias for ``pbc`` (the new resolution name).
``make_bcwd(kb, w, d, u=0)`` is the direct BC_{w,d,u} constructor.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder._build.data.kb import KB
from grounder._build.grounder.backward import BackwardGrounder

_PATTERN = re.compile(
    r"^(?P<resolution>sld|rtf|enum|pbc|closure)"
    r"(\.(?P<filter>fp_batch|fp_global|prune|provset|none))?"
    r"(?P<pd>\.pd)?"
    r"(\.full|\.w(?P<width>\d+))?"
    r"(\.d(?P<depth>\d+))?$"
)
_BCWD_PATTERN = re.compile(r"^bc(?P<width>\d)(?P<depth>\d)(?:u(?P<u>\d))?$")
_FILTER_ALIASES = {"prune": "fp_batch", "provset": "fp_global"}


def _default_pbc_filter(u: int) -> str:
    """u=0 (paper) → fp_batch (keras prune=True); u>0 → none."""
    return "fp_batch" if u == 0 else "none"


def parse_grounder_type(grounder_type: str) -> Dict[str, Any]:
    """Parse a type string into a config dict (resolution uses the new ``pbc`` name)."""
    m_bcwd = _BCWD_PATTERN.match(grounder_type)
    if m_bcwd is not None:
        u = int(m_bcwd.group("u")) if m_bcwd.group("u") is not None else 0
        return {"resolution": "pbc", "filter": _default_pbc_filter(u),
                "depth": int(m_bcwd.group("depth")), "width": int(m_bcwd.group("width")),
                "is_full": False, "step_prune_dead": False,
                "flat_intermediate": True, "u": u}

    clean = grounder_type.replace(".flat", "")
    m = _PATTERN.match(clean)
    if not m:
        raise ValueError(
            f"Unknown grounder type: {grounder_type!r}. Expected "
            f"'{{sld|rtf|pbc|closure}}[.{{filter}}][.wW][.dD]' or BC shorthand 'bcWD'.")

    resolution = m.group("resolution")
    if resolution == "enum":          # legacy alias
        resolution = "pbc"
    is_full = ".full" in grounder_type
    flat_intermediate = ".flat" in grounder_type
    depth = int(m.group("depth")) if m.group("depth") else 1
    width = int(m.group("width")) if m.group("width") else 1
    default_filter = _default_pbc_filter(0) if resolution == "pbc" else "none"
    filter_mode = m.group("filter") or default_filter
    filter_mode = _FILTER_ALIASES.get(filter_mode, filter_mode)
    return {"resolution": resolution, "filter": filter_mode, "depth": depth,
            "width": width, "is_full": is_full,
            "step_prune_dead": bool(m.group("pd")), "flat_intermediate": flat_intermediate, "u": 0}


def create_grounder(
    grounder_type: str,
    *,
    facts_idx: Tensor,
    rule_heads: Tensor,
    rule_bodies: Tensor,
    rule_lens: Tensor,
    constant_no: int,
    padding_idx: int,
    device: torch.device,
    predicate_no: Optional[int] = None,
    max_facts_per_query: int = 64,
    fact_index_type: str = "block_sparse",
    max_groundings: int = 32,
    max_total_groundings: int = 64,
    fc_method: str = "spmm",
    max_goals: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build a KB + grounder from a type string (the one construction entry)."""
    cfg = parse_grounder_type(grounder_type)
    kb = KB(facts_idx, rule_heads, rule_bodies, rule_lens,
            constant_no=constant_no, predicate_no=predicate_no, padding_idx=padding_idx,
            device=device, max_facts_per_query=max_facts_per_query, fact_index_type=fact_index_type)

    if cfg["resolution"] == "closure":
        from grounder._build.forward.grounder import ForwardGrounder
        return ForwardGrounder(kb, method=fc_method, **kwargs)

    g_kwargs: Dict[str, Any] = dict(
        resolution=cfg["resolution"], filter=cfg["filter"], depth=cfg["depth"],
        max_total_groundings=max_total_groundings, max_goals=max_goals)
    if cfg["resolution"] == "pbc":
        if cfg["is_full"]:
            g_kwargs["width"] = None
            g_kwargs["depth"] = 1
        else:
            g_kwargs["width"] = cfg["width"]
        g_kwargs["max_groundings_per_query"] = max_groundings
        g_kwargs["flat_intermediate"] = cfg["flat_intermediate"]
        g_kwargs["fc_method"] = fc_method
    else:
        g_kwargs["width"] = cfg["width"]
    g_kwargs.update(kwargs)
    return BackwardGrounder(kb, **g_kwargs)


def make_bcwd(kb: KB, w: int, d: int, u: int = 0, *,
              flat_intermediate: bool = True, filter: Optional[str] = None,
              **kwargs) -> BackwardGrounder:
    """The paper's BC_{w,d,u} grounder directly (resolution='pbc')."""
    if filter is None:
        filter = _default_pbc_filter(u)
    return BackwardGrounder(kb, resolution="pbc", depth=d, width=w, u=u,
                            filter=filter, flat_intermediate=flat_intermediate, **kwargs)


__all__ = ["parse_grounder_type", "create_grounder", "make_bcwd"]
