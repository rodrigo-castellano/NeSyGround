"""Grounder factory — parse grounder type string and instantiate BCGrounder.

Naming convention (dot-separated):
    {resolution}[.{filter}].d{depth}[.w{width}]

Examples:
    sld.d2           SLD resolution, no filter, depth 2
    sld.prune.d2     SLD + PruneIncompleteProofs, depth 2
    sld.provset.d3   SLD + provable-set filter, depth 3
    rtf.d2           Rule-Then-Fact, depth 2
    enum.prune.w1.d2 Enum resolution, prune, width 1, depth 2
    enum.full        Enum, all entities, depth 1
    lazy.enum.prune.w0.d1  Lazy-filtered enum
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from grounder.kb import KB
from grounder.bc.bc import BCGrounder
from grounder.bc.lazy import LazyGrounder


# ======================================================================
# Parser
# ======================================================================

_PATTERN = re.compile(
    r"^(?P<lazy>lazy\.)?"
    r"(?P<resolution>sld|rtf|enum)"
    r"(\.(?P<filter>prune|provset|none))?"
    r"(\.full|\.w(?P<width>\d+))?"
    r"(\.d(?P<depth>\d+))?$"
)


def parse_grounder_type(grounder_type: str) -> dict:
    """Parse a grounder type string into a config dict.

    Returns dict with keys: resolution, filter, depth, width, is_lazy, is_full.

    Examples:
        'sld.prune.d2'         → {resolution:'sld', filter:'prune', depth:2, ...}
        'enum.prune.w1.d2'     → {resolution:'enum', filter:'prune', depth:2, width:1}
        'enum.full'            → {resolution:'enum', filter:'prune', depth:1, is_full:True}
        'lazy.enum.prune.w0.d1'→ {resolution:'enum', filter:'prune', depth:1, width:0, is_lazy:True}
    """
    m = _PATTERN.match(grounder_type)
    if not m:
        raise ValueError(
            f"Unknown grounder type: {grounder_type!r}. "
            f"Expected format: {{resolution}}[.{{filter}}].d{{depth}}[.w{{width}}]  "
            f"(e.g. 'sld.prune.d2', 'enum.prune.w1.d2', 'rtf.d4')"
        )

    resolution = m.group("resolution")
    is_full = ".full" in grounder_type

    if resolution == "enum":
        default_filter = "prune"
    else:
        default_filter = "none"

    return {
        "resolution": resolution,
        "filter": m.group("filter") or default_filter,
        "depth": int(m.group("depth")) if m.group("depth") else 1,
        "width": int(m.group("width")) if m.group("width") else 1,
        "is_lazy": bool(m.group("lazy")),
        "is_full": is_full,
    }


# ======================================================================
# Factory
# ======================================================================

def create_grounder(
    grounder_type: str,
    *,
    # KB params
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
    # Algorithm params
    max_groundings: int = 32,
    max_total_groundings: int = 64,
    fc_method: str = "join",
    max_goals: int = 256,
    **kwargs,
) -> nn.Module:
    """Create a grounder from a type string.

    Builds a KB from the data params, then instantiates the right grounder.

    Args:
        grounder_type: e.g. 'sld.prune.d2', 'enum.prune.w1.d2', 'rtf.d4'.

    Returns:
        Grounder module instance (BCGrounder or LazyGrounder).
    """
    cfg = parse_grounder_type(grounder_type)

    kb = KB(
        facts_idx, rule_heads, rule_bodies, rule_lens,
        constant_no=constant_no, predicate_no=predicate_no,
        padding_idx=padding_idx, device=device,
        max_facts_per_query=max_facts_per_query,
        fact_index_type=fact_index_type,
    )

    bc_kwargs = dict(
        resolution=cfg["resolution"],
        filter=cfg["filter"],
        depth=cfg["depth"],
        max_total_groundings=max_total_groundings,
        max_goals=max_goals,
    )

    if cfg["resolution"] == "enum":
        if cfg["is_full"]:
            bc_kwargs["width"] = None
            bc_kwargs["depth"] = 1
        else:
            bc_kwargs["width"] = cfg["width"]
        bc_kwargs["max_groundings_per_query"] = max_groundings
        bc_kwargs["fc_method"] = fc_method

    bc_kwargs.update(kwargs)

    if cfg["is_lazy"]:
        return LazyGrounder(kb, **bc_kwargs)

    return BCGrounder(kb, **bc_kwargs)
