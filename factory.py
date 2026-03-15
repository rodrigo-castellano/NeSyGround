"""Grounder factory — parse grounder_type string and instantiate BCGrounder.

All grounder types map to BCGrounder (or LazyGrounder wrapping BCGrounder)
with appropriate resolution, filter, width, and depth settings.

Naming convention:
  BC grounders:       '{name}_{depth}'        (e.g., bcprune_2, bcprovset_3)
  Parametrized:       '{name}_{width}_{depth}' (e.g., lazy_0_1, soft_1_2)
  backward_W_D is also accepted (maps to BCGrounder with enum resolution).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.bc.bc import BCGrounder
from grounder.bc.lazy import LazyGrounder


# (prefix, resolution, filter, uses_width)
# Order matters: longer prefixes first to avoid prefix collisions.
_REGISTRY = [
    # NeSy grounders (enum resolution, hooks TBD) — width+depth: prefix_W_D
    ("softneural_", "enum", "prune", True),
    ("soft_",       "enum", "prune", True),
    ("sampler_",    "enum", "prune", True),
    ("kge_",        "enum", "prune", True),
    ("neural_",     "enum", "prune", True),
    # Lazy grounder — width+depth: lazy_W_D
    ("lazy_",       "enum", "prune", True),
    # SLD-based grounders — depth only: prefix_D
    ("bcprovset_",  "sld",  "provset", False),
    ("bcprune_",    "sld",  "prune",   False),
    ("bcsld_",      "sld",  "none",    False),
    ("bcprolog_",   "sld",  "none",    False),
    ("prolog_",     "sld",  "none",    False),
    # RTF grounder — depth only: rtf_D
    ("rtf_",        "rtf",  "none",    False),
    # Full BC — no suffix
    ("full",        "enum", "prune",   False),
    # Parametrized BC fallback — width+depth: backward_W_D
    ("backward_",   "enum", "prune",   True),
]


def parse_grounder_type(grounder_type: str) -> Tuple[int, int]:
    """Parse grounder name into (width, depth).

    Supports two naming conventions:
      - '{prefix}_{depth}'        → (1, depth)
      - '{prefix}_{width}_{depth}' → (width, depth)

    Examples:
        'bcprune_2'    → (1, 2)
        'backward_1_2' → (1, 2)
        'lazy_0_1'     → (0, 1)
        'full'         → (1, 1)
    """
    suffix = None
    for prefix, _, _, _ in _REGISTRY:
        if grounder_type.startswith(prefix):
            suffix = grounder_type[len(prefix):]
            break

    if suffix is None:
        parts = grounder_type.split("_")
        if len(parts) >= 3:
            return int(parts[-2]), int(parts[-1])
        return 1, 1

    if not suffix:
        return 1, 1

    suffix_parts = suffix.split("_")
    if len(suffix_parts) == 1:
        return 1, int(suffix_parts[0])
    elif len(suffix_parts) >= 2:
        return int(suffix_parts[-2]), int(suffix_parts[-1])
    return 1, 1


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
    max_groundings: int = 32,
    max_total_groundings: int = 64,
    fc_method: str = "join",
    kge_model: Optional[nn.Module] = None,
    predicate_no: Optional[int] = None,
    num_entities: Optional[int] = None,
    max_facts_per_query: int = 64,
    fact_index_type: str = "block_sparse",
    max_goals: int = 256,
    **kwargs,
) -> nn.Module:
    """Create a grounder from a type string.

    All type strings now map to BCGrounder (unified backward chaining) or
    LazyGrounder (predicate-filtered wrapper over BCGrounder).

    Mapping:
        'prolog_D' / 'bcprolog_D' → BCGrounder(resolution='sld', depth=D)
        'rtf_D'                   → BCGrounder(resolution='rtf', depth=D)
        'bcprune_D'               → BCGrounder(resolution='sld', filter='prune', depth=D)
        'bcprovset_D'             → BCGrounder(resolution='sld', filter='provset', depth=D)
        'bcsld_D'                 → BCGrounder(resolution='sld', filter='none', depth=D)
        'backward_W_D'            → BCGrounder(resolution='enum', width=W, depth=D, filter='prune')
        'lazy_W_D'                → LazyGrounder wrapping BCGrounder
        'full'                    → BCGrounder(resolution='enum', width=None, depth=1)
        'sampler_W_D' etc.        → BCGrounder(resolution='enum', width=W, depth=D)

    Args:
        grounder_type: String like 'bcprune_2', 'backward_1_2', 'full', etc.
        facts_idx: [F, 3] fact triples.
        rule_heads: [R, 3] rule head atoms.
        rule_bodies: [R, M, 3] rule body atoms (padded).
        rule_lens: [R] body lengths.
        constant_no: Highest constant index.
        padding_idx: Padding value.
        device: Target device.
        max_groundings: Max groundings per query per rule.
        max_total_groundings: Max total groundings per query.
        fc_method: FC method for provable set ('join').
        kge_model: KGE model (for future hook integration).
        predicate_no: Number of predicates.
        num_entities: Number of entities.
        max_facts_per_query: K_f for fact index.
        fact_index_type: Fact index type.
        max_goals: Max goals for BFS-based grounders.

    Returns:
        Grounder module instance (BCGrounder or LazyGrounder).
    """
    width, depth = parse_grounder_type(grounder_type)

    # Look up resolution and filter from registry
    resolution = "enum"
    filter_mode = "prune"
    uses_width = True
    is_lazy = False
    is_full = False

    for prefix, res, filt, uw in _REGISTRY:
        if grounder_type.startswith(prefix):
            resolution = res
            filter_mode = filt
            uses_width = uw
            is_lazy = prefix == "lazy_"
            is_full = prefix == "full"
            break

    # Base kwargs shared by all grounders (positional args for Grounder base)
    base_kwargs = dict(
        facts_idx=facts_idx,
        rules_heads_idx=rule_heads,
        rules_bodies_idx=rule_bodies,
        rule_lens=rule_lens,
        constant_no=constant_no,
        padding_idx=padding_idx,
        device=device,
        predicate_no=predicate_no,
        num_entities=num_entities,
        max_facts_per_query=max_facts_per_query,
        fact_index_type=fact_index_type,
    )

    # BCGrounder-specific kwargs
    bc_kwargs = dict(
        resolution=resolution,
        filter=filter_mode,
        depth=depth,
        max_total_groundings=max_total_groundings,
        max_goals=max_goals,
    )

    # Enum resolution: set width + enum-specific params
    if resolution == "enum":
        if is_full:
            bc_kwargs["width"] = None
            bc_kwargs["depth"] = 1
        else:
            bc_kwargs["width"] = width
        bc_kwargs["max_groundings_per_query"] = max_groundings
        bc_kwargs["fc_method"] = fc_method

    # Merge all kwargs
    gkw = {**base_kwargs, **bc_kwargs}

    if is_lazy:
        return LazyGrounder(**gkw)

    return BCGrounder(**gkw)
