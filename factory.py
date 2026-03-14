"""Grounder factory — parse grounder_type string and instantiate the right class.

Uses raw tensor interface (no ns_lib domain objects).

Naming convention:
  BC grounders:       '{name}_{depth}'        (e.g., bcprune_2, bcprovset_3)
  Parametrized:       '{name}_{width}_{depth}' (e.g., lazy_0_1, soft_1_2)
  backward_W_D is also accepted (maps to ParametrizedBCGrounder).
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from grounder.base import Grounder
from grounder.bc.bc import PrologGrounder, RTFGrounder
from grounder.types import ForwardResult


def _lazy_imports():
    """Lazy import all grounder classes to avoid circular imports."""
    from grounder.grounders.parametrized import ParametrizedBCGrounder
    from grounder.grounders.full import FullBCGrounder
    from grounder.grounders.prune import BCPruneGrounder
    from grounder.grounders.provset import BCProvsetGrounder
    from grounder.nesy.sampler import SamplerGrounder
    from grounder.nesy.kge import KGEGrounder, NeuralGrounder
    from grounder.nesy.soft import SoftGrounder
    from grounder.grounders.lazy import LazyGrounder
    return {
        "ParametrizedBCGrounder": ParametrizedBCGrounder,
        "FullBCGrounder": FullBCGrounder,
        "BCPruneGrounder": BCPruneGrounder,
        "BCProvsetGrounder": BCProvsetGrounder,
        "SamplerGrounder": SamplerGrounder,
        "KGEGrounder": KGEGrounder,
        "NeuralGrounder": NeuralGrounder,
        "SoftGrounder": SoftGrounder,
        "LazyGrounder": LazyGrounder,
    }


# (prefix, class_name, extra_kwargs)
# Order matters: longer prefixes first to avoid prefix collisions.
_REGISTRY = [
    # Parametrized grounders (use width+depth: prefix_W_D)
    ("lazy_", "LazyGrounder", {}),
    ("sampler_", "SamplerGrounder", {"_sample": True}),
    ("kge_", "KGEGrounder", {}),
    ("neural_", "NeuralGrounder", {}),
    ("softneural_", "SoftGrounder", {"mode": "neural"}),
    ("soft_", "SoftGrounder", {"mode": "kge"}),
    # BC grounders (use depth only: prefix_D)
    ("bcprovset_", "BCProvsetGrounder", {}),
    ("bcprune_", "BCPruneGrounder", {}),
    # SLD grounders (existing prolog/rtf)
    ("prolog_", "PrologGrounder", {"_prolog": True}),
    ("rtf_", "RTFGrounder", {"_rtf": True}),
    # Full BC (no width/depth suffix)
    ("full", "FullBCGrounder", {}),
    # Parametrized BC fallback (backward_W_D → ParametrizedBCGrounder)
    ("backward_", "ParametrizedBCGrounder", {}),
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
    for prefix, _, _ in _REGISTRY:
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
        kge_model: KGE model (for kge/neural/soft grounders).
        predicate_no: Number of predicates.
        num_entities: Number of entities.
        max_facts_per_query: K_f for fact index.
        fact_index_type: Fact index type.
        max_goals: Max goals for BFS-based grounders.

    Returns:
        Grounder module instance.
    """
    classes = _lazy_imports()
    width, depth = parse_grounder_type(grounder_type)

    # Base kwargs shared by all grounders that inherit from Grounder
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

    for prefix, class_name, extra in _REGISTRY:
        if not grounder_type.startswith(prefix):
            continue

        cls = classes.get(class_name)
        if cls is None:
            # Fall through for PrologGrounder/RTFGrounder which are already
            # imported at module level
            if class_name == "PrologGrounder":
                cls = PrologGrounder
            elif class_name == "RTFGrounder":
                cls = RTFGrounder
            else:
                raise ValueError(f"Unknown grounder class: {class_name}")

        gkw = {**base_kwargs, **extra}

        # Remove internal markers
        is_sample = gkw.pop("_sample", False)
        is_prolog = gkw.pop("_prolog", False)
        is_rtf = gkw.pop("_rtf", False)

        # Prolog/RTF grounders use max_goals, depth from BCGrounder
        if is_prolog or is_rtf:
            gkw["max_goals"] = max_goals
            gkw["depth"] = depth
            gkw["max_total_groundings"] = max_total_groundings
            return cls(**gkw)

        # ParametrizedBCGrounder and derivatives
        if class_name in ("ParametrizedBCGrounder", "FullBCGrounder",
                          "LazyGrounder", "SamplerGrounder",
                          "KGEGrounder", "NeuralGrounder",
                          "SoftGrounder"):
            gkw["depth"] = depth
            gkw["width"] = width
            gkw["max_groundings_per_query"] = max_groundings
            gkw["max_total_groundings"] = max_total_groundings

            if class_name == "FullBCGrounder":
                # Full grounder doesn't take width/depth/max_groundings_per_query
                gkw.pop("width", None)
                gkw.pop("depth", None)
                gkw.pop("max_groundings_per_query", None)

            if is_sample:
                gkw["max_sample"] = max_total_groundings

            if "mode" in extra:
                gkw["kge_model"] = kge_model

            if class_name in ("KGEGrounder", "NeuralGrounder"):
                gkw["kge_model"] = kge_model

            return cls(**gkw)

        # BC grounders (BCPruneGrounder, BCProvsetGrounder)
        if class_name in ("BCPruneGrounder", "BCProvsetGrounder"):
            gkw["depth"] = depth
            gkw["max_groundings_per_query"] = max_groundings
            gkw["max_total_groundings"] = max_total_groundings
            gkw["max_goals"] = max_goals
            return cls(**gkw)

        return cls(**gkw)

    # Fallback: ParametrizedBCGrounder
    cls = classes["ParametrizedBCGrounder"]
    return cls(
        **base_kwargs,
        depth=depth,
        width=width,
        max_groundings_per_query=max_groundings,
        max_total_groundings=max_total_groundings,
    )
