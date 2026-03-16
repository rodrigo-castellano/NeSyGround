"""NeSyGround — compiled CUDA-graph-safe FOL grounding for neuro-symbolic reasoning.

This package provides a fully compilable grounding engine with fixed-shape
tensors throughout.  All dynamic operations are replaced with masked operations
for torch.compile and CUDA graph compatibility.

Architecture
------------
The grounding system is built around a single ``BCGrounder`` class that is
configured at construction time via three orthogonal axes:

* **resolution** — how unification candidates are generated:
    ``sld``   standard SLD resolution (fact + rule unification),
    ``rtf``   Rule-Then-Fact (rule first, then match facts),
    ``enum``  full entity enumeration.

* **filter** — how incomplete / unreachable proof branches are pruned:
    ``prune``   BFS + PruneIncompleteProofs fixed-point,
    ``provset`` BFS + forward-chaining provable-set check,
    ``none``    no filtering (raw resolution output).

* **hooks** — optional neuro-symbolic callbacks injected at various points:
    ``ResolutionFactHook``   scores / filters fact candidates during resolution,
    ``ResolutionRuleHook``   scores / filters rule candidates during resolution,
    ``GroundingHook``        post-processes final groundings.

There is no class hierarchy of subclasses — just one ``BCGrounder`` that
composes the desired behaviour from the options above.  ``LazyGrounder``
wraps ``BCGrounder`` with predicate-level filtering for large KBs.

Usage::

    from grounder import BCGrounder, create_grounder

    # Factory construction (recommended)
    grounder = create_grounder(
        'bcprune_2', facts_idx=facts, rule_heads=heads,
        rule_bodies=bodies, rule_lens=lens, constant_no=C,
        padding_idx=P, device=dev,
    )
    result = grounder(queries, query_mask)
"""

# --- Primitives ---
from grounder.primitives import apply_substitutions, unify_one_to_one

# --- Fact indexing ---
from grounder.fact_index import (
    ArgKeyFactIndex,
    BlockSparseFactIndex,
    FactIndex,
    InvertedFactIndex,
    fact_contains,
    pack_triples_64,
)

# --- Rule indexing + compilation ---
from grounder.rule_index import RuleIndex, RuleIndexEnum, RulePattern, compile_rules

# --- Packing + post-processing ---
from grounder.bc.common import (
    compact_atoms,
    collect_groundings,
    pack_states,
    prune_ground_facts,
)

# --- Standardization ---
from grounder.resolution.standardization import (
    build_standardize_fn,
    standardize_vars_canonical,
    standardize_vars_offset,
)

# --- Filters ---
from grounder.filters import cap_ground_children, prune_dead_nonground_rules


# --- Types ---
from grounder.types import GroundingResult, PackResult, ResolveResult, StepResult

# --- Data loading ---
from grounder.data_loader import KGDataset


# --- Forward chaining ---
from grounder.fc.fc import run_forward_chaining

# --- Factory ---
from grounder.factory import create_grounder

# --- KB + Grounders ---
from grounder.kb import KB
from grounder.base import Grounder
from grounder.bc.bc import BCGrounder
from grounder.bc.lazy import LazyGrounder

# --- NeSy hooks ---
from grounder.nesy.hooks import (
    GroundingHook, ResolutionFactHook, ResolutionRuleHook, StepHook,
)
from grounder.nesy.scoring import kge_score_triples, kge_score_all_tails, kge_score_all_heads
from grounder.nesy.kge import KGEScorer, KGEFactFilter, KGERuleFilter
from grounder.nesy.neural import NeuralScorer, GroundingAttention
from grounder.nesy.soft import SoftScorer, ProvabilityMLP
from grounder.nesy.sampler import RandomSampler

__all__ = [
    # Primitives
    "apply_substitutions",
    "unify_one_to_one",
    # Fact indexing
    "ArgKeyFactIndex",
    "BlockSparseFactIndex",
    "FactIndex",
    "InvertedFactIndex",
    "fact_contains",
    "pack_triples_64",
    # Rule indexing
    "RuleIndex",
    "RuleIndexEnum",
    "RulePattern",
    "compile_rules",
    # Packing + post-processing
    "compact_atoms",
    "collect_groundings",
    "pack_states",
    "prune_ground_facts",
    # Standardization
    "standardize_vars_canonical",
    "standardize_vars_offset",
    # Filters
    "cap_ground_children",
    "prune_dead_nonground_rules",
    # Types
    "GroundingResult",
    "PackResult",
    "ResolveResult",
    "StepResult",
    # Data loading
    "KGDataset",
    # Forward chaining
    "run_forward_chaining",
    # Factory
    "create_grounder",
    # Grounders
    "BCGrounder",
    "KB",
    "Grounder",
    "LazyGrounder",
    # NeSy hooks
    "GroundingHook",
    "ResolutionFactHook",
    "ResolutionRuleHook",
    "StepHook",
    # Scoring primitives
    "kge_score_triples",
    "kge_score_all_tails",
    "kge_score_all_heads",
    # Hook implementations
    "KGEScorer",
    "KGEFactFilter",
    "KGERuleFilter",
    "NeuralScorer",
    "GroundingAttention",
    "SoftScorer",
    "ProvabilityMLP",
    "RandomSampler",
]
