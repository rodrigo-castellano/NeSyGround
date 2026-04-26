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
    ``fp_batch``  cross-query Kleene T_P fixed-point,
    ``fp_global`` forward-chaining provable-set check (precomputed),
    ``none``      no filtering (raw resolution output).

* **hooks** — optional neuro-symbolic callbacks injected at various points:
    ``ResolutionFactHook``   scores / filters fact candidates during resolution,
    ``ResolutionRuleHook``   scores / filters rule candidates during resolution,
    ``GroundingHook``        post-processes final groundings.

There is no class hierarchy of subclasses — just one ``BCGrounder`` that
composes the desired behaviour from the options above.

Usage::

    from grounder import BCGrounder, create_grounder

    # Factory construction (recommended)
    grounder = create_grounder(
        'sld.fp_batch.d2', facts_idx=facts, rule_heads=heads,
        rule_bodies=bodies, rule_lens=lens, constant_no=C,
        padding_idx=P, device=dev,
    )
    result = grounder(queries, query_mask)
"""

# --- Primitives ---
from grounder.resolution.primitives import apply_substitutions, unify_one_to_one

# --- Fact indexing ---
from grounder.data.fact_index import (
    ArgKeyFactIndex,
    BlockSparseFactIndex,
    FactIndex,
    InvertedFactIndex,
    fact_contains,
    pack_triples_64,
)

# --- Rule indexing + compilation ---
from grounder.data.rule_index import RuleIndex, RuleIndexEnum, RulePattern, compile_rules

# --- Packing + post-processing ---
from grounder.bc.common import (
    compact_atoms,
    collect_groundings,
    pack_states,
)
from grounder.filters.search.prune_facts import prune_ground_facts

# --- Standardization ---
from grounder.resolution.standardization import (
    build_standardize_fn,
    standardize_vars_canonical,
    standardize_vars_offset,
)

# --- Filters ---
from grounder.filters import filter_prune_dead, filter_width


# --- Types ---
from grounder.types import (
    GrounderOutput, GroundingResult, ProofEvidence, ProofState,
    ResolvedChildren, PackedStates, SyncParams,
)

# --- Data loading ---
from grounder.data.loader import KGDataset


# --- Forward chaining ---
from grounder.fc.fc import run_forward_chaining

# --- Factory ---
from grounder.factory import create_grounder

# --- KB + Grounders ---
from grounder.data.kb import KB
from grounder.bc.bc import BCGrounder

# --- NeSy hooks ---
from grounder.nesy.hooks import (
    GroundingHook, ResolutionFactHook, ResolutionRuleHook, StepHook,
)
from grounder.nesy.scoring import kge_score
from grounder.nesy.kge import KGEScorer, KGEFactFilter, KGERuleFilter
from grounder.nesy.neural import NeuralScorer, GroundingAttention
from grounder.nesy.soft import SoftScorer, ProvabilityMLP
from grounder.nesy.sampler import RandomSampler

# --- Utilities ---
from grounder.utils import timed_warmup

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
    "filter_prune_dead",
    "filter_width",
    # Types
    "GrounderOutput",
    "GroundingResult",
    "ProofEvidence",
    "ProofState",
    "ResolvedChildren",
    "PackedStates",
    "SyncParams",
    # Data loading
    "KGDataset",
    # Forward chaining
    "run_forward_chaining",
    # Factory
    "create_grounder",
    # Grounders
    "BCGrounder",
    "KB",
    # NeSy hooks
    "GroundingHook",
    "ResolutionFactHook",
    "ResolutionRuleHook",
    "StepHook",
    # Scoring primitives
    "kge_score",
    # Hook implementations
    "KGEScorer",
    "KGEFactFilter",
    "KGERuleFilter",
    "NeuralScorer",
    "GroundingAttention",
    "SoftScorer",
    "ProvabilityMLP",
    "RandomSampler",
    # Utilities
    "timed_warmup",
]
