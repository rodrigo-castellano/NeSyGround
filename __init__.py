"""NeSyGround — compiled CUDA-graph-safe FOL grounding for neuro-symbolic reasoning.

This package provides a fully compilable grounding engine with fixed-shape
tensors throughout.  All dynamic operations are replaced with masked operations
for torch.compile and CUDA graph compatibility.

Usage:
    from grounder import PrologGrounder, ParametrizedBCGrounder, create_grounder

    # Direct construction
    grounder = PrologGrounder(
        facts_idx=facts, rules_heads_idx=heads,
        rules_bodies_idx=bodies, rule_lens=lens,
        constant_no=C, padding_idx=P, device=dev,
        max_goals=G, depth=2,
    )
    result = grounder(queries, query_mask)

    # Factory construction
    grounder = create_grounder(
        'backward_1_2', facts_idx=facts, rule_heads=heads,
        rule_bodies=bodies, rule_lens=lens, constant_no=C,
        padding_idx=P, device=dev,
    )

Class hierarchy:
    Grounder(nn.Module)             — base: owns KB state
    ├── BCGrounder                  — backward chaining: step() + multi-depth forward()
    │   ├── PrologGrounder          — K = K_f + K_r, Prolog MGU resolution
    │   ├── RTFGrounder             — K = K_f * K_r, Rule-Then-Fact
    │   ├── BCPruneGrounder         — BFS + PruneIncompleteProofs fixed-point
    │   └── BCProvsetGrounder       — BFS + FC provable set check
    ├── ParametrizedBCGrounder      — vectorized enumeration with dual anchoring
    │   ├── SamplerGrounder         — random subsampling wrapper
    │   ├── KGEGrounder             — KGE min-conjunction scoring
    │   ├── NeuralGrounder          — learned attention scoring
    │   ├── SoftGrounder            — soft provability scoring
    │   └── LazyGrounder            — predicate-filtered wrapper
    └── FullBCGrounder              — full entity enumeration (BC_{∞,1})
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

# --- Rule indexing ---
from grounder.rule_index import RuleIndex

# --- Packing ---
from grounder.packing import compact_atoms, pack_combined, pack_fact_rule

# --- Post-processing ---
from grounder.postprocessing import (
    collect_groundings,
    prune_ground_facts,
)

# --- Standardization ---
from grounder.standardization import (
    standardize_vars_canonical,
    standardize_vars_offset,
)

# --- Filters ---
from grounder.filters import cap_ground_children, prune_dead_nonground_rules

# --- Feature encoding ---
from grounder.features import (
    build_predicate_var_count_table,
    build_rule_feature_encoding,
    build_rule_var_count_table,
    compute_shared_slot_indices,
)

# --- Types ---
from grounder.types import ForwardResult, PackResult, ResolveResult, StepResult

# --- Data loading ---
from grounder.data_loader import KGDataset

# --- Compilation ---
from grounder.compilation import CompiledRule, compile_rules

# --- Forward chaining ---
from grounder.forward_chaining import run_forward_chaining

# --- Factory ---
from grounder.factory import create_grounder, parse_grounder_type

# --- Class hierarchy ---
from grounder.grounders import (
    BCGrounder,
    BCPruneGrounder,
    BCProvsetGrounder,
    FullBCGrounder,
    GroundingAttention,
    Grounder,
    KGEGrounder,
    LazyGrounder,
    NeuralGrounder,
    ParametrizedBCGrounder,
    PrologGrounder,
    ProvabilityMLP,
    RTFGrounder,
    SamplerGrounder,
    SoftGrounder,
)

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
    # Packing
    "compact_atoms",
    "pack_combined",
    "pack_fact_rule",
    # Post-processing
    "collect_groundings",
    "prune_ground_facts",
    # Standardization
    "standardize_vars_canonical",
    "standardize_vars_offset",
    # Filters
    "cap_ground_children",
    "prune_dead_nonground_rules",
    # Feature encoding
    "build_predicate_var_count_table",
    "build_rule_feature_encoding",
    "build_rule_var_count_table",
    "compute_shared_slot_indices",
    # Data loading
    "KGDataset",
    # Types
    "ForwardResult",
    "PackResult",
    "ResolveResult",
    "StepResult",
    # Compilation
    "CompiledRule",
    "compile_rules",
    # Forward chaining
    "run_forward_chaining",
    # Factory
    "create_grounder",
    "parse_grounder_type",
    # Class hierarchy
    "BCGrounder",
    "BCPruneGrounder",
    "BCProvsetGrounder",
    "FullBCGrounder",
    "GroundingAttention",
    "Grounder",
    "KGEGrounder",
    "LazyGrounder",
    "NeuralGrounder",
    "ParametrizedBCGrounder",
    "PrologGrounder",
    "ProvabilityMLP",
    "RTFGrounder",
    "SamplerGrounder",
    "SoftGrounder",
]
