"""The library's single vocabulary — a controlled, enforced naming contract.

Registries (each a FIXED, enforced set):
  - ``GLOSSARY``        : concept terms (one agreed term per concept; no synonyms).
  - ``FIELDS``          : allowed data-structure field names → meaning.
  - ``SHAPES``          : tensor shape symbols → meaning; matches the Shapes dataclass.
  - ``LAYOUTS``         : layout values → meaning; matches the Layout enum.
  - ``INTEGRATIONS``    : compile integration points; matches the Integration enum.
  - ``COMPILE_BACKENDS``: compile backends; matches the CompileBackend enum.

This is the "ubiquitous language" (DDD) / single-source-of-truth for naming.
Every field on a dataclass/NamedTuple in ``state.py`` / ``types.py`` MUST use a
name in ``FIELDS``; every ``Shapes`` field MUST be in ``SHAPES`` — both enforced by
``contracts/test_vocabulary.py``. Adding a name forces a deliberate registry
entry, which surfaces any overlap with an existing name.
"""
from __future__ import annotations

from typing import Mapping

GLOSSARY: Mapping[str, str] = {
    "forward": "forward chaining only (ForwardGrounder); never the BC batch driver",
    "run_backward": "the backward proof-search batch driver (backward/loop.py)",
    "backward": "query-directed proof search (BackwardGrounder; sld/rtf/pbc)",
    "resolution": "the WHAT axis of BackwardGrounder: sld | rtf | pbc (config type, not subclass)",
    "pbc": "Parametrized Backward Chaining (IJCAI BC_{w,d,u}); pbc resolution + fp_batch filter",
    "firing": "one fired rule application (rule_idx, head, body); the FiringSet unit, PRIMARY for RuleGroundings (replaces legacy 'considered')",
    "completed-tree firings": "fired rule apps INSIDE completed proof trees (~3x undercount); the TREES tier — chunk-merge fallback (was 'evidence')",
    "filter": "post-hoc soundness prune; the ONLY filter is fp_batch (Kleene T_P)",
    "fp_global": "GONE as a filter — that capability is ForwardGrounder + KB.with_closure()",
    "layout": "materialization of one grounding step: flat (compact/eager) | dense (padded/compiled)",
    "strategy": "ExecStrategy — single owner of compile/layout/chunk/cudagraph policy",
    "cell": "one (layout, compile-mode) point a grounder DECLARES it supports",
    # ── cross-family core/ seam nouns (additive; nothing wired yet) ──
    "ground": "the single runtime verb (Grounder.ground); request in, GroundResult out",
    "grounder": "a family (BackwardGrounder | ForwardGrounder); ground + capability_row + producible_tiers + rebound",
    "tier": "a BACKWARD output stratum (proof_state | firings | trees); the OutputSpec.tiers set",
    "closure": "the ForwardGrounder result FAMILY (provable triples); never a resolution value",
}

# The FIXED data-structure field names. One name per concept; no synonyms.
FIELDS: Mapping[str, str] = {
    # ── identity / provenance ──
    "rule_idx": "rule (or anchor-variant) index",
    "head": "head atom(s), trailing dims [..,3]",
    "body": "body atom(s), trailing dims [..,M,3] (depth-structured in CompletedTreeFirings)",
    "current_head": "head of the current rule application [..,3]",
    "sub_rule_idx": "per-child rule index from rule resolution",
    "query_idx": "global query index (flat layouts)",
    # ── goals / search state ──
    "proof_goals": "remaining goals to prove [B,S,G,3]",
    "grounding_body": "the per-state working body atoms [..,M,3]",
    "accumulated_body": "per-depth accumulated body atoms [B,S,D,M,3]",
    "ridx_per_depth": "per-depth rule index [B,S,D]",
    "head_per_depth": "per-depth head atom [B,S,D,3]",
    "selected_goal": "the goal atom selected this step [..,3]",
    "next_var": "next free-variable id (standardization only)",
    # ── validity masks (the <thing>_valid family) ──
    "state_valid": "active proof-state mask [B,S]",
    "grounding_valid": "valid-grounding mask [B,C]",
    "body_atom_valid": "valid body-atom slots [..,M]",
    "fact_success": "fact-child unification succeeded [..,K_f]",
    "rule_success": "rule-child unification succeeded [..,K_r]",
    "has_new_body": "this state produced a new body atom this step",
    # ── counts / sizes ──
    "count": "number of groundings per query [B]",
    "body_count": "number of body atoms bound [..]",
    "num_atoms": "size of the atom_table pool",
    "num_rules": "number of rules",
    "M_max": "max body atoms across rules",
    # ── pool / CSR (RuleGroundings) ──
    "atom_table": "deduped atom pool [num_atoms,3]",
    "body_pool_idx": "atom_table index per body slot",
    "head_pool_idx": "atom_table index per head",
    "rule_offsets": "CSR per-rule firing offsets [num_rules+1]",
    "query_pool_idx": "atom_table slot per query [B]",
    # ── substitutions ──
    "fact_subs": "fact-resolution substitutions [..,2,2]",
    "rule_subs": "rule-resolution substitutions [..,2,2]",
    "flat_subs": "flat-path substitutions [T,2,2]",
    "winning_subs": "the winning substitution per packed state",
    "subs_noop": "substitutions are identity (skip the apply pass)",
    # ── resolution children / pack / sync (engine seams) ──
    "fact_goals": "fact-child goals [..,K_f,G,3]",
    "fact_grounding_body": "fact-child working body [..,K_f,M,3]",
    "rule_goals": "rule-child goals [..,K_r,G,3]",
    "rule_grounding_body": "rule-child working body [..,K_r,M,3]",
    "flat_goals": "flat-path goals [T,G,3]",
    "flat_grounding_body": "flat-path working body [T,A,3]",
    "flat_rule_idx": "flat-path rule index [T]",
    "flat_batch_idx": "flat-path batch index [T]",
    "flat_state_idx": "flat-path state index [T]",
    "flat_is_fact": "flat-path fact-vs-rule child tag [T]",
    "flat_top_ridx": "flat-path parent top (depth-0) rule index [T]",
    "top_rule_idx": "top (depth-0) rule index per state/tree",
    "current_rule_idx": "rule index of the current step's children",
    "parent_map": "parent-state index per packed state",
    "parent_body_count": "parent's per-depth body count",
    # ── shapes ──
    "shapes": "the Shapes static-symbol registry",
    # ── output bundle / request / run state ──
    "state": "the ProofState in BackwardResult",
    "completed_tree_firings": "the CompletedTreeFirings in BackwardResult",
    "rule_groundings": "the RuleGroundings in BackwardResult",
    "groundings": "OutputSpec: produce the proof-state tier",
    "firings": "OutputSpec: produce the rule-firings tier",
    "trees": "OutputSpec: produce the proof-tree tier",
    "spec": "the OutputSpec on RunState",
    # ── core/ GroundRequest + typed OutputSpec (additive; not wired) ──
    "queries": "the [B,3] query atoms on GroundRequest (None => data-directed FC)",
    "query_mask": "the [B] active-query mask on GroundRequest",
    "output_spec": "the OutputSpec carried on GroundRequest",
    "excluded_queries": "queries to exclude from groundings on GroundRequest",
    "closure_depth": "FC-only closure depth override on GroundRequest (None for BC)",
    "tiers": "the FrozenSet[Tier] on the typed OutputSpec (proof_state | firings | trees)",
    "chunk_query_offset": "running global-query offset across chunks",
    # ── misc tags / sizes used as fields ──
    "layout": "the Layout value tag (dense | flat | sparse)",
    "B": "batch size (as a carried field)",
    "S": "states per step (as a carried field)",
}

# The FIXED tensor shape symbols — must match the Shapes dataclass exactly.
SHAPES: Mapping[str, str] = {
    "B": "batch size — queries per forward (per chunk)",
    "S": "states per proof step (default 256)",
    "G": "goals per state (= M + (M-1)*D)",
    "M": "body atoms per rule (max over the KB's rules)",
    "D": "depth — number of proof steps",
    "A": "accumulated-body capacity (= D*M)",
    "C": "collected-groundings budget per query",
    "K_f": "fact children per state (from the fact index)",
    "K_r": "rules per predicate (from the rule index)",
    "G_r": "groundings per rule per query (the PBC G_r cap)",
    "K_v": "candidates per free variable (= min(K_f, G_r))",
    "V": "free variables per rule (from the rules)",
    "E": "number of entities (the all-entity candidate space)",
    "N": "flattened state count (= B*S)",
    "pad": "padding index — sentinel id (neither entity nor variable)",
}

# The FIXED layout values — must match the Layout enum exactly.
LAYOUTS: Mapping[str, str] = {
    "dense": "padded static [N,K_r,G_r,M,3] tensors — the compile / CUDA-graph path",
    "flat": "compact (nonzero) tensors sized to real survivors — eager, low memory",
    "sparse": "sparse relational / spmm representation — forward chaining",
}

# The FIXED compile axes — must match the execution enums exactly. (The proven
# combinations are the CompileSpec presets EAGER / COMPILED_STEP / OUTER_REDUCE_OVERHEAD.)
INTEGRATIONS: Mapping[str, str] = {
    "step": "per-step inner compile",
    "outer": "whole-batch outer compile",
}
COMPILE_BACKENDS: Mapping[str, str] = {
    "reduce_overhead": "inductor + CUDA graphs",
    "default": "inductor, no CUDA graphs",
    "max_autotune": "autotuned kernels, no CUDA graphs",
}

__all__ = ["GLOSSARY", "FIELDS", "SHAPES", "LAYOUTS", "INTEGRATIONS", "COMPILE_BACKENDS"]
