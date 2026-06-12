"""The library's single vocabulary — a controlled, enforced naming contract.

Registries (each a FIXED, enforced set):
  - ``GLOSSARY``        : concept terms (one agreed term per concept; no synonyms).
  - ``FIELDS``          : allowed data-structure field names → meaning.
  - ``SHAPES``          : tensor shape symbols → meaning; matches the Shapes dataclass.
  - ``SCALARS``         : derived quantities that are concepts, NOT tensor axes.
  - ``DYNAMIC``         : runtime counts (not static; live on per-step tuples).
  - ``KNOBS``           : ctor/config knob → the symbol it sets or caps + default.
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
    # ── goal/atom discipline (goal ≡ the MDP state, a SET of atoms) ──
    "goal": "a SET of atoms still to be proved = the MDP state; the frontier unit (G per step); NOT an atom",
    "atom": "one (predicate, arg0, arg1) triple; a goal holds up to L atoms",
    "frontier": "the set of G goals carried at one proof step",
    "width": "W — how many atoms of a goal may stay unknown (not yet facts) at a step",
    "forward": "forward chaining only (ForwardGrounder); never the BC batch driver",
    "run_backward": "the backward proof-search batch driver (backward/loop.py)",
    "backward": "query-directed proof search (BackwardGrounder; sld/rtf/pbc)",
    "resolution": "the WHAT axis of BackwardGrounder: sld | rtf | pbc (config type, not subclass)",
    "pbc": "Parametrized Backward Chaining (IJCAI BC_{w,d,u}); pbc resolution + fp_batch filter",
    "firing": "one fired rule application (rule_idx, head, body); the FiringSet unit, PRIMARY for RuleGroundings",
    "completed-tree firings": "fired rule apps INSIDE completed proof trees (~3x undercount); the TREES tier — chunk-merge fallback",
    "filter": "post-hoc soundness prune; the ONLY filter is fp_batch (Kleene T_P)",
    "layout": "materialization of one grounding step: flat (compact/eager) | dense (padded/compiled)",
    "strategy": "ExecStrategy — single owner of compile/layout/chunk/cudagraph policy",
    "cell": "one (layout, compile-mode) point a grounder DECLARES it supports",
    # ── cross-family core/ seam nouns ──
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
    "sub_rule_idx": "per-child rule index from rule resolution",
    "query_idx": "global query index (flat layouts)",
    # ── goals / search state ──
    "goal_atoms": "the atoms of each goal [B,G,L,3]",
    "grounding_body": "the per-goal working body atoms [..,M,3]",
    "accumulated_body": "per-depth accumulated body atoms [B,G,D,M,3]",
    "rule_idx_per_depth": "per-depth rule index [B,G,D]",
    "head_per_depth": "per-depth head atom [B,G,D,3]",
    "selected_atom": "the atom selected from the goal this step [..,3]",
    "next_var": "next free-variable id (standardization only)",
    "initial_next_var": "passed-in next_var; terminal-standardize base [B]",
    "initial_goals": "passed-in goal atoms; terminal-standardize parents [B,M_in,3]",
    # ── validity masks (the <thing>_valid family) ──
    "goal_valid": "active-goal mask [B,G]",
    "grounding_valid": "valid-grounding mask [B,Y_q]",
    "body_atom_valid": "valid body-atom slots [..,M]",
    "firing_valid": "per-firing validity mask [N_firings] (RuleGroundings; tkk fast path)",
    "fact_success": "fact-child unification succeeded [..,K_f]",
    "rule_success": "rule-child unification succeeded [..,K_r]",
    "has_new_body": "this goal produced a new body atom this step",
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
    "winning_subs": "the winning substitution per packed goal",
    "subs_noop": "substitutions are identity (skip the apply pass)",
    # ── resolution children / pack / sync (engine seams) ──
    "fact_child_goals": "fact-child goals (atom-sets) [..,K_f,L,3]",
    "fact_grounding_body": "fact-child working body [..,K_f,M,3]",
    "rule_child_goals": "rule-child goals (atom-sets) [..,K_r,L,3]; pbc-dense fused emit narrows to body-only [..,K_r,M,3] (pack_states rebuilds the parent tail)",
    "rule_grounding_body": "rule-child working body [..,K_r,M,3]",
    "flat_child_goals": "flat-path child goals (atom-sets) [T,L,3]",
    "flat_grounding_body": "flat-path working body [T,A,3]",
    "flat_rule_idx": "flat-path rule index [T]",
    "flat_batch_idx": "flat-path batch index [T]",
    "flat_state_idx": "flat-path state index [T]",
    "flat_is_fact": "flat-path fact-vs-rule child tag [T]",
    "flat_top_rule_idx": "flat-path parent top (depth-0) rule index [T]",
    "top_rule_idx": "top (depth-0) rule index per goal/tree",
    "current_rule_idx": "rule index of the current step's children",
    "parent_map": "parent-goal index per packed goal",
    # ── shapes ──
    "shapes": "the Shapes static-symbol registry",
    # ── output bundle / request / run state ──
    "goal_state": "the GoalState in BackwardResult",
    "completed_tree_firings": "the CompletedTreeFirings in BackwardResult",
    "rule_groundings": "the RuleGroundings in BackwardResult",
    "groundings": "OutputSpec: produce the goal-state tier",
    "firings": "OutputSpec: produce the rule-firings tier",
    "trees": "OutputSpec: produce the proof-tree tier",
    "spec": "the OutputSpec on RunState",
    # ── core/ GroundRequest + typed OutputSpec ──
    "queries": "the [B,3] query atoms on GroundRequest (None => data-directed FC)",
    "query_mask": "the [B] active-query mask on GroundRequest",
    "output_spec": "the OutputSpec carried on GroundRequest",
    "excluded_queries": "queries to exclude from groundings on GroundRequest",
    "closure_depth": "FC-only closure depth override on GroundRequest (None for BC)",
    "tiers": "the FrozenSet[Tier] on the typed OutputSpec (proof_state | firings | trees)",
    "chunk_query_offset": "running global-query offset across chunks",
    # ── misc tags used as fields ──
    "layout": "the Layout value tag (dense | flat | sparse)",
    # ── shape symbols carried AS scalar fields on engine structures (canonical axis order) ──
    "B": "batch size (as a carried field)",
    "G": "goals per step (as a carried field)",
    "M": "max body atoms per rule (as a carried field; PbcPlan)",
    "S": "states per step (as a carried field; RunPlan/work tuples)",
    "Y_q": "collected-groundings budget (as a carried field; RunPlan/PbcPlan)",
    "Y_r": "groundings per rule (as a carried field; PbcPlan/dense cand)",
    "K": "children-per-state cap (as a carried field; PbcPlan)",
    "K_r": "rule children per atom (as a carried field; PbcPlan/ClusteredRules)",
    "K_v": "candidates per free var (as a carried field; PbcPlan)",
    "V": "free vars per rule (as a carried field; PbcPlan)",
    "E": "entity count (as a carried field; Encoding/PbcPlan)",
    "pad": "padding index (as a carried field; Encoding)",
    "max_vars_per_rule": "var-id spacing per goal (as a carried field; RunPlan)",
    # ── resolution request / step inputs (StepInputs, ResolveRequest) ──
    "remaining": "the goal's remaining atoms to prove [B,G,L,3]",
    "active_mask": "per-state active mask [B,G] (or per-rule rule-exists mask)",
    "d": "the current depth index (int eager | 0-dim tensor compiled)",
    "depth": "proof depth D (carried as a field on plans/step inputs)",
    "is_last": "whether this is the last proof step (duality with d)",
    "w_last_depth": "last-step unknown-fact cap W_l (carried field)",
    "width": "intermediate-step unknown-fact cap W (carried field)",
    "dedup_goals": "dedup identical goals before enumeration (flat path)",
    "frontier": "the Frontier carried on a ResolveRequest",
    "depth_selector": "the DepthSelector on a ResolveRequest",
    "plan": "the RunPlan carried on a ResolveRequest",
    # ── pbc binding tables / plan (PbcTables, PbcPlan) ──
    "tables": "the PbcTables on a PbcPlan",
    "pred_rule_indices": "[P,K_r] predicate → candidate rule rows",
    "pred_rule_mask": "[P,K_r] which candidate slots are real rules",
    "has_free": "[R] rule has >=1 free var",
    "head_pred_mask": "[P] predicate is some rule's head",
    "body_preds": "[R,M] body predicates (dep order)",
    "body_preds_dep": "[R,M] dep-order body preds (cartesian fill)",
    "num_body_atoms": "[R] body-atom count per rule",
    "enum_pred": "[R] single-fv: enumerate predicate",
    "enum_bound": "[R] single-fv: bound-arg source",
    "enum_dir": "[R] single-fv: enumeration direction",
    "check_arg_source": "[R,M,2] body arg → source column",
    "arg_source_dep": "[R,M,2] dep-order arg sources (cartesian fill)",
    "fv_enum_pred": "[R,Fv] multi-fv: per-free-var enumerate pred",
    "fv_enum_bound_src": "[R,Fv] multi-fv: per-free-var bound source",
    "fv_enum_direction": "[R,Fv] multi-fv: per-free-var direction",
    "fv_enum_valid": "[R,Fv] multi-fv: per-free-var slot valid",
    "fv_any_valid": "per fv-slot: any rule uses it (PbcPlan)",
    "cartesian_product": "all-entities enumeration flag (carried on PbcPlan)",
    "variant_to_orig": "anchor-variant → original rule index map",
    # ── pbc candidate clusters / dense+flat cand & work tuples ──
    "active_idx": "[N,K_r] candidate rule rows per query",
    "has_free_q": "[N,K_r] per-query rule has a free var",
    "q": "the flattened query atoms [N,3] (dense work)",
    "q_preds": "the query predicates [N] (work tuples)",
    "q_valid": "the query-active mask [N] (work tuples)",
    "flat_q": "the compact query atoms [T,3] (flat work)",
    "active_pos": "[T] indices of active states (flat compaction)",
    "state_to_goal": "[T] flat state → deduped-goal map",
    "source": "[N,K_r,Y_r,3] candidate source atoms (dense cand)",
    "flat_source": "[T,3] candidate source atoms (flat cand)",
    "check_arg": "candidate body-arg source columns (dense cand)",
    "check_flat": "candidate body-arg source columns (flat cand)",
    "cmask": "candidate validity mask (dense cand)",
    "num_body_q": "[N,K_r] per-query body-atom count (dense cand)",
    "n_idx_eff": "[T] effective candidate index (flat cand)",
    "r_idx": "[T] local rule index within candidates (flat cand)",
    "rule_global_idx": "[T] global rule index (flat cand)",
    "bpreds_flat": "[T,M] body preds for flat candidates",
    "nbody_flat": "[T] body-atom count for flat candidates",
    # ── execution: capability / strategy / chunk / depth ──
    "eager": "CompileSpec: run eager (no torch.compile)",
    "integration": "CompileSpec: step | outer integration point",
    "backend": "CompileSpec: inductor backend",
    "dynamic": "CompileSpec: dynamic-shape tracing",
    "fullgraph": "CompileSpec: require a single graph",
    "compile": "the CompileSpec on a Cell",
    "cells": "the FrozenSet[Cell] on a CapabilityRow",
    "cell": "the chosen Cell on an ExecStrategy",
    "chunk": "the ChunkPolicy on an ExecStrategy",
    "batch_size": "queries per chunk (0 = whole input in one chunk)",
    "compiled": "DepthSelector: compiled-path flag",
    # ── backward run plan (RunPlan) ──
    "kb": "the KB on a RunPlan",
    "strategy": "the ExecStrategy on a RunPlan",
    "pbc": "the PbcPlan on a RunPlan (None for sld/rtf)",
    "resolution": "the resolution name on a RunPlan",
    "filter_mode": "the resolved filter name on a RunPlan",
    "max_atoms": "L cap on a RunPlan",
    "max_fact_pairs_body": "fact-pair body cap on a RunPlan",
    "collect_evidence": "emit trees tier (StepInputs flag; RunPlan derives it from output_spec)",
    "flat_prune": "pbc flat: in-enumeration width branch-prune (carried on RunPlan)",
    "guided_topk": "KGE-guided beam width (carried on RunPlan)",
    "guided_tnorm": "guided state-score t-norm (carried on RunPlan)",
    "guided_scorer": "KGE GuidedScorer reference (carried on RunPlan)",
    "guided_stats": "guided census counters (carried on RunPlan)",
    "pack_dedup": "dedup goals during pack (carried on RunPlan)",
    "prune_facts": "drop proved-fact leaves (carried on RunPlan)",
    "all_anchors": "anchor on every body position (carried on RunPlan)",
    "init_state_shape": "initial goal-state shape (carried on RunPlan)",
    "standardize": "standardization on (= standardize_fn is not None)",
    "standardize_fn": "terminal output-var renaming callable (RunPlan)",
    "fact_hook": "fact-resolution hook (carried on plans/requests)",
    "rule_hook": "rule-resolution hook (carried on plans/requests)",
    # ── backward collect / pack scratch tuples ──
    "collected_body": "[B,Y_q,D,M,3] collected proof bodies",
    "collected_mask": "[B,Y_q] collected-grounding validity",
    "collected_rule_idx": "[B,Y_q,D] collected per-depth rule index",
    "collected_body_count": "[B,Y_q,D] collected per-depth body count",
    "collected_head": "[B,Y_q,D,3] collected per-depth head",
    # ── forward closure / spmm rule descriptor / fire spec ──
    "hashes": "[N] sorted provable-atom hashes (Closure)",
    "n_provable": "number of provable atoms (Closure)",
    "num_entities": "entity count E for decode (Closure)",
    "kind": "result family tag (closure | backward)",
    "op": "the SpMMOp of a rule descriptor",
    "head_pred": "head predicate id (spmm rule desc)",
    "pred_A": "first body predicate id (spmm rule desc)",
    "pred_B": "second body predicate id (spmm rule desc)",
    "pred_C": "third body predicate id (spmm 3-body chain)",
    "transpose_A": "transpose the A matrix (spmm rule desc)",
    "transpose_B": "transpose the B matrix (spmm rule desc)",
    "transpose_C": "transpose the C matrix (spmm 3-body chain)",
    "transpose_result": "transpose the result matrix (spmm rule desc)",
    "reflexive_body0": "body atom 0 is reflexive (spmm rule desc)",
    "reflexive_body1": "body atom 1 is reflexive (spmm rule desc)",
    "reflexive_body2": "body atom 2 is reflexive (spmm 3-body chain)",
    "b00": "body0 arg0 binding column (spmm rule desc)",
    "b01": "body0 arg1 binding column (spmm rule desc)",
    "b10_binding": "body1 arg0 binding column (spmm rule desc)",
    "b11_binding": "body1 arg1 binding column (spmm rule desc)",
    "exist_body1_head_arg": "existence-check head arg for body1 (spmm)",
    "exist_filter_body0_arg": "existence-filter body0 arg (spmm)",
    "slot_mats": "per-slot predicate→matrix maps (FireSpec)",
    "slot_mats_T": "transposed per-slot matrix maps (FireSpec)",
    "fact_hashes": "base fact hashes (FireSpec)",
    "num_facts": "fact count (FireSpec)",
    "provable_hashes": "accumulated provable hashes (FireSpec)",
    # ── standardization config (StandardizationConfig) ──
    "mode": "standardization algorithm: offset | canonical",
    "constant_no": "entity count (standardization config)",
    "runtime_var_end_index": "runtime var-id range end (standardization)",
    "padding_idx": "padding index (carried on configs/step inputs)",
    "body_width": "body-atom width for offset standardization",
    "enforce_runtime_range": "assert var ids stay in runtime range",
}

# The FIXED tensor shape symbols — must match the Shapes dataclass exactly.
SHAPES: Mapping[str, str] = {
    "B": "queries per batch (batch size)",
    "G": "goals per step — the frontier width (= states)",
    "L": "atoms per goal — the goal length (= M + (M-1)*D)",
    "M": "body atoms per rule (max over the KB's rules)",
    "D": "depth — number of proof steps",
    "A": "accumulated atoms per goal (= D*M)",
    "Y_q": "groundings per query — the collected budget",
    "K_f": "fact children per atom (from the fact index)",
    "K_r": "rule children per atom (rules per predicate)",
    "Y_r": "groundings per rule (per query)",
    "K_v": "candidates per free variable (= min(K_f, Y_r))",
    "V": "free variables per rule (from the rules)",
    "E": "number of entities (the all-entity candidate space)",
    "N": "flattened goal count (= B*G)",
    "pad": "padding index — sentinel id (neither entity nor variable)",
}

# Derived quantities that are NOT tensor axes (concepts, not in SHAPES).
SCALARS: Mapping[str, str] = {
    "K": "children per state (SLD: K_f+K_r · RTF: K_f*K_r · PBC: K_r*Y_r), capped by max_children",
    "W": "width — unknown-atom tolerance at intermediate steps (BC_{w,d,u})",
    "W_l": "last-step width tolerance (paper convention 0: every leaf atom is a fact)",
    "P": "predicate count",
    "R": "rule count",
    "max_vars_per_rule": "variable-id spacing reserved per goal for standardization",
}

# Runtime counts (NOT static; NOT in SHAPES; live on per-step tuples).
DYNAMIC: Mapping[str, str] = {
    "T": "flat-path survivor count — real (nonzero) child rows after compaction this step (FlatResolvedChildren)",
    "G_out": "packed goals out — goals surviving pack this step (== G on dense; dynamic on flat)",
}

# Ctor/config knob → (sets|caps|selects, the symbol/concept it drives, default).
# Covers every typed-config field (PBC/SLD/RTF/Backward/Forward) + the make_grounder
# exec knobs + the KB construction knobs. Enforced by contracts/test_vocabulary.py:
# every config field is here (test_config_fields_registered), every ctor param is
# here (test_ctor_params_registered), and nothing here is unused (test_no_dead_knobs).
KNOBS: Mapping[str, tuple] = {
    # ── resolutions (PBC/SLD/RTF) — depth lives on the resolution ──
    "depth": ("sets", "D", "—"),
    "width": ("sets", "W", 1),
    "u": ("sets", "W_l", 0),
    "max_groundings_per_rule": ("caps", "Y_r", 32),
    "cartesian_product": ("selects", "all-entities candidate pool (ablation)", False),
    "flat_prune": ("selects", "pbc flat: in-enumeration width branch-prune (vs one-shot)", True),
    "guided_topk": ("caps", "KGE-guided beam width (None = exhaustive)", None),
    "guided_tnorm": ("selects", "guided state-score t-norm: min | product", "min"),
    # ── Backward (shared backward knobs) ──
    "resolution": ("selects", "the WHAT axis: PBC | SLD | RTF", "—"),
    "max_groundings_per_query": ("caps", "Y_q", 64),
    "pack_dedup": ("selects", "dedup goals during pack", True),
    "prune_facts": ("selects", "drop proved-fact leaves", False),
    "bump_s_to_k": ("selects", "enlarge S to K_r*K_v when depth>1 (pbc)", True),
    "init_state_shape": ("selects", "initial goal-state shape", "minimal"),
    "max_atoms": ("caps", "L", "M+(M-1)*D"),
    "max_goals": ("caps", "G", 256),
    "max_children": ("caps", "K", None),
    "standardization": ("selects", "terminal output-var renaming config", None),
    "filter": ("selects", "soundness filter (None → derived from resolution)", None),
    "hooks": ("sets", "generic per-step hook list", None),
    "fact_hook": ("sets", "fact-resolution hook (sld/rtf)", None),
    "rule_hook": ("sets", "rule-resolution hook (sld/rtf)", None),
    "guided_scorer": ("sets", "KGE GuidedScorer for the guided beam (PBC.guided_topk)", None),
    # ── Forward ──
    "method": ("selects", "forward method: spmm | staged", "spmm"),
    "join_algo": ("selects", "staged-method join: staged | chunked", "staged"),
    # ── make_grounder exec knobs ──
    "layout": ("selects", "exec layout: auto | dense | flat", "auto"),
    "compile": ("selects", "compile axis: off | graph | dynamic", "off"),
    "chunk_size": ("caps", "queries per chunk (None=auto, 0=one pass)", None),
    "transforms": ("sets", "ProgramTransform pipeline (AXIS 4)", ()),
    # ── KB construction knobs ──
    "max_facts_per_query": ("caps", "K_f", 64),
    "fact_index_type": ("selects", "fact index: arg_key | inverted | block_sparse", "arg_key"),
    "max_memory_mb": ("caps", "fact-index memory budget (MB)", 256),
    "fact_order": ("selects", "fact ordering: original | shuffle", "original"),
    "rule_order": ("selects", "rule ordering: original | shuffle", "original"),
    "order_seed": ("sets", "shuffle seed for fact/rule order", 42),
    "pack_base": ("sets", "block-sparse pack base (None=auto)", None),
}

# The FIXED layout values — must match the Layout enum exactly.
LAYOUTS: Mapping[str, str] = {
    "dense": "padded static [N,K_r,Y_r,M,3] tensors — the compile / CUDA-graph path",
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

__all__ = ["GLOSSARY", "FIELDS", "SHAPES", "SCALARS", "DYNAMIC", "KNOBS",
           "LAYOUTS", "INTEGRATIONS", "COMPILE_BACKENDS"]
