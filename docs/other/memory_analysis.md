# Memory Analysis — Unification Engine

Space complexity analysis of the vectorized unification engine (`unification/`),
the dominant memory consumer in the training pipeline.

## Role in the Pipeline

The unification engine is the MDP transition function. Every `env.step()` call
invokes `engine.get_derived_states_compiled()` to compute all possible successor
states for B environments in parallel:

```
runner_kge.py → builder.py → PrologEngine/RTFEngine + EnvVec + PPO

Each env.step():
  1. Agent picks an action (selects a derived state)
  2. EnvVec._compute_derived() → UnificationLogicComponent
  3. engine.get_derived_states_compiled(current_states, next_var_indices, ...)
  4. Strategy pipeline (Prolog or RTF) runs unification
  5. Returns [B, K, M, 3] successor states
  6. Env applies visited-pruning, compaction, end-action insertion
  7. Returns obs, reward, done to PPO
```

This runs every step of every rollout, making it the computational and memory
bottleneck of the system.

## Dimension Reference

| Symbol | Meaning | Source | Typical range |
|--------|---------|--------|---------------|
| **B** | Batch size (`n_envs`) | Config | 64–512 |
| **A** | Atoms per state (`padding_atoms`) | Config | 6–30 |
| **G** | Remaining atoms = A − 1 | Derived | 5–29 |
| **F** | Number of facts in KG | Dataset | 3k–272k |
| **R** | Number of rules | Dataset | 5–400 |
| **K_f** | `max_fact_pairs` — max facts per query | Auto from facts distribution, reduced if K > 550 | 1–1944 |
| **K_r** | `max_rule_pairs` — max rules per predicate (always full max) | Auto from data | 2–30 |
| **K** | `max_derived_per_state` (= `padding_states`, capped at 550) | Auto from facts distribution, **strategy-dependent** | 1–550 |
| **M** | Max atoms in a derived state | Auto: `B_max + A + 1` | 7–35 |
| **B_max** | Max rule body length | Auto from rules | 2–4 |
| **max_fp** | `_max_fact_pairs_body` (RTF body resolution) | Prolog: `1`, RTF: `K_f` | 1–843 |

All tensors are `int64` (8 bytes per element). Shapes are `[..., 3]` for atoms
(predicate, arg0, arg1), so each atom = 24 bytes.

## Parameters: Auto-Calculated vs User-Set

### All auto-calculated from facts distribution (default)

| Parameter | Computed where | Formula |
|-----------|---------------|---------|
| `max_fact_pairs` (K_f) | `_engine.py` via `_effective_max_fact_pairs` | Full max of per-query fact-hit counts, reduced if K exceeds 550 |
| `padding_states` (K) | `_engine.py` via `_effective_max_derived` | Full max of per-predicate derived-state counts, capped at 550. **Strategy-dependent**: prolog uses `fact_hits + rule_hits` (additive), rtf uses `fact_hits × rule_hits` (multiplicative) |
| `max_rule_pairs` (K_r) | `_engine.py` via `rule_seg_lens.max()` | **Always full max** — rules are never truncated |
| `M` | `_engine.py` | `max_rule_body + padding_atoms + 1` |
| `_max_fact_pairs_body` | `_engine.py` | Prolog: `1`, RTF: `K_f` |

**Capping rule**: K_r, K_f, and K are always computed as full worst-case max.
If K > 550, it is capped to 550 and K_f is reduced to fit:
- **Prolog**: `K_f = min(K_f, 550 - K_r)`
- **RTF**: `K_f = min(K_f, 550 // K_r)`
- If **K_r > 550** (e.g. nations with 2107 rules): K_r is also capped to leave
  room for `min(K_f, 50)` facts:
  - Prolog: `K_r = 550 - min(K_f, 50)`
  - RTF: `K_r = 550 // min(K_f, 50)`

A warning is emitted when capping occurs, showing the uncapped vs capped rollout
memory consumption at default n_envs=256, n_steps=256.

### User-configurable overrides

| Parameter | Config field | Effect |
|-----------|-------------|--------|
| `padding_states` | `config.py` | Explicit K override (bypasses auto-calc when ≠ -1) |
| `max_fact_pairs_cap` | `config.py` | Hard cap on K_f: `min(auto_K_f, cap)` |
| `max_rule_pairs` | `config.py` | Overrides K_r (not recommended — rules should not be truncated) |

### Fixed defaults (not derived from data)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `padding_atoms` (A) | 6 | Max atoms per state. Not auto-derived |
| `n_envs` (B) | 128 | Batch size |

## K_f: How It's Computed

K_f is computed per-fact via `targeted_fact_lookup`'s (pred, arg0) preference
logic: for each fact (p, a0, a1), K_f counts how many facts share (p, a0).
This is always the **full max** (no percentile). For dense KGs, K_f may then
be reduced if K exceeds 550 (see capping rule above).

For fb15k237: K_f_max = 843, reduced to 520 (prolog) or 18 (rtf) by capping.

## Step-by-Step Space Complexity

Each call to `get_derived_states_compiled` goes through these phases.
Tensors within a phase coexist in memory; tensors across phases are
allocated and freed sequentially (the previous phase's intermediates are
garbage-collected before the next phase's peak).

### Phase 0: Static Buffers (one-time, at engine init)

| Buffer | Shape | Size |
|--------|-------|------|
| `facts_idx` | `[F, 3]` | `24F` |
| `fact_hashes` | `[F]` | `8F` |
| `rules_heads_sorted` | `[R, 3]` | `24R` |
| `rules_idx_sorted` | `[R, B_max, 3]` | `24·R·B_max` |
| `arg0/arg1_order` | `[F]` each | `16F` |
| `arg0/arg1_starts/lens` | `[max_key]` each | `32·P·C` |

Total: ~48F + 24R·B_max. For F=272k (fb15k237): ~13 MB. Negligible.

### Phase 1: Terminal Detection (`detect_terminals`)

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `derived` (output pre-alloc) | `[B, K, M, 3]` | `24·B·K·M` |
| `queries` | `[B, 3]` | `24B` |
| `remaining` | `[B, G, 3]` | `24·B·G` |

**Peak**: `24·B·K·M`. This output buffer persists through the entire call.

### Phase 2: Fact Index Lookup (`targeted_fact_lookup`)

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `sorted_idx0/1` | `[B, K_f]` each | `16·B·K_f` |
| `orig_idx0/1` | `[B, K_f]` each | `16·B·K_f` |
| `valid0/1` | `[B, K_f]` each | `2·B·K_f` |

**Peak**: ~`34·B·K_f`.

### Phase 3: Fact Unification (`unify_with_facts`)

The dominant expansion step. Each of B queries is matched against K_f candidate
facts, and the remaining atoms are broadcast to all candidates:

| Tensor | Shape | Bytes | Notes |
|--------|-------|-------|-------|
| `fact_atoms` | `[B, K_f, 3]` | `24·B·K_f` | Gathered from facts |
| `flat_q`, `flat_f` | `[B·K_f, 3]` each | `48·B·K_f` | Flattened for unify |
| `ok_flat` | `[B·K_f]` | `B·K_f` | |
| `subs_flat` | `[B·K_f, 2, 2]` | `32·B·K_f` | |
| **`rem_flat`** | **`[B·K_f, G, 3]`** | **`24·B·K_f·G`** | Remaining expansion |
| `rem_subst` | `[B·K_f, G, 3]` | `24·B·K_f·G` | After substitution |
| `derived_states` | `[B, K_f, G, 3]` | `24·B·K_f·G` | Output (= reshaped rem_subst) |

**Peak**: ~`3 × 24·B·K_f·G` (rem_flat + rem_subst + derived coexist).

This is the **first major bottleneck** — K_f scales with dataset density.

### Phase 4: Rule Unification (`unify_with_rules`)

Same pattern but with K_r (much smaller than K_f):

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `rule_bodies_sel` | `[B, K_r, B_max, 3]` | `24·B·K_r·B_max` |
| `combined_flat` | `[B·K_r, B_max+G, 3]` | `24·B·K_r·(B_max+G)` |
| `combined_subst` | `[B·K_r, B_max+G, 3]` | `24·B·K_r·(B_max+G)` |

**Peak**: ~`3 × 24·B·K_r·(B_max+G)`. Much smaller than facts since K_r << K_f.

### Phase 5 (RTF only): Body Fact Resolution (`resolve_body_atom_with_facts`)

For each rule-derived state, resolve body[0] against facts:

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `flat_remaining` | `[B·K_r, G, 3]` | `24·B·K_r·G` |
| Inside `unify_with_facts`: | | |
| `rem_flat` | `[B·K_r·max_fp, G, 3]` | `24·B·K_r·max_fp·G` |
| `rem_subst` | same | same |
| Output | `[B, K_r·max_fp, G, 3]` | same |

**Peak**: ~`3 × 24·B·K_r·K_f·G`. Since max_fp = K_f in RTF mode, this is
the **single largest allocation in RTF mode** on dense KGs. For fb15k237
RTF (K_f=18, K_r=30, B=128, G=5): ~19 MB. Without capping (K_f=843): ~805 MB.

### Phase 6: Result Packing (`pack_results`)

Combines fact and rule results into the fixed `[B, K, M, 3]` output:

| Tensor | Shape | Bytes | Notes |
|--------|-------|-------|-------|
| `all_states_rev` | `[B, K_f+K_r, M, 3]` | `24·B·(K_f+K_r)·M` | Concatenation |
| `target_idx_exp` | `[B, K_f+K_r, M, 3]` | `24·B·(K_f+K_r)·M` | Scatter index |
| `derived` (output) | `[B, K, M, 3]` | `24·B·K·M` | |

**Peak**: `2 × 24·B·(K_f+K_r)·M` (concat + scatter coexist).

This is often the **overall peak** because it operates at the full K_f+K_r size
regardless of K — the truncation to K happens via scatter, but the input
tensors must still be full-sized.

### Phase 7: Ground Fact Pruning (`prune_ground_facts`)

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `flat_atoms` | `[B·K·M, 3]` | `24·B·K·M` |
| `is_fact_flat` | `[B·K·M]` | `B·K·M` |

**Peak**: ~`25·B·K·M`. Modest.

### Phase 8: Atom Compaction (`compact_atoms`)

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `sort_key` + `sorted_indices` | `[B, K, M]` each | `16·B·K·M` |
| `sorted_indices_exp` | `[B, K, M, 3]` | `24·B·K·M` |

**Peak**: ~`40·B·K·M`.

### Phase 9: Variable Standardization (`standardize_vars_offset`)

| Tensor | Shape | Bytes |
|--------|-------|-------|
| `standardized` (clone) | `[B, K, M, 3]` | `24·B·K·M` |

**Peak**: ~`24·B·K·M`.

For canonical mode (parity only), ~8 working tensors of `[B·K, M·2]`
each, totaling ~`88·B·K·M`.

## Measured Memory Profile (fb15k237)

Measured with B=128, K=200, K_f=3612 (old group-level), K_r=30, G=5,
M=26 (old formula), compilation disabled, Prolog strategy:

| Function | Calls | Avg Delta/call | Peak above baseline |
|----------|------:|---------------:|--------------------:|
| **get_derived_states_prolog** (total) | 771 | +16 MB | **681 MB** |
| `unify_with_facts` | 771 | +70.8 MB | — |
| `apply_substitutions` (×2) | 1542 | +28.1 MB | — |
| `prune_ground_facts` | 771 | +16.2 MB | — |
| `compact_atoms` | 771 | +16.0 MB | — |
| `unify_one_to_one` (×2) | 1542 | +7.7 MB | — |
| `targeted_fact_lookup` | 771 | +4.2 MB | — |
| `unify_with_rules` | 771 | +0.8 MB | — |

CUDA peak during profiling: **2983 MB** (on top of ~693 MB baseline).

### family dataset (K_f=28, K_r=22) for contrast

| Function | Calls | Avg Delta/call | Peak above baseline |
|----------|------:|---------------:|--------------------:|
| **get_derived_states_prolog** (total) | 771 | +10.5 MB | **54 MB** |
| `prune_ground_facts` | 771 | +10.7 MB | — |
| `compact_atoms` | 771 | +10.4 MB | — |
| `unify_with_facts` | 771 | +0.5 MB | — |

On sparse KGs, the output buffer and post-processing dominate. On dense KGs,
`unify_with_facts` and `pack_results` dominate.

## Bottleneck Ranking

Ranked by per-call peak memory on fb15k237 (B=128, prolog K_f=520, rtf K_f=18):

| Rank | Phase | Dominant tensor | Formula | fb15k237 (Prolog) | fb15k237 (RTF) |
|------|-------|----------------|---------|----------|----------|
| **1** | `pack_results` | concat + scatter | `O(B·(K_f+K_r)·M)` | **30 MB** (×2) | 3 MB (×2) |
| **2** | `unify_with_facts` | remaining expansion | `O(B·K_f·G)` | **24 MB** | 1 MB |
| **3** | RTF body resolution | remaining expansion | `O(B·K_r·K_f·G)` | — | **19 MB** |
| **4** | Post-processing | output + prune + compact | `O(B·K·M)` | 16 MB each | 16 MB each |
| **5** | `unify_with_rules` | combined body+remaining | `O(B·K_r·(B_max+G))` | 0.8 MB | 0.8 MB |

## Where Memory Actually Lives: Engine Transient vs Rollout Persistent

The engine phases above describe **transient** memory — allocated and freed
within a single `get_derived_states_compiled` call. But the dominant memory
consumer during training is the **persistent** rollout buffer, which stores
observations for every step × every environment:

| Component | Shape | Formula |
|-----------|-------|---------|
| **Rollout obs** | `[n_steps·B, K, A, 3]` | `24·N·K·A` |
| Rollout action masks | `[n_steps·B, K]` | `8·N·K` |
| Engine output | `[B, K, M, 3]` | `24·B·K·M` |
| Engine phases 7–9 | `[B, K, M, ...]` | ~`40·B·K·M` |

With n_steps=128, B=128 → N=16,384:
- K is the dominant multiplier for persistent memory (rollout buffer)
- K_f is the dominant multiplier for transient memory (engine internals)

## The Three Knobs That Matter

### 1. K_f (`max_fact_pairs`) — controls transient engine peak

Always uses full max of per-query fact-hit counts. If K > 550, K_f is
reduced to bring K within the cap (see K below).

### 2. K_r (`max_rule_pairs`) — always full max, never truncated

K_r is always the full max rules per predicate. Rules represent unique proof
strategies — truncating them means losing access to valid proof paths. K_r is
typically small (2–30) so memory impact is negligible.

### 3. K (`padding_states`) — controls persistent memory everywhere

K is the output cap from the engine and the action dimension for the policy.
Computed **per-predicate** from derived-state counts (not as global K_r × K_f),
then takes the max:
- **Prolog**: `K = max_P(fact_hits(P) + rule_count(P))` — additive
- **RTF**: `K = max_P(fact_hits(P) × rule_count(P))` — multiplicative

Since K_r and K_f maxima may come from different predicates, K is often
much smaller than global K_r × K_f (e.g. countries_s3: K_rtf=2, not 16×2=32).

**Capped at 550.** If the uncapped K exceeds 550, K_f is reduced to fit.
The K_f reduction uses the conservative global formula (`550 - K_r` for prolog,
`550 // K_r` for rtf) to guarantee the cap holds for all predicates.
If K_r itself exceeds 550, K_r is also capped to leave room for min(K_f, 50)
facts: `K_r = 550 - min(K_f, 50)` (prolog) or `K_r = 550 // min(K_f, 50)` (rtf).
A warning is emitted showing memory savings.

**K capping effect**: queries with more derived states than 550 see
truncated candidates. Since K_f is reduced (not K_r), only fact-derived
candidates are dropped — rule-derived candidates are always preserved.

## Dataset Reference: Auto-Calculated Defaults

Auto-calculated values for all datasets. K is computed per-predicate then
capped at 550; K_f is reduced to fit (K_r is never reduced).

**Prolog strategy** (K = max per-predicate fact_hits + rule_count):

| Dataset | K_r | K_r (used) | K_f | K_f (used) | K | K (used) |
|---|---:|---:|---:|---:|---:|---:|
| countries_s3 | 2 | 2 | 16 | 16 | 16 | 16 |
| family | 22 | 22 | 28 | 28 | 40 | 40 |
| wn18rr | 8 | 8 | 442 | 442 | 448 | 448 |
| fb15k237 | 30 | 30 | 843 | **520** | 849 | **550** |
| deep_chain_v5_100k_2k | 1 | 1 | 19 | 19 | 19 | 19 |
| nations | 23 | 23 | 11 | 11 | 34 | 34 |
| umls | 12 | 12 | 21 | 21 | 27 | 27 |
| pharmkg_full | 9 | 9 | 1944 | **541** | 1945 | **550** |

**RTF strategy** (K = max per-predicate fact_hits × rule_count):

| Dataset | K_r | K_r (used) | K_f | K_f (used) | K | K (used) |
|---|---:|---:|---:|---:|---:|---:|
| countries_s3 | 2 | 2 | 16 | 16 | 2 | 2 |
| family | 22 | 22 | 28 | 28 | 336 | 336 |
| wn18rr | 8 | 8 | 442 | **68** | 2652 | **550** |
| fb15k237 | 30 | 30 | 843 | **18** | 8470 | **550** |
| deep_chain_v5_100k_2k | 1 | 1 | 19 | 19 | 1 | 1 |
| nations | 23 | 23 | 11 | 11 | 253 | 253 |
| umls | 12 | 12 | 21 | 21 | 180 | 180 |
| pharmkg_full | 9 | 9 | 1944 | **61** | 3249 | **550** |

**Bold** = value was reduced by capping.
When K > 550, K_f is reduced via `550 - K_r` (prolog) or `550 // K_r` (rtf).
If K_r itself exceeds 550, K_r is also capped to leave room for min(K_f, 50) facts.

## Practical Recommendations

### For dense KGs (fb15k237, pharmkg_full)

K is automatically capped at 550 and K_f is reduced to fit. For further
reduction:

```bash
# Hard cap on K_f (independent of auto-calc)
python runner_kge.py --set dataset=fb15k237 --set max_fact_pairs_cap=200

# Explicit padding_states override (bypasses auto-calc)
python runner_kge.py --set dataset=fb15k237 --set padding_states=358
```

### For datasets with few rules (deep_chain variants)

Rules are always at full max — no action needed.

### For large batch sizes or padding_atoms

Both B and G appear as multipliers in every allocation. When scaling B > 256 or
A > 10, use `max_fact_pairs_cap` to prevent OOM.

### Summary: fb15k237 memory (B=128, M=9, n_steps=128, Prolog)

fb15k237 K is capped (uncapped 849 > 550):

| Scenario | K_f | K | Engine peak | Rollout obs | Total |
|----------|----:|------:|----------:|------------:|------:|
| Old group-level max (pre-fix) | 3612 | 358 | 211 MB | 845 MB | ~3 GB |
| **Capped at 550 (current default)** | **520** | **550** | **30 MB** | **1298 MB** | **~1.6 GB** |
| Uncapped max | 843 | 849 | 48 MB | 2003 MB | ~2.5 GB |
| Explicit `padding_states=358` | 520 | 358 | 30 MB | 845 MB | ~1.1 GB |
