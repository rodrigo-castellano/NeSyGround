# Architecture

NeSyGround separates grounding into two independent pipelines — backward chaining (BC) and forward chaining (FC) — that compose when needed. This document describes the system structure, each pipeline's stages, and how they interact.

---

## 1. System Context

```
KnowledgeBase ──> Compilation ──> Grounder ──> (body, mask, count, rule_idx) ──> Consumer
     (F, R)     (FactIndex,       (nn.Module)      [B, tG, M, 3]            (SBR, R2N,
                 RuleIndex,                                                    DCR, RL)
                 metadata)
```

**Separation of concerns:**

- **Compilation** transforms symbolic KB into indexed tensor structures. Done once per KB.
- **Grounder** computes ground rule instantiations. Purely structural — it produces groundings, not scores.
- **Consumer** scores groundings with t-norms, KGE models, or learned functions. This is downstream and outside NeSyGround's scope.

### Compiled KB components

| Component | Purpose | Built from |
|-----------|---------|------------|
| `FactIndex` | O(1) or O(log F) fact lookup and enumeration | Facts + entity/predicate indices |
| Rule tensors | Vectorized rule metadata (`head_preds`, `body_preds`, etc.) | Rules + predicate indices |
| `fact_hashes` | Sorted int64 hashes for fast membership tests | Facts |
| Rule clustering | Per-predicate rule mapping (`pred_rule_indices`) | Rules |
| Dual anchoring metadata | Enumeration directions for rules with free variables | CompiledRule analysis |

All tensors are registered as buffers on the grounder module, ensuring they move to the correct device automatically.

---

## 2. Backward Chaining Pipeline: SELECT, RESOLVE, PACK

The three irreducible operations for BC grounding. Every BC grounder implements this pipeline.

### Overview

```
queries [B, 3]
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   BC Proof Loop (D steps)                │
│                                                         │
│   proof_goals [B, S, G, 3]                              │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────┐                                           │
│   │ SELECT  │──> goal [B, S, 3], remaining [B, S, G, 3] │
│   └─────────┘                                           │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────┐    fact_children [B, S, K_f, G, 3]        │
│   │ RESOLVE │──> rule_children [B, S, K_r, G, 3]        │
│   └─────────┘                                           │
│       │                                                 │
│       ▼                                                 │
│   ┌─────────┐                                           │
│   │  PACK   │──> new_states [B, S, G, 3]                │
│   └─────────┘                                           │
│       │                                                 │
│       ▼ (repeat D times)                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
postprocess ──> (body, mask, count, rule_idx) [B, tG, M, 3]
```

### SELECT

Goal selection from proof states.

- Input: `proof_goals [B, S, G, 3]` — current proof states with G goal atoms each
- Output: `goal [B, S, 3]` — the selected goal atom per state; `remaining [B, S, G, 3]` — goals after removing the selected one; `active_mask [B, S]` — which states have unresolved goals
- Strategy: select the first valid (non-padding) goal atom
- Key invariant: selection does not depend on tensor values (no data-dependent branching)

### RESOLVE

Compositional resolution against facts and rules.

**Resolve facts**: match the selected goal against the fact index. Each match produces a child state where the goal is removed (proven).

**Resolve rules**: find rules whose head predicate matches the goal, compute the MGU between the rule head and the goal, and produce child states where the goal is replaced by the rule's body atoms (after substitution).

Sub-operations:
- `resolve_facts` — fact lookup via FactIndex
- `resolve_rule_heads` — head unification (shared across grounder variants)
- `resolve_rules` — full rule resolution (head unification + body substitution)

The RESOLVE stage produces up to `K = K_f + K_r` children per state.

### PACK

Compaction and truncation to maintain fixed tensor shapes.

- Input: `K` children per state, `S` states → `S * K` candidate states per query
- Output: `S` states per query (fixed)
- Operations:
  1. **Merge**: interleave fact and rule children
  2. **Deduplicate**: hash-based dedup via `pack_triples_64`
  3. **Truncate**: keep top `S` states (by validity, then by order)
  4. **Terminal detection**: mark states with no remaining goals as completed groundings

### Step as first-class operation

The `step()` method performs one SELECT → RESOLVE → PACK cycle. This is the API for RL agents:

```python
# RL agent calls step() once per decision
new_states = grounder.step(current_states, depth=d)
```

Batch mode calls `step()` D times in a loop. The outer structure is:

```
forward(queries) = pre_ground() → init_states() → step() × D → post_ground()
```

### Grounder variants via overrides

Different BC grounders customize the pipeline by overriding specific stages:

| Grounder | Overrides | What changes |
|----------|-----------|--------------|
| BCGrounder | (abstract base) | Pipeline skeleton — does not implement resolution |
| PrologGrounder | `_resolve_facts`, `_resolve_rules` | Single-level SLD: K = K_f + K_r (facts + rules independently) |
| RTFGrounder | `_resolve_facts`, `_resolve_rules` | Two-level: K = K_f * K_r (rules first, then body against facts) |
| BCPruneGrounder | `_postprocess` | Adds fixed-point pruning of unprovable body atoms |
| BCProvsetGrounder | `_postprocess` | Checks body atoms against FC provable set |

---

## 3. Forward Chaining Pipeline: MATCH, JOIN, MERGE

A **separate pipeline**, not forced into BC's stages. FC computes the provable set iteratively.

### Overview

```
facts [F, 3]
    │
    ▼
I_0 = facts (initial provable set)
    │
    ▼
┌─────────────────────────────────────────┐
│          FC Loop (up to D iterations)    │
│                                         │
│   delta [delta_max]  (new atoms)        │
│       │                                 │
│       ▼                                 │
│   ┌─────────┐                           │
│   │  MATCH  │──> applicable rules       │
│   └─────────┘                           │
│       │                                 │
│       ▼                                 │
│   ┌─────────┐                           │
│   │  JOIN   │──> new head atoms         │
│   └─────────┘                           │
│       │                                 │
│       ▼                                 │
│   ┌─────────┐                           │
│   │  MERGE  │──> updated I, fixpoint?   │
│   └─────────┘                           │
│       │                                 │
│       ▼ (stop if fixpoint or d == D)    │
│                                         │
└─────────────────────────────────────────┘
    │
    ▼
provable_hashes [I_max], n_provable
```

### MATCH

Identify applicable rules using delta facts (semi-naive evaluation).

- Only considers rules that can fire using **newly derived** facts from the previous iteration
- This avoids redundant derivation: facts already in `I_{d-1}` were processed in earlier iterations
- Output: applicable rule instantiations (rule index + partial bindings)

### JOIN

Bind rule body atoms against the fact set and compute head instantiations.

- For each applicable rule, find all substitutions `theta` such that `body(r) theta ⊆ I_d`
- Produce the corresponding `head(r) theta` atoms
- This is the expensive step — cubic in the worst case (joining M body atoms)

### MERGE

Deduplicate new atoms and merge into the provable set.

- Hash new atoms via `pack_triples_64`
- Remove duplicates (atoms already in `I_d`)
- If no new atoms: fixpoint reached, stop early
- Otherwise: `I_{d+1} = I_d ∪ new_atoms`, continue

### FC Implementations

| Implementation | Strategy | Trade-off |
|---------------|----------|-----------|
| `FCSemiNaiveGrounder` | Join-based semi-naive evaluation | General-purpose, moderate speed |
| `FCSPMMGrounder` | Sparse matrix multiply (SpMM) | Faster for dense rule sets, higher memory |

Both compute the same provable set for a given depth. The choice is a performance trade-off.

---

## 4. How BC and FC Compose

BC grounders use FC as a **pre-computation step** to determine which atoms are provable. This enables pruning:

```
                    ┌──────────────┐
                    │  FC Grounder │
                    │ (provable    │
                    │   set I_D)   │
                    └──────┬───────┘
                           │
                           ▼
┌──────────┐      ┌───────────────┐      ┌──────────┐
│  Queries │──────│  BC Grounder  │──────│  Output   │
│ [B, 3]   │      │  (uses I_D    │      │  [B,tG,M,3]
│          │      │   for pruning)│      │          │
└──────────┘      └───────────────┘      └──────────┘
```

### Composition patterns

**BCProvsetGrounder**: runs FC in `pre_ground()` to compute `I_D`. During postprocessing, filters out groundings where any body atom is not in `I_D`.

**ParametrizedBCGrounder**: runs FC at construction time. During grounding, checks each candidate body atom against `I_D` using binary search on `provable_hashes`. Atoms not in `I_D` count toward the width budget `W`.

**FullBCGrounder**: does **not** use FC. Enumerates all entities for free variables without any provability check.

### When each pattern applies

| Grounder | Uses FC? | FC purpose |
|----------|----------|------------|
| BCGrounder | No | Pure SLD resolution |
| BCPruneGrounder | No | Fixed-point pruning (no FC needed) |
| BCProvsetGrounder | Yes | Filter groundings by provability |
| ParametrizedBCGrounder | Yes | Width-check body atom provability |
| FullBCGrounder | No | Full enumeration, no filtering |
| LazyGrounder | Yes (inherited) | Reduced rule set → smaller provable set |
| SamplerGrounder | Yes (inherited) | Same as parent ParametrizedBC |
| KGEGrounder | Yes (inherited) | Same as parent ParametrizedBC |
| NeuralGrounder | Yes (inherited) | Same as parent ParametrizedBC |
| SoftGrounder | Yes (inherited) | Provability feeds into soft scoring |
