# SLD vs Enum Resolution

This document compares the two main resolution strategies in the grounder library
and how they interact with the terminal soundness filters (`fp_batch`, `fp_global`).

## Running example

Throughout this document we use:

```
Facts:
  parent(alice, bob)     parent(bob, charlie)

Rules:
  R1: ancestor(X, Y) :- parent(X, Y)
  R2: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)

Query: ancestor(alice, charlie)
```

Constants: `{alice=1, bob=2, charlie=3}`.  Predicates: `{parent=0, ancestor=1}`.
Padding index: `99`.  `constant_no = 3`.  Variables: indices `> 3`.

---

## 1. Resolution strategies

### SLD (Select-Literal-Descend)

Top-down, one goal at a time.  At each depth step, the **first open goal** is
selected and resolved against facts (via MGU) and rules (via head unification).

**Per-step operation** (all four phases are shared across resolutions; only
RESOLVE differs):
1. **SELECT** — pick first goal from `proof_goals [B, S, G, 3]`.
2. **RESOLVE** — produce K_f fact children + K_r rule children via MGU
   (`resolve_sld` in `resolution/sld.py`).
3. **PACK** — concatenate fact + rule children, scatter-compact into S_out
   states (`pack_states` in `bc/common.py`).
4. **POSTPROCESS** — prune known ground facts from goals, compact, collect
   terminal groundings.

Fact resolution unifies the goal with indexed facts.  Rule resolution unifies
the goal with rule heads, then substitutes into the rule body — but variables
not bound by the head MGU **remain as variables**.

**Example — SLD depth=2:**

```
Step 0 (init):
  State 0: goals=[ancestor(1,3)]    body=[pad]

Step 1 (SELECT ancestor(1,3)):
  Fact children: none (ancestor(1,3) not a fact)
  Rule R1 match: head ancestor(X,Y)→X=1,Y=3 → body=[parent(1,3)]
    → goals=[parent(1,3)]                 body=[parent(1,3)]
  Rule R2 match: head ancestor(X,Z)→X=1,Z=3 → body=[parent(1,Y4), ancestor(Y4,3)]
    → goals=[parent(1,Y4), ancestor(Y4,3)] body=[parent(1,Y4), ancestor(Y4,3)]
      (Y4 is a fresh variable, index > constant_no)

  POSTPROCESS:
    State from R1: parent(1,3) IS a fact → pruned → goals=[] → COLLECTED
      collected_body = [parent(1,3)]  ✓ ground
    State from R2: parent(1,Y4) NOT ground → stays as open goal

Step 2 (SELECT parent(1,Y4) from R2 state):
  Fact children: parent(1,2) matches → Y4=2 → remaining goal ancestor(2,3)
    substitute: ancestor(Y4,3) → ancestor(2,3)
    → goals=[ancestor(2,3)]

  POSTPROCESS:
    ancestor(2,3): is it a fact? No. Still open.
    → NOT collected (depth exhausted, state discarded)
```

Result after depth=2: **1 grounding collected** (from R1).
With depth=3, the R2 branch could continue resolving `ancestor(2,3)`.

Key observations:
- **Variables appear** in intermediate states (Y4 in step 1).
- **Only fully ground bodies** are collected (line 312-314 of `bc/common.py`:
  `is_ground = (body_args < E).all(...)`).
- The search is **depth-bounded**: unresolved goals at depth D are discarded.

### Enum (Entity Enumeration)

Bottom-up entity enumeration with pre-compiled binding tables.  Instead of
MGU, enum fills body templates by enumerating entity candidates from the fact
index.

**Per-step operation:**
The four phases (SELECT, RESOLVE, PACK, POSTPROCESS) are identical to SLD —
only RESOLVE differs:

2. **RESOLVE** — for each matching rule, enumerate entity candidates from the
   fact index to fill free variables, then fill body templates with the
   enumerated bindings.  All body atoms are **fully ground** by construction
   (`resolve_enum` in `resolution/enum.py`).  Returns the same 7-tensor
   format but with K_f = 0 (no separate fact-resolution children); all
   K_enum children are rule groundings whose entity bindings come from
   fact-anchored enumeration.

**Width parameter** controls how many body atoms may be "unknown" (not base
facts) per grounding:
- `width=None` (unlimited): all entity combinations enumerated.
- `width=w`: reject groundings with > w non-fact body atoms.
- Last step forces `width=0`: all body atoms must be facts (line 244 of
  `resolution/enum.py`).

**Example — Enum width=1, depth=2:**

```
Step 0 (init):
  State 0: goals=[ancestor(1,3)]    body=[pad]

Step 1 (SELECT ancestor(1,3)):
  Rule R1: body template = [parent(X,Y)] with X=1,Y=3 (both bound from head)
    → body=[parent(1,3)]   exists? YES → 0 unknowns ≤ 1 → ACCEPT
    → goals=[parent(1,3)]  body=[parent(1,3)]

  Rule R2: body template = [parent(X,Y), ancestor(Y,Z)] with X=1,Z=3
    Free variable: Y.  Enumerate Y from facts matching parent(1,*):
      Y=2 (from parent(1,2)):
        body=[parent(1,2), ancestor(2,3)]
        parent(1,2) exists=YES, ancestor(2,3) exists=NO → 1 unknown ≤ 1 → ACCEPT
        Also: ancestor pred IS a rule head → head_pred_mask OK

  POSTPROCESS:
    R1 state: parent(1,3) is a fact → pruned → goals=[] → COLLECTED
      collected_body = [parent(1,3)]  ✓
    R2 state: parent(1,2) is a fact → pruned.  ancestor(2,3) not a fact → stays.
      goals=[ancestor(2,3)]  body=[parent(1,2), ancestor(2,3)]

Step 2 (SELECT ancestor(2,3), width_d=0 for last step):
  Rule R1: body=[parent(2,3)].  parent(2,3) exists? YES → 0 unknowns ≤ 0 → ACCEPT
    → goals=[parent(2,3)]  body=[parent(1,2), ancestor(2,3)]

  Rule R2: body=[parent(2,Y5), ancestor(Y5,3)]
    Enumerate Y5 from parent(2,*): Y5=3 → body=[parent(2,3), ancestor(3,3)]
    parent(2,3) YES, ancestor(3,3) NO → 1 unknown > 0 → REJECT (width=0)

  POSTPROCESS:
    R1 state: parent(2,3) is a fact → pruned → goals=[] → COLLECTED
      collected_body = [parent(1,2), ancestor(2,3)]  ✓
```

Result: **2 groundings collected**.

Key observations:
- **All body atoms are always ground** — entities are enumerated, never left as
  variables.
- **Width filters during resolution** (inside `_apply_enum_filters`), before the
  topk selection that caps children to K_enum.
- The last step forces `width=0` for **new** body atoms introduced at that step.
  However, `grounding_body` is frozen at the first rule application
  (`pack_states` line 128-136: set when `top_ridx == -1`, inherited unchanged
  after).  So collected groundings **can still contain ground non-fact body
  atoms** from intermediate steps where `width > 0`.  This is why terminal
  soundness filters (`fp_batch` / `fp_global`) are necessary — they verify
  that ALL body atoms are provable, not just the leaf-level ones.

---

## 2. Terminal soundness filters

After the depth loop completes, `filter_terminal` (`bc/bc.py:338`) runs a
soundness filter on the **collected groundings** `[B, tG, M, 3]`.

Both filters answer the same question — "are all body atoms in this grounding
provable from base facts?" — but differ in **scope**: what evidence they use.

### No filter (`filter='none'`)

Returns all collected groundings as-is.  Some may have body atoms that are
not base facts and not provable.

### `fp_batch` — per-batch fixed-point (Kleene T_P iteration)

Computes provability by iterating over **collected groundings within the same
batch element**.  This is the batched analog of keras-ns
`PruneIncompleteProofs`.

**Algorithm** (reference: keras-ns `backward_chaining_grounder.py:172`):

The correct algorithm (as in keras-ns) works in two phases:

**Phase 1 — Build proof dependencies** (during backward chaining):
For each collected grounding, record `(query, proof_atoms)` where `proof_atoms`
is the list of body atoms that are NOT base facts (the "unknowns" that must be
proved by other groundings).

**Phase 2 — Fixed-point iteration** (Kleene T_P):
```
atom2proved = {}
for step in range(num_steps):
    snapshot = copy(atom2proved)              ← freeze state from previous iter
    for (query, proof_atoms) in all_proofs:
        if all proof_atoms are proved in snapshot:
            atom2proved[query] = True
```

The **snapshot** is critical for soundness — it ensures synchronous iteration
(Kleene semantics).  All lookups in iteration k use the state from end of
iteration k-1.  This prevents circular reasoning:

```
Circular rules:  p(X) :- q(X),  q(X) :- p(X)
No relevant facts.

Iteration 0: snapshot = {}
  p(a): needs q(a), q(a) not in snapshot → FALSE
  q(a): needs p(a), p(a) not in snapshot → FALSE
Iteration 1: snapshot = {} (unchanged)
  Same → both stay FALSE forever ✓
```

Neither atom ever bootstraps from base facts → neither is ever proved.
The snapshot prevents intra-iteration cascading that could mask circular
dependencies.

**Example with our running KB:**

The batch processes multiple queries.  Suppose the batch contains queries for
`ancestor(1,3)`, `ancestor(2,3)`, and `ancestor(1,2)`:

```
Collected groundings (across the batch):
  For ancestor(1,2): G_a body=[parent(1,2)]          proof=[]
  For ancestor(2,3): G_b body=[parent(2,3)]          proof=[]
  For ancestor(1,3): G_c body=[parent(1,3)]          proof=[]
  For ancestor(1,3): G_d body=[parent(1,2), ancestor(2,3)]  proof=[ancestor(2,3)]

Iteration 0: snapshot = {}
  G_a: proof=[] → all([]) = True → atom2proved[ancestor(1,2)] = True
  G_b: proof=[] → all([]) = True → atom2proved[ancestor(2,3)] = True
  G_c: proof=[] → all([]) = True → atom2proved[ancestor(1,3)] = True
  G_d: proof=[ancestor(2,3)] → ancestor(2,3) not in snapshot → FALSE

Iteration 1: snapshot = {ancestor(1,2), ancestor(2,3), ancestor(1,3)}
  G_d: proof=[ancestor(2,3)] → ancestor(2,3) in snapshot → TRUE ✓

Result: G_d PROVED (its body atom ancestor(2,3) was proved by G_b)
```

The key is that `fp_batch` operates **across queries within the batch** — G_b
(from the ancestor(2,3) query) proves ancestor(2,3), which G_d (from the
ancestor(1,3) query) depends on.

**Properties:**
- Sound: Kleene fixed-point ensures no circular proofs (snapshot mechanism).
- Completeness depends on batch composition: body atoms provable only if
  their groundings are also in the same batch.
- Cost: O(num_steps × N_proofs) per batch, where N_proofs is total proof
  entries across all queries.

**Current torch-ns limitation:** The current `filters/prune.py` implementation
assigns ALL groundings within a batch element the same head hash (the query).
Since query exclusion prevents body atoms from equaling the query, the
fixed-point iterations are effectively a no-op — it degenerates to "all body
atoms must be base facts."  This needs to be fixed to match the keras-ns
cross-query semantics described above.

### `fp_global` — global fixed-point (forward-chaining provable set)

Precomputes the provable set I_D by running **forward chaining over the entire
KB** at init time.  At filter time, checks each body atom independently.

**Algorithm** (`filters/provset.py` + `fc/fc.py`):

**Phase 1 — Forward chaining at init** (one-time cost):
```
I_0 = facts
for step in range(fc_depth):
    I_{k+1} = I_k ∪ {head(r,θ) | r is a rule, θ grounds body(r) ⊆ I_k}
I_D = final provable set
```

This is the same Kleene T_P iteration as `fp_batch`, but applied globally
over ALL rules × ALL entities rather than over collected groundings.

**Phase 2 — Filter at query time** (per grounding):
```
for each grounding:
    for each body atom:
        if atom ∈ facts OR atom ∈ I_D:  OK
        else: REJECT grounding
```

**Example:**

```
Forward chaining builds I_D:
  Step 0: I_0 = {parent(1,2), parent(2,3)}
  Step 1: R1 fires → add ancestor(1,2), ancestor(2,3)
  Step 2: R2 fires with ancestor(2,3) ∈ I_1 → add ancestor(1,3)
  I_D = {parent(1,2), parent(2,3), ancestor(1,2), ancestor(2,3), ancestor(1,3)}

Filter:
  G_c body=[parent(1,3)]: parent(1,3) is a fact → PROVED
  G_d body=[parent(1,2), ancestor(2,3)]:
    parent(1,2) ∈ facts → OK
    ancestor(2,3) ∈ I_D → OK
    → PROVED ✓
```

**Properties:**
- Sound: I_D contains only atoms genuinely derivable from facts + rules.
- Complete: does not depend on batch composition — every derivable atom is in
  I_D regardless of which queries are in the current batch.
- Cost at filter time: O(tG × M × log(|I_D|)) per query (single searchsorted).
- Cost at init: O(fc_depth × R × E^arity) — can be expensive for large KBs.
- Currently only implemented for enum resolution (`_build_provable_set` in
  `bc/bc.py`).

### Relationship between `fp_batch` and `fp_global`

Both compute provability via Kleene T_P fixed-point.  They differ in scope:

| | `fp_batch` | `fp_global` |
|---|---|---|
| **Evidence** | Collected groundings in the current batch | All rules × all entities (entire KB) |
| **When computed** | At query time (after BC loop) | At init time (before any queries) |
| **Depends on batch** | Yes — different batches may prove different atoms | No — I_D is fixed for the KB |
| **Multi-hop proofs** | Only if intermediate atoms are in the same batch | Always (I_D contains all derivable atoms) |
| **Circular safety** | Snapshot mechanism (Kleene semantics) | Forward chaining is inherently acyclic |

`fp_global` is strictly stronger than `fp_batch`: every atom provable by
`fp_batch` is also in I_D, but I_D may contain atoms whose groundings weren't
collected in the current batch.  The trade-off is init cost — `fp_global`
requires precomputing I_D over the full KB.

In practice, `fp_batch` is sufficient when training batches are large enough
that intermediate atoms are likely in the same batch.  `fp_global` is needed
when batches are small or proofs are deep.

---

## 3. Comparison table

### Resolution

| Aspect | SLD | Enum |
|--------|-----|------|
| Unification | MGU (Most General Unifier) | Pre-compiled binding tables |
| Variables in intermediate states | Yes (fresh vars from standardize-apart) | No (all ground by enumeration) |
| Children per state per step | K_f fact + K_r rule = K | K_enum (rules only, K_f=0) |
| Free variable handling | Deferred to next depth step | Enumerated from fact index immediately |
| Width parameter | Not used | Controls max unknown body atoms |
| Body atoms at collection | Always ground (enforced by `is_ground` check) | Always ground (by construction) |

### Terminal filters

| Aspect | `fp_batch` | `fp_global` |
|--------|------------|-------------|
| When | After depth loop | After depth loop |
| Precomputation | None | FC provable set at init |
| Algorithm | Kleene T_P fixed-point over collected groundings | Single-pass searchsorted against I_D |
| Multi-hop | Yes, if intermediate atoms are in the same batch | Yes, always (I_D is global) |
| Batch-dependent | Yes — provability depends on batch composition | No — I_D is fixed for the KB |
| Circular safety | Snapshot mechanism (synchronous iteration) | FC is inherently acyclic |
| Available for | All resolutions | Enum only (needs FC init; could be generalized) |

### Combined configurations

| Config | Resolution | Terminal filter | Behavior |
|--------|-----------|-----------------|----------|
| `sld.none.d2` | SLD | none | Raw SLD, no soundness check |
| `sld.fp_batch.d2` | SLD | fp_batch | SLD + cross-query fixed-point |
| `enum.fp_batch.w1.d2` | Enum (width=1) | fp_batch | Enum with width + batch fixed-point |
| `enum.fp_global.w1.d2` | Enum (width=1) | fp_global | Enum with width + global FC provable set |

---

## 4. Time and space complexity

### Notation

| Symbol | Meaning | Typical range |
|--------|---------|---------------|
| B | Batch size (queries) | 32–512 |
| S | Max states per query per step | K or max_states |
| D | Depth (number of BC steps) | 1–4 |
| K_f | Max fact children per state (SLD) | 10–100 |
| K_r | Max rule children per state (SLD) | num matching rules |
| K | K_f + K_r (SLD total children) | ≤ K_MAX (550) |
| K_enum | Max children per state (enum) | R_eff × G_per_query |
| R_eff | Effective rules per predicate (enum) | 1–10 |
| G | Max groundings per rule per query (enum) | 32 |
| M | Max body atoms per rule | 2–3 |
| F | Number of facts | 100–100k |
| E | Number of entities | 10–10k |
| tG | Max collected groundings (`effective_total_G`) | 32–128 |
| w | Width (enum only) | 0–2 or None |

### SLD — per step

| Operation | Time | Tensors / Memory |
|-----------|------|------------------|
| Fact lookup (`targeted_lookup`) | O(B·S) index ops | `[B·S, K_f]` indices |
| Fact MGU (`unify_one_to_one`) | O(B·S·K_f) | `[B·S·K_f, 3]` atoms |
| Fact substitutions | O(B·S·K_f·(G+M)) | `[B·S·K_f, G+M, 3]` combined |
| Rule lookup (segment) | O(B·S) | `[B·S, K_r]` indices |
| Standardize apart | O(B·S·K_r·M) | `[B·S·K_r, M, 3]` bodies |
| Rule MGU | O(B·S·K_r) | `[B·S·K_r, 3]` heads |
| Rule substitutions | O(B·S·K_r·(M+G+M_g)) | `[B·S·K_r, M+G+M_g, 3]` combined |
| Pack (scatter-compact) | O(B·S·K) | `[B, S_out, G, 3]` output |
| Prune ground facts | O(B·S·G) searchsorted | `[B, S, G, 3]` goals |
| Collect groundings | O(B·(tG+S)) topk | `[B, tG, M, 3]` buffer |

**Total per step**: O(B · S · K · (G + M))

**Peak memory per step**: `[B, S, K_r, G+M+M_g, 3]` for the substitution
combined tensor — this is the largest allocation.

**Across D steps**: S states feed into the next step.  Since pack compacts to
S_out ≤ S, memory is **constant per step** (not exponential).  Total time is
O(D · B · S · K · (G + M)).

### Enum — per step

| Operation | Time | Tensors / Memory |
|-----------|------|------------------|
| Rule clustering (`pred_rule_indices`) | O(B·S) gather | `[B·S, R_eff]` |
| Enumerate candidates | O(B·S·R_eff) index ops | `[B·S·R_eff, G]` candidates |
| Fill body templates (`_fill_body`) | O(B·S·R_eff·G·M) gather | `[B·S, R_eff, G, M, 3]` bodies |
| `fact_index.exists()` | O(B·S·R_eff·G·M) searchsorted | `[B·S·R_eff·G·M]` bool |
| Width + filters (`_apply_enum_filters`) | O(B·S·R_eff·G·M) | `[B·S, R_eff, G]` mask |
| topk (cap to K_enum) | O(B·S·K_total·log K_enum) | `[B·S, K_enum]` indices |
| Pack | O(B·S·K_enum) | `[B, S_out, G, 3]` output |

**Total per step**: O(B · S · R_eff · G · M)

**Peak memory per step**: `[B·S, R_eff, G, M, 3]` for the body atoms tensor.

**Width impact on enum**:
- `width=None`: all entity combinations enumerated (G = `max_groundings_per_query`).
  Memory for dual anchoring doubles: `[B·S, R_eff, G/2, M, 3]` × 2 directions.
- `width=w < M`: enables dual anchoring (direction B).  Fewer candidates pass
  the filter, but **tensor shapes are the same** — width only affects the mask,
  not the allocation.

### Terminal filters

| Filter | Time | Memory |
|--------|------|--------|
| `fp_batch` | O(num_steps × N_proofs) per batch — Kleene iteration over proof entries | `atom2proved` dict (or tensor equivalent) |
| `fp_global` filter | O(tG · M · log(\|I_D\|)) per query — single searchsorted pass | `[B, tG, M]` hashes + `[\|I_D\|]` provable set |
| `fp_global` init (FC) | O(fc_depth · R · E^arity) — forward chaining at construction time | `[I_D]` provable atoms |

### Scaling summary

| Axis | SLD | Enum |
|------|-----|------|
| More entities (E↑) | K_f may grow (more fact matches per goal) | G candidates grow (more enumerations) |
| More rules (R↑) | K_r grows linearly | R_eff grows (rules per predicate) |
| More facts (F↑) | Fact lookup O(1) via CSR, K_f capped | `exists()` cost O(log F) per atom |
| Deeper (D↑) | G grows: max_goals = 1 + D·(M-1) | S grows: more states to carry |
| Wider (w↑, enum only) | N/A | More candidates pass filter → more packing work |

---

## 5. When to use which

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Few rules, many facts | SLD | K_r small, fact lookup is O(1) via CSR |
| Many rules, few entities | Enum | Pre-compiled bindings amortize rule overhead |
| Deep proofs (D ≥ 3) | SLD | Enum's last-step width=0 constraint is limiting |
| Width-bounded search | Enum | Native width support prunes search early |
| Multi-hop soundness, small batch | `fp_global` | Provability independent of batch composition |
| Multi-hop soundness, large batch | `fp_batch` | Cheaper than FC init; batch likely covers intermediates |
| CUDA-compiled training | Either | Both produce fixed-shape tensors for `torch.compile` |
