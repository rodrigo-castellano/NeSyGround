# Grounding Report: Backward Chaining Engine

## 1. Problem Statement

Given a knowledge graph (facts) and a set of Horn rules, the **grounder** finds all valid rule instantiations that prove a batch of **grounded** query atoms (both arguments are constants, e.g. `parent(john, mary)`). These groundings feed into a neural-symbolic model (SBR/GSBR) that computes differentiable confidence scores.

The core challenge: backward chaining creates a **search tree** that grows exponentially with proof depth. The grounder must handle this efficiently in batched, GPU-compiled tensor operations.

---

## 2. Architecture

The grounder lives in `torch-ns/grounder/` as a standalone compiled package.

```
grounder/
  grounders/
    base.py        -- Grounder: owns facts, rules, indices
    backward.py    -- BCGrounder: 5-stage step() + multi-depth forward()
    prolog.py      -- PrologGrounder: K = K_f + K_r (single-level)
    rtf.py         -- RTFGrounder: K = K_f * K_r (two-level, not covered here)
  primitives.py    -- unify_one_to_one, apply_substitutions
  packing.py       -- pack_combined, pack_fact_rule, compact_atoms
  postprocessing.py -- prune_ground_facts, collect_groundings
  fact_index.py    -- ArgKeyFactIndex, InvertedFactIndex, BlockSparseFactIndex
  rule_index.py    -- RuleIndex (segment-based lookup)
  types.py         -- ForwardResult, StepResult, ResolveResult, PackResult
```

### Class Hierarchy

```
Grounder (base: owns KB state, builds indices)
  └── BCGrounder (5-stage pipeline: SELECT → RESOLVE → PACK → POSTPROCESS)
        └── PrologGrounder (single-level: facts + rules independently, K = K_f + K_r)
        └── RTFGrounder (two-level: rules then facts, K = K_f * K_r)
```

Only the **Prolog strategy** is covered here.

---

## 3. Tensor Conventions

| Symbol | Meaning | Typical values |
|--------|---------|----------------|
| B | Batch size (queries) | 128-512 |
| S | Concurrent states per query | 1 (RL) or K (TS) |
| G | Max atoms per proof state | 8 (RL) or 26 (TS) |
| M | Max body atoms per rule | 2-3 |
| K | Max derived children per state | K_f + K_r |
| K_f | Max fact matches per query | data-dependent |
| K_r | Max rule matches per query | data-dependent |
| tG | Max collected groundings | 64 |
| D | Proof depth | 1-3 |

Every atom is a triple `(pred, arg0, arg1)` stored as 3 int64 values.

**Variable representation**: constants are indices `[0, constant_no]`. Variables are indices `>= constant_no + 1`. Padding is a special index above all variables.

**Queries are always grounded**: both arguments are constants. Variables only appear in proof states after rule application introduces free body variables (e.g., `parent(X, Z), parent(Z, Y)` with `Z` free).

---

## 4. Worked Examples

### Example 1: 1-body rule (1 depth)

```
KB:   father(john, mary), father(john, bob), mother(mary, sue)
Rule: parent(X, Y) :- father(X, Y)     [r1]
      parent(X, Y) :- mother(X, Y)     [r2]

Query: parent(john, mary)

── DEPTH 0 ─────────────────────────────────────────────────────

1. SELECT
   goal = parent(john, mary)       remaining = []

2. RESOLVE
   FACTS: parent(john,mary) not a base fact           -> K_f = 0
   RULES: 2 rules match head predicate "parent"       -> K_r = 2
     r1: Standardize: X->v0, Y->v1
         Unify: parent(john,mary) ~ parent(v0,v1)
           sigma = {v0->john, v1->mary}
         Apply to body: father(v0,v1) -> father(john,mary)
         child_1: goals=[father(john,mary)]

     r2: Standardize: X->v2, Y->v3
         Unify: parent(john,mary) ~ parent(v2,v3)
           sigma = {v2->john, v3->mary}
         Apply to body: mother(v2,v3) -> mother(john,mary)
         child_2: goals=[mother(john,mary)]

3. PACK
   K_f=0 fact children + K_r=2 rule children -> compact to S=2 states
   state_0: goals=[father(john,mary)]
   state_1: goals=[mother(john,mary)]

4. POSTPROCESS
   PRUNE: father(john,mary) IS a base fact -> remove from goals
     state_0: goals=[]  (empty!)
   PRUNE: mother(john,mary) is NOT a base fact -> keep
     state_1: goals=[mother(john,mary)]  (still open)

   COLLECT: state_0 has empty goals -> proof found! state_0 deactivated.

   Result: 1 grounding collected, 1 state still active (but depth exhausted)
```

### Example 2: 2-body rule with free variables (3 depths)

```
KB:   father(john, mary), mother(mary, sue)
Rules: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)    [r1]
       parent(X, Y) :- father(X, Y)                        [r2]
       parent(X, Y) :- mother(X, Y)                        [r3]

Query: grandparent(john, sue)

── DEPTH 0 ─────────────────────────────────────────────────────

1. SELECT
   goal = grandparent(john, sue)    remaining = []

2. RESOLVE
   FACTS: no match                                      -> K_f = 0
   RULES: r1 matches head predicate "grandparent"       -> K_r = 1
     r1: Standardize: X->v0, Y->v1, Z->v2
         Unify: grandparent(john,sue) ~ grandparent(v0,v1)
           sigma = {v0->john, v1->sue}
         Apply to body: parent(v0,v2), parent(v2,v1) -> parent(john,v2), parent(v2,sue)
         child: goals=[parent(john,v2), parent(v2,sue)]
                       ^^^ v2 is FREE (not bound by head)

3. PACK
   1 valid child -> S=1
   state_0: goals=[parent(john,v2), parent(v2,sue)]

4. POSTPROCESS
   PRUNE: parent(john,v2) has variable v2 -> not ground -> skip
   PRUNE: parent(v2,sue) has variable v2 -> not ground -> skip
   COLLECT: goals not empty -> no proof yet

── DEPTH 1 ─────────────────────────────────────────────────────

1. SELECT
   goal = parent(john, v2)    remaining = [parent(v2, sue)]

2. RESOLVE
   FACTS: parent(john,v2) not a base fact               -> K_f = 0
   RULES: r2, r3 match head predicate "parent"          -> K_r = 2
     r2: Standardize: X->v3, Y->v4
         Unify: parent(john,v2) ~ parent(v3,v4)
           sigma = {v3->john, v4->v2}
         Apply to body+remaining:
           father(v3,v4) -> father(john,v2)
           parent(v2,sue) unchanged (v2 not in sigma domain as target)
         child_1: goals=[father(john,v2), parent(v2,sue)]

     r3: Standardize: X->v5, Y->v6
         Unify: parent(john,v2) ~ parent(v5,v6)
           sigma = {v5->john, v6->v2}
         Apply: mother(v5,v6) -> mother(john,v2)
         child_2: goals=[mother(john,v2), parent(v2,sue)]

3. PACK
   2 valid children -> S=2

4. POSTPROCESS
   PRUNE: all goals still have variables -> nothing pruned
   COLLECT: no empty goals -> no proofs yet

── DEPTH 2 ─────────────────────────────────────────────────────

(processing state_0: goals=[father(john,v2), parent(v2,sue)])

1. SELECT
   goal = father(john, v2)    remaining = [parent(v2, sue)]

2. RESOLVE
   FACTS: lookup (father, john) -> father(john,mary)    -> K_f = 1
     Unify: father(john,v2) ~ father(john,mary)
       sigma = {v2->mary}
     Apply to remaining: parent(v2,sue) -> parent(mary,sue)
     child: goals=[parent(mary,sue)]

   RULES: no rule has head "father"                     -> K_r = 0

3. PACK
   1 valid child -> S=1
   state_0: goals=[parent(mary,sue)]

4. POSTPROCESS
   PRUNE: parent(mary,sue) is NOT a base fact -> keep
     (but mother(mary,sue) IS a fact -- different predicate!)
   COLLECT: goals not empty -> no proof yet
   -> would need DEPTH 3 to resolve parent(mary,sue) via r3 -> mother(mary,sue)
```

---

## 5. The 5-Stage Pipeline (Implementation Detail)

### Stage 1: SELECT

Extract the first unresolved goal atom from each active state.

```python
queries = proof_goals[:, :, 0, :]      # [B, S, 3] -- first atom
remaining = proof_goals.clone()
remaining[:, :, 0, :] = padding_idx    # mask selected atom
active_mask = proof_goals[:, :, 0, 0] != padding_idx  # [B, S]
```

Active states have a non-padding first atom. Inactive states (all padding) are skipped.

### Stage 2: RESOLVE FACTS

For each query atom, find matching facts via `ArgKeyFactIndex.targeted_lookup`:

1. **Lookup**: composite key `(pred, bound_arg)` -> up to K_f fact indices
2. **Unify**: `unify_one_to_one(query, fact)` -> substitution pairs
3. **Apply**: `apply_substitutions(remaining + gbody, subs)` -> successor goals

The fact index uses a hash table keyed by `(predicate, arg0)` or `(predicate, arg1)` depending on which argument is bound (ground constant). The initial query is always fully grounded, but after rule application, subgoals may have one free variable -- the bound argument is used for lookup. For each key, it stores up to K_f matching fact indices in contiguous memory.

**Result**: `fact_goals [B, S, K_f, G, 3]`, `fact_success [B, S, K_f]`

### Stage 3: RESOLVE RULES

For each query atom, find matching rules and perform MGU:

1. **Lookup**: `RuleIndex.lookup_by_segments(pred)` -> up to K_r rules by head predicate
2. **Standardize Apart**: rename template variables `(constant_no+1, constant_no+2, ...)` to fresh runtime IDs unique per (batch, state) pair via offset arithmetic
3. **Unify**: `unify_one_to_one(query, standardized_head)` -> substitution pairs
4. **Apply**: apply subs to `[body, remaining, grounding_body]`
5. **Assemble**: new goals = `[substituted_body | substituted_remaining]`

**Variable Standardization** is critical: without it, two states using the same rule would share variable IDs and interfere. Each state gets a namespace:
```
runtime_var = next_var_base + state_idx * max_vars_per_rule + (template_var - template_start)
```

**Result**: `rule_goals [B, S, K_r, G, 3]`, `rule_success [B, S, K_r]`

### Stage 4: PACK

Flatten `S * K_f` fact children and `S * K_r` rule children into `S_out` output slots.

Uses **scatter-based compaction** (not topk or nonzero, which break torch.compile):

```python
cumsum = success.long().cumsum(dim=1)           # [B, K_total]
target_idx = where(success, cumsum - 1, K)      # valid -> 0,1,2,...; invalid -> garbage slot
output = full((B, K+1, G, 3), padding)
output.scatter_(1, target_idx_expanded, states)  # place valid entries
output = output[:, :K, :, :]                     # discard garbage slot
```

**Two pack modes**:
- **TS mode** (`track_grounding_body=True`): `pack_fact_rule` -- tracks grounding body and rule index through pack
- **RL mode** (`track_grounding_body=False`): `pack_combined` -- lightweight, skips gbody/ridx

### Stage 5: POSTPROCESS

1. **Prune ground facts**: for each goal atom, check if it's a known fact via hash-based membership (`searchsorted` on sorted fact hashes). If so, remove it (the goal is already satisfied).
2. **Compact atoms**: left-align remaining atoms after pruning (remove gaps via `argsort`-based stable sort).
3. **Collect groundings** (TS only): detect terminal states (all goals resolved = all padding), store their grounding bodies in the output buffer, deactivate them.

---

## 6. The Unification Primitive

`unify_one_to_one(queries, terms, constant_no, padding_idx)` is the core operation.

**Input**: `queries [L, 3]`, `terms [L, 3]` (pairwise, not cross-product).

**Logic per argument position**:

| Query arg | Term arg | Action |
|-----------|----------|--------|
| constant a | constant a | OK (equal) |
| constant a | constant b | FAIL (conflict) |
| variable X | constant a | bind X -> a |
| constant a | variable X | bind X -> a |
| variable X | variable Y | bind Y -> X |
| padding | anything | no sub |

**Output**: `mask [L]` (success), `subs [L, 2, 2]` (two (from, to) pairs, one per arg position).

**Consistency check**: if both arg positions bind the same variable to different values, unification fails.

**Key property**: since queries are always grounded, the initial unification against a rule head always binds both head variables to constants. But multi-body rules introduce **free variables** in the body (variables not in the head). MGU doesn't enumerate these -- they stay as variables in the successor state and get bound at a later depth when matched against facts. This is fundamentally different from the cartesian product approach (keras-ns) which eagerly enumerates all entities for free variables.

---

## 7. Multi-Depth: RL vs TS

Both RL and torch-ns use the same BC operator (see section 8). The difference is how they handle the $K$ children produced at each step.

Let $S_d$ be the set of active proof states at depth $d$. Initially $S_0 = \{[\text{query}]\}$ (one state, one goal). Applying BC to a set means applying it to every state and collecting all children: $\text{BC}(S) = \bigcup_{s \in S} \text{BC}(s)$.

**RL — applies BC once, policy selects one child:**

$$S_d \xrightarrow{\text{BC}} S'_d \xrightarrow{\pi} S_{d+1} \quad \text{where } |S_{d+1}| = 1$$

BC produces $K$ children. The RL agent's policy $\pi$ picks **one** to keep. Memory stays constant.

**torch-ns — applies BC iteratively, keeping all children:**

$$S_0 \xrightarrow{\text{BC}} S_1 \xrightarrow{\text{BC}} S_2 \xrightarrow{\text{BC}} \dots \xrightarrow{\text{BC}} S_D$$

Each step keeps **all** children. The frontier grows: $|S_d| \leq K^d$. Complete up to depth $D$, but memory grows exponentially.

| | Active states after step d | Memory per step |
|---|---|---|
| **RL** | 1 (policy selects one child) | O(B x K x G x 3) — constant |
| **torch-ns** | up to K^d (keeps all children) | O(B x K^d x K x G x 3) — exponential in d |

The subsections below detail the implementation of each mode.

### 7.1 RL (batched-env): One Depth Per Call

The RL environment calls `grounder.step()` once per action:

```
Input:  [B, 1, G, 3]    -- 1 state per query
Output: [B, K, G, 3]    -- K children per query

RL agent picks 1 child -> next state [B, 1, G, 3]
Repeat for next depth
```

**Configuration**:
- `S = 1` (one active state per query)
- `track_grounding_body = False` (no need to collect groundings; the RL agent only needs successor states)
- `max_goals = G` (typically 8, smaller than TS)

**Memory per step**: `B * K * G * 3 * 8` bytes. With B=512, K=120, G=8: **~14 MB**. Constant across all depths.

The RL agent controls which children to expand, making it **selective** and **depth-adaptive**. It can go arbitrarily deep without memory explosion because it only ever holds 1 state per query.

### 7.2 torch-ns: All Depths at Once

`grounder.forward(queries, mask)` runs D steps internally:

```python
for d in range(D):
    # Clone for CUDA graph boundary (depth > 0)
    (gbody, goals, ridx, valid, next_var) = step(gbody, goals, ridx, valid, next_var)
    # Postprocess: prune + compact + collect groundings
    (goals, collected_body, collected_mask, collected_ridx, valid) = postprocess(...)
```

**Configuration**:
- `S = K` (each step expands ALL children as new states)
- `track_grounding_body = True` (collecting rule instantiations)
- `max_goals = G` (typically 26)
- `depth = D` (fixed at construction, typically 1-3)

**Frontier growth**: after step d, there are up to `K^d` active states per query.

**Memory per step**: `B * S * K * G * 3 * 8` bytes. S grows as K^d:

| Depth | Active states (S) | Memory (B=512, K=120, G=26) |
|-------|-------------------|-----------------------------|
| D=1 | 120 | 38 MB |
| D=2 | 14,400 | 4.5 GB |
| D=3 | 1,728,000 | **impossible** |

In practice, S is capped (default 5000 or via `max_states`), and K is capped (`K_MAX`). But even with capping, the exponential pressure is real.

### 7.3 Comparison Table

| Aspect | RL (1-depth) | TS (D-depth) |
|--------|-------------|-------------|
| States per query | **1** (constant) | K^D (exponential, capped) |
| Memory | O(B * K) per step | O(B * S * K) per step, S grows |
| Depth control | Agent policy (adaptive) | Fixed D at construction |
| Search strategy | Selective (policy-guided) | Exhaustive BFS |
| Output | K successor states | All collected groundings |
| Grounding body | Not tracked | Tracked + collected |
| Scalability | Large KGs, any depth | Limited by K^D explosion |
| Completeness | Depends on policy quality | Complete (up to depth D) |

### 7.4 Dataset-Specific Explosion

Real datasets show how fast the explosion hits:

| Dataset | Facts | K_f | K_r | K_f+K_r | D=2 states | D=3 states |
|---------|-------|-----|-----|---------|------------|------------|
| family | 19,845 | 28 | 22 | 50 | 250K | 12.5M |
| wn18rr | 86,835 | 473 | 8 | 481 | 23M | **impossible** |
| fb15k237 | 272,115 | 3,612 | 30 | 3,642 | **1.3B** | -- |
| deep_chain | 100,310 | 107 | 1 | 108 | 1.2M | 125M |

The RL agent sidesteps this entirely: 1 state regardless of dataset or depth.

---

## 8. Operator Reference

### 8.1 MATCH_FACT

$$\text{MATCH}(q, \mathcal{F}) = \{f \in \mathcal{F} \mid \text{pred}(f) = \text{pred}(q) \wedge \text{arg}_i(f) = \text{arg}_i(q) \text{ for bound } i\}$$

**ArgKey index**: hash table keyed by `pack(pred, bound_arg)`. Lookup is O(1) per query, returns up to K_f facts.

**Inverted index**: stores per-predicate, per-direction (subj or obj) lists of entity-to-fact mappings. Enumerate returns all facts matching `(pred, bound_arg, direction)`.

**Complexity**: O(K_f) per query atom.

### 8.2 MATCH_RULE

$$\text{MATCH}(q, \mathcal{R}) = \{r \in \mathcal{R} \mid \text{pred}(r.\text{head}) = \text{pred}(q)\}$$

**Segment index**: rules sorted by head predicate. `lookup_by_segments` does a table lookup to find the segment `[start, end)` for a given predicate, then returns up to K_r rule positions.

**Complexity**: O(1) lookup + O(K_r) gather.

### 8.3 STANDARDIZE

$$\text{STD}(r, b, s) : v_i \mapsto \text{nv}[b] + s \cdot V + (v_i - E)$$

where nv[b] is the per-batch variable counter, s is the state index, V = max_vars_per_rule, E = constant_no + 1.

This ensures every (batch, state) pair uses a disjoint variable namespace. `next_var_indices` is incremented by `S * V` after each step.

### 8.4 UNIFY (MGU)

$$\text{MGU}(q, t) = \sigma \text{ such that } q\sigma = t\sigma$$

Implemented as `unify_one_to_one`. Returns at most 2 substitution pairs (one per argument position). Handles var-const, const-var, var-var bindings. Detects conflicts (const-const mismatch, same var bound to different values).

**Key property**: no enumeration of free variables. Unbound vars stay as vars in the successor state.

### 8.5 APPLY

$$\text{APPLY}(\text{atoms}, \sigma) : \text{atoms}[v \mapsto \sigma(v)]$$

Implemented as `apply_substitutions`. Optimized for S=2 (2 sub pairs): loop-unrolled `torch.where` chain. General case uses gather-based lookup.

### 8.6 PACK

$$\text{PACK}(\text{children}_f, \text{children}_r, S_{out}) \to \text{states}[B, S_{out}, G, 3]$$

Scatter-based: cumsum of valid mask gives target indices. Invalid entries go to a garbage slot (index S_out) that's discarded after scatter. torch.compile compatible (no topk, no nonzero).

### 8.7 PRUNE

$$\text{PRUNE}(\text{goals}, \mathcal{F}) = \{a \in \text{goals} \mid \neg(\text{ground}(a) \wedge a \in \mathcal{F})\}$$

Hash-based: `pack_triples_64(pred, arg0, arg1)` computes a hash. `searchsorted` against sorted fact hashes determines membership. O(log F) per atom.

After pruning, `compact_atoms` removes gaps via stable argsort on a `(valid, position)` key.

### 8.8 COLLECT (TS only)

$$\text{COLLECT} = \{(g.\text{body}, g.\text{ridx}) \mid \text{all goals of } g \text{ are padding} \wedge g.\text{body is ground}\}$$

Terminal detection: `(proof_goals[:,:,:,0] == pad).all(dim=2)`. Deduplication via polynomial hash on `(ridx, body atoms)`. Terminal states are deactivated to prevent re-expansion.

---

## 9. Operators in Detail

### Why define BC as an operator?

The grounding engine is a complex system — fact indices, variable namespaces, scatter-based packing, hash-based pruning. Defining it as a single algebraic operator $\text{BC}$ lets us reason about what the engine **does** without worrying about **how** it does it.

This gives us three things:

1. **Composability**: a depth-$D$ proof is just $D$ applications of the same operator: $\text{BC}^D(s_0)$. We don't need a separate description for each depth — the operator composes with itself.

2. **RL vs torch-ns as a one-line difference**: both use the exact same $\text{BC}$. The only difference is what happens to the output:
   - torch-ns: $S_{d+1} = \bigcup_{s \in S_d} \text{BC}(s)$ — keep all children
   - RL: $s_{d+1} = \pi\big(\text{BC}(s_d)\big)$ — policy selects one child

   The scalability analysis (section 7) follows directly from this: torch-ns frontier grows as $K^d$, RL stays at 1. No need to trace through implementation details to see why.

3. **Correctness by construction**: each sub-operator (SELECT, MATCH, UNIFY, APPLY, PACK, PRUNE) has a clear contract. If each one is correct, their composition $\text{BC}$ is correct. This is how we verify the tensor implementation — check each primitive against its algebraic definition, not end-to-end debugging.

In short: the operator is the algebraic specification of the grounding engine. The code is an implementation of that spec.

### Walkthrough

This section walks through every sub-operator in the BC step from section 4 of the slide. We use the grandparent example throughout:

```
KB (facts):  father(john, mary), mother(mary, sue)
Rules:       grandparent(X, Y) :- parent(X, Z), parent(Z, Y)    [r1]
             parent(X, Y) :- father(X, Y)                        [r2]
             parent(X, Y) :- mother(X, Y)                        [r3]

Query:       grandparent(john, sue)
```

Recall the full BC operator from the slide:

$$\text{BC}(s, \mathcal{F}, \mathcal{R}) = \text{PRUNE}\Big(\text{PACK}\big(\text{RESOLVE}(g_1, \mathcal{F}, \mathcal{R}),\ [g_2, \dots]\big),\ \mathcal{F}\Big)$$

We break this down step by step.

---

### 9.1 SELECT

**What it does**: picks the **first** unresolved goal from the proof state, and sets aside the rest.

$$g_1 = s[0], \quad \text{remaining} = s[1:]$$

**Example** (depth 0):
```
State:     s = [grandparent(john, sue)]
           ↓ SELECT
Goal:      g₁ = grandparent(john, sue)
Remaining: []                              (no other goals)
```

**Example** (depth 1, after r1 was applied):
```
State:     s = [parent(john, v2), parent(v2, sue)]
           ↓ SELECT
Goal:      g₁ = parent(john, v2)
Remaining: [parent(v2, sue)]               (carried forward to children)
```

Always picks position 0. The remaining goals are **inherited** by every child — they don't disappear, they just wait their turn.

---

### 9.2 MATCH

**What it does**: finds all facts and rules whose predicate matches the goal's predicate.

$$\text{MATCH}(g, \mathcal{F}) = \{f \in \mathcal{F} \mid \text{pred}(f) = \text{pred}(g) \wedge \text{args compatible}\}$$
$$\text{MATCH}(g, \mathcal{R}) = \{r \in \mathcal{R} \mid \text{pred}(r.\text{head}) = \text{pred}(g)\}$$

This is a **lookup**, not unification — it just filters by predicate (and bound arguments for facts).

**Example** (depth 0, goal = `grandparent(john, sue)`):
```
Fact matches:  none (no fact has predicate "grandparent")     → K_f = 0
Rule matches:  r1 has head "grandparent"                      → K_r = 1
```

**Example** (depth 1, goal = `parent(john, v2)`):
```
Fact matches:  none (no fact has predicate "parent")           → K_f = 0
Rule matches:  r2, r3 have head "parent"                      → K_r = 2
```

**Example** (depth 2, goal = `father(john, v2)`):
```
Fact matches:  father(john, mary) matches (pred + arg0)       → K_f = 1
Rule matches:  none (no rule has head "father")                → K_r = 0
```

For facts, the index uses the **bound argument** as a key: if the goal is `father(john, v2)`, it looks up `(father, john)` and finds `father(john, mary)`. If both arguments are bound (e.g., `father(john, mary)`), it uses arg0 for lookup and checks arg1 after.

---

### 9.3 UNIFY (MGU)

**What it does**: given the goal $g$ and a matched term $t$ (a fact or a rule head), finds a substitution $\sigma$ that makes them equal.

$$\text{MGU}(g, t) = \sigma \text{ such that } g\sigma = t\sigma, \quad \text{or } \bot \text{ if impossible}$$

The substitution $\sigma$ is a set of bindings `{variable → value}`. Each argument position can produce at most one binding, so $\sigma$ has at most 2 pairs.

**Case table** (what happens for each argument position):

| Goal arg | Term arg | Result |
|----------|----------|--------|
| constant `a` | constant `a` | OK, no binding needed |
| constant `a` | constant `b` | FAIL — unification impossible |
| variable `X` | constant `a` | bind `X → a` |
| constant `a` | variable `X` | bind `X → a` |
| variable `X` | variable `Y` | bind `Y → X` (chain) |

**Example** (depth 0, goal vs r1 head):
```
Goal:        grandparent(john, sue)
Rule head:   grandparent(X, Y)         (standardized to v0, v1)

Arg0:  john (const) vs v0 (var)  →  bind v0 → john
Arg1:  sue  (const) vs v1 (var)  →  bind v1 → sue

σ = {v0 → john, v1 → sue}     ✓ success
```

**Example** (depth 2, goal vs fact):
```
Goal:        father(john, v2)
Fact:        father(john, mary)

Arg0:  john (const) vs john (const)  →  equal, no binding
Arg1:  v2   (var)   vs mary (const)  →  bind v2 → mary

σ = {v2 → mary}               ✓ success
```

**Consistency check**: if both arg positions try to bind the **same variable** to **different values**, unification fails. E.g., goal `p(X, X)` vs term `p(a, b)` would give `{X→a}` and `{X→b}` — conflict → $\bot$.

---

### 9.4 APPLY

**What it does**: takes the substitution $\sigma$ from UNIFY and applies it to all atoms that need updating — the rule body **and** the remaining goals.

$$\text{APPLY}(\text{atoms}, \sigma) = \text{atoms}[v \mapsto \sigma(v) \text{ for each } v \in \sigma]$$

Variables that appear in $\sigma$ get replaced by their bound value. Variables **not** in $\sigma$ stay as variables (they're free — they'll be resolved at a later depth).

**Example** (depth 0, applying σ = {v0→john, v1→sue} to r1's body):
```
Rule body:   parent(v0, v2), parent(v2, v1)
                              ↓ APPLY σ
Result:      parent(john, v2), parent(v2, sue)

v0 → john  ✓ replaced
v1 → sue   ✓ replaced
v2          not in σ, stays as variable (FREE)
```

**Example** (depth 2, applying σ = {v2→mary} to remaining goals):
```
Remaining:   [parent(v2, sue)]
                  ↓ APPLY σ
Result:      [parent(mary, sue)]

v2 → mary  ✓ replaced — the free variable is now ground!
```

This is the moment where free variables from earlier depths get resolved. When a fact match binds `v2→mary`, APPLY propagates that binding to **all** atoms in the state, including remaining goals that were waiting with `v2` unbound.

---

### 9.5 PACK

**What it does**: collects all the valid children (from both fact matches and rule matches) and flattens them into the output state tensor.

$$\text{PACK}: K_f \text{ fact children} + K_r \text{ rule children} \to S' \text{ output states}$$

Each child becomes a new proof state. The new state's goals are:

```
child goals = [substituted body atoms] + [substituted remaining goals]
```

**Example** (depth 1, goal = `parent(john, v2)`, K_r = 2):
```
Child from r2:  goals = [father(john, v2),  parent(v2, sue)]
Child from r3:  goals = [mother(john, v2),  parent(v2, sue)]
                         ↑ new body atom    ↑ inherited remaining

PACK → S' = 2 states
```

**Why scatter-based**: the grounder runs inside `torch.compile`, which requires fixed tensor shapes. Operations like `topk` or `nonzero` produce dynamic-length outputs and would break compilation. Instead, PACK uses `cumsum` on a validity mask to compute target indices, then `scatter` to place valid children into fixed-size output slots:

```
valid:      [True, False, True, True]
cumsum:     [1,    1,     2,    3]
target_idx: [0,    garbage, 1, 2]     → scatter into output[0], output[1], output[2]
```

---

### 9.6 PRUNE

**What it does**: looks at every goal atom in every state. If a goal is **ground** (no variables) **and** exists as a known fact, it's already satisfied — remove it.

$$\text{PRUNE}(\text{goals}, \mathcal{F}) = \text{remove } \{a \in \text{goals} \mid \text{ground}(a) \wedge a \in \mathcal{F}\}$$

**Example** (depth 0, after PACK):
```
State 0: goals = [father(john, mary)]
  father(john, mary) is ground? YES. In KB? YES → PRUNE it
  State 0: goals = []  → PROOF FOUND!

State 1: goals = [mother(john, mary)]
  mother(john, mary) is ground? YES. In KB? NO → keep
  State 1: goals = [mother(john, mary)]  → still open
```

**Why "ground" matters**: if a goal still has variables (e.g., `parent(v2, sue)`), we can't check it against the KB — we don't know what `v2` is yet. Prune skips non-ground atoms entirely.

**After pruning**: `compact_atoms` left-aligns the remaining goals (fills gaps from removed atoms). Then `collect_groundings` checks if any state has **all goals pruned** (= empty = proof found) and saves its grounding body.

---

### 9.7 Putting It All Together

One full BC step on the grandparent query at depth 0:

```
s = [grandparent(john, sue)]

  1. SELECT     →  g₁ = grandparent(john, sue),  remaining = []

  2. MATCH      →  facts: K_f = 0,  rules: K_r = 1 (r1)

  3. UNIFY      →  grandparent(john,sue) ~ grandparent(v0,v1)
                    σ = {v0→john, v1→sue}

  4. APPLY      →  body: parent(v0,v2), parent(v2,v1)
                         → parent(john,v2), parent(v2,sue)

  5. PACK       →  1 child → S' = 1
                    state: [parent(john,v2), parent(v2,sue)]

  6. PRUNE      →  parent(john,v2) has variable → skip
                    parent(v2,sue) has variable → skip
                    no pruning, no proof yet
```

Result: one state with two open goals, one free variable `v2`. The next BC step (depth 1) will SELECT `parent(john,v2)` and try to resolve it — eventually binding `v2→mary` via a fact match, which propagates to the remaining goal.

---

## 10. Compilation

The grounder is designed for `torch.compile(fullgraph=True, mode='reduce-overhead')`:

- **Fixed tensor shapes**: all intermediate tensors are padded to compile-time constants (K_MAX, G, S). No dynamic shapes.
- **No Python control flow**: no `.item()`, no data-dependent branching, no CPU-GPU sync in the forward path.
- **Scatter-based packing**: avoids `topk`/`nonzero` (which trigger dynamic shapes).
- **Clone at depth boundaries**: between depths, tensors are cloned to break the CUDA graph computation chain (otherwise the graph accumulates across all depths).
- **RL mode**: `step()` is compiled individually (not the multi-depth loop). The RL environment calls it from Python.
- **TS mode**: `step()` is compiled; the multi-depth loop runs in Python but each step is a single compiled graph call.

---

## 11. Summary

The Prolog backward chaining grounder is a **batched, compiled, 5-stage pipeline** that:

1. **Resolves** query atoms against facts and rules via MGU (no cartesian product, no domain enumeration)
2. **Packs** children into fixed-shape tensors via scatter (torch.compile compatible)
3. **Prunes** known facts via hash-based membership and **collects** completed proofs

The fundamental scalability tradeoff:
- **TS (all-depths)**: complete search up to depth D, but memory grows as K^D
- **RL (one-depth)**: policy-guided selective search, constant memory, any depth

The RL approach replaces the BFS frontier explosion with a learned policy that picks 1 state per step, making backward chaining tractable for large knowledge graphs and deep proofs.
