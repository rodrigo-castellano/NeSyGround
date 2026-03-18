# Prolog MGU Engine: Memory 

## Notation

| Symbol | Meaning |
|--------|---------|
| $B$ | Batch size (number of queries) |
| $A$ | Atoms per state ($= M_{\max} = \text{padding\_atoms} + \text{max\_rule\_body} \times 10$; typically 26) |
| $K_f$ | Max facts matching a `(pred, arg)` lookup — **data-dependent** |
| $K_r$ | Max rules sharing the same head predicate — **data-dependent** |
| $K_{\max}$ | Compile-time cap on successors per state (default 120). Output tensor is always `[B, K_max, A, 3]`, padded. |
| $D$ | Number of BFS steps (= proof depth) |
| $F_{\max}$ | Frontier cap (default $2 \times 10^6$ states) |

---

## 1. One MGU Step

**Pipeline for one goal atom:**

```
goal: parent(john, ?Y)          rule head: parent(X, Z) :- father(X, Z)
                                         │
             ┌──────────────────────────┘
             ▼
   1. Standardize Apart
      rename template vars → fresh runtime IDs
      parent(X=v100, Z=v101) :- father(v100, v101)
             │
             ▼
   2. unify_one_to_one(goal, renamed_head)
      pred match? ✓
      arg0: john (const) vs v100 (var) → bind v100 = john
      arg1: ?Y (var) vs v101 (var) → bind v101 = ?Y
      substitution σ = {v100 → john, v101 → ?Y}
             │
             ▼
   3. apply_substitutions(body + remaining, σ)
      father(v100, v101) → father(john, ?Y)
             │
             ▼
   successor state: [father(john, ?Y)]   ← still has variable ?Y
```

**Tensor shapes (logical → physical):**

| Tensor | Logical shape | Physical (compiled) | Contents |
|--------|--------------|---------------------|----------|
| Goal atom | `[B, 3]` | `[B, 3]` | `(pred, arg0, arg1)` — args can be constants or variables |
| Fact candidates | `[B, K_f]` | `[B, K_f_max]` | Fact indices matching `(pred, bound_arg)` |
| Fact-derived states | `[B, K_f, A, 3]` | `[B, K_f_max, A, 3]` | Remaining atoms with substitutions applied |
| Rule candidates | `[B, K_r]` | `[B, K_r_max]` | Rule indices matching goal predicate |
| Rule-derived states | `[B, K_r, A, 3]` | `[B, K_r_max, A, 3]` | Body atoms + remaining, with substitutions |
| Substitutions | `[B, K_f+K_r, 2, 2]` | `[B, K_max, 2, 2]` | `(from, to)` pairs — 2 pairs per unification |
| **Combined output** | `[B, K_f+K_r, A, 3]` | **`[B, K_max, A, 3]`** | Padded to fixed $K_{\max}$ (default 120) |

**Memory per step:** $O(B \cdot K_{\max} \cdot A)$ (physical allocation, always).

---

## 2. Detailed Sub-steps

Each state's **first atom** (the selected goal) is resolved against facts and rules simultaneously.

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| i. Detect terminal | Check if first atom is True/False or state is empty (= proof found). | $O(B \cdot A)$ | $O(B)$ |
| ii. Fact unification | Targeted lookup `(pred, arg)` → up to $K_f$ facts. Unify → substitute into remaining atoms. | $O(B \cdot K_f \cdot A)$ | $O(B \cdot K_f \cdot A)$ |
| iii. Rule unification | Segment lookup by head predicate → up to $K_r$ rules. Unify head → substitute into body. | $O(B \cdot K_r \cdot A)$ | $O(B \cdot K_r \cdot A)$ |
| iv. Pack + cap | Merge fact-derived and rule-derived → `[B, K_max, A, 3]`. Priority sort, cap at $K_{\max}$. | $O(B \cdot (K_f + K_r) \cdot A)$ | $O(B \cdot K_{\max} \cdot A)$ |
| v. Standardization | Offset-based: shift variable IDs to avoid collision with parent state. | $O(B \cdot K_{\max} \cdot A)$ | $O(B \cdot K_{\max} \cdot A)$ |
| vi. Fact pruning | Remove body atoms that are already ground facts. Hash-based membership test. | $O(B \cdot K_{\max} \cdot A \cdot \log F)$ | $O(B \cdot K_{\max} \cdot A)$ |

**$K_f$, $K_r$ vs $K_{\max}$:** Steps ii–iii do work proportional to the *actual* matches ($K_f$, $K_r$, data-dependent). Step iv pads the output to the compile-time constant $K_{\max}$ (default 120). On FB15k237, ~97% of `[B, K_max, A, 3]` is padding.

**Peak intermediate:** $B \times K_{\max} \times A \times 3 \times 8$ bytes.

---

## 3. Empirical $K_f$, $K_r$ per Dataset

$K_f$ = max facts per `(pred, arg)` lookup. $K_r$ = max rules per head predicate. Extracted from train.txt and rules.txt.

Peak per engine call = $B \times K_{\max} \times A \times 3 \times 8$ bytes ($B$=512, $A$=26).

| Dataset | Facts | Entities | $K_f$ | $K_r$ | $K_f + K_r$ | p99 | Peak (data-fitted) | Peak ($K_{\max}$=120) | Fits 120? |
|---------|-------|----------|-------|-------|-------------|-----|--------------------|-----------------------|-----------|
| **family** | 19,845 | 2,968 | 28 | 22 | **50** | 10 | 16 MB | 38 MB | Yes |
| **wn18rr** | 86,835 | 40,559 | 473 | 8 | **481** | 9 | 154 MB | 38 MB | No — hub truncated |
| **fb15k237** | 272,115 | 14,505 | 3,612 | 30 | **3,642** | 34 | 1,164 MB | 38 MB | No — hub truncated |
| **deep_chain** | 100,310 | 2,000 | 107 | 1 | **108** | 79 | 35 MB | 38 MB | Yes |

wn18rr and fb15k237 have extreme skew: worst-case $K_f$ is 50–100$\times$ the p99, driven by hub entities (WordNet root synsets, Freebase "Male"/"Official website"). At $K_{\max}$=120, 99% of queries fit; only hub-node queries are truncated.

---

## 4. Peak Memory for Static D-step BFS

**1-step (RL):** engine tensor `[B, K_max, A, 3]` only — no frontier accumulation.

**D-step without cap:** frontier after step $D$ = $B \cdot (K_f + K_r)^D$ states, each $A \times 3 \times 8 = 624$ bytes. With $B$=100:

| Dataset | $K_f + K_r$ | D=1 | D=2 | D=3 | D=5 |
|---------|-------------|-----|-----|-----|-----|
| **family** | 50 | 5K (3 MB) | 250K (149 MB) | 12.5M (**7.5 GB**) | 31.2B (**impossible**) |
| **wn18rr** | 481 | 48K (29 MB) | 23M (**13.8 GB**) | 11.1B (**impossible**) | — |
| **fb15k237** | 3,642 | 364K (217 MB) | 1.3B (**800 GB**) | — | — |
| **deep_chain** | 108 | 10.8K (6 MB) | 1.2M (700 MB) | 125M (**75 GB**) | 1.4T (**impossible**) |

Without a cap, frontier memory grows as $O(B \cdot (K_f+K_r)^D)$ — exponential in $D$. fb15k237 is already impossible at D=2; wn18rr at D=3; family and deep_chain at D=5.

**D-step with cap ($F_{\max} = 2 \times 10^6$):** frontier capped at $F_{\max} \times 624$ = **1.25 GB** (same for all datasets).

| Dataset | 1-step ($K_{\max}$=120) | D-step (capped) | D where cap kicks in |
|---------|------------------------|-----------------|----------------------|
| **family** | **38 MB** | 38 MB + **1.25 GB** | D=3 |
| **wn18rr** | **38 MB** | 38 MB + **1.25 GB** | D=2 |
| **fb15k237** | **38 MB** | 38 MB + **1.25 GB** | D=2 |
| **deep_chain** | **38 MB** | 38 MB + **1.25 GB** | D=3 |

The 1-step RL setting avoids the frontier entirely — constant 38 MB regardless of $D$ or dataset.

---

## 5. Peak Memory for Dynamic D-step BFS

In dynamic mode (no `torch.compile`), tensor shapes adapt to the actual data — no $K_{\max}$ padding, no truncation, no $F_{\max}$ cap. Since shapes vary per query, the **average** branching factor $\bar{K}_f + \bar{K}_r$ determines typical frontier size, not the worst-case max.

**Average branching factors** (from train.txt and rules.txt):

| Dataset | $\bar{K}_f$ | $\bar{K}_r$ | $\bar{K}_f + \bar{K}_r$ | max $K_f + K_r$ |
|---------|-------------|-------------|--------------------------|-----------------|
| **family** | 2.1 | 11.9 | **14.0** | 50 |
| **wn18rr** | 1.7 | 3.8 | **5.5** | 481 |
| **fb15k237** | 3.6 | 4.7 | **8.3** | 3,642 |
| **deep_chain** | 22.2 | 1.0 | **23.2** | 108 |

**D=1 (engine output only).** With $B$=100, $A$=26:

| Dataset | D=1 dynamic (avg) | D=1 dynamic (max) | D=1 compiled ($K_{\max}$=120) |
|---------|--------------------|--------------------|-------------------------------|
| **family** | **0.9 MB** | 3 MB | 7.5 MB |
| **wn18rr** | **0.3 MB** | 30 MB | 7.5 MB |
| **fb15k237** | **0.5 MB** | 227 MB | 7.5 MB |
| **deep_chain** | **1.4 MB** | 7 MB | 7.5 MB |

At D=1, dynamic with average branching is 5–25$\times$ cheaper than compiled. Even worst-case dynamic is cheaper for family and deep_chain. Only wn18rr/fb15k237 hub nodes exceed the compiled cost.

**D=2..5 (frontier growth, average branching).** Frontier at step $D$ = $B \cdot (\bar{K}_f + \bar{K}_r)^D$ states, each 624 bytes. No $F_{\max}$ cap.

| Dataset | $\bar{K}_f{+}\bar{K}_r$ | D=1 | D=2 | D=3 | D=4 | D=5 |
|---------|--------------------------|-----|-----|-----|-----|-----|
| **family** | 14.0 | 0.9 MB | 12 MB | 170 MB | 2.4 GB | **34 GB** |
| **wn18rr** | 5.5 | 0.3 MB | 1.9 MB | 10 MB | 57 MB | 310 MB |
| **fb15k237** | 8.3 | 0.5 MB | 4.3 MB | 36 MB | 300 MB | 2.5 GB |
| **deep_chain** | 23.2 | 1.4 MB | 34 MB | 780 MB | **18 GB** | **419 GB** |

With average branching, wn18rr fits in 310 MB even at D=5, and fb15k237 in 2.5 GB. Compare section 4's worst-case table where wn18rr explodes at D=2 (14 GB) and fb15k237 at D=2 (800 GB) — those used $\max(K_f + K_r)$ driven by hub nodes.

**Comparison with compiled (section 4):**

| | Compiled | Dynamic |
|-|----------|---------|
| Per-step engine tensor | Fixed $B \times K_{\max} \times A \times 24$ | Variable $B \times (\bar{K}_f{+}\bar{K}_r) \times A \times 24$ |
| Truncation | Yes — cap at $K_{\max}$, lossy | No — all successors kept |
| Frontier cap | $F_{\max} = 2 \times 10^6$ states (1.25 GB) | None — grows until OOM |
| CUDA graphs | Yes — fixed shapes enable caching | No — re-traced each call |

---

## 6. BCDynamicPrune with MGU

Why MGU in Prolog but cartesian product in keras-ns? The difference comes down to how each system handles free variables (body variables not in the rule head):

**MGU (Prolog):** doesn't enumerate free variables at all. After unifying the goal with the rule head, unbound variables stay as variables in the successor state. They get resolved at the next depth when fact unification binds them. This handles any variable placement naturally — there's no topological constraint. Per-step space: $O(K_f{+}K_r)$ successor states, each $A \times 24$ bytes — independent of the number of free variables. The cost of free variables is deferred to deeper steps, where each variable binding multiplies the frontier by $\bar{K}_f$.

**Cartesian product over domains (keras-ns):** for free variables remaining after anchoring on one body atom, iterates over `product(*[domain.constants])` — all entities in the domain, not just facts matching a specific predicate. Per-step space: $O(|D|^{n_{\text{free}}})$ groundings, each $M \times 24$ bytes. For family with $|D|$=2,968 and 1 free variable, that's up to 2,968 combinations per anchor per query.

BCDynamicPrune performs BFS with **independent subgoal expansion** + a
**PruneIncompleteProofs** fixed-point filter. Each goal atom is resolved
via `unify_one_to_one` against facts and rule heads, producing successor
states that may contain variables.

### 6.1 One Depth (D=1)

For each query $q = (p, s, o)$, a single BFS step:

1. **Match rules.** Find rules with head predicate $= p$. Produces up to $K_r$ candidates.
2. **Standardize apart.** Rename rule variables to fresh IDs.
3. **Unify** goal with each rule head → substitution $\sigma$.
4. **Apply $\sigma$** to body atoms → successor states (may contain variables).
5. **Match facts.** Targeted lookup `(p, s)` or `(p, o)` → up to $K_f$ facts. Unify → substitution resolves remaining atoms.
6. **Collect groundings.** Body atoms where all arguments are ground constants and are base facts → valid groundings. Body atoms with unresolved variables → become subgoals (but at D=1, no further expansion).

At D=1, only groundings whose body atoms are **all base facts** survive.
No subgoal expansion occurs.

**Per-query work:** $K_r$ rule unifications + $K_f$ fact unifications, each $O(A)$.
**Per-query output:** up to $K_f + K_r$ successor states, capped to $G$ groundings.

### 6.2 Example — Disconnected Graph (why pruning is needed)

Facts: `edge(a,b)`, `edge(c,d)` (disconnected components).
Rules: `reach(X,Y) :- edge(X,Y)`, `reach(X,Y) :- edge(X,Z), reach(Z,Y)`.
Query: `reach(a,d)?` — answer should be **0 groundings** (a and d are disconnected).

**Phase 1 (BFS) — collects all groundings unconditionally (torch-ns BCDynamicPrune):**

```
D=1: expand reach(a,d)
  Rule 1 (1-body): body = [edge(a,d)]
    edge(a,d) not a fact → COLLECTED (grounding stored, subgoal edge(a,d) queued)
    subgoal: edge(a,d) — no rules with head `edge` → dead end

  Rule 2 (2-body): body = [edge(a,Z), reach(Z,d)]
    Z=b → [edge(a,b), reach(b,d)]
    edge(a,b) ✓ fact,  reach(b,d) ✗ not a fact → COLLECTED
    subgoal: reach(b,d)

D=2: expand subgoals {edge(a,d), reach(b,d)}
  edge(a,d): no rules with head `edge` → nothing to expand
  reach(b,d):
    Rule 1: body = [edge(b,d)] → not a fact → COLLECTED, subgoal edge(b,d)
    Rule 2: body = [edge(b,Z), reach(Z,d)]
      no Z where edge(b,Z) exists → no candidates

Phase 1 result: 3 groundings collected
  g1: [edge(a,d)]              via r1 for reach(a,d)
  g2: [edge(a,b), reach(b,d)]  via r2 for reach(a,d)
  g3: [edge(b,d)]              via r1 for reach(b,d)
```

> **Note — keras-ns differs:** 1-body rules are also collected unconditionally (shortcut at line 70). For multi-body rules, non-fact body atoms are only kept if `predicate ∈ head_predicates` (line 141). In this example, `edge` is not a head predicate, so keras-ns would reject g1 and g3 at collection time. The final result is the same (0 groundings), but keras-ns filters earlier while torch-ns defers to Phase 2.

**Without pruning (torch-ns):** g1 and g2 are top-level groundings for the query.
Output = **2 groundings. WRONG** — reach(a,d) is not provable.

**Phase 2 (PruneIncompleteProofs fixed-point):**

```
proved = {edge(a,b), edge(c,d)}                ← base facts only
iteration 1:
  reach(a,d) via g1 needs edge(a,d) proved    → NO
  reach(a,d) via g2 needs reach(b,d) proved   → NO
  reach(b,d) via g3 needs edge(b,d) proved    → NO
iteration 2: no change → converged
```

**Phase 3 (filter):**

```
g1: edge(a,d) not proved      → REJECTED
g2: reach(b,d) not proved     → REJECTED

Result WITH pruning: 0 groundings ✓ CORRECT
```

The pruning phase removes groundings that depend on atoms that were
never transitively proved from facts. Without it, any grounding where
at least one body atom is a fact but others are unresolved would leak
through as a false positive.

### 6.3 Peak Memory for One Depth

At D=1, processing is **sequential per query** (Python loop over $B$ queries). Peak memory = one query's engine call + output buffer.

**Per-query engine call:** $(K_f + K_r) \times A \times 24$ bytes (successor states).

**Output buffer (pre-allocated):** $B \times G \times M \times 3 \times 8$ bytes.

With $B$=100, $G$=64, $M$=2, $A$=26:

| Dataset | $\bar{K}_f{+}\bar{K}_r$ | Per-query engine (avg) | Per-query engine (max) | Output buffer |
|---------|--------------------------|------------------------|------------------------|---------------|
| **family** | 14 | 8.7 KB | 31 KB | 0.6 MB |
| **wn18rr** | 5.5 | 3.4 KB | 300 KB | 0.6 MB |
| **fb15k237** | 8.3 | 5.2 KB | 2.3 MB | 0.6 MB |
| **deep_chain** | 23.2 | 14.5 KB | 67 KB | 0.6 MB |

Output buffer ($B \times G \times M \times 24$) = $100 \times 64 \times 2 \times 24$ = **0.3 MB** dominates over the per-query engine cost. Total peak ≈ **1 MB** for any dataset at D=1.

Compare with sections 4–5: compiled 1-step = 7.5 MB (fixed), dynamic BFS 1-step = 0.3–227 MB (varies). BCDynamicPrune is cheapest because it processes one query at a time — no batch-wide tensor allocation.

### 6.4 Peak Memory for D-depth

At D>1, each depth expands subgoals from the previous depth. The **goal stack** grows:

- D=1: $K_r$ rules applied to the original query → up to $K_r \times \bar{K}_f$ subgoals
- D=2: each subgoal branches again → up to $K_r \times \bar{K}_f$ new subgoals each
- D=$d$: goal stack ≤ $\min\!\left((K_r \times \bar{K}_f)^d,\; G_{\max}\right)$ (capped by `max_goals`)

Each subgoal is a triple $(p, s, o)$ = 24 bytes. The goal stack and `seen_goals` set are Python dicts, not tensors.

**Memory components at depth $D$:**

| Component | Size | Notes |
|-----------|------|-------|
| Goal stack | $\min((K_r \cdot \bar{K}_f)^D,\; G_{\max})$ triples | Capped at $G_{\max}$ (default 256) |
| `seen_goals` set | $\leq \sum_{d=0}^{D} (K_r \cdot \bar{K}_f)^d$ triples | Monotonically grows, uncapped |
| `all_groundings` list | $\leq \sum_{d=0}^{D} |\text{goals}_d| \cdot K_r \cdot \bar{K}_f$ entries | Each entry: body atoms + metadata |
| `proofs` dict | Same size as `all_groundings` | Dependency tracking for pruning |
| Output buffer | $B \times G \times M \times 24$ bytes | Fixed, independent of $D$ |

**Per-query memory at each depth** (Python objects, ~100 bytes per entry):

| Dataset | $K_r \cdot \bar{K}_f$ | D=1 | D=2 | D=3 | D=5 |
|---------|------------------------|-----|-----|-----|-----|
| **family** | $12 \times 2.1 = 25$ | 2.5 KB | 63 KB | 1.6 MB | **1.0 GB** |
| **wn18rr** | $4 \times 1.7 = 7$ | 0.7 KB | 5 KB | 34 KB | 1.6 MB |
| **fb15k237** | $5 \times 3.6 = 18$ | 1.8 KB | 32 KB | 583 KB | **190 MB** |
| **deep_chain** | $1 \times 22.2 = 22$ | 2.2 KB | 49 KB | 1.1 MB | **530 MB** |

These are **per-query**. Total = per-query × $B$ (sequential, so only one active at a time). The `max_goals` cap ($G_{\max}$=256) truncates the goal stack, bounding the per-depth expansion but not the cumulative `seen_goals` and `all_groundings`.

**BCDynamicPrune vs BFS (sections 4–5):**

| | BFS (sections 4–5) | BCDynamicPrune |
|-|---------------------|----------------|
| Parallelism | All $B$ queries in one tensor | Sequential (one query at a time) |
| Peak allocation | $B \times K_{\max} \times A \times 24$ (batch-wide) | $(K_r \cdot \bar{K}_f)^D \times 100$ bytes (one query) |
| Frontier growth | $B \cdot (K_f{+}K_r)^D$ states (exponential, batch-wide) | $(K_r \cdot \bar{K}_f)^D$ entries (exponential, one query) |
| Caps | $K_{\max}$ (compiled) or $F_{\max}$ (BFS) | $G_{\max}$ (goal stack) + $G$ (output groundings) |
| Pruning | None (BFS) or tabling (SLD) | PruneIncompleteProofs fixed-point |
| GPU utilization | High (batched tensor ops) | None (Python loops) |
