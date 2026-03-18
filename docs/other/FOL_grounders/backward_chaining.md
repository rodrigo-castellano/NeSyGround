# Backward Chaining Grounders

Per-query grounders for SBR training. All eight classes produce the same output
format `(B, tG, M, 3)` and are registered in `factory.py`.

## Table of Contents

1. [Overview](#1-overview)
2. [BCDynamicSLD — SLD Resolution](#2-bcdynamic--sld-resolution)
3. [BCDynamicPrune — PruneIncompleteProofs](#3-bcdynamicprune--pruneincompleteproofs)
4. [BCDynamicProvset — FC Provable Set](#4-bcdynamicprovset--fc-provable-set)
   - [4.1 Why BCDynamicProvset Uses Less Memory Than BCStaticProvset](#41-why-bcdynamicprovset-uses-less-memory-than-bcstatic)
5. [BCStaticProvset — K_max-Capped + Provable Set](#5-bcstaticprovset--kmax-capped--provable-set)
6. [BCStaticPrune — K_max-Capped + PruneIncompleteProofs](#6-bcstaticprune--kmax-capped--pruneincompleteproofs)
7. [BCStaticSLD — K_max-Capped SLD Resolution + Tabling](#7-bcstaticsld--kmax-capped-sld-resolution--tabling)
8. [BCPrologDynamic / BCPrologStatic — Prolog MGU Resolution](#8-bcprologdynamic--bcprologstatic--prolog-mgu-resolution)
9. [Comparison](#9-comparison)
10. [SBR Layer Stacking](#10-sbr-layer-stacking)
11. [Worked Example — Disconnected Graph](#11-worked-example--disconnected-graph)
12. [Output Format](#12-output-format)
13. [Shared Helpers](#13-shared-helpers)
14. [Benchmarks](#14-benchmarks)

---

## 1. Overview

| Class | Prefix | Sound | GPU compile | Key idea |
|-------|--------|-------|-------------|----------|
| `BCDynamicSLD` | `bcslddyn_` | Yes | No (eager) | SLD resolution with compound states |
| `BCDynamicPrune` | `bcprunedyn_` | Yes | No (eager) | Independent subgoals + PruneIncompleteProofs |
| `BCDynamicProvset` | `bcprovsetdyn_` | Yes | No (eager) | Single-step + FC provable-set check |
| `BCStaticProvset` | `bcprovset_` | Yes | Yes (fullgraph) | K_max-capped + FC provable-set check |
| `BCStaticPrune` | `bcprune_` | Yes | Yes (fullgraph) | K_max-capped + PruneIncompleteProofs |
| `BCStaticSLD` | `bcsld_` | Yes | Yes (fullgraph) | K_max-capped SLD resolution + tabling |
| `BCPrologDynamic` | `bcprologdyn_` | Yes | No (eager) | Prolog-style MGU resolution (verification) |
| `BCPrologStatic` | `bcprolog_` | Yes | Yes (fullgraph) | Prolog-style MGU resolution (compile-safe) |

Legacy alias: `backward_` → `ParametrizedBCGrounder`.

All dynamic variants use `@torch._dynamo.disable` (Python dicts for uncapped
fact lookups). All static variants use `fact_index.enumerate()` with K_max cap
and are fully compatible with `torch.compile(fullgraph=True, mode='reduce-overhead')`.

Usage: `--grounder_type bcslddyn_2` (SLD dynamic, depth 2), `--grounder_type bcprune_2` (static prune, depth 2), etc.

**Notation:**
- `E`: number of entities
- `P`: number of predicates
- `F`: base facts ⊆ P × E × E
- `R`: number of rules
- `R_q`: rules matching a given goal predicate (avg ≈ R/P)
- `m` (or `N_max`): max body atoms per rule
- `K`: branching factor (avg out-degree per `(pred, entity)` PS/PO lookup)
- `K_max`: capped fanout (BCStatic)
- `D`: depth (num_steps)
- `B`: batch size (queries)
- `G_max`: max groundings per query (tG parameter)
- `|I_D|`: provable set size (base facts + derived) at depth D
- `max_goals`: BFS goal queue capacity (default 256)

---

## 2. BCDynamic — SLD Resolution

**Soundness mechanism**: Compound state tracking.

Each proof state maintains ALL remaining goals as a set. When a rule fires on
goal `g`, `g` is replaced by the rule's body atoms — the other goals carry
forward unchanged. Only terminal states (all goals are base facts) produce
groundings.

### Algorithm

**Preprocessing (once at init):**

1. Build PS/PO dicts and `fact_set` frozenset from the training facts.
   - **Time:** $O(F)$ — one pass over all facts.
   - **Space:** $O(F)$ for PS/PO dicts + $O(F)$ for `fact_set` frozenset.

**Per query:**

1. **Seed — Initial rule application.** Apply all $R_q$ rules whose head predicate matches the query. For each rule, resolve head variables with query arguments and enumerate free variables via PS/PO dict lookups.

   For a 2-body rule `h(a,b) :- b1(a,Z), b2(Z,b)`: lookup `PS(b1, a)` → $K$ candidates for `Z`. For each candidate, resolve all body atoms and check if they are base facts (O(1) `fact_set` lookup per atom). Terminal candidates (all body atoms are facts) → collect grounding immediately. Non-terminal candidates → create compound state `(unresolved_goals: frozenset, rule_idx, body_atoms)` and enqueue for BFS.

   - **Time:** $O(R_q \cdot K \cdot m)$ — $R_q$ rules, $K$ candidates each, $m$ body atom checks per candidate.
   - **Space:** $O(R_q \cdot K)$ initial compound states, each storing a frozenset of $\leq m$ goals.

2. **BFS expansion** ($D$ iterations). For each compound state in the queue, pick the first unresolved goal and apply $R_q$ matching rules with $K$ candidates each.

   Let $S_d$ = active states at depth $d$, capped by `max_goals`. Each state spawns up to $R_q \cdot K$ new states. Deduplication via `seen_states` hash set (frozenset keys, O(1) lookup) prevents revisiting identical goal sets.

   - **Time at depth** $d$: $O(\min(\text{max\_goals},\, S_d) \cdot R_q \cdot K \cdot m)$ — each state expands one goal.
   - **Space at depth** $d$: $O(\min(\text{max\_goals},\, S_d))$ compound states. Each state's goal set grows by up to $m{-}1$ atoms per depth (rule body replaces one goal with $m$ new ones minus the resolved one).

3. **Collect.** Terminal states (all goals resolved to base facts) produce groundings, capped at $G_{\max}$ per query.
   - **Time:** $O(G)$ where $G$ = terminal states found ($\leq G_{\max}$).
   - **Space:** $O(G_{\max} \cdot m)$ for the output buffer.

> **Why `max_goals` dominates**
>
> Without the cap, BFS states would grow as $(R_q \cdot K)^D$ — exponential in depth. The `max_goals=256` default ensures linear-in-D cost at the expense of completeness: queries requiring more than 256 intermediate states may miss proofs. For kinship_family with $R_q \approx 12$ and $K \approx 5$, one depth step can produce $12 \times 5 = 60$ new states per existing state — hitting the 256 cap within 2 expansions of a high-branching goal.

### Why it's sound

A grounding is only emitted when the compound state reaches terminal
(empty goals). Every atom in the proof tree has been verified against
base facts through the same substitution chain. No cross-parent
variable contamination is possible.

### Complexity Summary

Steps 1–2 dominate. With `max_goals` capping the BFS queue at each depth:

| | |
|---|---|
| **Time per query** | $O(D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m)$ |
| **Total time (batch)** | $O(B \cdot D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m)$ |
| **Peak space per query** | $O(\text{max\_goals} \cdot m \cdot D)$ — BFS queue + growing goal sets |
| **Persistent space** | $O(F)$ — PS/PO dicts + `fact_set` frozenset |

> **What $K$ (fanout) means in backward chaining**
>
> $K$ is the number of results returned by a PS/PO dict lookup for a given `(pred, entity)` pair. When expanding a goal `b(a, Z)` where `a` is bound and `Z` is free, the kernel fetches all facts `b(a, ?)` — the count returned is $K$ for that `(b, a)` pair. Each result produces one candidate substitution for `Z`.
>
> For multi-body rules ($m > 1$), each free variable adds a factor of $K$: the first lookup gives $K$ candidates, the second gives up to $K$ for each, etc. Total candidates per rule: $O(K^{m-1})$. For 2-body rules ($m=2$) there is one free variable, so $K^1 = K$ candidates per rule.

### Example: kinship_family

**Dataset:** E=2,968, P=12, F=19,845, R=143 (all 2-body), D=2, `max_goals`=256.

$R_q \approx 143/12 \approx 12$ rules/predicate. $K_{\text{avg}} \approx 5$ (max base degree = 28, but most `(pred, entity)` pairs have degree 1–5).

| Step | Formula | Substitution | Cost |
|------|---------|-------------|------|
| Seed | $R_q \cdot K \cdot m$ | $12 \times 5 \times 2$ | 120 ops |
| BFS × 2 depths | $2 \times \text{max\_goals} \times R_q \times K \times m$ | $2 \times 256 \times 12 \times 5 \times 2$ | 61,440 ops |
| **Per query** | | | **~61.6K ops** |
| **Batch (B=1000)** | | | **~61.6M ops** |

**Peak space per query:** $256 \times 2 \times 2 \times 8\text{B} = 8\text{ KB}$ (states). **Persistent:** PS/PO dicts + `fact_set` ≈ 4 MB.

**Empirical (B=1000, D=2):** 8.658s, 7 MB ΔRSS, 27.78 groundings/query.

### Example: FB15k237

**Dataset:** E=14,505, P=237, F=272,115, R=199 (all 2-body), D=2, `max_goals`=256.

$R_q \approx 199/237 \approx 0.84$ rules/predicate (sparse — many predicates have no rules). But concentrated: popular predicates may have 10+ rules. $K$ varies widely: max degree = 3,612; most pairs have degree 1–10.

| Step | Formula | Cost (avg R_q=1, K=10) | Cost (dense R_q=10, K=100) |
|------|---------|------------------------|----------------------------|
| Seed | $R_q \cdot K \cdot m$ | 20 ops | 2,000 ops |
| BFS × 2 | $2 \times 256 \times R_q \times K \times m$ | 10,240 ops | 1,024,000 ops |
| **Per query** | | **~10K ops** | **~1M ops** |

For high-degree predicates ($K = 3{,}612$), a single expansion fills the `max_goals` queue immediately: $S_1 = \min(256, R_q \times 3{,}612) = 256$. The cap prevents blowup but truncates proofs.

**Peak space per query:** $256 \times 2 \times 2 \times 8\text{B} = 8\text{ KB}$ (same cap). **Persistent:** PS/PO dicts ≈ 30 MB (272K facts).

---

## 3. BCDynamicPrune — PruneIncompleteProofs

**Soundness mechanism**: Post-hoc fixed-point pruning.

Matches keras `ApproximateBackwardChainingGrounder` + `PruneIncompleteProofs`.

### Algorithm

**Preprocessing (once at init):**

Same as BCDynamic: build PS/PO dicts + `fact_set` from training facts.
   - **Time:** $O(F)$.
   - **Space:** $O(F)$.

**Per query (3 phases):**

1. **Phase 1 — Independent BFS.** Same BFS structure as BCDynamic but with **independent subgoals**: each body atom is expanded separately rather than tracking compound proof states. Collect ALL groundings and track proof dependencies: `atom → [list of body atom lists]`.

   Independent expansion is simpler (no frozenset creation for compound states) but produces spurious groundings from disconnected subgoals. The dependency graph records which atoms were used to prove which, enabling Phase 2 to filter.

   - **Time:** $O(D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m)$ — same BFS structure as BCDynamic.
   - **Space:** $O(\text{max\_goals})$ for the goal queue + $O(|\text{groundings}|)$ for all collected groundings + $O(|\text{proofs}|)$ for the dependency graph, where $|\text{proofs}|$ = number of distinct atoms with recorded proof dependencies.

2. **Phase 2 — PruneIncompleteProofs fixed-point.** Compute the set of transitively provable atoms from base facts using the dependency graph.

   ```
   proved = {all base facts}
   repeat (≤ D+1 times) until no change:
     for each atom with dependency lists:
       if ANY dependency list has ALL atoms proved:
         mark atom as proved
   ```

   Each iteration checks all atoms in the `proofs` dict against the `proved` set. The fixed-point converges in at most $D+1$ iterations (one per derivation depth).

   - **Time:** $O((D{+}1) \cdot |\text{proofs}| \cdot p \cdot m)$ where $p$ = average proof paths per atom.
   - **Space:** $O(|\text{proofs}|)$ for the `proved` set.

3. **Phase 3 — Filter.** Keep only top-level groundings where ALL body atoms are in the `proved` set.
   - **Time:** $O(|\text{groundings}| \cdot m)$ — one `proved` lookup per body atom.
   - **Space:** $O(G_{\max} \cdot m)$ output buffer (surviving groundings).

### Why it's sound

Phase 1 may produce spurious groundings (atoms proved via disconnected
subgoals). Phase 2 computes transitive provability from base facts.
Phase 3 filters: only groundings with transitively-provable body atoms survive.

### Complexity Summary

Phase 1 dominates time; Phase 2 adds the pruning overhead.

| | |
|---|---|
| **Time per query** | $O(D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m) + O((D{+}1) \cdot \lvert\text{proofs}\rvert \cdot p \cdot m)$ |
| **Total time (batch)** | $O\!\left(B \cdot \bigl[D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m + (D{+}1) \cdot \lvert\text{proofs}\rvert \cdot p \cdot m\bigr]\right)$ |
| **Peak space per query** | $O(\text{max\_goals} + \lvert\text{groundings}\rvert \cdot m + \lvert\text{proofs}\rvert)$ |
| **Persistent space** | $O(F)$ — PS/PO dicts + `fact_set` frozenset |

> **Why BCDynamicPrune is slower than BCDynamic**
>
> Despite independent subgoal expansion being simpler per state, BCDynamicPrune pays twice: Phase 1 explores broadly (no early rejection of inconsistent substitutions), producing more intermediate states, and Phase 2 iterates over the full proof dependency graph. In the kinship_family benchmark, BCDynamicPrune (10.53s) is 22% slower than BCDynamic (8.66s). The pruning overhead exceeds the savings from avoiding compound states.

### Example: kinship_family

**Dataset:** E=2,968, P=12, F=19,845, R=143, D=2.

| Phase | Formula | Cost |
|-------|---------|------|
| Phase 1 (BFS) | $D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m$ | $2 \times 256 \times 12 \times 5 \times 2 = 61{,}440$ |
| Phase 2 (Prune) | $(D{+}1) \cdot \lvert\text{proofs}\rvert \cdot p \cdot m$ | $3 \times {\sim}500 \times 2 \times 2 = 6{,}000$ |
| Phase 3 (Filter) | $\lvert\text{groundings}\rvert \cdot m$ | ${\sim}10{,}000 \times 2 = 20{,}000$ |
| **Per query** | | **~87K ops** |

**Peak space:** BFS queue $\approx$ 8 KB + groundings $\approx$ 80 KB + proofs $\approx$ 20 KB. **Persistent:** ≈ 4 MB.

**Empirical (B=1000, D=2):** 10.530s, 7 MB ΔRSS, 10.09 groundings/query.

### Example: FB15k237

**Dataset:** E=14,505, P=237, F=272,115, R=199, D=2.

Phase 1 cost is similar to BCDynamic (same BFS structure). Phase 2 adds a factor proportional to the number of discovered proof dependencies. For high-degree predicates, more groundings are found in Phase 1, making Phase 2 more expensive.

| Phase | Cost (avg predicate) | Cost (dense predicate) |
|-------|---------------------|------------------------|
| Phase 1 | ~10K ops/query | ~1M ops/query |
| Phase 2 | ~2K ops/query | ~200K ops/query |
| Phase 3 | ~1K ops/query | ~50K ops/query |

**Peak space:** same per-query cap as BCDynamic. **Persistent:** ≈ 30 MB.

---

## 4. BCDynamicProvset — FC Provable Set

**Soundness mechanism**: Forward chaining provable set as oracle.

### Algorithm

**Preprocessing (once at init):**

1. **Build PS/PO dicts + `fact_set`** from training facts.
   - **Time:** $O(F)$.
   - **Space:** $O(F)$.

2. **Compute provable set $I_D$** via forward chaining (`FCDynamic`) at depth $D$. Returns all atoms derivable from base facts within $D$ rule applications. Store as sorted hash tensor (for `searchsorted`) and Python frozenset (for O(1) lookup in eager code).
   - **Time:** $O(T_{\text{FC}})$ — see [forward chaining document](forward_chaining.md) for details. For kinship_family at $D=2$: ~0.2s. For FB15k237 at $D=2$: ~16s.
   - **Space:** $O(|I_D|)$ for sorted hash tensor + $O(|I_D|)$ for frozenset.

**Per query:**

1. **Apply rules (single step).** For each of $R_q$ rules whose head predicate matches the query, resolve head variables with query arguments and enumerate free variables via PS/PO dict lookups.

   For a 2-body rule `h(a,b) :- b1(a,Z), b2(Z,b)`: lookup `PS(b1, a)` → $K$ candidates for `Z`. Each candidate produces a fully-resolved set of $m$ body atoms.

   - **Time:** $O(R_q \cdot K^{m-1})$ — one lookup per free variable, total $m{-}1$ free variables for $m$-body rules. For $m=2$: $O(R_q \cdot K)$.
   - **Space:** $O(R_q \cdot K^{m-1} \cdot m)$ candidate body atoms (transient, freed after the query).

2. **Check body atoms.** For each candidate, check every body atom against `facts ∪ I_D`:
   - `fact_set` frozenset lookup: $O(1)$ per atom.
   - If not a base fact: `provable_set` frozenset lookup: $O(1)$ per atom.
   - Accept the grounding only if ALL $m$ body atoms are provable.

   - **Time:** $O(R_q \cdot K^{m-1} \cdot m)$ — $m$ checks per candidate.
   - **Space:** $O(1)$ — in-place checks, no new allocation per candidate.

3. **Collect.** Accepted groundings stored in output buffer, capped at $G_{\max}$.
   - **Time:** $O(G)$ where $G$ = accepted groundings ($\leq G_{\max}$).
   - **Space:** $O(G_{\max} \cdot m)$ output buffer.

> **Why single-step suffices**
>
> The provable set $I_D$ already encodes all atoms derivable within $D$ rule applications (FC is sound and complete at depth $D$). Instead of recursively expanding subgoals (as BCDynamic does), BCDynamicProvset applies rules once and verifies body atoms against the precomputed oracle. This eliminates the BFS loop entirely — $O(R_q \cdot K \cdot m)$ per query versus $O(D \cdot \text{max\_goals} \cdot R_q \cdot K \cdot m)$.

### Why it's sound

If atom $\in I_D$, it IS provable (FC is sound and complete at depth D).
Each body atom in a grounding has fully-bound variables (Z is already resolved
before checking), so there's no cross-parent contamination. The provable set
is an exact characterization at depth D.

### Complexity Summary

| | |
|---|---|
| **Init time** | $O(T_{\text{FC}})$ — forward chaining at depth $D$ (amortized over all training) |
| **Init space** | $O(F + \lvert I_D\rvert)$ — PS/PO dicts + provable set |
| **Time per query** | $O(R_q \cdot K^{m-1} \cdot m)$ |
| **Total time (batch)** | $O(B \cdot R_q \cdot K^{m-1} \cdot m)$ |
| **Peak space per query** | $O(R_q \cdot K^{m-1} \cdot m)$ — transient candidate buffer |
| **Persistent space** | $O(F + \lvert I_D\rvert)$ |

### Example: kinship_family

**Dataset:** E=2,968, P=12, F=19,845, R=143 (all 2-body), D=2.

**Provable set init:** FC at $D=2$ derives 47,052 atoms in ~0.2s. $|I_D| = 19{,}845 + 47{,}052 = 66{,}897$. Hash buffer: $66{,}897 \times 8\text{B} = 535\text{ KB}$.

| Step | Formula | Substitution | Cost |
|------|---------|-------------|------|
| Apply rules | $R_q \cdot K$ | $12 \times 5$ | 60 candidates |
| Check body | $R_q \cdot K \cdot m$ | $12 \times 5 \times 2$ | 120 lookups |
| **Per query** | | | **~180 ops** |
| **Batch (B=1000)** | | | **~180K ops** |

Compare to BCDynamic's 61.6K ops/query — **BCDynamicProvset is ~340× fewer operations per query**, trading BFS depth for a precomputed oracle.

**Peak space per query:** $12 \times 5 \times 2 \times 24\text{B} = 2.9\text{ KB}$ (transient). **Persistent:** dicts + provable set ≈ 5 MB.

**Empirical (B=1000, D=2):** 1.143s, 6 MB ΔRSS, 10.41 groundings/query.

### Example: FB15k237

**Dataset:** E=14,505, P=237, F=272,115, R=199 (all 2-body), D=2.

**Provable set init:** FC at $D=2$ derives 8,767,095 atoms in ~16s. $|I_D| \approx 9{,}039{,}210$. Hash buffer: $9{,}039{,}210 \times 8\text{B} = 72\text{ MB}$.

| Step | Formula | Cost (avg, K=10) | Cost (dense, K=100) |
|------|---------|-----------------|---------------------|
| Apply rules | $R_q \cdot K$ | $1 \times 10 = 10$ | $10 \times 100 = 1{,}000$ |
| Check body | $R_q \cdot K \cdot m$ | $1 \times 10 \times 2 = 20$ | $10 \times 100 \times 2 = 2{,}000$ |
| **Per query** | | **~30 ops** | **~3K ops** |

**Persistent:** dicts ≈ 30 MB + provable set ≈ 72 MB = **~102 MB**. The FC init is the bottleneck — 16s for FB15k237 — but is paid once and amortized over all training epochs.

### 4.1 Why BCDynamicProvset Uses Less Memory Than BCStatic

BCDynamicProvset and BCStatic share the same soundness oracle (FC provable set) but differ fundamentally in how they store and access facts during query processing.

**BCDynamicProvset — sparse, per-query:**

```python
ps_dict[(pred, subj)] → List[obj]     # only existing facts, uncapped
po_dict[(pred, obj)]  → List[subj]    # only existing facts, uncapped
fact_set = frozenset(all_facts)        # O(1) membership
```

Memory scales with $F$ (existing facts only). Empty `(pred, entity)` pairs consume zero space.

**BCStatic — dense, pre-allocated:**

```python
fact_index = Tensor[P, E, K_max]       # every (pred, entity) slot reserved
fact_mask  = Tensor[P, E, K_max]       # K_max slots per slot, mostly empty
```

Memory scales with $P \times E \times K_{\max}$ regardless of how many facts actually exist. The density gap is:

$$\text{Density} = \frac{F}{P \times E \times K_{\max}}$$

**The sparse/dense gap:**

| Property | BCDynamicProvset | BCStatic |
|---|---|---|
| Fact storage | Python dicts (sparse) | Pre-allocated tensors (dense) |
| Iterates over | existing facts only | all $P \times E \times K_{\max}$ slots |
| Fact memory | $O(F)$ | $O(P \cdot E \cdot K_{\max})$ |
| Sparsity | fully exploited | ignored |
| Per-query intermediates | $O(R_q \cdot K \cdot m)$, freed after each query | $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m)$ tensors |
| Execution | sequential Python loops | single vectorized kernel |

| Dataset | $F$ | $P \cdot E \cdot K_{\max}$ | Density | Provset ΔRSS | Static ΔRSS | Ratio |
|---------|-----|---------------------------|---------|--------------|-------------|-------|
| kinship_family | 19,845 | $12 \times 2{,}968 \times 56 = 2.0\text{M}$ | 1.0% | **6 MB** | **98 MB** | 16× |
| FB15k237 | 272,115 | $237 \times 14{,}505 \times 64 = 220\text{M}$ | 0.12% | **est. ~102 MB** | **est. ~1.8 GB** | ~18× |

For kinship_family, BCStatic allocates 2.0M tensor slots (16 MB for the fact index alone) where only 1% are non-empty. BCDynamicProvset stores only the 19,845 existing facts in Python dicts — paying for the actual data, not the empty space.

For FB15k237, the gap widens: BCStatic's fact index would require $220\text{M} \times 8\text{B} = 1.76\text{ GB}$ for the tensor alone, where only 0.12% of slots contain data. BCDynamicProvset stores 272K facts in dicts plus the 72 MB provable-set buffer — an order of magnitude less.

> **When BCStatic wins despite higher memory**
>
> BCStatic pays the memory premium to enable `torch.compile(fullgraph=True, mode='reduce-overhead')`. On GPU with CUDA graph caching, the compiled kernel avoids Python overhead entirely — each forward pass is a single kernel launch. For production training with thousands of batches, the constant factor advantage of compiled tensor operations outweighs the per-query efficiency of Python dict lookups, even though BCDynamicProvset does less total work.

---

## 5. BCStaticProvset — K_max-Capped + Provable Set

**Soundness mechanism**: FC provable set via `torch.searchsorted` (fullgraph-safe).

Same logical approach as BCDynamicProvset but fully vectorized with K_max-capped tensor lookups, enabling `torch.compile(fullgraph=True, mode='reduce-overhead')`.

### Algorithm

**Preprocessing (once at init):**

1. **Compute provable set $I_D$** via forward chaining (same as BCDynamicProvset). Store as sorted hash tensor for `searchsorted`.
   - **Time:** $O(T_{\text{FC}})$.
   - **Space:** $O(|I_D|)$ sorted hash buffer.

2. **Build K_max-capped tensor fact index.** Sort facts by `(pred, entity)`, truncate to $K_{\max}$ per pair, store in dense tensors `[P, E, K_max]`.
   - **Time:** $O(F \cdot \log F)$ for sorting.
   - **Space:** $O(P \cdot E \cdot K_{\max})$ for the tensor buffers (long + bool mask).

3. **Build rule buffers.** Precompute per-predicate rule clustering, metadata tensors (`_pred_rule_indices`, `_body_preds`, `_body_args`, etc.).
   - **Time:** $O(R \cdot m)$.
   - **Space:** $O(R \cdot m)$ tensor buffers + $O(P \cdot R_{\text{eff}})$ clustering index, where $R_{\text{eff}}$ = max rules per predicate.

**Per depth $d$ ($d = 0 \ldots D{-}1$), vectorized over batch $B$:**

Let $G_{\text{proc}} = \min(\text{max\_G},\, 32)$ = goals processed per depth.

1. **Rule clustering.** Map goal predicates to rule indices via precomputed `pred → rules` table.
   - **Time:** $O(B \cdot G_{\text{proc}})$ — gather from clustering index.
   - **Space:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}})$ rule indices.

2. **Enumerate candidates.** $K_{\max}$-capped lookup for each `(rule, goal)` pair via `fact_index.enumerate()`.
   - **Time:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max})$ — one capped gather per combination.
   - **Space:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max})$ candidate entities + masks.

3. **Resolve body atoms.** Vectorized gather to fill body atom triples `(pred, subj, obj)` for all $m$ body atoms of each candidate.
   - **Time:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m)$.
   - **Space:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m \cdot 3)$ — the body atom tensor (long). This is the peak intermediate allocation.

4. **Existence + provable check.** For each body atom: check base-fact existence via hash lookup + check provable-set membership via `torch.searchsorted`.
   - Hash computation: $\text{hash} = p \cdot E^2 + s \cdot E + o$.
   - `searchsorted` into the sorted provable hash buffer: $O(\log |I_D|)$ per atom.
   - Fully proved = `(exists | in_provable)` for ALL active body atoms (`.all(dim=-1)`).
   - **Time:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m \cdot \log |I_D|)$ — dominated by `searchsorted`.
   - **Space:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m)$ boolean masks.

5. **Collect + dedup new goals.** Accept fully-proved candidates as groundings (cap at $G_{\max}$ via `topk`). Extract unproved body atoms as new goals for the next depth, deduplicate via hash + sort + adjacent-diff.
   - **Time:** $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m)$ for hashing + sorting.
   - **Space:** $O(B \cdot \text{max\_G})$ new goals for next depth.

### K_max defaults

K_max differs per grounder variant:

| Dataset | Provset/Prune | SLD | Prolog |
|---------|--------------|-----|--------|
| kinship_family | 56 | 16 | 64 |
| kinship | 48 | 16 | 64 |
| wn18rr | 128 | 32 | 64 |
| FB15k237 | 64 | 16 | 64 |

### Complexity Summary

Step 4 (`searchsorted`) dominates time per depth. Step 3 (body atom tensor) dominates space.

| | |
|---|---|
| **Init time** | $O(T_{\text{FC}} + F \cdot \log F + R \cdot m)$ |
| **Init space** | $O(P \cdot E \cdot K_{\max} + \lvert I_D\rvert + R \cdot m)$ |
| **Time per depth** | $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m \cdot \log \lvert I_D\rvert)$ |
| **Total time** | $O(D \cdot B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m \cdot \log \lvert I_D\rvert)$ |
| **Peak space (intermediates)** | $O(B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max} \cdot m)$ — body atom + mask tensors |
| **Persistent space** | $O(P \cdot E \cdot K_{\max} + \lvert I_D\rvert + B \cdot G_{\max} \cdot m)$ |

> **CUDA Graph Notes**
>
> - No `.item()` calls inside the forward pass.
> - No data-dependent branching — rule clustering and candidate selection use pre-computed index tensors and `torch.where`, not Python `if` statements.
> - The `for depth in range(num_steps)` Python loop is the only dynamic control flow.
> - Buffer updates and provable-set init run **outside** the compiled region.

### Example: kinship_family

**Dataset:** E=2,968, P=12, F=19,845, R=143, D=2, $K_{\max}=56$, $R_{\text{eff}} \approx 12$.

**Init:** FC at $D=2$: ~0.2s, $|I_D| = 66{,}897$. Fact index: $12 \times 2{,}968 \times 56 \times 8\text{B} = 16\text{ MB}$.

| Step | Formula | Substitution | Cost |
|------|---------|-------------|------|
| Rule clustering | $B \cdot G_{\text{proc}}$ | $1000 \times 32$ | 32K |
| Enumerate | $B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max}$ | $1000 \times 32 \times 12 \times 56$ | 21.5M |
| Resolve body | $\times\, m$ | $\times\, 2$ | 43.0M |
| Check (searchsorted) | $\times\, \log \lvert I_D\rvert$ | $\times\, 17$ | 730M |
| **Per depth** | | | **~730M ops** |
| **Total (D=2)** | | | **~1.46B ops** |

**Peak intermediate space:** $1000 \times 32 \times 12 \times 56 \times 2 \times 3 \times 8\text{B} = 103\text{ MB}$ (body atom tensor). **Persistent:** fact index 16 MB + provable hashes 535 KB + output buffer ~3 MB ≈ 20 MB.

**Empirical (B=1000, D=2):** 4.372s, 98 MB ΔRSS, 10.45 groundings/query.

> **Why BCStatic is slower than BCDynamicProvset on CPU but wins on GPU**
>
> BCStatic performs $K_{\max}$-capped enumeration for ALL `(goal, rule)` pairs simultaneously — 21.5M candidates per depth versus BCDynamicProvset's ~60 candidates per query ($\times$ 1000 = 60K total). The overhead comes from processing empty $K_{\max}$ slots: with $K_{\max}=56$ and avg degree ≈ 5, ~91% of candidate slots are padded zeros. On CPU this wasted computation dominates. On GPU, the vectorized kernel with CUDA graph caching eliminates Python loop overhead, making BCStatic faster in production training despite the higher operation count.

### Example: FB15k237

**Dataset:** E=14,505, P=237, F=272,115, R=199, D=2, $K_{\max}=64$, $R_{\text{eff}} \approx 5$.

**Init:** FC at $D=2$: ~16s, $|I_D| \approx 9{,}039{,}210$. Fact index: $237 \times 14{,}505 \times 64 \times 8\text{B} = 1.76\text{ GB}$.

| Step | Formula | Substitution | Cost |
|------|---------|-------------|------|
| Enumerate | $B \cdot G_{\text{proc}} \cdot R_{\text{eff}} \cdot K_{\max}$ | $1000 \times 32 \times 5 \times 64$ | 10.2M |
| Resolve body | $\times\, m$ | $\times\, 2$ | 20.5M |
| Check (searchsorted) | $\times\, \log \lvert I_D\rvert$ | $\times\, 23$ | 471M |
| **Per depth** | | | **~471M ops** |
| **Total (D=2)** | | | **~942M ops** |

**Peak intermediate space:** $1000 \times 32 \times 5 \times 64 \times 2 \times 3 \times 8\text{B} = 49\text{ MB}$.

**Persistent:** fact index 1.76 GB + provable hashes 72 MB + output buffer ~3 MB ≈ **1.84 GB**. The fact index dominates — $220\text{M}$ pre-allocated slots where only $0.12\%$ are non-empty.

---

## 6. BCStaticPrune — K_max-Capped + PruneIncompleteProofs

**Soundness mechanism**: PruneIncompleteProofs fixed-point filter (no FC).

Same BFS structure as BCStaticProvset but without the FC provable-set computation
at init. Instead, collects ALL valid candidate groundings across depths into
pre-allocated buffers, then runs a PruneIncompleteProofs fixed-point to keep only
transitively-provable groundings.

### Key differences from BCStaticProvset

- **No FC at init** — no `run_forward_chaining()`, no `_provable_hashes` buffer → faster init
- **BFS Phase**: Stores ALL valid candidates (not just proved) into depth-indexed slots
- **Prune Phase**: `num_steps + 1` iterations of `searchsorted`-based fixed-point:
  initially proved = all-fact body atoms; each iteration extends via head hashes of
  newly proved groundings
- **Output**: depth-0 proved groundings selected via `topk`

Uses the same `_DATASET_K_MAX` table as BCStaticProvset.

Compatible with `torch.compile(fullgraph=True, mode='reduce-overhead')`.

---

## 7. BCStaticSLD — K_max-Capped SLD Resolution + Tabling

**Soundness mechanism**: Compound state tracking (same as BCDynamicSLD).

SLD resolution with fixed-size tensor state tracking. Each proof state maintains
ALL remaining goals. Only terminal states (no remaining goals) produce groundings.
Most restrictive soundness — fewest groundings per query.

### Key features

- **Compound states**: `(B, max_states, max_gps, 3)` tensor with per-state goal mask
- **max_gps** = `1 + (M-1) * num_steps` (tight upper bound on goals per state)
- **max_states** = 64 (cap on compound states per batch element)
- **Tabling**: Direct-mapped hash table (`T=512`) caches proved subgoals.
  When a goal has any valid expansion with all body-fact atoms, its hash is inserted.
  Goals found in the table are removed from compound states, potentially making
  them terminal earlier. +4% overhead, significant completeness benefit.
- **XOR compound hash dedup**: Order-independent state comparison via per-goal-atom
  XOR hash, then sort + adjacent_diff
- **Smaller K_max defaults**: `{kinship_family: 16, wn18rr: 32, FB15k237: 16}`
  to bound state explosion

Compatible with `torch.compile(fullgraph=True, mode='reduce-overhead')`.

---

## 8. BCPrologDynamic / BCPrologStatic — Prolog MGU Resolution

**Soundness mechanism**: Most General Unifier (MGU) — true Prolog-style unification.

Source: `ns_lib/grounding/backward_chaining_prolog.py`.

Uses variable-carrying proof states resolved through substitution chains, unlike
the cascaded-enumeration approach of BCStaticSLD. States carry unresolved variables
that are progressively grounded through MGU unification against facts and rule heads.

### Key features

- **BCPrologDynamic**: Eager-mode Python BFS with MGU unification. For verification/debugging.
- **BCPrologStatic**: Compile-safe version for `torch.compile(fullgraph=True)`.
- **K_max defaults**: 64 for all datasets (`_PROLOG_DATASET_K_MAX`).
- **max_states**: `_PROLOG_MAX_STATES = 128`.
- **max_fact_candidates**: `_PROLOG_MAX_FACT_CANDIDATES = 64`.
- **Unification source**: `ns_lib/grounding/unification.py` (ported from Batched_env-swarm).

See [backward_chaining_sld.md](backward_chaining_sld.md) for a detailed comparison
of MGU vs cascaded enumeration resolution strategies.

---

## 9. Comparison

| Property | BCDynamicSLD | BCDynamicPrune | BCDynamicProvset | BCStaticProvset | BCStaticPrune | BCStaticSLD | BCPrologDynamic | BCPrologStatic |
|----------|-------------|----------------|------------------|-----------------|---------------|-------------|-----------------|----------------|
| Sound | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Complete at depth D | Yes (up to max_goals) | Yes (up to pruning) | No (single step) | No (K_max cap) | No (K_max cap) | No (K_max + max_states cap) | Yes (up to max_goals) | No (K_max cap) |
| GPU compile | No | No | No | Yes (fullgraph) | Yes (fullgraph) | Yes (fullgraph) | No | Yes (fullgraph) |
| Init cost | Low | Low | Medium (FC) | Medium (FC) | Low | Low | Low | Low |
| Forward speed (CPU) | Slow | Slow | Fast | Medium | Medium | Fast | Slow | Medium |
| Forward speed (GPU) | N/A | N/A | N/A | Fast | Fast | Fastest | N/A | Fast |
| Memory | ~7 MB | ~7 MB | ~6 MB | ~260 MB | ~240 MB | ~240 MB | ~7 MB | ~240 MB |

**When to use what**:
- **BCDynamicSLD**: Reference implementation, best for correctness validation
- **BCDynamicPrune**: Keras parity testing
- **BCDynamicProvset**: Fast sound grounder when GPU compile not needed
- **BCStaticProvset**: Production training with torch.compile (most groundings)
- **BCStaticPrune**: Production training without FC init overhead (stricter soundness)
- **BCStaticSLD**: Production training with strongest soundness + tabling
- **BCPrologDynamic**: Prolog MGU verification and debugging
- **BCPrologStatic**: Production training with true unification (supports non-ground queries)

---

## 10. SBR Layer Stacking

`reasoner_depth > 1` chains SBR layers: output of layer i becomes input to
layer i+1, propagating derived-atom confidence through proof chains.

### How it works

```
1. KGE scores all atoms → all_scores
2. Grounder finds body atoms (once)
3. Hash body atoms to atom-table indices (once, via searchsorted)
4. Layer loop:
   for layer in range(reasoner_depth):
     body_scores = current_scores[body_atom_indices]
     concept = current_scores[query_indices]
     task = sbr_layer(concept, body_scores, masks, resnet=(last layer))
     current_scores[query_indices] = max(current_scores[query_indices], task)
```

### Example

```
edge(a,b) score = 0.8, edge(b,c) score = 0.9
reach(b,c) KGE score = 0.3 (raw, before reasoning)
reach(a,c) KGE score = 0.3

Depth=1:
  reach(b,c) = max(0.3, min(0.9)) = 0.9
  reach(a,c) = max(0.3, min(0.8, 0.3)) = 0.3  ← raw score for reach(b,c)

Depth=2:
  Layer 1: reach(b,c) updated to 0.9
  Layer 2: reach(a,c) = max(0.3, min(0.8, 0.9)) = 0.8  ← propagated!
```

### Parameters

- `reasoner_depth`: Number of SBR layers (default=1 for backward compat)
- `reasoner_single_model`: If True, share weights across all layers (matches
  keras `reasoner_single_model` option)

### Limitation

Stacking only propagates scores for atoms in the current batch. For full
keras equivalence, ALL provable atoms must be in the batch.

---

## 11. Worked Example — Disconnected Graph

Facts: `edge(a,b)`, `edge(c,d)` (disconnected).
Rules: `reach(X,Y) :- edge(X,Z), reach(Z,Y)`, `reach(X,Y) :- edge(X,Y)`.
Query: `reach(a,d)?`

### BCDynamic (SLD)

```
Initial: apply rules to reach(a,d)
  Rule 1: reach(a,d) :- edge(a,Z), reach(Z,d)
    Z=b → goals = {reach(b,d)}    ← reach(b,d) not a fact
    No candidates for Z where edge(a,Z) and reach(Z,d) resolves
  Rule 2: reach(a,d) :- edge(a,d)
    edge(a,d) not a fact → skip

Depth 2: expand reach(b,d)
  Rule 1: reach(b,d) :- edge(b,Z), reach(Z,d)
    No Z: edge(b,Z) only matches edge(b,?) — no such fact with second arg usable
  Rule 2: reach(b,d) :- edge(b,d)
    edge(b,d) not a fact → skip

Result: 0 groundings ✓ (correct — a and d are disconnected)
```

### BCDynamicPrune

```
Phase 1 BFS: expands reach(b,d) as subgoal, no groundings found with all-fact body
Phase 2: reach(b,d) not proved → not in proved set
Phase 3: no top-level groundings survive
Result: 0 groundings ✓
```

### BCDynamicProvset

```
Init: FC computes I_D = {reach(a,b), reach(c,d)} (only connected pairs)
Forward: reach(a,d) → rule 1 body = [edge(a,b), reach(b,d)]
  reach(b,d) not in I_D → rejected
Result: 0 groundings ✓
```

### BCStatic

```
Same as BCDynamicProvset but with tensor ops + K_max cap.
reach(b,d) not in provable_hashes → rejected
Result: 0 groundings ✓
```

---

## 12. Output Format

All grounders return:
```python
collected_body:  (B, tG, M, 3)  # body atoms [pred, subj, obj]
collected_mask:  (B, tG)         # valid groundings
collected_count: (B,)            # count per query
collected_ridx:  (B, tG)         # rule index per grounding
```

Where:
- `B` = batch size (number of queries)
- `tG` = `effective_total_G` (max groundings per query)
- `M` = `max_body_atoms` (max body atoms across all rules)

---

## 13. Shared Helpers

- `_bc_init()`: Registers buffers (rule tensors, clustering, metadata) on the module
- `_build_python_dicts()`: Builds PS/PO dicts + fact_set from TensorFactIndex
- `_check_in_provable()`: Sorted-hash searchsorted check (fullgraph-safe)
- `_tensorize_rules()`: Converts CompiledRules to tensor buffers
- `_build_rule_clustering()`: Per-predicate rule mapping for vectorized dispatch

---

## 14. Benchmarks

Benchmark script: `analysis/kg_complexity.py --bc-benchmark`

```bash
cd torch-ns

# All 6 variants on kinship_family test set (1000 queries, D=2, CUDA)
python -u analysis/kg_complexity.py \
    -d kinship_family --bc-benchmark --query-split test \
    --bc-num-steps 2 --device cuda --max-queries 1000
```

### kinship_family (B=1000 test queries, D=2, CUDA RTX 3090)

**Dataset:** 19,845 train facts, E=2,968 entities, P=12 predicates, 143 rules (141 two-body, 2 one-body).
max_total_groundings=2048, max_groundings_per_query=32. Times are avg of 3 forward passes.

| Variant | Time (s) | Provable (/1000) | Groundings | Peak VRAM (MB) |
|---------|----------|------------------|------------|----------------|
| BCDynamicSLD | 21.929 | 977 | 9,474 | 252 |
| BCDynamicPrune | 29.582 | 977 | 10,293 | 252 |
| BCDynamicProvset | 3.438 | 979 | 10,646 | 252 |
| **BCStaticProvset** | **0.166** | **979** | **10,680** | 13,157 |
| **BCStaticPrune** | **0.375** | **977** | **10,201** | 18,487 |
| **BCStaticSLD** | **0.067** | **977** | **9,166** | 4,947 |

#### Static variants — full test set (B=5626, D=2, CUDA RTX 3090)

11 chunks of 512 queries. Times are avg of 3 full passes over all chunks.

| Variant | Init (s) | Forward (s) | Provable (/5626) | Groundings | Peak VRAM (MB) |
|---------|----------|-------------|------------------|------------|----------------|
| **BCStaticSLD** | 0.252 | **0.386** | 5,488 | 50,260 | 2,498 |
| **BCStaticProvset** | 0.259 | 0.939 | **5,515** | 58,123 | 6,676 |
| **BCStaticPrune** | 0.021 | 2.132 | 5,487 | 55,514 | 9,429 |

#### Static variants — full test set (B=5626, D=5, CUDA RTX 3090)

Same setup, deeper search. 11 chunks of 512 queries.

| Variant | Init (s) | Forward (s) | Provable (/5626) | Groundings | Peak VRAM (MB) |
|---------|----------|-------------|------------------|------------|----------------|
| **BCStaticSLD** | 0.252 | **1.763** | 5,493 | 50,587 | 4,814 |
| **BCStaticProvset** | 0.523 | 2.356 | **5,521** | 58,151 | 6,678 |
| **BCStaticPrune** | 0.021 | 8.372 | 5,492 | 56,102 | 19,237 |

#### Key observations

1. **Static variants are 80-440× faster** than dynamic counterparts on GPU. BCStaticSLD is fastest: 67 ms for 1000 queries (D=2), 386 ms for the full 5626-query test set (D=2).
2. **BCStaticSLD scales best** — at D=2: 2.4× faster than BCStaticProvset and 5.5× faster than BCStaticPrune. At D=5: 1.3× faster than Provset and 4.7× faster than Prune.
3. **Provable queries plateau at D=2.** D=2→D=5 adds only 5–6 provable queries per variant (e.g. SLD: 5,488→5,493). BCStaticProvset consistently finds ~28 more than SLD/Prune thanks to the exhaustive FC oracle.
4. **Groundings plateau at D=2.** SLD: 50,260→50,587 (+0.6%), Provset: 58,123→58,151 (+0.05%). Deeper search yields negligible new groundings for kinship_family.
5. **Time scales linearly with depth.** SLD: 0.39s→1.76s (4.5×), Provset: 0.94s→2.36s (2.5×), Prune: 2.13s→8.37s (3.9×). D=2 is the sweet spot for this dataset.
6. **VRAM scales with depth for Prune** (9.4→19.2 GB, D=2→D=5) due to extra pruning iteration buffers. SLD and Provset grow moderately (2.5→4.8 GB and 6.7→6.7 GB).
7. **Grounding counts agree** across dynamic/static pairs after output dedup was added to BCStaticSLD. At B=1000 D=2: DynSLD=9,474 vs StaticSLD=9,166 (small gap from K_max=16 capping).
8. **All 6 variants are sound** — no false groundings. The 3 static variants compile with `torch.compile(fullgraph=True, mode='reduce-overhead')`.

### wn18rr (B=2924 test queries, D=2, CUDA RTX 3090)

**Dataset:** 86,835 train facts, E=40,559 entities, P=11 predicates, 42 rules (all two-body).
6 chunks of 512 queries. Times are avg of 3 full passes.

| Variant | Init (s) | Forward (s) | Provable (/2924) | Groundings | Peak VRAM (MB) |
|---------|----------|-------------|------------------|------------|----------------|
| **BCStaticSLD** | 0.255 | **0.144** | 1,220 | 1,410 | 1,827 |
| **BCStaticProvset** | 0.997 | 0.214 | **1,250** | 1,483 | 2,838 |
| **BCStaticPrune** | 0.006 | 0.463 | 1,221 | 1,415 | 3,930 |

**Observations:**
- **Lower provability** than kinship_family: only 41.7–42.7% of test queries are provable (vs 97.5–98.0%). wn18rr has sparser rule coverage.
- **BCStaticSLD is fastest** (144 ms), 1.5× faster than Provset (214 ms) and 3.2× faster than Prune (463 ms).
- **BCStaticProvset finds 30 more provable queries** (1,250 vs 1,220/1,221) — same pattern as kinship_family, the FC oracle covers atoms missed by per-query BFS.
- **Fewer groundings per query** (~0.5 g/q vs ~9 g/q in kinship_family) — wn18rr rules are sparser.
- **Lower VRAM** than kinship_family despite 14× more entities — fewer rules (42 vs 143) and sparser fact index mean smaller intermediate tensors.
- **BCStaticProvset init is slower** (1.0s) due to FC computing 2.68M provable atoms (step 0: 546K, step 1: 2.13M). BCStaticPrune/SLD skip FC entirely (6–255 ms).
