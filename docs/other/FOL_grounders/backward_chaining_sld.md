# BCStaticSLD vs Prolog Batched BFS

Side-by-side complexity analysis of the two compiled, GPU-native proof
engines. See `analysis/grounding_utils.py` for shared helpers.

## Table of Contents

1. [Notation](#1-notation)
2. [Two Resolution Strategies: MGU vs Cascaded Enumeration](#2-two-resolution-strategies-mgu-vs-cascaded-enumeration)
3. [BCStaticSLD — Step-by-Step](#3-bcstaticsld--step-by-step)
4. [Prolog Batched BFS — Step-by-Step](#4-prolog-batched-bfs--step-by-step)
5. [Memory Limits and Pre-set Hyperparameters](#5-memory-limits-and-pre-set-hyperparameters)
6. [Side-by-Side Comparison](#6-side-by-side-comparison)
7. [When Each System Hits Its Memory Ceiling](#7-when-each-system-hits-its-memory-ceiling)
8. [Prolog in RL — Interactive Proof Search](#8-prolog-in-rl--interactive-proof-search)
9. [Prolog: 1-Step vs N-Steps](#9-prolog-1-step-vs-n-steps)

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $B$ | Batch size (queries processed per forward call) |
| $E$ | Number of entities |
| $P$ | Number of predicates |
| $F$ | Number of base facts ($\subseteq P \times E \times E$) |
| $R$ | Total rules |
| $R_q$ (or $R_{\text{eff}}$) | Max rules matching a single predicate |
| $m$ (or $M$) | Max body atoms per rule |
| $D$ | Max proof depth (= `num_steps` in BCStaticSLD, `max_depth` in Prolog) |
| $K$ | Average branching factor (out-degree per `(pred, entity)` lookup) |
| $K_{\max}$ | Capped fanout (BCStaticSLD: per-predicate; Prolog: `_PROLOG_DATASET_K_MAX`, default 64) |
| $A$ | Max atoms per Prolog state (`M_max = padding_atoms + max_rule_body * 10`) |
| $S_{\max}$ | `max_states` — BCStaticSLD: `_MAX_STATES=64`; Prolog: `_PROLOG_MAX_STATES=128` |
| $F_{\max}$ | `max_frontier` in Prolog BFS (configurable, used in BFS mode) |
| $Q_{\max}$ | `max_per_query` in Prolog BFS (configurable, used in BFS mode) |
| $T$ | Tabling hash table size in BCStaticSLD (default 512) |
| $G_t$ | `G_total_max` — combo budget per (goal, rule) in BCStaticSLD |

---

## 2. Two Resolution Strategies: MGU vs Cascaded Enumeration

BCStaticSLD and the Prolog engine solve the same problem — given a goal atom
and a set of rules, produce successor proof states — but they use
fundamentally different resolution mechanisms. This choice ripples through
every aspect of their memory and compute profiles.

### 2.1 MGU — Most General Unifier (Prolog)

Source: `ns_lib/grounding/unification.py` (ported from Batched_env-swarm).

**Core idea.** Given a goal atom (which may contain variables) and a
fact or rule head, compute the *most general substitution* that makes
them identical. Apply that substitution to the remaining atoms. The
resulting state still carries **unresolved variables** that will be bound
by later unification steps.

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

**Key properties:**

- **States carry variables.** A successor state like `father(john, ?Y)` is
  not fully ground — `?Y` will only be resolved when `father(john, ?Y)` is
  itself unified against a fact like `father(john, mary)`.
- **One unification = one substitution.** Each MGU step binds at most 2
  variables (one per argument for binary atoms). Free variables accumulate
  across the proof tree and are progressively resolved.
- **Variable bookkeeping.** Each batch element tracks a `next_var_index`
  counter. Every rule application allocates fresh variable IDs via
  standardization-apart to avoid cross-rule contamination. The state
  representation must store these variable IDs alongside constants.

**Per-step tensor shapes:**

| Tensor | Logical shape | Physical (compiled) shape | Contents |
|--------|--------------|--------------------------|----------|
| Goal atom | `[B, 3]` | `[B, 3]` | `(pred, arg0, arg1)` — args can be constants or variables |
| Fact candidates | `[B, K_f]` | `[B, K_f_max]` | Fact indices matching `(pred, bound_arg)` |
| Fact-derived states | `[B, K_f, G, 3]` | `[B, K_f_max, G, 3]` | Remaining atoms with substitutions applied |
| Rule candidates | `[B, K_r]` | `[B, K_r_max]` | Rule indices matching goal predicate |
| Rule-derived states | `[B, K_r, A, 3]` | `[B, K_r_max, A, 3]` | Body atoms + remaining, with substitutions |
| Substitutions | `[B, K, 2, 2]` | `[B, K_max, 2, 2]` | `(from, to)` pairs — 2 pairs per unification |
| **Combined output** | `[B, K_f+K_r, A, 3]` | **`[B, K_max, A, 3]`** | **Padded to fixed `K_max`** (default 120) |

**Important: logical vs physical complexity.** $K_f$ and $K_r$ are the
*actual* number of matching facts and rules for a given query — these are
**data-dependent** and typically small (e.g. 3–5 on FB15k237). But in
**compiled mode**, the engine allocates a **fixed** `[B, K_max, A, 3]`
output tensor (via `pack_fn` in `_strategy_prolog.py:240`), where
`K_max` is a compile-time constant (default 120, set via
`max_derived_per_state`). Unused slots are zero-padded.

**Memory per step (logical work):** $O(B \cdot (K_f + K_r) \cdot A)$ —
the actual compute touches only the populated slots.

**Memory per step (physical allocation):** $O(B \cdot K_{\max} \cdot A)$
— **always**, regardless of how many facts/rules match. Plus $\sim 2\times$
for standardization intermediates. This is what matters for OOM.

### 2.2 Cascaded Enumeration (BCStaticSLD)

Source: `ns_lib/grounding/backward_chaining_sld.py`, `BCStaticSLD._expand_goals`.

**Core idea.** Queries in BCStaticSLD are always **fully ground** — both
arguments are entity IDs, never variables. Instead of computing a symbolic
MGU, the grounder **enumerates** all concrete entity substitutions for free
variables by looking them up in the fact index. Each free variable is
resolved by a separate lookup, and the results are cross-producted.

**Pipeline for one goal atom:**

```
goal: reach(alice, bob)       rule: reach(X, Y) :- edge(X, Z), reach(Z, Y)
                                         │
             ┌──────────────────────────┘
             ▼
   1. Bind head variables from goal
      X = alice, Y = bob   (head args are always ground)
             │
             ▼
   2. Identify free variable: Z (not in head)
      body atom edge(X, Z) = edge(alice, Z) — Z is free
             │
             ▼
   3. Enumerate Z via fact_index.enumerate(pred=edge, bound=alice, dir=PS)
      → candidates: [charlie, dave, ...]   (up to K_max results)
             │
             ▼
   4. For EACH candidate z_i, resolve ALL body atoms:
      edge(alice, z_i), reach(z_i, bob)    ← fully ground!
             │
             ▼
   5. Check existence: is edge(alice, z_i) a base fact?
                       is reach(z_i, bob) a base fact?
      → terminal if ALL are facts; else compound state
```

**Key properties:**

- **States are always ground.** Every atom in every state contains only
  entity IDs (integers in $[0, E)$) and predicate IDs. No variable
  representation, no variable bookkeeping, no standardization-apart.
- **Enumeration replaces unification.** Instead of symbolically binding
  `Z = ?` and deferring resolution, the grounder concretely enumerates
  all possible values of `Z` from the fact index. This turns a single
  symbolic unification into $K_{\max}$ concrete candidates.
- **Cross-product for multiple free vars.** If a rule has $F$ free
  variables, each is enumerated in cascade (topological order by variable
  dependency). The candidate set grows as $K_{\max}^F$, capped at
  `_G_BUDGET = 128`.

**Per-step tensor shapes:**

| Tensor | Shape | Contents |
|--------|-------|----------|
| Goal atoms | `[B, G_p, 3]` | Goals to expand (always ground) |
| Rule clustering | `[B, G_p, R_eff]` | Rule indices per goal predicate |
| Candidate combos | `[B, G_p, R_eff, G_t, F]` | Entity values per free var per combo |
| Body atoms (resolved) | `[B, G_p, R_eff, G_t, M, 3]` | **Fully ground** body atoms |
| Fact existence | `[B, G_p, R_eff, G_t, M]` | Boolean — is this a base fact? |

**Memory per step:** $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot M)$ for the
body-atom tensor. This is the **peak intermediate** and scales with the
combinatorial expansion of free variables.

### 2.3 Memory Comparison

The fundamental tradeoff: MGU defers variable binding (compact states, more
steps needed); cascaded enumeration resolves variables eagerly (large
intermediate tensors, fewer steps needed).

| Property | MGU (Prolog) | Cascaded Enumeration (BCStaticSLD) |
|----------|-------------|-----------------------------------|
| **State contents** | Constants + variables (mixed) | Constants only (fully ground) |
| **State width** | $A$ atoms × 3 ints (some are var IDs) | $\text{max\_gps}$ atoms × 3 ints (all entity IDs) |
| **Per-step work** | 1 unification per (goal, fact/rule) pair: $O(A)$ per pair | 1 enumeration per free var × cross-product: $O(K_{\max}^F \cdot M)$ per rule |
| **Branching** | $K_f + K_r$ successors (facts + rules) | $R_{\text{eff}} \times G_t$ successors (rules × combos) |
| **Peak tensor** | $O(B \cdot K_{\max} \cdot A)$ | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot M)$ |
| **Variable overhead** | `next_var_indices [B]` + standardization temps ($\approx 2\times$ output) | Zero |
| **Depth scaling** | State width $A$ is fixed (vars carry forward) | `max_gps = 1+(m-1)*D` grows linearly with depth |

**Per-step comparison on kinship** ($R_{\text{eff}}=12$, $K_{\max}=16$,
$M=2$, $B=512$):

- **MGU**: resolves goal against ~12 rules → $B \times 12 \times A \times 3 \times 8$ bytes.
  At $A=26$: **4.6 MB** expansion tensor. Plus $\sim 2\times$
  standardization intermediates → **~14 MB total**.
- **Cascade (depth 0)**: $G_p = 1$ → $B \times 1 \times 12 \times 16 \times 2 \times 3 \times 8$ = **4.5 MB**.
- **Cascade (depth $\geq 1$)**: $G_p = 32$ → $B \times 32 \times 12 \times 16 \times 2 \times 3 \times 8$ = **144 MB**.

At depth 0, both use comparable memory (~4.5–14 MB). At depth $\geq 1$,
cascade's expansion tensor is larger (144 MB) because it materializes
$G_p \times K_{\max}$ concrete candidates in a single pass — but it
does **no standardization** and works on $M = 2$ atoms per candidate
instead of $A = 26$. The cascade pays upfront for what MGU defers across
multiple sub-steps. See Section 2.6 for the full D-step analysis showing
that total work across a complete proof is comparable, with cascade
having smaller constant factors.

### 2.4 Worked Example: FB15k237

**Dataset:** E = 14,505 entities, P = 237 predicates, F = 272,115 facts,
R = 199 rules (all 2-body). Max base degree = 3,612 but median ≈ 5.
$R_{\text{eff}} \approx 5$ (max rules per predicate). Sparse rule coverage:
$199/237 \approx 0.84$ rules per predicate on average, but concentrated on
popular predicates (up to 10+).

**Concrete query:** `film/directed_by(Inception, ?)`

Suppose 3 rules match `film/directed_by`:
```
film/directed_by(X,Y) :- film/produced_by(X,Z), person/colleague(Z,Y)
film/directed_by(X,Y) :- film/written_by(X,Z), person/spouse(Z,Y)
film/directed_by(X,Y) :- film/starred_in(X,Z), person/mentor(Z,Y)
```

#### MGU (Prolog) — one step

**1. Rule unification.** Unify `film/directed_by(Inception, ?Y)` against
each rule head. 3 rules → 3 unifications, each producing a substitution
and a successor state with variables:

```
Rule 1 → σ = {X→Inception, Y→?Y}
  successor: [film/produced_by(Inception, ?Z₁), person/colleague(?Z₁, ?Y)]

Rule 2 → σ = {X→Inception, Y→?Y}
  successor: [film/written_by(Inception, ?Z₂), person/spouse(?Z₂, ?Y)]

Rule 3 → σ = {X→Inception, Y→?Y}
  successor: [film/starred_in(Inception, ?Z₃), person/mentor(?Z₃, ?Y)]
```

**2. Fact unification.** Targeted lookup for `film/directed_by(Inception, ?)`
→ say 2 matching facts: `film/directed_by(Inception, Nolan)`,
`film/directed_by(Inception, Nolan_alt)`. Each fact resolves `?Y` directly
→ 2 terminal successors (proof found).

**Total successors:** $K_r + K_f = 3 + 2 = 5$ states.

**Tensor shape:** `[B, K_max, A, 3]` where `K_max = 200`, `A = M_max`.
For FB15k237 with 2-body rules and `padding_atoms = 20`:
$A = 20 + 2 \times 10 = 40$.

**Why $A = 40$?** The formula `padding_atoms + max_rule_body * 10` reserves
space for ~10 levels of rule expansion within a single state representation.
At any given proof depth $d$, a state actually needs only $1 + (m-1) \cdot d$
atoms (the same formula as cascade's `max_gps`). At depth 5 with $m=2$,
that is 6 atoms — meaning **85% of $A = 40$ is zero-padding**. Every
unification and substitution operation processes all 40 slots including the
~34 padding slots. This fixed padding is the core per-step overhead of MGU
relative to cascade, which works on exactly $M$ atoms per grounding.

**Per-step memory at B=512:**
$512 \times 200 \times 40 \times 3 \times 8 = \textbf{98 MB}$
(plus $\sim 2\times$ for standardization intermediates → **~294 MB total**).

But only 5 out of 200 $K_{\max}$ slots are populated per query — **97.5% is
padding**. The tensor is pre-allocated at worst-case width regardless of
actual branching.

#### Cascaded Enumeration (BCStaticSLD) — one step

**1. Rule clustering.** Map `film/directed_by` → 3 rules (via
`pred_rule_indices`). $R_{\text{eff}} \leq 5$.

**2. Enumerate free variable Z.** For Rule 1, the free variable Z appears
in `film/produced_by(Inception, Z)`. Lookup
`fact_index.enumerate(pred=film/produced_by, bound=Inception, dir=PS)`:

```
→ candidates for Z: [WB_Pictures, Syncopy, LegendaryEnt, ...]
  up to K_max = 16 results (capped)
```

**3. Resolve all body atoms.** For each candidate $z_i$, fill in both body
atoms with concrete entities:

```
z = WB_Pictures →  [film/produced_by(Inception, WB_Pictures),
                     person/colleague(WB_Pictures, ?)]
                                                     ↑ wait — what is ?
```

Here is the key difference: in cascaded enumeration, the second body atom
`person/colleague(Z, Y)` becomes `person/colleague(WB_Pictures, bob)` where
`bob` is the **query's object argument** (already ground from the original
query `film/directed_by(Inception, bob)`). Both arguments are concrete
entity IDs. The system then checks: *is this triple a base fact?*

**4. Fact existence check.** For each of the $3 \times 16 = 48$ candidate
body-atom pairs, check all $m = 2$ body atoms against the fact index:

```
film/produced_by(Inception, WB_Pictures) — exists? hash lookup → True
person/colleague(WB_Pictures, bob)       — exists? hash lookup → False
→ NOT terminal (second body atom is not a fact)
→ becomes a compound state with 1 unresolved goal
```

**Total candidates per batch element:** At **depth 0**, only the single
query goal is expanded, so $G_p = 1$: candidates $= 1 \times 5 \times 16 = 80$.
At **depth $\geq 1$**, multiple compound states are expanded:
$G_p = \min(S_{\max}, \text{active\_states}) \leq 32$, giving up to
$32 \times 5 \times 16 = 2{,}560$ candidates.

**Tensor shape:** `[B, G_p, R_eff, K_max, M, 3]`.

| Depth | $G_p$ | Shape | Memory at $B = 512$ |
|-------|--------|-------|---------------------|
| 0 | 1 | `[512, 1, 5, 16, 2, 3]` | $512 \times 1 \times 5 \times 16 \times 2 \times 3 \times 8 = \textbf{1.9 MB}$ |
| $\geq 1$ | 32 | `[512, 32, 5, 16, 2, 3]` | $512 \times 32 \times 5 \times 16 \times 2 \times 3 \times 8 = \textbf{60 MB}$ |

No standardization overhead (no variables). Every depth pays its expansion
cost again.

#### Side-by-side for FB15k237 at B=512

| Metric | MGU (Prolog) | Cascade (BCStaticSLD) |
|--------|-------------|----------------------|
| **Successors per query** | $K_f + K_r \approx 5$ (actual data-dependent) | $R_{\text{eff}} \times K_{\max} = 80$ (materialized) |
| **Useful fraction** | ~5/200 = 2.5% of $K_{\max}$ slots | ~5/80 = 6.3% of combos have all-fact bodies |
| **Per-step tensor** | `[512, 200, 40, 3]` = **98 MB** (200 = $K_{\max}$, fixed) | `[512, 32, 5, 16, 2, 3]` = **60 MB** (depth $\geq 1$; 1.9 MB at depth 0) |
| **Standardization temps** | +196 MB ($\sim 2\times$ output) | 0 |
| **Per-step total** | **~294 MB** | **~60 MB** (depth $\geq 1$) |
| **Contains variables?** | Yes — must be resolved at next depth | No — fully ground, check against facts |
| **Depth-5 state width** | 40 atoms (fixed, ~85% padding) | $1 + 1 \times 5 = 6$ goals per state (exact) |
| **Depth-5 compound states** | N/A (frontier-based) | `[512, 64, 6, 3]` = 4.7 MB |

**Per-step memory:** MGU is **~5× more expensive** per step on FB15k237.
The overhead comes from two sources: (1) the $A = 40$ fixed state width
where most slots are padding — at depth 5 only 6 of 40 atoms are
meaningful, so every unification and substitution wastes ~85% of its
compute on zeros; (2) the $\sim 2\times$ standardization intermediates
that cascade avoids entirely by keeping states ground.

**Across depth:** However, the Prolog BFS frontier can grow to
$F_{\max} = 2{,}000{,}000$ states between depths:

| Depth | Prolog frontier (worst) | Prolog frontier memory | BCStaticSLD compound states |
|-------|------------------------|------------------------|----------------------------|
| 1 | $N = 512$ | $512 \times 40 \times 24 = 0.5\text{ MB}$ | $512 \times 64 \times 4 \times 24 = 3.1\text{ MB}$ |
| 2 | $512 \times 200 = 102{,}400$ | $102{,}400 \times 40 \times 24 = 98\text{ MB}$ | $512 \times 64 \times 5 \times 24 = 3.9\text{ MB}$ |
| 3 | $\min(102{,}400 \times 200,\, 2\text{M}) = 2\text{M}$ | $2\text{M} \times 40 \times 24 = 1.92\text{ GB}$ | $512 \times 64 \times 6 \times 24 = 4.7\text{ MB}$ |
| 5 | $2\text{M}$ (capped) | **1.92 GB** (frontier) + engine per-chunk | $512 \times 64 \times 8 \times 24 = 6.3\text{ MB}$ |

At depth 3+, the Prolog frontier alone exceeds the cascade's **total**
memory footprint. At depth 1, a single MGU engine call is ~5× more
expensive than a single cascade expansion.

> **FB15k237 summary:** MGU has cheaper *total* memory at shallow depth
> (one engine call, small frontier), but frontier growth makes it
> expensive by depth 3. Cascade has a ~60 MB expansion tensor at depth
> $\geq 1$ that stays constant — but it also pays for a 440 MB dense
> fact index upfront (`[237, 14505, 16]`). The crossover point is around
> depth 2–3: below that, MGU's small frontier wins; above that,
> cascade's capped compound states win.

**Why MGU uses more memory across depth:**

MGU's per-step cost is moderate (~294 MB), but each successor state must
be expanded again at the next depth — and the number of states can grow
combinatorially (the frontier explosion from Section 4). Cascaded
enumeration's per-step cost is lower (~60 MB), and the number of compound
states is capped at $S_{\max}=64$, preventing frontier growth.

> **In short:** MGU trades *wide steps* for *many states*; cascaded
> enumeration trades *few states* for *wide steps*. The memory bottleneck
> for MGU is the frontier; for cascaded enumeration, it's the expansion
> tensor.

### 2.5 Thought Experiment: What If BCStaticSLD Used MGU?

Suppose we replaced cascaded enumeration inside BCStaticSLD with the same
MGU engine that Prolog uses. The per-step expansion tensor would change,
but **five architectural differences would remain** — and these are
arguably more important than the resolution mechanism itself.

#### Difference 1 — Frontier Model: Fixed Cap vs Dynamic Growth

This is the most consequential difference.

**BCStaticSLD** keeps at most $S_{\max} = 64$ compound states per batch
element, stored in a pre-allocated tensor `[B, S_max, max_gps, 3]`.
If an expansion produces more than 64 non-terminal states, excess states
are discarded via `topk`. The buffer size never changes.

**Prolog BFS** maintains a shared frontier `[F_d, A, 3]` across all
queries. After each depth, every surviving state spawns up to $K_{\max}$
children. The frontier can grow from $N$ to $N \cdot K_{\max}$ to
$N \cdot K_{\max}^2$ before the caps ($F_{\max}$, $Q_{\max}$) kick in.

Even with MGU inside BCStaticSLD, the compound-state cap would still
prevent frontier growth. The memory profile would change per-step
(MGU's `[B, S_max, K_max, A, 3]` instead of cascade's
`[B, G_p, R_eff, G_t, M, 3]`) but would remain **fixed and predictable**.
Prolog BFS's frontier would still be data-dependent.

```
BCStaticSLD (with MGU):     memory ────────────────────── (flat)
                            depth 1   depth 2   depth 3

Prolog BFS:                 memory ─────────╱─────────── (growing)
                            depth 1   depth 2   depth 3
                                              ↑ hits F_max cap
```

#### Difference 2 — Compound States vs Flat States

**BCStaticSLD** tracks the full goal set per proof state. Each compound
state records ALL unresolved goals as a `[max_gps, 3]` sub-tensor.
When a rule fires on one goal, that goal is replaced by the rule's
body atoms — the other goals carry forward in the same state. A
grounding is only emitted when ALL goals in the state are resolved.

**Prolog BFS** uses flat states: a padded list of atoms `[A, 3]`.
The engine always resolves the first atom and produces successor states
with the remaining atoms. There is no explicit "goal set" — the list
order implicitly determines resolution order.

The compound-state model has a structural cost: `max_gps = 1 + (m-1)*D`
grows linearly with depth. At $D=5$, $m=2$: each state carries 6 goal
slots. But it also enables **tabling** (Difference 3) because individual
goals can be checked and resolved against the cache independently.

With MGU, BCStaticSLD's compound states would carry variables — but the
*structure* (fixed-size goal sets, per-state tracking) would be unchanged.
Prolog's flat states would remain flat.

#### Difference 3 — Tabling

**BCStaticSLD** maintains a direct-mapped hash table `[B, T]` (T=512)
for proved subgoals. When any expansion of a goal produces an all-fact
proof, the goal's hash is inserted into the table. At subsequent depths,
every goal in every compound state is checked against the cache — cached
goals are removed from the state, potentially making it terminal without
further expansion.

**Prolog BFS** has no tabling. If the same subgoal appears in states
owned by different queries (or in different proof branches of the same
query), it will be expanded from scratch each time.

Tabling is orthogonal to MGU vs cascade. BCStaticSLD with MGU would
still benefit from tabling — the hash table stores goal-atom hashes,
which work identically for ground atoms and atoms with variables
(as long as variables are canonicalized before hashing).

#### Difference 4 — Compilation Model

**BCStaticSLD** runs under `torch.compile(fullgraph=True, mode='reduce-overhead')`.
Every tensor dimension is a compile-time constant. The only Python loop
is `for depth in range(num_steps)`. This enables CUDA graph caching:
after a warmup pass, every subsequent forward call reuses the captured
kernel sequence with zero Python overhead.

**Prolog BFS** compiles the per-step engine (`get_derived_states_compiled`)
under `fullgraph=True, reduce-overhead`. This is made possible by:
fixed-shape `[B, K_max, M_max, 3]` output tensors, scatter-based
compaction (no `.nonzero()` or boolean fancy indexing), canonical
standardization (all tensor ops, `torch._assert_async` for bounds
checks), and `mark_static_address` for persistent buffers. However,
the outer BFS loop — frontier management, chunking, deduplication —
remains eager Python because the frontier's data-dependent size
changes each iteration.

If BCStaticSLD used MGU, it would need to add variable standardization
inside the compiled region. This is already proven feasible — the Prolog
engine's canonical standardizer is `fullgraph=True` compatible. The key
point is that **BCStaticSLD's fixed shapes enable full end-to-end
compilation** (outer loop included), while Prolog BFS can only compile
the inner per-step engine, not the frontier management.

#### Difference 5 — Fact Storage: Dense vs Sparse

**BCStaticSLD** uses a dense `[P, E, K_max]` tensor fact index.
Every `(predicate, entity)` pair has $K_{\max}$ reserved slots, even
if most are empty. This enables O(1) batched gather operations that
are fullgraph-compatible.

**Prolog BFS** uses a sparse sorted fact list with segment indices.
Facts are stored contiguously, sorted by `(pred, arg)` keys. Lookup
is O(1) via pre-computed segment starts, but the storage is proportional
to $F$ (actual facts), not $P \cdot E \cdot K_{\max}$ (all possible slots).

This is independent of MGU — both resolution strategies need fact
lookups. But the dense index is what makes BCStaticSLD's
`fact_index.exists()` and `fact_index.enumerate()` fullgraph-compatible.
With MGU, BCStaticSLD would still need the dense index for fact
unification (or would need a new fullgraph-compatible sparse lookup).

#### Summary Table

| Difference | Depends on MGU vs Cascade? | Impact |
|------------|---------------------------|--------|
| **Fixed vs dynamic frontier** | No | Determines memory predictability and depth scaling |
| **Compound vs flat states** | No | Determines goal-set tracking and `max_gps` growth |
| **Tabling** | No | Completeness gain, avoids redundant expansion |
| **Fullgraph compilation** | No (but requires fixed shapes) | 10-100× throughput on GPU via CUDA graph caching |
| **Dense vs sparse facts** | No | Dense wastes space on sparse KGs; enables O(1) batched gather |

> **Bottom line:** If BCStaticSLD adopted MGU, the per-step tensor would
> shrink (no $G_p \times R_{\text{eff}} \times G_t$ cross-product), but
> states would now carry variables (requiring standardization). The five
> differences above would remain. The most important — **fixed frontier
> vs dynamic frontier** — is what makes BCStaticSLD's memory constant
> across depth and Prolog BFS's memory data-dependent. That difference
> has nothing to do with how individual goals are resolved.

### 2.6 D-Step Analysis Without Caps

The comparisons in Sections 2.3–2.5 include implementation caps
($F_{\max}$, $S_{\max}$, $G_{\text{BUDGET}}$, $K_{\max}$) that
conflate algorithmic properties with engineering choices. This section
strips away the caps and compares the two resolution strategies on
equal footing.

#### 2.6.1 Completeness

**Both are equally complete** for the KG completion setting (finite
entities, no function symbols, ground queries).

At every branching point, both discover the same set of candidate
entities. Consider a rule `p(X,Y) :- q(X,Z), r(Z,Y)` and ground
query `p(a,b)`:

- **MGU:** unifies goal → state `[q(a, ?Z), r(?Z, b)]` → resolves
  `q(a, ?Z)` against facts → `?Z` bound to each entity $c$ where
  `q(a,c)` is a fact.
- **Cascade:** binds X=a, Y=b → enumerates Z from `q(a,?)` in fact
  index → gets the **same set** of entities $c$.

MGU defers naming the candidates (carries `?Z`); cascade materializes
them eagerly. At every step, the set of discovered branches is identical.
This holds for multi-step proofs: cascade's compound states carry forward
the same unresolved goals that MGU would resolve at the next step. For
multiple free variables, cascade's topological enumeration produces the
same $(Z, W)$ pairs that MGU discovers through sequential unification.

The only setting where MGU is genuinely more complete is with **function
symbols** (infinite Herbrand universe), which does not apply to KG
completion.

> **The document's earlier framing of MGU as "more complete" (Section 2.3)
> is an artifact of the implementation caps, not of the algorithms
> themselves.**

#### 2.6.2 Frontier Growth

Without caps, **both have the same exponential frontier growth**:

$$F_d = B \cdot (R \cdot K)^d$$

where $R$ = rules per predicate, $K$ = avg entities per enumeration
(degree), and $d$ = proof depth (number of rule applications).

- **MGU:** produces $R$ symbolic successors per rule-resolution step
  (additive: $K_f + K_r$). The $K$ branching is deferred to fact-
  resolution steps. Over the $M$ steps needed per depth, the frontier
  expands by factor $R \cdot K$ total.
- **Cascade:** produces $R \cdot K^F$ ground candidates per depth
  iteration. For $F = 1$ (typical 2-body rules): same $R \cdot K$
  factor.

The frontier grows at the same rate — cascade materializes it in one
shot per depth, MGU spreads it across $M$ sub-steps.

> **The claim that frontier explosion is MGU-specific (Section 2.3,
> line 363) is wrong without caps.** Cascade has the same exponential
> growth; the difference is that cascade's $S_{\max} = 64$ cap is easy
> to apply (ground states, fixed shapes), while MGU's $F_{\max}$ cap
> is lossy (drops variable-carrying states whose downstream value is
> unknown).

#### 2.6.3 Steps vs Depths

The two engines count progress differently:

- **MGU resolves one atom per step.** A depth-$D$ proof (i.e. $D$
  rule applications) requires $\sim D \cdot M$ total steps — $D$ rule-
  resolution steps plus $D \cdot (M-1)$ fact-resolution steps to bind
  the free variables introduced by each rule.
- **Cascade resolves one goal per depth iteration**, but within that
  iteration it enumerates all free variables and checks all body atoms
  at once. A depth-$D$ proof takes exactly $D$ iterations.

Cascade does $D$ kernel launches where MGU does $D \cdot M$, but each
cascade launch does more work per call.

#### 2.6.4 Per-Depth Cost

At depth $d$, both must expand every state in the frontier
$F_d = B \cdot (R \cdot K)^d$.

| Metric | MGU (summed over $M$ steps) | Cascade (1 iteration) |
|--------|----------------------------|----------------------|
| **Calls** | $M$ steps | 1 iteration |
| **Expansion tensor per call** | $F_d \cdot K_{\max} \cdot A \cdot 3$ | $F_d \cdot R \cdot G_t \cdot M \cdot 3$ |
| **Standardization overhead** | $+2{-}3\times$ per step ($M$ times) | 0 |
| **Per-candidate work** | $O(A)$ — unify + substitute across all atom slots | $O(M)$ — fact-existence hash lookups for $M$ body atoms |

For $M=2$, $R=5$, $K=10$, $A=40$:

- **MGU per depth:** $F_d$ states $\times$ 2 steps $\times$ $(R+K) \cdot A \cdot 3 = 15 \times 40 \times 3 = 1{,}800$ work units per state, $\times 3$ standardization → **$\sim 5{,}400 \cdot F_d$**
- **Cascade per depth:** $F_d$ states $\times$ $R \cdot K \cdot M \cdot 3 = 5 \times 10 \times 2 \times 3 = 300$ work units per state → **$300 \cdot F_d$**

**MGU does ~18× more work per depth** per state. The dominant factor is
the $A = 40$ padding: every unification and substitution touches all 40
atom slots, but at depth 5 only 6 contain meaningful data. Cascade works
on exactly $M = 2$ atoms per grounding — no padding waste.

#### 2.6.5 Where the $A$ Padding Overhead Comes From

MGU's state tensor has fixed width $A = \text{padding\_atoms} + \text{max\_rule\_body} \times 10$.
The `× 10` factor reserves space for ~10 levels of rule expansion within
a single state. The actual number of atoms grows as $1 + (m-1) \cdot d$:

| Depth $d$ | Atoms needed | $A$ (padded) | Waste |
|-----------|-------------|--------------|-------|
| 0 | 1 | 40 | 97.5% |
| 1 | 2 | 40 | 95.0% |
| 3 | 4 | 40 | 90.0% |
| 5 | 6 | 40 | 85.0% |
| 10 | 11 | 40 | 72.5% |

Cascade's `max_gps = 1 + (m-1) \cdot d` grows exactly as needed (known
at compile time per depth), so its waste is zero. This is the single
largest constant-factor difference between the two approaches: **every
MGU operation pays for $A$ atoms; every cascade operation pays for $M$
atoms.**

In principle, $A$ could be reduced (e.g. `padding_atoms=5,
max_rule_body * 3` → $A = 11$), but then deeper proofs would overflow
the state tensor. It is a compile-time tradeoff: larger $A$ = supports
deeper proofs but wastes more compute on padding per step.

#### 2.6.6 Total Across $D$ Depths

| | MGU | Cascade |
|---|---|---|
| **Total work** | $\sum_{d=0}^{D} M \cdot F_d \cdot K_{\max} \cdot A \cdot 3 \cdot (1 + \text{std})$ | $\sum_{d=0}^{D} F_d \cdot R \cdot K^F \cdot M \cdot 3$ |
| **Dominant term** | $(R \cdot K)^D \cdot B \cdot M \cdot K_{\max} \cdot A$ | $(R \cdot K)^D \cdot B \cdot R \cdot K \cdot M$ |
| **Frontier storage** | $(R \cdot K)^D \cdot B \cdot A \cdot 24$ bytes | $(R \cdot K)^D \cdot B \cdot \text{max\_gps}(D) \cdot 24$ bytes |

Both are **dominated by the $(R \cdot K)^D$ exponential frontier**. The
asymptotic complexity is the same. The difference is the constant factor:
MGU pays $K_{\max} \cdot A \cdot (1 + \text{std\_overhead})$ per state;
cascade pays $R \cdot K \cdot M$ per state.

#### 2.6.7 Summary

| Property | MGU | Cascade | Verdict |
|----------|-----|---------|---------|
| **Completeness** | Complete | Complete | **Tie** |
| **Frontier growth** | $O((R \cdot K)^D)$ | $O((R \cdot K)^D)$ | **Tie** |
| **Kernel launches per depth** | $M$ | 1 | **Cascade** |
| **Per-state overhead** | $A$ (large fixed padding) + $2{-}3\times$ standardization | $M$ (exact, no padding) | **Cascade** (~18× less for typical params) |
| **Compilability** | Per-step engine: `fullgraph=True` ✓ | Full end-to-end: `fullgraph=True` ✓ | **Cascade** (outer loop too) |
| **Expressiveness** | Non-ground queries (`p(a, ?Y)`) | Ground queries only | **MGU** |

**Without caps, the two approaches are algorithmically equivalent in
completeness and asymptotic frontier growth.** The practical differences
are: (1) cascade's per-step efficiency (~18× less work due to no $A$
padding and no standardization), (2) cascade's single kernel launch per
depth vs MGU's $M$ launches, and (3) MGU's ability to handle non-ground
queries. For KG completion — where queries are always ground — cascade
wins on constant factors at every depth.

---

## 3. BCStaticSLD — Step-by-Step

Source: `ns_lib/grounding/backward_chaining_sld.py`, class `BCStaticSLD`.

BCStaticSLD is a **fixed-shape, fully compiled** SLD grounder. Every tensor
has a compile-time-known shape, enabling `torch.compile(fullgraph=True,
mode='reduce-overhead')` with CUDA graph caching.

### Step 0 — Preprocessing (once at init)

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 0a. Build K_max-capped fact index | Sort facts, truncate to $K_{\max}$ per `(pred, entity)`, store in dense `[P, E, K_{\max}]` tensors | $O(F \log F)$ | $O(P \cdot E \cdot K_{\max})$ |
| 0b. Compile rules | `CompiledRule` per rule: topological sort of body atoms, cascaded enumeration metadata | $O(R \cdot m^2)$ | $O(R \cdot m)$ |
| 0c. Build rule clustering | Per-predicate rule grouping → `pred_rule_indices [P, R_{\text{eff}}]` | $O(R)$ | $O(P \cdot R_{\text{eff}})$ |
| 0d. Allocate tabling buffer | Direct-mapped hash table `[B, T]` (filled at runtime) | $O(1)$ | $O(B \cdot T)$ |

**No forward chaining** — soundness comes from SLD proof completion, not an
FC provable-set oracle.

**Total init time:** $O(F \log F + R \cdot m^2)$.
**Persistent space:** $O(P \cdot E \cdot K_{\max} + R \cdot m + P \cdot R_{\text{eff}})$.

### Step 1 — Depth 0: Expand Queries

Each query $q = (p, s, o)$ is expanded by rule clustering + cascaded
enumeration + body-atom resolution.

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 1a. Rule clustering | Map query predicate → $R_{\text{eff}}$ candidate rules via precomputed index | $O(B)$ | $O(B \cdot R_{\text{eff}})$ |
| 1b. Cascaded enumeration | For each (query, rule): enumerate free variables via `fact_index.enumerate()`. For $F_{\max\text{-vars}}$ free vars per rule, cross-product capped at $G_t$ combos | $O(B \cdot R_{\text{eff}} \cdot F_{\max\text{-vars}} \cdot K_{\max})$ | $O(B \cdot R_{\text{eff}} \cdot G_t \cdot F_{\max\text{-vars}})$ |
| 1c. Resolve body atoms | Gather `(pred, subj, obj)` for all $m$ body atoms of each combo using binding metadata | $O(B \cdot R_{\text{eff}} \cdot G_t \cdot m)$ | $O(B \cdot R_{\text{eff}} \cdot G_t \cdot m)$ |
| 1d. Fact existence check | Hash-based lookup in fact index: $\text{hash} = p \cdot E^2 + s \cdot E + o$. Boolean per body atom. | $O(B \cdot R_{\text{eff}} \cdot G_t \cdot m)$ | $O(B \cdot R_{\text{eff}} \cdot G_t \cdot m)$ |
| 1e. Terminal collection | Groundings where ALL active body atoms are facts → output buffer. Query self-exclusion filter applied. Dedup via polynomial hash sort. | $O(B \cdot R_{\text{eff}} \cdot G_t \cdot m)$ | $O(B \cdot tG \cdot m)$ |
| 1f. Compound state init | Non-terminal expansions → `states [B, S_{\max}, \text{max\_gps}, 3]` with per-state goal mask. XOR hash dedup. | $O(B \cdot R_{\text{eff}} \cdot G_t)$ | $O(B \cdot S_{\max} \cdot \text{max\_gps})$ |

where $\text{max\_gps} = 1 + (m-1) \cdot D$ (tight upper bound on goals per compound state).

**Peak intermediate tensor** (step 1c): the body-atom tensor
$B \times 1 \times R_{\text{eff}} \times G_t \times m \times 3$ (long).

### Step 2 — Depth Loop ($d = 1 \ldots D{-}1$)

Each iteration picks the first unresolved goal from each compound state,
expands it, and updates the state.

| Sub-step | What | Time per depth | Space per depth |
|----------|------|----------------|-----------------|
| 2a. Goal selection | `argmax` on goal mask → first active goal per state | $O(B \cdot G_p \cdot \text{max\_gps})$ | $O(B \cdot G_p)$ |
| 2b. Expand goals | Same as Step 1 (clustering + enum + resolve + exist) but over $G_p = \min(S_{\max}, 32)$ goals instead of 1 | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ |
| 2c. State update | Remove expanded goal, insert unresolved new body atoms into empty goal slots | $O(B \cdot n_{\exp} \cdot m \cdot \text{max\_gps})$ | $O(B \cdot n_{\exp} \cdot \text{max\_gps})$ |
| 2d. Tabling — insert | Goals with ANY all-fact expansion → hash into `tbl [B, T]` | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t)$ | $O(B \cdot T)$ (reused) |
| 2e. Tabling — resolve | Check ALL goals in new states against tabling cache. Cached goals removed from state → may make it terminal earlier. | $O(B \cdot n_{\exp} \cdot \text{max\_gps})$ | $O(1)$ (in-place) |
| 2f. Terminal collection | Same dedup + topk as Step 1e | $O(B \cdot n_{\exp} \cdot m)$ | $O(B \cdot tG \cdot m)$ |
| 2g. State selection | Keep up to $S_{\max}$ non-terminal states via topk. XOR hash dedup. | $O(B \cdot n_{\exp})$ | $O(B \cdot S_{\max} \cdot \text{max\_gps})$ |

where $n_{\exp} = G_p \cdot R_{\text{eff}} \cdot G_t$.

### Complexity Summary

| | |
|---|---|
| **Init time** | $O(F \log F + R \cdot m^2)$ |
| **Init space** | $O(P \cdot E \cdot K_{\max} + R \cdot m)$ |
| **Time per depth** | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ |
| **Total forward time** | $O(D \cdot B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ |
| **Peak intermediate space** | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ — body atom tensor |
| **Persistent space** | $O(P \cdot E \cdot K_{\max} + B \cdot S_{\max} \cdot \text{max\_gps} + B \cdot T + B \cdot tG \cdot m)$ |

All shapes are **compile-time constants** (no data-dependent sizing).
The only Python-level loop is `for depth in range(num_steps)`.

---

## 4. Prolog Batched BFS — Step-by-Step

Source: `ns_lib/grounding/backward_chaining_prolog.py` (torch-ns port) and
`ns_lib/grounding/unification.py` (ported from Batched_env-swarm).

The Prolog engine is a **batched, GPU-accelerated BFS** that processes all
queries simultaneously. Unlike BCStaticSLD (which has fixed output shapes
and produces groundings), Prolog BFS tracks a **growing frontier of proof
states** and only needs to determine provability (depth or unprovable).

### Step 0 — Preprocessing (once at init)

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 0a. Index constants + predicates | Map strings → integer IDs. Reserve runtime variable pool: `[constant_no+1 .. runtime_var_end_index]`. | $O(E + P)$ | $O(E + P)$ |
| 0b. Build fact index | Sort facts lexicographically by $p \cdot V^2 + s \cdot V + o$. Build `predicate_range_map [P, 2]` for O(log F) predicate lookup. Build arg-indexed structures for O(1) `(pred, arg)` fact lookup. | $O(F \log F)$ | $O(F + P)$ |
| 0c. Compile rules | Group rules by head predicate → `rule_seg_starts [P]`, `rule_seg_lens [P]`. Pack into `rules_idx_sorted [R, M_{\max}, 3]`. | $O(R \cdot m)$ | $O(R \cdot A)$ |
| 0d. Compute safe batch size | $\text{bytes\_per\_state} = K_{\max} \cdot A \cdot 3 \cdot 8$. Safe $B = \min(\text{batch\_size},\, \lfloor 4\text{GB} / (\text{bytes\_per\_state} \cdot 6) \rfloor)$. | $O(1)$ | — |

where $V$ = total vocabulary size (constants + vars + padding).

**Total init time:** $O(F \log F + R \cdot m)$.
**Persistent space:** $O(F + R \cdot A + E + P)$.

### Step 1 — Seed Frontier

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 1a. Create initial states | Each query $(p, s, o)$ becomes a single-atom state: `frontier_states [N, A, 3]` with atoms 1..A zeroed (padding). | $O(N)$ | $O(N \cdot A)$ |
| 1b. Init metadata | `frontier_owner [N]`: query index per state. `frontier_vars [N]`: next variable ID. `visited_hashes`: empty. | $O(N)$ | $O(N)$ |

### Step 2 — BFS Depth Loop ($d = 1 \ldots D$)

The frontier is processed in **chunks** of size $B$ (safe batch size).
Within each chunk, the unification engine expands every state in parallel.

#### 2a. Chunk the frontier

| What | Time | Space |
|------|------|-------|
| Split `frontier_states [F_d, A, 3]` into $\lceil F_d / B \rceil$ chunks. | $O(1)$ (views) | — |

#### 2b. Unification — per chunk ($B$ states)

Each state's **first atom** (the selected goal) is resolved against facts
and rules simultaneously.

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 2b-i. Detect terminal atoms | Check if first atom is the True/False predicate or if state is empty (= proof found). | $O(B \cdot A)$ | $O(B)$ |
| 2b-ii. Fact unification | Targeted lookup: `(pred, arg0)` or `(pred, arg1)` → up to $K_f$ matching facts per state. Unify variables → substitute into remaining atoms. Output: `[B, K_f, G, 3]` successor states (goal removed, variables bound). | $O(B \cdot K_f \cdot A)$ | $O(B \cdot K_f \cdot A)$ |
| 2b-iii. Rule unification | Segment lookup: head predicate → up to $K_r$ matching rules. Unify head with goal → substitute into body atoms. Output: `[B, K_r, A, 3]` successor states (goal replaced by rule body). | $O(B \cdot K_r \cdot A)$ | $O(B \cdot K_r \cdot A)$ |
| 2b-iv. Pack + cap | Merge fact-derived ($K_f$) and rule-derived ($K_r$) → `[B, K_{\max}, A, 3]`. Priority sort, cap at $K_{\max}$. | $O(B \cdot (K_f + K_r) \cdot A)$ | $O(B \cdot K_{\max} \cdot A)$ |
| 2b-v. Variable standardization | Offset-based: shift variable IDs in derived states to avoid collision with parent state variables. | $O(B \cdot K_{\max} \cdot A)$ | $O(B \cdot K_{\max} \cdot A)$ |
| 2b-vi. Ground fact pruning | Remove body atoms that are already ground facts (no further resolution needed). Vectorized hash-based membership test against `fact_hashes`. | $O(B \cdot K_{\max} \cdot A \cdot \log F)$ | $O(B \cdot K_{\max} \cdot A)$ |

**Note on $K_f$, $K_r$ vs $K_{\max}$:** In sub-steps 2b-ii and 2b-iii,
$K_f$ and $K_r$ denote the *actual* number of matching facts/rules for a
given query — these are **data-dependent** (typically 3–5 on FB15k237).
The time complexity is proportional to these. But the **space
allocation** is always padded to the compile-time constant $K_{\max}$
(default 120, set via `max_derived_per_state`) in step 2b-iv. This is
required for `torch.compile(fullgraph=True)` — all tensor shapes must be
static. On FB15k237, ~97% of the `[B, K_max, A, 3]` output is padding.

**Peak engine intermediate**: the derived-states tensor
$B \times K_{\max} \times A \times 3$ (long) = $B \cdot K_{\max} \cdot A \cdot 24$ bytes.

#### 2c. Collect results across chunks

| What | Time | Space |
|------|------|-------|
| Concatenate valid derived states from all chunks into new frontier. Record proved queries. | $O(F_d \cdot K_{\max})$ | $O(F_d \cdot K_{\max} \cdot A)$ — **this is the raw expansion** |

#### 2d. Within-depth deduplication

| What | Time | Space |
|------|------|-------|
| Position-weighted hash per state (combining owner ID + atom content). Sort hashes, mark adjacent duplicates. | $O(F_{d+1} \cdot A)$ for hashing + $O(F_{d+1} \log F_{d+1})$ for sort | $O(F_{d+1})$ hash buffer |

#### 2e. Cross-depth deduplication

| What | Time | Space |
|------|------|-------|
| Binary search new state hashes against sorted `visited_hashes` buffer. Filter already-seen states. | $O(F_{d+1} \cdot \log H)$ where $H$ = cumulative visited | $O(H)$ visited buffer |

#### 2f. Frontier capping

| What | Time | Space |
|------|------|-------|
| Per-query cap: keep at most $Q_{\max}$ states per owner query. Global cap: keep at most $F_{\max}$ states total (random selection if over). | $O(F_{d+1})$ | $O(\min(F_{d+1},\, F_{\max}))$ |

#### 2g. Filter proved owners

| What | Time | Space |
|------|------|-------|
| Remove frontier states whose owner query was already proved at a shallower depth. | $O(F_{d+1})$ | — |

### Complexity Summary

| | |
|---|---|
| **Init time** | $O(F \log F + R \cdot m)$ |
| **Init space** | $O(F + R \cdot A + E + P)$ |
| **Time per depth** | $O\!\left(\frac{F_d}{B} \cdot B \cdot K_{\max} \cdot A + F_d \cdot K_{\max} \cdot A + F_{d+1} \log F_{d+1}\right)$ |
| | $= O(F_d \cdot K_{\max} \cdot A + F_{d+1} \log F_{d+1})$ |
| **Total BFS time** | $O\!\left(\sum_{d=1}^{D} F_d \cdot K_{\max} \cdot A + F_{d+1} \log F_{d+1}\right)$ |
| **Peak engine space (per chunk)** | $O(B \cdot K_{\max} \cdot A)$ |
| **Peak frontier space** | $O(\min(F_{\max},\, N \cdot Q_{\max}) \cdot A)$ — **data-dependent** |
| **Visited set** | $O(H)$ — grows monotonically across depths |

**Key difference from BCStaticSLD**: the frontier size $F_d$ is
**data-dependent** — it grows with branching and shrinks with deduplication.
Capping at $F_{\max}$ and $Q_{\max}$ prevents unbounded growth but the
actual memory used varies per dataset and depth.

---

## 5. Memory Limits and Pre-set Hyperparameters

### 5.1 BCStaticSLD — All Shapes Are Compile-Time Constants

Every tensor dimension is determined **before** the first forward pass:

| Buffer | Shape | Formula (bytes) | Example: kinship_family |
|--------|-------|------------------|------------------------|
| Fact index | $[P, E, K_{\max}]$ long | $P \cdot E \cdot K_{\max} \cdot 8$ | $12 \times 2{,}968 \times 16 \times 8 = 4.6\text{ MB}$ |
| Fact mask | $[P, E, K_{\max}]$ bool | $P \cdot E \cdot K_{\max}$ | $0.6\text{ MB}$ |
| Compound states | $[B, S_{\max}, \text{max\_gps}, 3]$ long | $B \cdot 64 \cdot \text{gps} \cdot 24$ | $512 \times 64 \times 3 \times 24 = 2.4\text{ MB}$ |
| State goal mask | $[B, S_{\max}, \text{max\_gps}]$ bool | $B \cdot 64 \cdot \text{gps}$ | $0.1\text{ MB}$ |
| Tabling cache | $[B, T]$ long | $B \cdot T \cdot 8$ | $512 \times 512 \times 8 = 2.1\text{ MB}$ |
| Output buffer | $[B, tG, M, 3]$ long | $B \cdot tG \cdot M \cdot 24$ | $512 \times 128 \times 2 \times 24 = 3.1\text{ MB}$ |
| **Peak expansion** | $[B, G_p, R_{\text{eff}}, G_t, M, 3]$ long | $B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot M \cdot 24$ | $512 \times 32 \times 12 \times 16 \times 2 \times 24 = 150\text{ MB}$ |

**Hyperparameters that must be set in advance:**

| Param | Default | Effect on memory | How to choose |
|-------|---------|------------------|---------------|
| `K_max` (via `_DATASET_K_MAX`) | 16 (kinship), 32 (wn18rr), 16 (FB15k237) | Peak expansion $\propto K_{\max}^{F_{\max\text{-vars}}}$ (capped by `_G_BUDGET=128`) | Set per-dataset based on max base degree. Lower = less memory + fewer groundings. |
| `max_states` (`_MAX_STATES`) | 64 | Compound state buffer $\propto S_{\max}$. Expansion tensor $\propto \min(S_{\max}, 32)$. | Rarely needs tuning. 64 is ample for most KGs. |
| `max_total_groundings` | 128 | Output buffer size $\propto tG$ | Increase if queries need many groundings for training signal. |
| `num_steps` | 2 | $\text{max\_gps} = 1 + (m{-}1) \cdot D$ → compound state width. Also number of depth iterations. | 2 is sufficient for most datasets (groundings plateau). |
| `_TABLING_SIZE` | 512 | Tabling cache. Collision rate ~ $\text{unique\_proved} / T$. | 512 is safe; increase for very rule-dense KGs. |
| `_G_BUDGET` | 128 | Caps cross-product for multi-free-var rules. | Only matters for 3+-body rules with $\geq 2$ free vars. |

**Memory is fully predictable.** Given the hyperparameters above, peak GPU
memory can be computed exactly before training starts.

### 5.2 Prolog BFS — Data-Dependent Frontier Growth

The Prolog engine's memory is bounded by caps but the **actual** usage is
data-dependent.

| Buffer | Shape | Formula (bytes) | Example: kinship_family |
|--------|-------|------------------|------------------------|
| Frontier states | $[F_d, A, 3]$ long | $F_d \cdot A \cdot 24$ | $2{,}000{,}000 \times 20 \times 24 = 960\text{ MB}$ (worst) |
| Frontier owner | $[F_d]$ long | $F_d \cdot 8$ | $16\text{ MB}$ (worst) |
| Frontier vars | $[F_d]$ long | $F_d \cdot 8$ | $16\text{ MB}$ (worst) |
| Visited hashes | $[H]$ long | $H \cdot 8$ | Grows each depth |
| **Engine per-chunk** | $[B, K_{\max}, A, 3]$ long | $B \cdot K_{\max} \cdot A \cdot 24$ | $2{,}048 \times 64 \times 20 \times 24 = 60\text{ MB}$ |

**Hyperparameters that must be set in advance:**

| Param | Default | Effect on memory | How to choose |
|-------|---------|------------------|---------------|
| `max_states` (`_PROLOG_MAX_STATES`) | 128 | Cap on BFS states per batch element. | Rarely needs tuning. |
| `max_fact_candidates` (`_PROLOG_MAX_FACT_CANDIDATES`) | 64 | Cap on fact unification candidates per goal. | Increase for high-degree predicates. |
| `max_derived_per_state` ($K_{\max}$) | 64 (all datasets via `_PROLOG_DATASET_K_MAX`) | Engine output shape per chunk. Per-chunk memory $\propto B \cdot K_{\max} \cdot A$. | Higher = more complete resolution. |
| `batch_size` | 2,048 (or auto-computed) | Chunk size for engine calls. Auto-capped: $B = \min(\text{batch\_size},\, \lfloor 4\text{GB} / (K_{\max} \cdot A \cdot 24 \cdot 6) \rfloor)$. | Usually auto. Override for memory-constrained GPUs. |
| `padding_atoms` ($A_{\text{base}}$) | 20 | Base state width. $A = A_{\text{base}} + \max\_rule\_body \times 10$. | Increase for deeply nested rules. Default 20 is safe for 2-body rules. |
| `max_depth` | 5 | Number of BFS iterations. More depths = larger visited set + more frontier growth. | 5 is generous; most KGs plateau at depth 2–3. |

**Memory is NOT fully predictable.** The frontier can grow from $N$ (initial)
to $F_{\max}$ (cap) depending on dataset branching. The auto batch-size
computation prevents per-chunk OOM, but frontier growth can still exhaust
memory if $F_{\max}$ is set too high.

### 5.3 Safe Memory Budget Formula

**BCStaticSLD:**
$$\text{peak\_MB} \approx \underbrace{\frac{P \cdot E \cdot K_{\max} \cdot 9}{10^6}}_{\text{fact index}} + \underbrace{\frac{B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m \cdot 24}{10^6}}_{\text{expansion tensor}} + \underbrace{\frac{B \cdot S_{\max} \cdot \text{gps} \cdot 24}{10^6}}_{\text{compound states}}$$

**Prolog BFS:**
$$\text{peak\_MB} \approx \underbrace{\frac{F_{\max} \cdot A \cdot 24}{10^6}}_{\text{frontier (worst case)}} + \underbrace{\frac{B \cdot K_{\max} \cdot A \cdot 24}{10^6}}_{\text{engine per-chunk}} + \underbrace{\frac{H \cdot 8}{10^6}}_{\text{visited set}}$$

---

## 6. Side-by-Side Comparison

### 6.1 Algorithmic Differences

| Property | BCStaticSLD | Prolog Batched BFS |
|----------|-------------|-------------------|
| **Goal** | Produce groundings (body atom triples) for SBR training | Determine provability (return proof depth or -1) |
| **Resolution** | SLD with compound state tracking | SLD with standard Prolog selection (leftmost atom) |
| **Frontier model** | Fixed `[B, S_max, max_gps, 3]` — constant shape | Dynamic `[F_d, A, 3]` — grows/shrinks per depth |
| **Compilation** | `torch.compile(fullgraph=True)` + CUDA graph caching | `torch.compile` on standardization; rest is eager |
| **Deduplication** | XOR compound hash (order-independent) | Position-weighted hash + sort + adjacent-diff |
| **Tabling** | Direct-mapped hash table (`T=512`), proved subgoals cached and reused | None — revisits previously proved goals |
| **Fact pruning** | Body atoms checked via `fact_index.exists()` (O(1) hash) | Ground facts pruned from states via `searchsorted` on `fact_hashes` |
| **Variable handling** | No variables — all atoms are fully ground (entities only) | Full variable standardization (offset-based renaming) |
| **Cross-depth state** | Compound states carry full goal sets + tabling cache | Frontier states carry remaining atoms; visited set prevents re-expansion |

### 6.2 Complexity Comparison

| Metric | BCStaticSLD | Prolog Batched BFS |
|--------|-------------|-------------------|
| **Init time** | $O(F \log F + R \cdot m^2)$ | $O(F \log F + R \cdot m)$ |
| **Time per depth** | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ | $O(F_d \cdot K_{\max} \cdot A + F_{d+1} \log F_{d+1})$ |
| **Total time** | $O(D \cdot B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ | $O(\sum_d F_d \cdot K_{\max} \cdot A)$ |
| **Peak space** | $O(B \cdot G_p \cdot R_{\text{eff}} \cdot G_t \cdot m)$ — **fixed** | $O(F_{\max} \cdot A + B \cdot K_{\max} \cdot A)$ — **data-dependent** |
| **Space predictability** | Fully predictable from hyperparams | Only upper-bounded; actual varies by dataset |

### 6.3 Hyperparameter Sensitivity

| Hyperparameter | BCStaticSLD impact | Prolog BFS impact |
|---|---|---|
| **Depth ($D$)** | Linear in time ($D$ iterations). `max_gps` grows: $1+(m{-}1) \cdot D$. Memory grows mildly. | Can cause frontier explosion: $F_d \propto K^d$ before capping. Visited set grows monotonically. |
| **Branching ($K$, $K_{\max}$)** | $G_t$ grows as $K_{\max}^{F_{\text{vars}}}$ but hard-capped at `_G_BUDGET=128`. Peak tensor ∝ $G_t$. | $K_{\max}$ directly controls engine output width. Frontier growth ∝ $K_{\max}$ per depth. |
| **Batch size ($B$)** | Peak tensor ∝ $B$. Fully under user control. | Engine tensor ∝ $B$, but $B$ is auto-capped. Frontier size independent of $B$. |
| **Rules ($R$)** | $R_{\text{eff}}$ enters peak tensor. More rules = wider expansion. | More rules = more derived states per resolution = faster frontier growth. |
| **Entities ($E$)** | Fact index ∝ $P \cdot E \cdot K_{\max}$ (dense). Dominates for large $E$. | Fact storage ∝ $F$ (sparse). Large $E$ only matters if $F$ is large. |

### 6.4 Soundness and Completeness

| Property | BCStaticSLD | Prolog Batched BFS |
|----------|-------------|-------------------|
| **Sound** | Yes — compound states ensure consistent substitutions | Yes — standard SLD resolution |
| **Complete at depth $D$** | No — $K_{\max}$ capping + $S_{\max}$ state limit can miss proofs | No — $K_{\max}$ per-state + $F_{\max}$ frontier + $Q_{\max}$ per-query caps can miss proofs |
| **Tabling benefit** | Yes — proved subgoals cached, avoids re-expansion. +4% overhead, significant completeness gain. | No tabling — may re-expand proved subgoals if they appear in different contexts. |

---

## 7. When Each System Hits Its Memory Ceiling

### BCStaticSLD — Dense Fact Index Is the Bottleneck

The fact index is allocated as a **dense** tensor `[P, E, K_max]` regardless
of sparsity. For large KGs:

| Dataset | $P$ | $E$ | $K_{\max}$ | Fact index (MB) | Density ($F / P \cdot E \cdot K_{\max}$) |
|---------|-----|-----|------------|-----------------|------------------------------------------|
| kinship_family | 12 | 2,968 | 16 | 4.6 | 3.5% |
| wn18rr | 11 | 40,559 | 32 | 115 | 0.6% |
| FB15k237 | 237 | 14,505 | 16 | 440 | 0.5% |
| Hypothetical | 500 | 100,000 | 32 | 12,800 | — |

For the hypothetical large KG, the fact index alone is 12.8 GB — close to
the limit of a 16 GB GPU. **The dense fact index is the first thing to
exceed memory on large KGs.**

**Mitigation**: Lower `K_max` (reduces completeness) or switch to
BCDynamicProvset (sparse dicts, no dense allocation).

### Prolog BFS — Frontier Growth Is the Bottleneck

The frontier grows dynamically. For a KG with high average branching $K$:

| Depth | Frontier (no cap) | Frontier (capped at $F_{\max}=2\text{M}$) | Memory |
|-------|-------------------|-------------------------------------------|--------|
| 1 | $N$ | $N$ | $N \cdot A \cdot 24$ |
| 2 | $N \cdot K_{\max}$ | $\min(N \cdot K_{\max},\, 2\text{M})$ | $\leq 960\text{ MB}$ |
| 3 | $N \cdot K_{\max}^2$ | $\min(N \cdot K_{\max}^2,\, 2\text{M})$ | $\leq 960\text{ MB}$ |

With $N=1000$, $K_{\max}=200$, $A=20$: depth 2 already yields 200K states
(96 MB). Depth 3 could hit the 2M cap (960 MB). The **visited set** also
grows: after 3 depths it can hold millions of hashes.

**Mitigation**: Lower `max_frontier`, lower `max_per_query`, or lower
`max_depth`. All reduce completeness.

### Summary: Which Runs Out of Memory First?

| Scenario | BCStaticSLD | Prolog BFS |
|----------|-------------|-----------|
| **Many entities, sparse facts** | Dense fact index wastes memory on empty slots | Efficient — only stores actual states |
| **High branching factor** | Capped at compile time ($G_t \leq 128$) — safe but incomplete | Frontier can explode to $F_{\max}$; engine buffer $\propto K_{\max}$ per chunk |
| **Deep proofs ($D \geq 4$)** | `max_gps` grows, compound states widen; mild | Frontier and visited set grow per depth; can be significant |
| **Large batch ($B$)** | Peak tensor $\propto B$; set $B$ to fit | Frontier $\propto N$ (all queries); engine auto-caps $B$ |
| **Multi-free-var rules** | $G_t$ grows as $K_{\max}^F$ but capped at 128 | Each free var multiplies derived states; no special cap |

**BCStaticSLD is memory-predictable but wastes space on sparse KGs.**
**Prolog BFS is memory-adaptive but can surprise you with frontier growth.**

For production training (fixed batch, repeated forward passes), BCStaticSLD's
predictability is preferred. For one-shot provability analysis over many
queries, Prolog BFS's adaptive frontier is more flexible.

---

## 8. Prolog in RL — Interactive Proof Search

Source: external `Batched_env-swarm` project (`env/env.py`, `EnvVec`) and
its components. Note: the RL env uses the same unification engine but with
different defaults (e.g. `max_derived_per_state=200`) than the torch-ns
BCPrologStatic grounder (`_PROLOG_DATASET_K_MAX=64`).

The RL environment wraps the same `UnificationEngineVectorized` as Prolog
Batched BFS (Section 3), but uses it **one step at a time**: an RL agent
selects which derived state to follow, rather than BFS exploring all
frontiers simultaneously. This fundamentally changes the memory model.

### 8.1 Architecture Overview

```
            ┌──────────────────────────────────────────────────┐
            │                  EnvVec (B envs)                 │
            │                                                  │
            │  current_states [B, A, 3]    ← agent's position │
            │         │                                        │
            │         ▼                                        │
            │  UnificationEngine.get_derived_states_compiled   │
            │         │                                        │
            │         ▼                                        │
            │  derived_states [B, S, A, 3] ← action space     │
            │         │                                        │
            │         ▼ (agent picks action index)             │
            │  new current = derived[b, action_b]              │
            │         │                                        │
            │         ▼                                        │
            │  reward + done check                             │
            │  history update (visited pruning)                │
            │  if done: auto-reset with new query              │
            └──────────────────────────────────────────────────┘
```

Unlike BFS (which expands ALL derived states at every depth), the RL env
expands only the **single state chosen by the agent**. The agent sees the
full action space `[B, S, A, 3]` but commits to one action per step.

### 8.2 Additional Notation

| Symbol | Meaning |
|--------|---------|
| $B$ | `batch_size` — number of parallel environments (default 100) |
| $A$ | `padding_atoms` — max atoms per state (default 6) |
| $S$ | `padding_states` — max derived states / actions per step (default 120) |
| $D_{\max}$ | `max_depth` — max steps per episode before truncation (default 20) |
| $H$ | `max_history_size` = $D_{\max} + 1$ — visited state buffer depth |
| $K_{\max}^{\text{eng}}$ | `max_derived_per_state` in engine (default 200) |
| $A^{\text{eng}}$ | Engine's `M_max` = `padding_atoms + max_rule_body * 10` |

### 8.3 Step-by-Step

#### Step 0 — Initialization (once)

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 0a. Engine setup | Same as Prolog BFS Step 0 (fact index, rule compilation, segment lookups) | $O(F \log F + R \cdot m)$ | $O(F + R \cdot A^{\text{eng}})$ |
| 0b. Allocate fixed buffers | Pre-allocate ALL environment tensors at their final shapes. No dynamic allocation during episodes. | $O(1)$ | See buffer table below |
| 0c. Build index tensors | `_arange_B [B]`, `_arange_S [S]`, `_arange_A [A]`, `_positions_S [1, S]`, `_ones_B [B]`, `_compact_zeros [B, S, A*3]` | $O(B + S + A)$ | $O(B \cdot S \cdot A)$ |
| 0d. Build special states | `end_state [A, 3]` (end-proof action), `_false_state_base [S, A, 3]` (fallback for zero-action states) | $O(S \cdot A)$ | $O(S \cdot A)$ |

#### Step 1 — Reset (per episode start)

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 1a. Sample queries | Draw $B$ queries from pool (uniform, weighted, or sequential). Apply negative sampling: some fraction become corrupted negatives. | $O(B)$ | $O(B)$ |
| 1b. Create initial states | Pad query into `[B, A, 3]`. Compute initial hash. | $O(B \cdot A)$ | $O(B \cdot A)$ |
| 1c. Compute initial derived | Call engine on initial states → raw `[B, K_{\max}^{\text{eng}}, A^{\text{eng}}, 3]`, truncate/pad to `[B, S, A, 3]`. Apply visited pruning + validity check + compaction. | $O(B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ | $O(B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ (engine) + $O(B \cdot S \cdot A)$ (env) |
| 1d. Skip-unary advance | If `skip_unary_actions=True`: auto-advance through states with only 1 non-terminal action (up to `_MAX_UNARY_ITERATIONS=2` times). Each iteration re-calls the engine. | $O(2 \cdot B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ | Same as 1c (reused) |
| 1e. Build TensorDict state | Pack all buffers into `TensorDict` with 20+ keys (current, derived, counts, queries, vars, depths, done, success, labels, history, rewards, pointers, ...) | $O(B)$ | See buffer table |

#### Step 2 — Agent Step (repeated up to $D_{\max}$ times)

| Sub-step | What | Time | Space |
|----------|------|------|-------|
| 2a. Action selection | Agent provides `actions [B]` — index into `derived_states`. Gather: `next_state = derived[b, action_b]`. | $O(B \cdot A)$ | $O(B \cdot A)$ |
| 2b. Reward + termination | Check for proof (all atoms = True), contradiction (any False), end-action, or depth exceeded. Dispatch to reward function (7 types available). | $O(B \cdot A)$ | $O(B)$ |
| 2c. History update | Hash new state, append to `history_hashes [B, H]`. For exact memory: append full state to `_state_history [B, H, A, 3]`. | $O(B \cdot A)$ | $O(1)$ (in-place scatter) |
| 2d. Compute derived | **Same as Step 1c** — engine call on new `current_states`, then visited pruning + compaction. This is the **dominant cost per step**. | $O(B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ | $O(B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ |
| 2e. Visited pruning | Hash-based: compare `derived_states` hashes against `history_hashes`. Match → mark invalid → compact. For exact memory: full state-by-state comparison via `_state_history`. | Hash: $O(B \cdot S \cdot H)$. Exact: $O(B \cdot S \cdot H \cdot A)$. | $O(B \cdot S \cdot H)$ |
| 2f. Skip-unary advance | Same as Step 1d (conditional). | $O(2 \cdot B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ | Reused |
| 2g. Done environments | For `step_and_reset`: done envs are immediately reset with new queries (fused step + reset). Fresh engine call for reset envs. | $O(B_{\text{done}} \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}})$ | Reused |

### 8.4 Memory Layout — All Buffers

Every buffer is allocated at init and never resized. This is the **complete**
memory footprint.

#### Core state buffers (per-env)

| Buffer | Shape | Bytes | Example ($B{=}100$, $S{=}120$, $A{=}6$) |
|--------|-------|-------|------------------------------------------|
| `current_states` | $[B, A, 3]$ | $B \cdot A \cdot 3 \cdot 8$ | 14.4 KB |
| `derived_states` | $[B, S, A, 3]$ | $B \cdot S \cdot A \cdot 3 \cdot 8$ | **1.73 MB** |
| `derived_counts` | $[B]$ | $B \cdot 8$ | 0.8 KB |
| `original_queries` | $[B, A, 3]$ | $B \cdot A \cdot 3 \cdot 8$ | 14.4 KB |
| `next_var_indices` | $[B]$ | $B \cdot 8$ | 0.8 KB |

#### History / memory buffers

| Buffer | Shape | Bytes | Example |
|--------|-------|-------|---------|
| `history_hashes` | $[B, H]$ | $B \cdot H \cdot 8$ | 16.8 KB ($H{=}21$) |
| `_state_history` (exact) | $[B, H, A, 3]$ | $B \cdot H \cdot A \cdot 3 \cdot 8$ | 302 KB |
| `_state_history_count` | $[B]$ | $B \cdot 8$ | 0.8 KB |

#### Helper / index buffers

| Buffer | Shape | Bytes | Example |
|--------|-------|-------|---------|
| `_compact_zeros` | $[B, S, A \cdot 3]$ | $B \cdot S \cdot A \cdot 3 \cdot 8$ | **1.73 MB** |
| `_false_state_base` | $[S, A, 3]$ | $S \cdot A \cdot 3 \cdot 8$ | 17.3 KB |
| `_arange_B`, `_ones_B`, etc. | $[B]$ each | $B \cdot 8$ each | 0.8 KB each |
| `_positions_S`, `_arange_S` | $[S]$ each | $S \cdot 8$ each | 1 KB each |

#### Scalar / metadata buffers

| Buffer | Shape | Bytes | Example |
|--------|-------|-------|---------|
| `depths`, `done`, `success` | $[B]$ each | $B \cdot 8$ each | 0.8 KB each |
| `current_labels`, `step_rewards` | $[B]$ each | $B \cdot 4{-}8$ each | 0.4–0.8 KB each |
| `per_env_ptrs`, `neg_counters` | $[B]$ each | $B \cdot 8$ each | 0.8 KB each |
| Pre-allocated reward scalars | 10 × scalar | 80 B | 80 B |

#### Engine internal (transient per step)

| Buffer | Shape | Bytes | Example |
|--------|-------|-------|---------|
| `derived_raw` (engine output) | $[B, K_{\max}^{\text{eng}}, A^{\text{eng}}, 3]$ | $B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}} \cdot 3 \cdot 8$ | **28.8 MB** ($K{=}200$, $A^{\text{eng}}{=}60$) |
| Variable standardization temps | $\approx 2 \times$ engine output | $\approx 2 \times 28.8$ | **57.6 MB** |

#### Total env memory (example: $B{=}100$, $S{=}120$, $A{=}6$, $K_{\max}^{\text{eng}}{=}200$)

| Category | MB |
|----------|----|
| Env buffers (state + history + helpers) | ~4 |
| Engine persistent (facts, rules, indices) | ~2–30 (dataset-dependent) |
| Engine transient per step | ~90 |
| **Total** | **~100–125 MB** |

### 8.5 Memory Limits and Pre-set Hyperparameters

| Param | Default | Memory effect | How to choose |
|-------|---------|---------------|---------------|
| `batch_size` ($B$) | 100 | All env buffers $\propto B$. Engine transient $\propto B$. | Scale to fill GPU. Typical: 100–2048. |
| `padding_states` ($S$) | 120 | `derived_states` + `_compact_zeros` $\propto B \cdot S \cdot A$. This is the **action space width**. | Must be $\geq K_{\max}^{\text{eng}}$ truncated output + 1 (end action). 120 is generous; most states have $<50$ valid derived. |
| `padding_atoms` ($A$) | 6 | All state tensors $\propto A$. Low default keeps env small. | 6 is tight for 2-body rules at depth $>3$. Increase for deep proofs or many-body rules. States wider than $A$ are silently truncated. |
| `max_depth` ($D_{\max}$) | 20 | History buffer $\propto D_{\max}$. Episode length cap. | 20 is generous for most KGs. Lower for faster rollouts; higher for deep proofs. |
| `max_derived_per_state` ($K_{\max}^{\text{eng}}$) | 200 | **Dominant**: engine transient $\propto B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}}$. | 200 is safe. Lower if GPU-constrained. Engine auto-caps batch size if $B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}} \cdot 24 \cdot 6 > 4\text{ GB}$. |
| `memory_pruning` | True | Adds hash comparison cost $O(B \cdot S \cdot H)$ per step. | Always on for training. Prevents cycles. |
| `use_exact_memory` | False | Adds `_state_history [B, H, A, 3]` buffer. Pruning becomes $O(B \cdot S \cdot H \cdot A)$. | Only for correctness debugging. Hash-based is sufficient in practice. |
| `skip_unary_actions` | False | Up to `_MAX_UNARY_ITERATIONS=2` extra engine calls per step (for states with exactly 1 non-terminal action). | Reduces effective episode length. Costs 2× engine time when triggered. |
| `end_proof_action` | True | Adds 1 extra action slot per step (the "I give up" action). | Always on for negative query handling. |
| `negative_ratio` | 1.0 | No direct memory effect. Controls pos/neg query mix. | Affects training signal, not memory. |

### 8.6 Key Differences from Prolog BFS

| Property | Prolog BFS (Section 3) | Prolog in RL (this section) |
|----------|------------------------|----------------------------|
| **Expansion per depth** | ALL frontier states expanded | Only 1 state per env expanded (agent's choice) |
| **Frontier size** | Grows: $F_d \rightarrow F_d \cdot K_{\max}$ (data-dependent) | Fixed: always $B$ states (one per env) |
| **Memory model** | Data-dependent frontier + visited set | **Fully static** — all buffers pre-allocated |
| **Memory predictability** | Upper-bounded by $F_{\max}$; actual varies | **Exact** — computable from hyperparams before training |
| **Completeness** | BFS-complete up to caps | Agent-dependent — agent may choose suboptimal paths |
| **Variable handling** | Offset-based standardization | Same engine, same standardization |
| **Deduplication** | Within/cross-depth hash + sort | Per-env hash history (ring buffer of $H$ hashes) |
| **Output** | Proof depth (provable/unprovable) | Reward signal (success/failure/truncation) |
| **Bottleneck** | Frontier growth | Engine call per step ($B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}}$) |

### 8.7 Safe Memory Budget Formula

$$\text{peak\_MB} \approx \underbrace{\frac{B \cdot S \cdot A \cdot 24 \cdot 2}{10^6}}_{\text{env buffers (derived + compact)}} + \underbrace{\frac{B \cdot K_{\max}^{\text{eng}} \cdot A^{\text{eng}} \cdot 24 \cdot 3}{10^6}}_{\text{engine transient (output + temps)}} + \underbrace{\frac{\text{facts} + \text{rules}}{10^6}}_{\text{engine persistent}}$$

The $\times 3$ factor on engine transient accounts for the output tensor plus
variable standardization intermediates.

#### Example budgets

| Config | Env buffers | Engine transient | Engine persistent | **Total** |
|--------|-------------|-----------------|-------------------|-----------|
| $B{=}100$, $S{=}120$, $A{=}6$, $K{=}200$, $A^e{=}60$, kinship | 3.5 MB | 86 MB | ~4 MB | **~94 MB** |
| $B{=}512$, $S{=}120$, $A{=}6$, $K{=}200$, $A^e{=}60$, kinship | 18 MB | 442 MB | ~4 MB | **~464 MB** |
| $B{=}2048$, $S{=}120$, $A{=}6$, $K{=}200$, $A^e{=}60$, kinship | 71 MB | 1,769 MB | ~4 MB | **~1.8 GB** |
| $B{=}100$, $S{=}120$, $A{=}6$, $K{=}200$, $A^e{=}60$, FB15k237 | 3.5 MB | 86 MB | ~30 MB | **~120 MB** |

**The engine transient tensor dominates.** For large batch sizes, $K_{\max}^{\text{eng}}$
is the main lever to reduce memory.

### 8.8 When the RL Env Hits Its Memory Ceiling

Unlike Prolog BFS, the RL env has **no data-dependent memory growth** —
all buffers are static. The ceiling is hit when the pre-allocated buffers
exceed GPU memory:

1. **Engine transient is the bottleneck.** At $B{=}2048$, $K_{\max}^{\text{eng}}{=}200$,
   $A^{\text{eng}}{=}60$: engine output alone is 1.77 GB, plus ~2× for
   standardization temps → **~5.3 GB** just for one engine call.

2. **$A^{\text{eng}}$ is not directly configurable.** It is computed as
   `padding_atoms + max_rule_body * 10`. For 2-body rules: $A^{\text{eng}} = 6 + 2 \times 10 = 26$.
   For 5-body rules: $A^{\text{eng}} = 6 + 5 \times 10 = 56$. Many-body
   rules inflate the engine tensor without user control.

3. **$S$ (padding_states) is usually not the bottleneck** because $S < K_{\max}^{\text{eng}}$
   and env buffers use the smaller $A$ (not $A^{\text{eng}}$).

4. **Exact memory mode doubles history cost** but history is small
   relative to the engine tensor (302 KB vs 86 MB at $B{=}100$).

**Mitigation strategies:**

| Strategy | Effect | Tradeoff |
|----------|--------|----------|
| Lower $B$ | All buffers $\propto B$ | Fewer parallel envs → lower throughput |
| Lower $K_{\max}^{\text{eng}}$ | Engine transient $\propto K_{\max}^{\text{eng}}$ | Fewer actions → may miss proof paths |
| Lower $S$ | Env action buffer $\propto S$ | Minor savings (env buffers are small relative to engine) |
| Lower $A$ | All state tensors $\propto A$ | States truncated → deep proofs may fail |
| Disable exact memory | Saves `_state_history [B, H, A, 3]` | Negligible saving; hash-based pruning is sufficient |

---

## 9. Prolog: 1-Step vs N-Steps

The RL environment (Section 8) calls the Prolog engine **once per agent
step** — the agent picks one successor state, and the engine expands
only that state next. BCStaticSLD and Prolog BFS call the engine
**N times in a loop**, expanding all surviving states at each depth.

This section analyzes the memory consequences of 1-step vs N-step
execution, in both compiled (`torch.compile fullgraph`) and dynamic
(eager Python) modes.

### 9.1 Compiled Prolog: 1-Step vs N-Steps

#### 1-step compiled (= RL env with torch.compile)

Each forward call expands exactly $B$ states. The engine output tensor
has a fixed shape:

```
engine_output: [B, K_max, A, 3]          ← same shape every call
```

**Memory per call:**
$$M_1 = B \cdot K_{\max} \cdot A \cdot 24 \cdot c$$

where $c \approx 3$ accounts for the output tensor plus standardization
intermediates.

This shape never changes across the episode. CUDA graph capture works
perfectly: capture once at warmup, replay for every subsequent step.
The memory is identical at step 1 and step 20.

#### N-step compiled (= BCStaticSLD-like loop)

To run N depths in a single compiled forward pass, we need **all
intermediate tensors to have compile-time-constant shapes**. This
forces us to introduce caps:

- **Frontier cap** $S_{\max}$: maximum states carried between depths
  (BCStaticSLD uses 64).
- **Expansion cap** $G_p$: goals expanded per depth iteration
  ($\min(S_{\max}, 32)$ in BCStaticSLD).

The depth loop then looks like:

```python
for depth in range(N):                     # Python loop (unrolled by compiler)
    expand [B, G_p] goals                  # fixed shape
    → expansion tensor [B, G_p, K, A, 3]  # fixed shape
    select top S_max states via topk       # fixed shape
    dedup via sort + adjacent-diff         # fixed shape
```

**Memory per depth (inside the loop):**
$$M_d = B \cdot G_p \cdot K_{\max} \cdot A \cdot 24 \cdot c$$

**Total persistent state (carried across depths):**
$$M_{\text{state}} = B \cdot S_{\max} \cdot A_{\text{state}} \cdot 24$$

where $A_{\text{state}}$ is the compound-state width (for Prolog-style
flat states: $A$; for BCStaticSLD compound states: `max_gps`).

**Comparison — compiled 1-step vs compiled N-step:**

| | 1-step (RL) | N-step (BCStaticSLD-like) |
|---|---|---|
| **Engine call shape** | `[B, K_max, A, 3]` | `[B, G_p, K_max, A, 3]` — **G_p× wider** |
| **Calls per query** | $D_{\max}$ separate calls (one per agent step) | 1 call with $N$ internal depth iterations |
| **State between depths** | `[B, A, 3]` (one state per env) | `[B, S_max, A, 3]` — **S_max× wider** |
| **Peak memory** | $B \cdot K_{\max} \cdot A \cdot 24 \cdot c$ | $B \cdot G_p \cdot K_{\max} \cdot A \cdot 24 \cdot c + B \cdot S_{\max} \cdot A \cdot 24$ |
| **Ratio** | 1× | $G_p + S_{\max} \cdot A / (K_{\max} \cdot A \cdot c) \approx G_p \times$ |

The N-step compiled version is roughly **$G_p$ times more expensive**
per forward pass because it expands $G_p$ goals simultaneously instead
of 1. But it also completes $N$ depths in a single CUDA-graph-cached
call, where the 1-step version requires $D_{\max}$ separate engine
invocations with Python overhead between them.

**Concrete numbers at $B=512$, $K_{\max}=200$, $A=40$ (FB15k237):**

| | 1-step | N-step ($G_p=32$, $S_{\max}=64$) |
|---|---|---|
| Engine expansion tensor | $512 \times 200 \times 40 \times 24 \times 3 = 2.9\text{ GB}$ | $512 \times 32 \times 200 \times 40 \times 24 \times 3 = 94\text{ GB}$ |

94 GB is clearly impossible — which is exactly why BCStaticSLD uses
**cascaded enumeration instead of MGU** and uses a **much smaller $K_{\max}$**
(16 instead of 200). The cascade trades $K_{\max}$ for $R_{\text{eff}} \times G_t$
with smaller constants:

| | 1-step MGU | N-step cascade (BCStaticSLD actual) |
|---|---|---|
| Expansion tensor | $512 \times 200 \times 40 \times 24 \times 3 = 2.9\text{ GB}$ | $512 \times 32 \times 5 \times 16 \times 2 \times 24 = 125\text{ MB}$ |

The N-step cascade is **23× cheaper** than a single MGU step because:
$R_{\text{eff}} \times K_{\max}^{\text{cascade}} \times M = 5 \times 16 \times 2 = 160$
vs $K_{\max}^{\text{MGU}} \times A = 200 \times 40 = 8{,}000$ (50× fewer elements per goal).

> **Key insight:** A compiled N-step Prolog with MGU would need to
> dramatically lower $K_{\max}$ (e.g. from 200 to ~10) to fit in GPU
> memory. This would severely limit completeness on high-degree KGs.
> BCStaticSLD avoids this by using cascaded enumeration, which has a
> naturally smaller per-candidate footprint ($M=2$ body atoms instead
> of $A=40$ atoms per state).

### 9.2 Dynamic Prolog: 1-Step vs N-Steps

In eager (non-compiled) mode, tensor shapes can be data-dependent.
No CUDA graph caching, no fullgraph requirement. The frontier can
grow freely.

#### 1-step dynamic (= RL env, eager mode)

Same as compiled: expand $B$ states, get `[B, K_max, A, 3]`, agent
picks one. Memory is constant.

$$M_1 = B \cdot K_{\max} \cdot A \cdot 24 \cdot c$$

No frontier to manage — one state per env, always.

#### N-step dynamic (= Prolog BFS, eager mode)

The frontier is a Python list of states that grows between depths.
No cap is enforced by the engine — caps are applied post-hoc
(`max_frontier`, `max_per_query`).

At each depth $d$, the frontier $F_d$ is processed in chunks of $B$:

```
depth 1:  expand N queries     → F_1 states
depth 2:  expand F_1 states    → F_2 ≤ F_1 × K_max states
depth 3:  expand F_2 states    → F_3 ≤ F_2 × K_max states
...
```

**Uncapped growth (worst case):**
$$F_d = N \cdot K_{\max}^{d-1}$$

**Capped growth:**
$$F_d = \min\!\left(N \cdot K_{\max}^{d-1},\; F_{\max},\; N \cdot Q_{\max}\right)$$

**Memory at depth $d$:**
$$M_d = \underbrace{F_d \cdot A \cdot 24}_{\text{frontier}} + \underbrace{B \cdot K_{\max} \cdot A \cdot 24 \cdot c}_{\text{engine per-chunk}} + \underbrace{H_d \cdot 8}_{\text{visited hashes}}$$

**Comparison — dynamic 1-step vs dynamic N-step:**

| | 1-step (RL) | N-step (Prolog BFS) |
|---|---|---|
| **Frontier at depth $d$** | $B$ (always) | $\min(N \cdot K_{\max}^{d-1},\, F_{\max})$ |
| **Engine call** | `[B, K_max, A, 3]` | Same per chunk, but $\lceil F_d / B \rceil$ chunks |
| **Visited set** | `[B, H]` hashes (ring buffer, $H = D_{\max}+1$) | $[H_d]$ sorted hashes (grows monotonically) |
| **Peak memory** | $B \cdot K_{\max} \cdot A \cdot 24 \cdot c$ (constant) | $F_{\max} \cdot A \cdot 24 + B \cdot K_{\max} \cdot A \cdot 24 \cdot c$ (frontier-dominated) |
| **Depth dependence** | None | Exponential until capped |

### 9.3 All Four Variants on FB15k237

**Dataset:** E=14,505, P=237, F=272,115, R=199, max_degree=3,612.

Let $B=512$, $K_{\max}^{\text{MGU}}=200$, $A=40$, $N=512$ queries.

| Variant | Expansion tensor | Frontier/state buffer | Total peak | Depth dependent? |
|---------|-----------------|----------------------|------------|-----------------|
| **1-step compiled** (RL) | $512 \times 200 \times 40 \times 72 = 2.9\text{ GB}$ | `[512, 40, 3]` = 0.05 MB | **~2.9 GB** | No |
| **1-step dynamic** (RL eager) | Same: 2.9 GB | Same: 0.05 MB | **~2.9 GB** | No |
| **N-step compiled** (hypothetical) | $512 \times 32 \times 200 \times 40 \times 72 = 94\text{ GB}$ | `[512, 64, 40, 3]` = 38 MB | **~94 GB** (OOM) | Per-depth (fixed shape) |
| **N-step dynamic** (Prolog BFS) | $512 \times 200 \times 40 \times 72 = 2.9\text{ GB}$ /chunk | frontier: up to $2\text{M} \times 40 \times 24 = 1.9\text{ GB}$ | **~4.8 GB** at depth 3+ | Yes (frontier grows) |

**With BCStaticSLD's cascade instead of MGU** ($R_{\text{eff}}=5$,
$K_{\max}=16$, $M=2$, $G_p=32$):

| Variant | Expansion tensor | Frontier/state buffer | Total peak |
|---------|-----------------|----------------------|------------|
| **N-step compiled** (BCStaticSLD actual) | $512 \times 32 \times 5 \times 16 \times 2 \times 24 = 125\text{ MB}$ | `[512, 64, 6, 3]` = 4.7 MB | **~570 MB** (incl. fact index) |

### 9.4 Where 1-Step (RL) Works but N-Step Fails

The question: *what kind of dataset makes N-step BFS run out of memory
while 1-step RL still fits comfortably?*

The answer comes directly from the frontier growth formula:
$$F_d = \min\!\left(N \cdot K^{d-1},\; F_{\max}\right)$$

**1-step RL is immune to all three factors** ($N$, $K$, $d$) because it
never builds a frontier. N-step BFS is sensitive to all three.

#### Factor 1 — High branching factor ($K$)

Datasets where many entities share the same predicate have high $K$.
The frontier multiplies by $K$ at each depth.

| Dataset | Avg $K$ | Max $K$ | $N \cdot K$ (depth 2 frontier) | Fits in 24 GB? (1-step) | Fits in 24 GB? (N-step BFS) |
|---------|---------|---------|-------------------------------|------------------------|----------------------------|
| kinship_family | 5 | 28 | $512 \times 5 = 2.6\text{K}$ | Yes (2.9 GB) | Yes (frontier 0.1 MB) |
| wn18rr | 4 | 442 | $512 \times 200 = 102\text{K}$ | Yes (2.9 GB) | Yes (frontier 98 MB) |
| FB15k237 | 19 | 3,612 | $512 \times 200 = 102\text{K}$ | Yes (2.9 GB) | Yes (frontier 98 MB) |
| **Social network** | 500 | 5,000 | $512 \times 200 = 102\text{K}$ | Yes (2.9 GB) | Marginal (hits $F_{\max}$ at depth 2) |
| **Citation graph** | 50 | 50,000 | $512 \times 200 = 102\text{K}$ | Yes (2.9 GB) | Marginal |

With $K_{\max}=200$, even moderate branching fills the frontier quickly.
The $F_{\max}$ cap prevents OOM but sacrifices completeness.

#### Factor 2 — Many simultaneous queries ($N$)

The frontier size is $N \times$ branching. More queries = larger frontier.

| $N$ (queries) | Frontier at depth 2 ($K=200$) | Frontier memory ($A=40$) | 1-step RL |
|---------------|-------------------------------|--------------------------|-----------|
| 512 | 102,400 | 98 MB | 2.9 GB (same always) |
| 2,048 | 409,600 | 393 MB | 2.9 GB (same always) |
| 10,000 | 2,000,000 (capped) | 1.92 GB | 2.9 GB (same always) |
| 50,000 | 2,000,000 (capped) | 1.92 GB | 2.9 GB (same always) |

At $N = 10{,}000$ the frontier hits the 2M cap at depth 2, losing
completeness for most queries. RL processes all 10K queries across
episodes without ever holding more than $B = 512$ states at once.

#### Factor 3 — Deep proofs ($D$)

Each additional depth multiplies the frontier (until capped) and grows
the visited set.

| Depth | Frontier ($N=512$, $K=200$) | Frontier + visited memory | 1-step RL |
|-------|----------------------------|--------------------------|-----------|
| 1 | 512 | 0.5 MB | 2.9 GB |
| 2 | 102,400 | 98 MB + 0.8 MB visited | 2.9 GB |
| 3 | 2,000,000 (capped) | 1.92 GB + 8 MB | 2.9 GB |
| 5 | 2,000,000 (capped) | 1.92 GB + 16 MB | 2.9 GB |
| 10 | 2,000,000 (capped) | 1.92 GB + 32 MB | 2.9 GB |

#### The datasets that break N-step but not 1-step

Combining all three factors, the failure profile for N-step BFS is:

> **High $K$ + large $N$ + deep proofs = frontier explosion.**

Concrete dataset profiles that would work in RL but fail in N-step BFS:

| Dataset profile | $K$ | $N$ | $D$ | Why N-step fails | Why 1-step survives |
|----------------|-----|-----|-----|-------------------|---------------------|
| **Dense social graph** (Facebook-like) | ~500 avg, 5K max | 10K+ | 3+ | Frontier hits 2M cap at depth 2. 95%+ queries lose coverage. Engine must chunk 2M states → 10K+ chunks. | B=512 envs, each expands 1 state. Agent explores depth-first in ~500 actions per episode. |
| **Citation network** (Semantic Scholar) | ~50 avg, 50K max | 50K+ | 5+ | High-degree "hub" papers produce 50K successors each. Frontier dominated by hub neighborhoods. | Agent can navigate around hubs by choosing low-degree successors. |
| **E-commerce product graph** (co-purchase) | ~200 avg, 10K max | 100K+ | 4+ | 100K queries × 200 branching = 20M states at depth 2 (10× over cap). | RL processes queries sequentially across episodes. Never >512 states. |
| **Biomedical KG** (drug-gene-disease) | ~30 avg, 1K max | 5K | 6+ | Moderate branching but deep proofs needed (drug → target → pathway → disease). Visited set reaches tens of millions. | Agent learns to select promising pathways early. Depth 6 = 6 engine calls. |

#### Why RL tolerates what BFS cannot

The fundamental reason is **selection vs exhaustion**:

- **BFS (N-step)** must expand ALL surviving states to guarantee
  completeness at depth $D$. It cannot skip a state — that might be
  the only path to a proof. Cost: $O(F_d)$ per depth.

- **RL (1-step)** expands only the state the agent selects. It
  sacrifices completeness (the agent may pick wrong) but gains
  constant memory. Cost: $O(B)$ per step, regardless of the proof
  tree's true branching factor.

```
BFS at depth 3 on dense social graph:

  query ──┬── 500 states ──┬── 250,000 states ──┬── 2,000,000 (CAPPED)
          │                │                    │
          └── must expand  └── must expand      └── must expand ALL
              ALL of these     ALL of these         (or lose proofs)

RL at depth 3 on the same graph:

  query ── 1 state ── 1 state ── 1 state
           (agent     (agent     (agent
            picks)     picks)     picks)

  Memory: [B, K_max, A, 3] at EVERY step. Same as depth 1.
```

The tradeoff is clear: **RL trades completeness for scalability**.
On datasets where BFS's frontier explodes, RL can still operate —
but the agent must learn to navigate the proof tree efficiently,
which is a harder learning problem than having all proofs enumerated.
