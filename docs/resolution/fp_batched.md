# fp_batch: Forward-Chaining Soundness Filter on Collected Groundings

This document explains the `fp_batch` filter (hook 5 — terminal soundness
filter) and its relationship to the keras-ns `PruneIncompleteProofs` function.

Prerequisites: `resolution_and_filters.md` (resolution + filter framework),
`filters.md` (hook architecture), `grounding_basics.md` (T_P operator).

---

## 1. The problem fp_batch solves

Backward chaining with bounded width collects ground rule instances that may
have **unprovable body atoms**.  A grounding like:

```
ancestor(tom, alice) :- parent(tom, bob), ancestor(bob, alice)
```

is useful only if `ancestor(bob, alice)` is itself provable.  The backward
chaining search cannot verify this during expansion — it would need to see
the full set of collected groundings first.

fp_batch runs **after** all depth steps complete.  It takes the full set of
collected groundings and asks: which ones form valid, complete proofs?

---

## 2. Algorithm

### What it is

PruneIncompleteProofs / fp_batch is a **forward propagation of provability**
over a fixed dependency graph of ground atoms.

It is **not** a tree traversal.  Backward chaining built a dependency graph
of ground atoms:

```
"atom Q is provable if atoms A, B, C are all provable"
```

PruneIncompleteProofs propagates provability **forward** through this graph,
starting from the leaves (atoms whose bodies were entirely base facts) and
working up toward the original queries, one level per iteration.

### Why it can be seen as forward chaining

Forward chaining in its essential form is (Russell & Norvig, AIMA):

> Start from known facts → find rules whose body is satisfied → fire them
> to derive new facts → repeat until fixpoint.

In first-order settings, "find rules whose body is satisfied" requires
unification and substitution to ground the variables.  But backward chaining
already did all the grounding — it produced a set of fully ground rule
instances with no variables.  What PruneIncompleteProofs receives is
effectively a **ground (propositional) program**: a fixed set of ground
implications like `ancestor(tom, alice) :- father(tom, bob), ancestor(bob, alice)`.

On this ground program, forward chaining reduces to pure set-membership
checks — no unification needed.  PruneIncompleteProofs does exactly this:

1. **Known facts:** atoms with empty needed lists (their body was all facts).
2. **Find satisfied rules:** check if all needed atoms are proved.
3. **Derive new facts:** mark the query atom as proved.
4. **Repeat** D times.

So PruneIncompleteProofs is forward chaining on the ground rule instances
collected by backward chaining.  The only difference from textbook FC is
representational: instead of storing the full ground rules and re-checking
facts every iteration, it stores a compressed proof obligation table where
facts are pre-resolved (excluded from the needed lists).  This compression
is lossless because the fact set is static.

### Pseudocode: PruneIncompleteProofs (keras-ns)

During backward chaining, the grounder records proof obligations: for each
query atom it tries to prove, it stores a `(query, proof)` pair where `proof`
is the list of **non-fact** body atoms that still need proving.  If all body
atoms are facts, `proof` is empty.

```
proved = {}

repeat num_steps times:
    snapshot = frozen copy of proved

    for each proof obligation (query, needed_atoms):
        if query is already proved:
            skip                           ── latch: once True, stays True

        if needed_atoms is empty:
            mark query as proved           ── all body atoms were facts
        else if every atom in needed_atoms is proved in the snapshot:
            mark query as proved           ── all dependencies satisfied
        else:
            mark query as not proved

keep only groundings where every body atom is a base fact or is proved
```

Key properties:
- **No separate seed phase.**  In the first iteration, the snapshot is empty.
  Obligations with an empty `needed_atoms` list (all body atoms were facts)
  get proved immediately.  Obligations with non-empty lists stay unproved
  because their dependencies are not in the empty snapshot.
- **`needed_atoms` contains only non-fact atoms**, not the full body.  Facts
  were already resolved during backward chaining — they are not re-checked
  here.  The fact check only appears in the final grounding filter.
- **Fixed iteration count.**  Runs exactly `num_steps` times, no early exit.
- **Atom-level tracking.**  Each atom is tracked individually, not each
  grounding.  Multiple groundings may prove the same atom.
- **Latch guard.**  Once a query is proved True, it is never re-evaluated.
  This matters because the same query atom can appear in multiple proof
  obligations (from different groundings with different bodies).  Without
  the guard, a later entry whose needed atoms are not yet proved would
  overwrite the True with False — un-proving an atom that was already proved
  by a different grounding.  The guard implements the existential: an atom
  is provable if **any** of its proof obligations succeeds.

### Pseudocode: apply_fp_batch (torch-ns)

fp_batch operates on the collected grounding tensors directly.  It does not
have access to separate proof obligation lists — it works with the full body
of each grounding (including facts and padding).

```
mark which body atoms are base facts (one-time check)
mark which body atom slots are active (not padding)

seed: for each grounding, if every active body atom is a fact, mark it proved

repeat up to depth+1 times:
    collect the head atoms of all proved groundings into a global pool
    sort the pool for fast lookup

    for each grounding that is not yet proved:
        for each active body atom:
            ok if it is a base fact OR its head hash is in the pool
        if all body atoms are ok:
            mark grounding as proved

    if nothing changed since last iteration:
        stop early

output: the proved mask
```

Key properties:
- **Grounding-level tracking.**  Each grounding is individually marked as
  proved or not.  When a grounding is proved, its head atom hash enters the
  global pool, making it available for other groundings that depend on it.
- **Full body checked every iteration.**  No separate proof obligation list.
  Facts are handled inline — if a body atom is a base fact, it is always ok
  regardless of the pool.
- **Global cross-query pool.**  The pool spans the entire batch (all queries,
  all groundings).  A grounding proved for query A can satisfy a body atom
  in query B's grounding.
- **Early convergence exit.**  Stops when no new groundings are proved.
- **Hash-based matching.**  Each atom (predicate, arg0, arg1) is packed into
  a single integer.  Pool membership is checked via binary search.

### Are they equivalent?

**Yes — both compute the same result.**  A grounding survives iff all its
body atoms are transitively derivable from facts through other collected
groundings.  But the mechanics differ:

| Aspect | PruneIncompleteProofs | apply_fp_batch |
|--------|----------------------|----------------|
| **What is tracked** | Per-atom: `atom2proved[query] = bool` | Per-grounding: `proved[b, n] = bool` |
| **What is checked** | Only non-fact obligations (`proof` list) | Full body (facts handled by `is_fact` tensor) |
| **Fact handling** | Facts excluded from `proof` during BC; checked only in final filter | Facts checked every iteration via `is_fact` |
| **Snapshot mechanism** | Explicit `snapshot = copy(dict)` | Implicit: read `proved`, write `new_proved` |
| **Cross-query scope** | Yes: all `rule2proofs` are in one dict | Yes: pool is global across B×N |
| **Iterations** | Exactly `num_steps` | Up to `depth + 1`, with early exit |
| **Lookup method** | Python dict `.get()` | Integer hash + `searchsorted` |

**Why they produce the same output despite tracking different things:**

PruneIncompleteProofs tracks atoms: once `ancestor(bob, alice)` is proved
(by any grounding), any other proof obligation referencing it is satisfied.

fp_batch tracks groundings: when a grounding is proved, its head hash enters
the pool.  If two groundings G1 and G2 both prove `ancestor(bob, alice)`,
both add the same head hash to the pool — but that is fine, duplicates in
the sorted pool do not affect `searchsorted`.  What matters is that the hash
is present, so any body atom matching `ancestor(bob, alice)` passes.

The atom-level vs. grounding-level distinction collapses because: an atom is
provable ↔ at least one grounding proving it exists and is itself proved ↔
its head hash appears in the pool.  Both representations encode the same
reachability over the same dependency graph.

### Why a snapshot?

The snapshot prevents **cascading within a single iteration**.  Without it,
proving atom A could immediately let atom B (which depends on A) become proved
in the same pass, which could then prove C, etc.  This would allow circular
dependencies to bootstrap:

```
p(a) :- q(a)    # grounding 1
q(a) :- p(a)    # grounding 2
```

Without snapshot: iteration 0 could prove p(a) from grounding 1 (checking q(a)
against the in-progress dict which just got set by grounding 2), creating a
false proof.

With snapshot: neither p(a) nor q(a) is in the snapshot at iteration 0.
Neither is in the snapshot at iteration 1.  Neither is ever proved.  Correct.

---

## 3. Worked example

### Setup

```
Facts:
  father(tom, bob)
  father(bob, alice)

Rules:
  R1: ancestor(X, Y) :- father(X, Y)
  R2: ancestor(X, Z) :- father(X, Y), ancestor(Y, Z)

Queries (batch of 3):
  Q0: ancestor(tom, bob)
  Q1: ancestor(bob, alice)
  Q2: ancestor(tom, alice)
```

In the torch-ns implementation, `queries` is `[B, 3]` where B is the batch
size.  Each query b has up to N collected groundings.  The head of every
grounding for query b is the query atom itself (queries[b]).

### Step 1: backward chaining (depth=2, width=1)

Each query is grounded independently.  In torch-ns, the grounder accumulates
body atoms across depth steps into a single grounding per proof path.  A
grounding is collected only when all goals are resolved (terminal state).

**Q0: ancestor(tom, bob)**
- Depth 1, R1: body = `father(tom, bob)` → fact → all goals resolved → terminal.
- Collected with accumulated body = `[father(tom, bob)]`.

**Q1: ancestor(bob, alice)**
- Depth 1, R1: body = `father(bob, alice)` → fact → all goals resolved → terminal.
- Collected with accumulated body = `[father(bob, alice)]`.

**Q2: ancestor(tom, alice)**
- Depth 1, R1: body = `father(tom, alice)` → not a fact, `father` is not
  a head predicate → rejected by head-predicate pruning.
- Depth 1, R2: body = `father(tom, bob), ancestor(bob, alice)`.
  `father(tom, bob)` is a fact, `ancestor(bob, alice)` is unknown (1 unknown
  ≤ width=1, and `ancestor` is a head predicate) → accepted.
  Remaining goal: `ancestor(bob, alice)`.  Not terminal yet.
- Depth 2 (last step, width=0): resolve `ancestor(bob, alice)` with R1:
  body = `father(bob, alice)` → fact → all goals resolved → terminal.
- Collected with accumulated body = `[father(tom, bob), ancestor(bob, alice),
  father(bob, alice)]`.

The key point: Q2 does NOT produce two separate groundings.  The depth-1
rule application (R2) and depth-2 rule application (R1) are fused into a
single grounding with the full accumulated body.  The intermediate atom
`ancestor(bob, alice)` ends up inside the body.

```
Collected groundings:
  G0: head=ancestor(tom, bob)    body=[father(tom, bob)]
  G1: head=ancestor(bob, alice)  body=[father(bob, alice)]
  G2: head=ancestor(tom, alice)  body=[father(tom, bob), ancestor(bob, alice), father(bob, alice)]
```

### Step 2: fp_batch

The head hash of each grounding comes from the original query (`queries[b]`),
broadcast to all N groundings of that query.

```
head hashes:
  G0: hash(ancestor(tom, bob))      ← from Q0
  G1: hash(ancestor(bob, alice))    ← from Q1
  G2: hash(ancestor(tom, alice))    ← from Q2
```

**Fact check** (one-time):

```
G0 body: father(tom, bob) → fact ✓
G1 body: father(bob, alice) → fact ✓
G2 body: father(tom, bob) → fact ✓,  ancestor(bob, alice) → NOT a fact,  father(bob, alice) → fact ✓
```

**Seed** (groundings where ALL active body atoms are facts):

```
G0: all facts → proved=True     head=ancestor(tom, bob)
G1: all facts → proved=True     head=ancestor(bob, alice)
G2: has non-fact body atom → proved=False
```

**Iteration 1:**

Build proved pool from the entire batch:
```
proved heads = { hash(ancestor(tom, bob)),     ← from G0
                 hash(ancestor(bob, alice)) }   ← from G1
```

Check unproved groundings:
```
G2 body atom: ancestor(bob, alice)
  → search in proved pool → FOUND (hash matches G1's head)
G2: father(tom, bob) fact ✓, ancestor(bob, alice) in pool ✓, father(bob, alice) fact ✓
  → all body atoms ok → proved=True
```

**Convergence check:** proved changed (G2 went False→True) → continue.

**Iteration 2:**

All groundings proved.  `new_proved == proved` → converge, stop.

**Result:** all three groundings survive.  G2 is valid because its
intermediate dependency `ancestor(bob, alice)` was proved by G1 from a
different query in the batch.  This cross-query provability is the key
feature of fp_batch — the global pool spans the entire batch.

---

## 4. What about circular dependencies?

```
Facts: (none)

Rules:
  R1: p(a) :- q(a)
  R2: q(a) :- p(a)

Collected groundings:
  G1: p(a) :- q(a)     head=p(a)
  G2: q(a) :- p(a)     head=q(a)
```

**Fact check:** q(a) not a fact, p(a) not a fact.

**Seed:** G1 has non-fact body → proved=False.  G2 has non-fact body → proved=False.

**Iteration 1:**
- Proved pool = {} (empty — no proved groundings).
- G1: q(a) not in pool → still False.
- G2: p(a) not in pool → still False.

**Iteration 2:** Same.  Pool still empty.  Converged.

**Result:** Both pruned.  Circular proofs correctly rejected because the
snapshot mechanism prevents self-bootstrapping.

---

## 5. Relationship to keras-ns PruneIncompleteProofs

`PruneIncompleteProofs` in `keras-ns/ns_lib/grounding/backward_chaining_grounder.py`
does the exact same thing with different data structures.

### Data structure mapping

| Concept | keras-ns | torch-ns fp_batch |
|---------|----------|-------------------|
| Per-grounding head | `query` in `(query, proof)` tuple | `queries[:, b]` broadcast to N groundings |
| Per-grounding body obligations | `proof` list (only non-fact atoms) | `body[b, n, :, :]` (all M body atom slots, with padding) |
| Fact membership check | `fact_index._index.get(atom)` | `fact_index.exists(body)` → `is_fact` bool tensor |
| Proved tracking | `atom2proved` Python dict | `proved` `[B, N]` bool tensor + hash pool |
| Snapshot | `snapshot = dict(atom2proved)` | Implicit: `proved` read, `new_proved` written separately |
| Cross-grounding matching | Dict key lookup: `snapshot.get(atom)` | Global hash pool + `searchsorted` binary search |
| Iteration count | `num_steps` (= grounder depth) | `depth + 1` (one extra for convergence margin) |
| Early exit | No | Yes: `if (new_proved == proved).all(): break` |

### Semantic differences

1. **What is tracked as a "proof obligation":**
   - keras-ns: `rule2proofs` stores only the non-fact body atoms (the atoms
     that need proving).  A grounding with all-fact body has an empty proof list.
   - torch-ns: `body` tensor stores ALL body atoms (facts + non-facts + padding).
     The fact check `is_fact` separates them.  This is more uniform for tensors.

2. **Granularity of the proved map:**
   - keras-ns: `atom2proved` maps individual atoms.  A query is proved if all
     its proof obligations are proved.  Then the final filter checks groundings.
   - torch-ns: `proved` is per-grounding `[B, N]`, not per-atom.  A grounding is
     proved when all its body atoms pass.  The head hash is then added to the
     pool for other groundings to use.

   These are equivalent because: a query atom is "proved" iff at least one of
   its groundings is proved.  In torch-ns, any proved grounding adds its head
   hash to the pool, so other groundings depending on that atom will find it.

3. **Iteration count:**
   - keras-ns uses exactly `num_steps` iterations (matching the backward
     chaining depth).
   - torch-ns uses up to `depth + 1` iterations with early convergence exit.
   - In practice, depth iterations are sufficient for proofs of depth d.  The
     +1 is a safety margin.

### Algorithmic equivalence

Both compute the **minimal Herbrand model restricted to the collected
groundings**.  The algorithms are:

1. keras-ns: Forward propagation on `(query, proof_obligations)` pairs,
   `num_steps` iterations, snapshot per step.
2. torch-ns: Forward propagation on `[B, N]` grounding mask with global hash
   pool, `depth+1` iterations, snapshot implicit in read/write separation.

Both guarantee:
- Atoms provable from facts in k hops are marked proved at iteration k.
- Circular dependencies never bootstrap (snapshot/separation prevents it).
- The final output is the same set of sound groundings.

---

## 6. Torch implementation details

The torch-ns implementation (`grounder/filters/soundness/fp_batch.py`) uses
integer hashing and binary search instead of dict lookups:

### Hash scheme

Each atom `(predicate, arg0, arg1)` is packed into a single long:

```
atom_hash = predicate * (pack_base²) + arg0 * pack_base + arg1
```

where `pack_base` is large enough that the packing is injective (no hash
collisions within the vocabulary).

### Cross-query pooling

The key insight: proved head hashes from ALL queries in the batch are collected
into a single 1D `proved_pool` tensor:

```python
all_heads = head_hashes.reshape(B * N)          # flatten all B×N groundings
proved_pool = where(proved, all_heads, -1)       # keep only proved heads
proved_pool_sorted = proved_pool.sort()          # sort for binary search
```

Then for each body atom:

```python
pos = searchsorted(proved_pool_sorted, body_hash)
found = proved_pool_sorted[pos] == body_hash
```

This gives O(log(B·N)) lookup per body atom, fully vectorized.

### Compile compatibility

The entire function uses only static-shape tensor operations:
- No Python dicts, no dynamic allocation.
- `searchsorted` + `clamp` + elementwise comparison.
- Fixed iteration count (or early-exit via `all()` which is still graph-safe).
- Compatible with `torch.compile(fullgraph=True)`.

---

## 7. When fp_batch is insufficient

fp_batch only knows about groundings **in the current batch**.  If a body atom
is provable via a grounding that was never collected (because width or depth
or budget limits excluded it), fp_batch will reject the grounding even though
a proof exists.

Example:

```
Facts: f(a, b), f(b, c), f(c, d)
Rule: path(X, Z) :- f(X, Y), path(Y, Z)
Rule: path(X, Y) :- f(X, Y)

Query: path(a, d)
```

With depth=2, the grounder may collect:

```
G1: path(a, b) :- f(a, b)                     [depth 1, all facts]
G2: path(b, c) :- f(b, c)                     [depth 1, all facts]
G3: path(a, c) :- f(a, b), path(b, c)         [depth 2, body needs path(b,c)]
```

But `path(a, d)` requires `path(b, d)` which requires `path(c, d)` — a
depth-3 proof.  With depth=2, `path(c, d)` is never collected, so
`path(a, d) :- f(a, b), path(b, d)` — if it were collected — would fail
fp_batch because `path(b, d)` is not in the batch.

This is not a bug.  fp_batch is sound by design.  Completeness requires
sufficient depth and width to collect all necessary intermediate groundings.

---

## 8. Summary

| Property | Value |
|----------|-------|
| **Type** | Terminal soundness filter (hook 5) |
| **Algorithm** | Bottom-up Kleene T_P fixpoint on collected groundings |
| **Direction** | Forward (leaves → root), opposite to the backward chaining that built them |
| **Snapshot** | Yes — prevents circular reasoning |
| **Cross-query** | Yes — grounding proved for query A can satisfy body atom in query B's grounding |
| **Sound** | Yes — only keeps transitively provable groundings |
| **Complete** | No — limited to atoms provable within the batch |
| **Compile-safe** | Yes — fixed-shape tensors, no Python dicts |
| **keras-ns equivalent** | `PruneIncompleteProofs` (same semantics, Python dict-based) |
