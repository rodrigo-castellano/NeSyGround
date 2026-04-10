# Resolution and Filters: A Unified Framework for Query-Driven Grounding

This document formalises the decomposition of query-driven grounding into two
orthogonal components — **resolution** (how ground rule instances are produced)
and **filters** (how they are validated) — and analyses the soundness,
completeness, and interactions of each.

Prerequisites: `grounding_basics.md` (Datalog evaluation, SLD, semi-naive,
Magic Sets) and `grounding_probDB.md` (probabilistic setting, ProbLog,
TensorLog).

---

## 1. Definitions

### 1.1 Ground rule instance

A **ground rule instance** is a rule with all variables replaced by constants:

```
ancestor(alice, charlie) :- parent(alice, bob), ancestor(bob, charlie).
```

This is a valid instantiation of `ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)`
under θ = {X=alice, Y=bob, Z=charlie}.

### 1.2 Proof

A **proof** of a query Q is a tree of ground rule instances where:
- The root is a ground rule instance whose head is Q.
- Every body atom that is not a base fact is the head of another ground rule
  instance in the tree.
- Every leaf body atom is a base fact in F.

A ground rule instance is a proof **only if all its body atoms are provable**
(derivable from F ∪ R).

### 1.3 Soundness and completeness of a grounder

A grounder takes a query Q, facts F, and rules R, and returns a set of ground
rule instances G.

- **Sound**: every g ∈ G is a proof (all body atoms are provable).
- **Complete**: every proof of Q is in G.

### 1.4 Provable set (provset)

The **provable set** I* is the set of all ground atoms derivable from F ∪ R.
It equals the minimal Herbrand model and can be computed by iterating the T_P
operator to fixpoint (see `grounding_basics.md` §2.2).

A **bounded provset** I_d is the set of atoms derivable in at most d iterations
of T_P:
```
I_0 = F
I_k = F ∪ T_P(I_{k-1})
I_d = result after d iterations
```
I_d is always sound (every atom in I_d is truly derivable) but may be
incomplete (atoms requiring more than d iterations are missing).

---

## 2. The resolution + filter decomposition

**Thesis**: every query-driven grounder can be decomposed into two components:

```
Query → [Resolution] → candidate ground rule instances → [Filter] → proofs
```

1. **Resolution** produces candidate ground rule instances for the query.
   It determines WHAT ground instances to consider and HOW to find them.

2. **Filter** validates candidates by checking whether their body atoms are
   provable.  It determines WHICH candidates are actual proofs.

These two components are **orthogonal**: any resolution can be paired with any
filter.

### 2.1 Resolution: sound or unsound

A resolution is **sound** if every candidate it produces is a proof (all body
atoms provable).  In that case, the filter is optional — it can only **prune
the search space** (optimisation), not correct errors.

A resolution is **unsound** if some candidates have unprovable body atoms.
In that case, the filter is **required for correctness** — without it, the
grounder returns false proofs.

```
                    Resolution
                sound          unsound
             ┌──────────┬──────────────┐
Filter role: │  prune   │  correctness │
             │  (optim) │  (required)  │
             └──────────┴──────────────┘
```

### 2.2 Filter: sound, ideally incomplete

When we do resolution and then use a provable set to filter, the provset
**should be sound but cannot be complete** — because a complete provset
(i.e. the full fixpoint I*) would make the resolution redundant.  Computing I*
is equivalent to running full forward chaining, which is itself a complete
grounder.  If we had I*, we would not need query-driven resolution at all.

The value of query-driven resolution is precisely that it avoids computing I*.
The filter is a **cheaper approximation**: a bounded provset I_d that is sound
(never marks an unprovable atom as provable) but incomplete (may miss atoms
derivable in more than d iterations).

---

## 3. Two fundamental resolution mechanisms

At the semantic level, there are two mechanisms for producing ground rule
instances (see `grounding_basics.md` §4):

### 3.1 Unification-based (top-down)

Match a goal with a rule head via MGU (Most General Unifier), apply the
substitution to the rule body.  Variables may remain in intermediate states,
resolved at later depth steps.

**SLD resolution** (Kowalski, 1974) is the canonical instance.  It is the
basis of Prolog.

### 3.2 Join-based (bottom-up)

Evaluate the rule body as a relational join over fact tables.  Each body atom
corresponds to a table; shared variables are join keys.  The result is a set of
ground rule instances.

Different join algorithms:

| Algorithm | How candidates are found | Anchoring |
|-----------|------------------------|-----------|
| Constant-anchored enumeration | Index probe on bound argument from query | Query constant |
| Domain enumeration | Cartesian product over all entities | None (brute force) |
| Semi-naive T_P step | Incremental join against newly derived facts | None (data-driven) |
| Matrix multiplication | Sparse mat-vec product | One-hot query vector |

All are different implementations of the same semantic operation: find all
substitutions θ such that the rule body under θ produces ground atoms.

---

## 4. Analysis of specific configurations

### 4.1 SLD, all depths (sound, complete)

SLD resolution with no depth bound (using tabling for termination) is both
sound and complete.  It explores all proof paths and only collects groundings
where every goal is resolved to a fact.

**Filter role**: optional, for pruning only.  Two options for building a provset:

1. **Tabling** (lazy): as SLD proves subgoals, store results in a table.  This
   is SLG resolution (Chen & Warren, 1996).  The provset is built
   incrementally, containing only atoms that SLD actually encounters.
   Query-directed, zero extra cost (it is memoisation of SLD's own work).

2. **Forward chaining to depth D** (eager): pre-compute I_D independently of
   any query.  Contains all atoms derivable in D steps from ALL facts.  Not
   query-directed unless Magic Sets are applied.

| | Tabling | FC depth D |
|--|---------|-----------|
| When built | Lazily, during BC | Eagerly, before BC |
| Scope | Only atoms BC encounters | All atoms derivable in ≤ D steps |
| Query-directed | Yes | No (unless Magic Sets) |
| Extra cost | None (memoisation) | O(rules × E^arity × D) upfront |

### 4.2 SLD, depth d (sound, incomplete)

SLD with a depth bound d is sound (unfinished proofs are discarded, never
collected) but incomplete (misses proofs requiring more than d steps).

**Filter role**: pruning only.  The provset can help SLD skip branches whose
subgoals are not in the provset, but it **cannot help SLD find new proofs**.
Standard SLD discards groundings when depth is exhausted (goals ≠ []), so there
are no "unknowns" for the filter to validate.

**Maximum proof depth**: d (the depth bound of BC).  The provset does not
extend this.

**To get the d+D benefit** (§5), SLD would need to be modified to keep
groundings with unresolved goals — which is essentially ENUM behaviour.

### 4.3 Enumeration anchored against facts, width=0 (sound, incomplete)

Enumerate entity candidates from the fact index using the query's bound
arguments as anchor.  With width=0, reject any grounding where a body atom is
not a base fact.

This is sound: every accepted grounding has all body atoms in F, so it is
a valid proof.  It is incomplete: only single-rule-application proofs are found
(multi-hop proofs require body atoms that are derived, which width=0 rejects).

**Depth**: depth > 1 adds nothing, because width=0 means no unknowns survive
to be resolved at the next step.  Equivalent to a single fact lookup per rule.

**Filter role**: optional, for pruning only.  A provset could skip rules whose
head predicates are not in I_D, but cannot find new proofs (there are no
unknowns to validate).

### 4.4 Enumeration anchored against facts, width > 0 (unsound, incomplete)

Same enumeration, but accept groundings with up to W unknown body atoms (atoms
that are ground but not base facts).  Unknowns are carried forward as goals for
subsequent depth steps.

This is unsound: some unknowns may not be provable.  It is incomplete: limited
by the depth bound and width schedule.

**Filter role**: required for correctness.  The provset validates unknown body
atoms: if an unknown is in I_D, it is provable; otherwise the grounding is
discarded.  After filtering: sound but still incomplete.

**Maximum proof depth**: d + D (see §5), where d is the backward depth and D is
the forward depth used to build the provset.

### 4.5 Enumeration anchored against all entities (unsound, incomplete)

Same as §4.4, but entity candidates come from domain constants (all entities)
rather than the fact index.  This produces the same set of sound groundings
after filtering, but wastes computation on entity combinations that cannot
participate in any proof.

|                           | Entities from facts    | Entities from all constants          |
|---------------------------|------------------------|--------------------------------------|
| width=0 (reject unknowns) | Sound, no waste        | Sound, wastes computation            |
| width>0 (keep unknowns)   | Unsound, needs filter  | Unsound, needs filter, wastes computation |

### 4.6 Magic Sets (sound, complete)

Magic Sets (Bancilhon et al., 1986) can be decomposed into the resolution +
filter framework:

- **Resolution** = adornment.  Analyse the query's binding pattern (bound/free),
  propagate through rules via the Sideways Information Passing Strategy (SIPS),
  determine which rules are relevant and how bindings flow.  This is the
  query-directed part.

- **Filter** = semi-naive FC to fixpoint on the adorned (rewritten) rules,
  seeded with the query constants.  This is both sound and complete — it
  produces exactly the query-relevant provable atoms.

Because the filter is complete, Magic Sets is both sound and complete.  The cost
is that the filter (FC to fixpoint) must run **per query** — there is no
shortcut.  This is why Magic Sets "cannot avoid query-driven grounding": the
completeness of the filter is tied to the specific query seed.

| Component | Magic Sets | Enumeration + fp |
|-----------|-----------|-----------------|
| Resolution | Adornment (binding analysis) | Binding table compilation |
| Filter | FC to fixpoint (sound, complete) | Bounded FC (sound, incomplete) |
| Per-query cost | FC to fixpoint (expensive) | Bounded FC or batched (cheaper) |
| Completeness | Yes | No |

---

## 5. The d+D completeness result

When the resolution **keeps unknowns** (e.g. enumeration with width > 0), the
filter can validate them using a provset built by forward chaining.  This
creates a **bidirectional** proof construction:

```
Query ←←← BC (d steps) ←←← meeting point →→→ FC (D steps) →→→ Facts
```

- **BC at depth d**: decomposes the query into body atoms, covering proof
  structure up to d rule applications from the query.
- **FC for D iterations**: builds a provset I_D, covering atoms derivable in up
  to D rule applications from base facts.
- **Validation**: a body atom at depth d is proved if it appears in I_D.

**Maximum proof depth**: d + D.  A proof of total depth k is found if the
backward part covers the top k₁ levels and the forward part covers the bottom
k₂ levels, with k₁ + k₂ ≥ k.

**Comparison with using either alone at the same budget**:

| Strategy | Budget | Max proof depth | Branching explored |
|----------|--------|----------------|-------------------|
| BC only (depth d+D) | d+D backward steps | d+D | B_back^(d+D) |
| FC only (depth d+D) | d+D forward steps | d+D | B_fwd^(d+D) |
| BC(d) + FC(D) | d backward + D forward | d+D | B_back^d + B_fwd^D |

Where B_back is the backward branching factor (rules per predicate × matches
per rule) and B_fwd is the forward branching factor (rules × entities^arity per
iteration).

The combined cost B_back^d + B_fwd^D is much less than either B_back^(d+D) or
B_fwd^(d+D) alone — the same insight as bidirectional BFS in graph search.

**Optimal split**: for a fixed total budget d + D = k, the minimum cost is
achieved when:
```
d · log(B_back) ≈ D · log(B_fwd)
```
If branching factors are equal (B_back ≈ B_fwd), the optimum is d = D = k/2.
If backward branching is larger, spend fewer steps backward (smaller d, larger
D), and vice versa.

**Prerequisite**: the resolution must **keep unknowns** for d+D to apply.  SLD
(which discards unknowns) does not benefit — its max proof depth is always d
regardless of the filter.

```
                  keeps unknowns?
                  no              yes
               ┌──────────┬──────────────┐
d+D benefit:   │  no      │  yes         │
               │  (SLD)   │  (ENUM)      │
               └──────────┴──────────────┘
```

---

## 6. Filter taxonomy

Filters can be classified along two axes: **when** they are built and **scope**.

### 6.1 When built

| Type | When | Cost | Query-dependent |
|------|------|------|-----------------|
| **Init-time** | Once, before any queries | O(rules × E^arity × D) | No |
| **Per-batch** | Once per batch of queries | O(batch groundings × steps) | Yes (batch composition) |
| **Per-query** | Once per query | O(query-relevant × steps) | Yes (specific query) |
| **Tabling** | Incrementally during resolution | Zero extra (memoisation) | Yes (what BC encounters) |

### 6.2 Scope

| Filter | Scope | Soundness | Completeness |
|--------|-------|-----------|-------------|
| **None** | — | N/A | N/A |
| **Width** | Per-state heuristic | N/A (prunes, doesn't verify) | N/A |
| **Provset (FC depth D, global)** | All atoms derivable in D steps from F | Sound | Incomplete if D < fixpoint |
| **Provset (FC depth D, batched)** | Atoms provable from collected groundings in the batch | Sound | Incomplete (batch-dependent) |
| **Provset (FC fixpoint, global)** | All derivable atoms I* | Sound | Complete |
| **Provset (FC fixpoint, per-query)** | Query-relevant derivable atoms (Magic Sets) | Sound | Complete (for the query) |
| **Tabling** | Atoms encountered and proved during BC | Sound | Depends on BC depth |

### 6.3 Width as a filter

Width is a **heuristic bound on proof complexity per state**: reject any state
where more than W goal atoms are ground but unknown (not base facts).  It
applies to any resolution that produces ground body atoms:

| Resolution | What width filters on |
|-----------|---------------------|
| Enumeration | Number of ground-but-unknown body atoms |
| SLD | Number of ground-but-unknown goals (after partial unification) |
| Semi-naive | N/A (everything is ground and verified) |

Width is not a soundness filter — it does not verify provability.  It is a
search space pruner that trades completeness for efficiency.

---

## 7. Unified view

Every configuration in this framework is defined by three choices:

1. **Resolution mechanism**: unification-based (SLD) or join-based (enumeration)
2. **Resolution depth bound**: d (or unbounded)
3. **Filter**: none, width, bounded provset (FC depth D), or complete provset (fixpoint)

| Configuration | Resolution | Filter | Sound | Complete | Max proof depth |
|---------------|-----------|--------|-------|----------|----------------|
| SLD unbounded + tabling | SLD (all depths) | Tabling (prune) | Yes | Yes | ∞ |
| SLD depth d | SLD (depth d) | None | Yes | No | d |
| SLD depth d + provset D | SLD (depth d) | FC depth D (prune) | Yes | No | d |
| Enum w=0 | Enum (facts, depth 1) | Width=0 | Yes | No | 1 |
| Enum w>0 depth d | Enum (depth d) | Width=W | No | No | d |
| Enum w>0 depth d + fp D | Enum (depth d) | Width + FC depth D | Yes | No | d+D |
| Magic Sets | Adornment | FC fixpoint (per-query) | Yes | Yes | ∞ |

### Practical configurations

| System | Configuration | Notes |
|--------|--------------|-------|
| Prolog | SLD unbounded (+ cut) | DFS, first solution, may loop |
| XSB Prolog | SLD unbounded + tabling | SLG resolution, terminates |
| ProbLog | SLD + tabling → WMC | Sound, complete, exact P(Q) |
| ProbLog T_P | Adornment + FC fixpoint → WMC | Magic Sets variant |
| Datalog engines | Semi-naive FC to fixpoint | Sound, complete, unfocused |
| keras-ns | Enum (all entities) depth d + batched fp D | Approximate, GPU-batched |
| TensorLog | Matrix compilation (implicit FC) | Differentiable, depth-bounded |

---

## 8. Open question: formalising d+D incompleteness

The combination of backward depth d and forward depth D produces proofs of
total depth at most d+D.  Several questions remain:

1. **Characterisation**: for a given rule set R, what fraction of proofs have
   depth ≤ d+D?  How does this depend on the structure of R (chain rules,
   tree-shaped rules, cyclic rules)?

2. **Optimal split**: given a fixed budget k = d+D and known branching factors,
   what is the optimal (d, D) split?  The bidirectional BFS analogy suggests
   d · log(B_back) ≈ D · log(B_fwd), but this needs formalisation for the
   logic programming setting.

3. **Batch effects**: when the provset is built per-batch (fp_batch), the
   effective D depends on the batch composition — queries in the batch may prove
   intermediate atoms for each other.  How does batch size affect the effective
   D?

4. **Interaction with width**: width limits the number of unknowns per state,
   which affects how many body atoms the filter must validate.  What is the
   interaction between width W, backward depth d, and forward depth D?

These questions are relevant for understanding the approximation quality of
bounded neuro-symbolic grounding and could be formalised as a contribution.

---

## References

### Resolution
- Kowalski, R. (1974). *Predicate logic as a programming language*. — SLD resolution.
- Robinson, J.A. (1965). *A machine-oriented logic based on the resolution principle*. — General resolution.

### Bottom-up evaluation
- van Emden, M. & Kowalski, R. (1976). *The semantics of predicate logic as a programming language*. — T_P operator.
- Bancilhon, F. (1986). *Naive evaluation of recursively defined relations*. — Semi-naive evaluation.

### Magic Sets
- Bancilhon, F., Maier, D., Sagiv, Y. & Ullman, J. (1986). *Magic sets and other strange ways to implement logic programs*. — Magic Sets transformation.

### Tabled resolution
- Chen, W. & Warren, D.S. (1996). *Tabled evaluation with delaying for general logic programs*. — SLG resolution.

### Probabilistic logic programming
- De Raedt, L., Kimmig, A. & Toivonen, H. (2007). *ProbLog: A probabilistic Prolog*. — ProbLog.
- Vlasselaer, J. et al. (2020). *Beyond the grounding bottleneck*. — Datalog techniques for ProbLog.
- Manhaeve, R. et al. (2018). *DeepProbLog*. — Neural extension of ProbLog.

### Differentiable reasoning
- Cohen, W. (2016). *TensorLog: A differentiable deductive database*. — Matrix compilation.
- Rocktäschel, T. & Riedel, S. (2017). *End-to-end differentiable proving*. — Soft unification (NTP).

### Combining forward and backward
- Haemmerlé, R. (2014). *On combining backward and forward chaining in constraint logic programming*. — BC+FC combination.

### Bidirectional search
- Pohl, I. (1971). *Bi-directional search*. — Bidirectional BFS optimality.
