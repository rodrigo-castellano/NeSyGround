# Grounding Basics: Rule Evaluation in Datalog and Logic Programming

A self-contained introduction to the problem of finding all ground rule
instances (proofs) in a database of facts and Horn clause rules.

---

## 1. Setting

**Datalog** is a query language for deductive databases.  A Datalog program
consists of:

- **Facts** (the extensional database, EDB): ground atoms.
  ```
  parent(alice, bob).
  parent(bob, charlie).
  parent(bob, diana).
  ```

- **Rules** (the intensional database, IDB): Horn clauses of the form
  `head :- body₁, body₂, …, bodyₙ` where head and each bodyᵢ are atoms.
  Variables are implicitly universally quantified.
  ```
  ancestor(X, Y) :- parent(X, Y).                          (R1)
  ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).          (R2)
  ```

- **Query**: an atom, possibly with constants and variables.
  ```
  ?- ancestor(alice, Z).
  ```

Datalog is function-free (no compound terms), which guarantees that the set of
ground atoms is finite.  This is the key difference from full Prolog.

### 1.1 What is grounding?

**Grounding** a rule means replacing every variable with a constant to produce a
**ground rule instance**.  For example, grounding R2 with {X=alice, Y=bob,
Z=charlie} gives:

```
ancestor(alice, charlie) :- parent(alice, bob), ancestor(bob, charlie).
```

This is one ground instance.  The **grounding problem** is: given a query, find
all ground rule instances that participate in a valid derivation of the query
from the facts.

---

## 2. Semantics: what does a Datalog program mean?

Three equivalent ways to define the meaning of a Datalog program:

### 2.1 Model-theoretic semantics

A **model** of a program is a set of ground atoms I such that for every ground
instance of every rule, if all body atoms are in I then the head atom is in I.
The meaning of the program is its **minimal model** — the smallest such set.

### 2.2 Fixpoint semantics (T_P operator)

Define the **immediate consequence operator** T_P:

```
T_P(I) = { head(θ) | (head :- body₁, ..., bodyₙ) ∈ R,
                      θ is a ground substitution,
                      body₁(θ) ∈ I, ..., bodyₙ(θ) ∈ I }
```

T_P takes a set of known facts I and produces all heads derivable in one step.
Starting from the EDB facts F:

```
I₀ = F
I₁ = F ∪ T_P(I₀)
I₂ = F ∪ T_P(I₁)
...
```

Since the domain is finite and T_P is monotone, this sequence reaches a fixpoint
I* = T_P(I*) in finitely many steps.  This fixpoint equals the minimal model.

### 2.3 Proof-theoretic semantics

A ground atom A is derivable iff there exists a **proof tree**: a tree where the
root is A, each internal node is the head of a ground rule instance, its
children are the body atoms of that instance, and every leaf is a fact in F.

All three definitions yield the same set of ground atoms.

---

## 3. Evaluation strategies

### 3.1 Bottom-up: naive evaluation

Directly implements the T_P fixpoint computation:

```
I = F                          -- start with facts
repeat:
    I_new = I ∪ T_P(I)
    if I_new = I: break        -- fixpoint reached
    I = I_new
return I
```

At each iteration, for every rule, try every possible ground substitution θ and
check if all body atoms are in I.  If so, add head(θ) to I.

**Properties:**
- Sound and complete: computes exactly the minimal model.
- Terminates: finite domain, monotone operator.
- Unfocused: derives ALL facts, including those irrelevant to the query.
- Redundant: at iteration k, re-derives everything from iterations 0..k-1.

**Complexity:** For a program with P predicates, E entities, and rules of
maximum body size m, each iteration costs O(P · E^m) in the worst case (trying
all substitutions).  The number of iterations is bounded by O(P · E²) (the
maximum number of derivable binary ground atoms).

### 3.2 Bottom-up: semi-naive evaluation

The key observation: at iteration k, the only NEW facts come from rule
instances where at least one body atom was derived at iteration k−1 (i.e. is in
ΔI_{k-1} = I_{k-1} \ I_{k-2}).  Instances using only "old" facts were already
computed.

```
I = F
ΔI = F                        -- everything is "new" at first
repeat:
    ΔI_new = T_P^Δ(I, ΔI)     -- only consider instances using ≥1 fact from ΔI
    ΔI_new = ΔI_new \ I        -- remove already-known facts
    I = I ∪ ΔI_new
    ΔI = ΔI_new
    if ΔI = ∅: break
return I
```

Where T_P^Δ(I, ΔI) computes rule heads where at least one body atom comes from
ΔI (and the rest from I).

**Properties:**
- Produces the same result as naive evaluation.
- Avoids redundant derivations: each fact is derived at most once.
- Standard in all modern Datalog engines (Soufflé, LogicBlox, etc.).

### 3.3 Top-down: SLD resolution

**SLD resolution** (Linear resolution with Selection function for Definite
clauses, Kowalski 1972) is the computational model of Prolog.  Instead of
computing all facts bottom-up, it starts from a query and works backward.

**Algorithm:**
1. Start with a **goal list** G = [query].
2. **Select** a goal atom A from G (the selection function picks which one).
3. Find a rule `H :- B₁, ..., Bₙ` whose head H unifies with A via a Most
   General Unifier (MGU) θ.  Standardise apart (rename rule variables) first.
4. Replace A in G with B₁θ, ..., Bₙθ.  Apply θ to the rest of G as well.
5. Repeat until G is empty (success) or no rule matches (failure → backtrack).

**Example:**
```
Goal: ancestor(alice, Z)?

Step 1: Select ancestor(alice, Z).
        Unify with R2 head: ancestor(X₁, Z₁), θ = {X₁=alice, Z₁=Z}
        New goals: [parent(alice, Y₁), ancestor(Y₁, Z)]

Step 2: Select parent(alice, Y₁).
        Unify with fact parent(alice, bob), θ = {Y₁=bob}
        New goals: [ancestor(bob, Z)]

Step 3: Select ancestor(bob, Z).
        Unify with R1 head: ancestor(X₂, Y₂), θ = {X₂=bob, Y₂=Z}
        New goals: [parent(bob, Z)]

Step 4: Select parent(bob, Z).
        Unify with fact parent(bob, charlie), θ = {Z=charlie}
        New goals: []  →  SUCCESS, Z=charlie

    Backtrack to step 4:
        Unify with fact parent(bob, diana), θ = {Z=diana}
        New goals: []  →  SUCCESS, Z=diana

    Backtrack to step 3:
        Unify with R2 head → deeper recursion...
```

**Properties:**
- Goal-directed: only explores rules and facts reachable from the query.
- Sound and refutation-complete for Horn clauses.
- **Not terminating** in general for recursive rules without a depth bound.
- Prolog uses depth-first search with backtracking, which is incomplete
  (left-recursion loops).  Tabled resolution (SLG) fixes this.
- Variables exist in intermediate states (resolved via unification at later
  steps).

### 3.4 Top-down: tabled resolution (SLG)

**Tabled resolution** (Chen & Warren, 1996) extends SLD with memoisation:

- The first time a subgoal S is encountered, it is registered in a **table**.
- Answers to S are added to the table as they are derived.
- Subsequent calls to S read answers from the table instead of re-deriving them.
- Recursive subgoals that would loop under SLD are **suspended** and resumed
  when new answers arrive.

This gives SLD the termination and optimality properties of bottom-up
evaluation while retaining goal-directed search.  Used in XSB Prolog.

### 3.5 Hybrid: Magic Sets

**Magic Sets** (Bancilhon et al., 1986) transforms a Datalog program so that
bottom-up evaluation only computes facts **relevant to the query**.  It gets the
best of both worlds: bottom-up execution with top-down focus.

The transformation has three steps:

#### Step 1: Adornment

Label each predicate argument as **b**ound or **f**ree based on information flow
from the query.  For `ancestor(alice, Z)`, the adornment is `bf`.

Adornments propagate through rules: if `ancestor^bf(X,Z) :- parent(X,Y),
ancestor(Y,Z)`, then X is bound (from the head), so parent gets adornment `bf`,
and Y becomes bound after evaluating parent, so the recursive ancestor also gets
`bf`.

#### Step 2: Generate magic predicates

For each adorned predicate, create a **magic predicate** that carries only the
bound arguments.  The magic predicate represents "we need to evaluate this
predicate with these bound arguments."

```
magic_ancestor(alice).                -- seed from the query
magic_ancestor(Y) :- magic_ancestor(X), parent(X, Y).   -- propagate
```

#### Step 3: Modify the original rules

Add the magic predicate as a guard to each rule:

```
ancestor(X,Y) :- magic_ancestor(X), parent(X,Y).
ancestor(X,Z) :- magic_ancestor(X), parent(X,Y), ancestor(Y,Z).
```

Now bottom-up evaluation of this modified program:
1. Starts with `magic_ancestor(alice)`.
2. Derives `magic_ancestor(bob)` (from parent(alice,bob)).
3. Only computes ancestor(alice,_) and ancestor(bob,_) — not ancestors of
   charlie or diana or any other entity.

**Properties:**
- Produces the same answers as the original program for the given query.
- Bottom-up execution: no backtracking, no loops.
- Query-directed: only computes query-relevant facts.
- Equivalent in power to top-down evaluation (under the Sideways Information
  Passing Strategy / SIPS).

---

## 4. Rule evaluation as join processing

A key insight from deductive databases: evaluating a rule body is equivalent to
a **relational join** over fact tables.

```
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

This is a join of the `parent` table and the `ancestor` table on variable Y:

```
SELECT p.arg1 AS X, a.arg2 AS Z
FROM   parent p, ancestor a
WHERE  p.arg2 = a.arg1           -- join on Y
```

Different grounding strategies correspond to different **join algorithms**:

| Strategy | Join analogy |
|----------|-------------|
| Naive bottom-up | Nested-loop join: try all X,Y,Z combinations |
| Semi-naive | Incremental join: only new tuples per iteration |
| SLD (top-down) | Index probe: given X, find matching Y in parent, then Z in ancestor |
| Magic Sets | Index probe driven by bound constants, executed bottom-up |

The **constant anchor** idea is natural here: given a query `ancestor(alice, ?)`,
the constant `alice` is used as an index key into the `parent` table to retrieve
Y candidates.  This avoids scanning the entire table.

---

## 5. Soundness and completeness

### Soundness
A grounding strategy is **sound** if every ground rule instance it produces is
part of a valid derivation (i.e. all body atoms are derivable from facts and
rules).

- SLD resolution is sound: each step is justified by a unification.
- Bottom-up (naive/semi-naive) is sound: only heads whose body is in I are
  derived.
- Magic Sets is sound: produces a subset of the full bottom-up result.

### Completeness
A grounding strategy is **complete** if it finds ALL ground rule instances that
participate in valid derivations of the query.

- Full bottom-up (naive/semi-naive) is complete: computes the entire minimal
  model.
- SLD with exhaustive search (all branches) is complete for Horn clauses.
- SLD with depth-first search and depth bound is **incomplete**: may miss proofs
  beyond the bound.
- Magic Sets is complete (relative to the query).

In the neuro-symbolic setting, both soundness and completeness matter: missing
valid proofs loses probability mass; including invalid proofs corrupts scores.

---

## 6. Summary: which strategy for which situation

| | Bottom-up | Top-down (SLD) | Magic Sets |
|--|-----------|---------------|------------|
| **Computes** | All derivable facts | Answers to the query | Answers to the query |
| **Execution** | Iterate T_P to fixpoint | Goal-directed search | Bottom-up on rewritten program |
| **Variables** | No (always ground) | Yes (intermediate) | No (always ground) |
| **Termination** | Always | Needs depth bound | Always |
| **Redundancy** | Semi-naive avoids it | May re-derive subgoals | Fixpoint avoids it |
| **Best for** | Small databases, need all facts | Large DB, specific query | Large DB, specific query, no backtracking |

---

## References

- Kowalski, R. (1974). *Predicate logic as a programming language*. Proc. IFIP Congress. — SLD resolution.
- van Emden, M. & Kowalski, R. (1976). *The semantics of predicate logic as a programming language*. JACM 23(4). — T_P operator, fixpoint semantics.
- Bancilhon, F. (1986). *Naive evaluation of recursively defined relations*. In: On Knowledge Base Management Systems. — Semi-naive evaluation.
- Bancilhon, F., Maier, D., Sagiv, Y. & Ullman, J. (1986). *Magic sets and other strange ways to implement logic programs*. Proc. PODS. — Magic Sets.
- Ullman, J.D. (1988). *Principles of Database and Knowledge-Base Systems*, Vols. 1 & 2. — Comprehensive textbook treatment.
- Abiteboul, S., Hull, R. & Vianu, V. (1995). *Foundations of Databases*. Addison-Wesley. — Ch. 12-15 on Datalog evaluation.
- Chen, W. & Warren, D.S. (1996). *Tabled evaluation with delaying for general logic programs*. JACM 43(1). — SLG / tabled resolution.
- Ceri, S., Gottlob, G. & Tanca, L. (1990). *Logic Programming and Databases*. Springer. — Deductive database techniques.
