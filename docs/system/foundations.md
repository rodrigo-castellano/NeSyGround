# Foundations

This document defines the formal concepts that NeSyGround operates on. It starts with standard first-order logic definitions, then defines grounding precisely for each paradigm, and finally maps everything to tensor representations.

---

## 1. First-Order Logic Definitions

### Atoms and predicates

A **predicate** `p` is a named relation with a fixed arity. In NeSyGround, all predicates are binary: `p(a, b)`.

An **atom** is a predicate applied to arguments: `p(t_1, t_2)`. Arguments are either:

- **Constants** (entities): concrete values like `john`, `mary`, `sue`
- **Variables**: placeholders like `X`, `Y`, `Z` that range over constants

A **ground atom** is an atom with no variables — all arguments are constants. Example: `father(john, mary)`.

### Substitutions and unification

A **substitution** `theta` is a mapping from variables to terms. Applying `theta` to an atom replaces each variable with its mapped value. For example, `theta = {X -> john, Y -> mary}` applied to `parent(X, Y)` yields `parent(john, mary)`.

The **most general unifier (MGU)** of two atoms `A` and `B` is the most general substitution `theta` such that `A theta = B theta`. If no such substitution exists, the atoms do not unify. For example:

- `parent(X, Y)` and `parent(john, Z)` unify with MGU `{X -> john, Y -> Z}` (or equivalently `{X -> john, Z -> Y}`)
- `parent(X, X)` and `parent(john, mary)` do not unify (would require `X = john` and `X = mary`)

### Horn clauses and rules

A **Horn clause** is a disjunction of literals with at most one positive literal. In NeSyGround, we use the equivalent implication form:

```
head :- body_1, body_2, ..., body_m
```

This reads: "head is true if all body atoms are true." The head is a single atom. The body is a conjunction of one or more atoms.

A **rule** is a Horn clause with variables. Example:

```
grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
```

A **fact** is a ground atom asserted as true. Facts can be viewed as rules with an empty body.

### Knowledge base

A **knowledge base** `KB = (F, R)` consists of:

- `F`: a set of ground atoms (facts)
- `R`: a set of rules (Horn clauses with variables)

---

## 2. Grounding Definitions

**Grounding** is the process of finding substitutions that make rules applicable given a knowledge base. Different paradigms define this differently.

### Backward chaining grounder

Backward chaining (BC) starts from a query and works backward through rules to find supporting facts. The engine is **SLD resolution** — the standard Prolog resolution strategy using MGU-based unification.

```
Ground_BC(Q, KB, D) -> G

  Q  : set of query atoms (ground or partially ground)
  KB : knowledge base (F, R)
  D  : maximum proof depth (number of resolution steps)
  G  : set of ground rule instantiations reachable within D steps
```

At each step, BC selects a goal atom, finds rules whose head unifies with it, and replaces the goal with the rule's body atoms (after applying the unifier). This continues until all goals are resolved against facts or the depth bound is reached.

There is no distinction between "SLD resolution" and "Prolog resolution" — they are the same algorithm. NeSyGround implements one BC engine.

### Parametrized backward chaining grounder

The parametrized BC grounder adds a **width** parameter that controls how many unproven body atoms are allowed in each grounding:

```
Ground_Param(Q, KB, D, W) -> G

  Q  : set of query atoms
  KB : knowledge base (F, R)
  D  : maximum proof depth
  W  : maximum width (number of unproven body atoms allowed per grounding)
  G  : set of ground rule instantiations reachable within D steps
       with at most W unproven body atoms
```

Width semantics:

| W value | Behavior |
|---------|----------|
| `W = 0` | Only groundings where ALL body atoms are facts (proven-only) |
| `W = 1` | Allow 1 unproven body atom (provability-checked via FC) |
| `W = 2` | Allow 2 unproven body atoms |
| `W = None` | Full enumeration — no width restriction (delegates to FullBCGrounder) |

The width parameter trades completeness for scalability. See [soundness.md](soundness.md) for formal analysis.

### Forward chaining grounder

Forward chaining (FC) starts from facts and derives new atoms by applying rules iteratively. It computes the **provable set** — all atoms derivable from the knowledge base.

```
Ground_FC(KB, D) -> I_D

  KB  : knowledge base (F, R)
  D   : maximum number of T_P iterations
  I_D : provable atoms after min(D, fixpoint) iterations
```

Key differences from BC:

- **No query parameter**: FC computes the full provable set, not query-specific groundings
- **D bounds iterations, not depth**: the process may reach a fixpoint before D iterations
- **Used as a sub-component**: BC grounders (e.g., ParametrizedBCGrounder) use FC internally to pre-compute which atoms are provable

---

## 3. Operators

### Backward chaining operator

`BC(s, F, R)` performs one step of SLD resolution:

1. **Select** a goal atom `g` from the current proof state `s`
2. **Resolve**: find all rules `r in R` whose head unifies with `g` via MGU `theta`, producing new states with body atoms of `r` substituted by `theta`
3. **Resolve against facts**: check if `g` unifies with any fact `f in F`, producing a state where `g` is removed (proven)
4. **Pack**: merge the resulting child states, deduplicate, and truncate to fixed capacity

### Forward chaining operator (T_P)

The **immediate consequence operator** `T_P` computes one step of forward derivation:

```
T_P(I) = I ∪ { head(r)theta | r in R, body(r)theta ⊆ I }
```

That is: for each rule `r`, find all substitutions `theta` such that every body atom of `r` (under `theta`) is in the current provable set `I`. Add the corresponding head atom to `I`.

The provable set after `D` iterations is:

```
I_0 = F
I_{d+1} = T_P(I_d)
```

FC reaches a **fixpoint** when `I_{d+1} = I_d` — no new atoms can be derived. The `D` parameter bounds the number of iterations; the process may stop earlier.

---

## 4. Tensor Mapping

Every FOL concept maps to a tensor representation for GPU computation. All shapes are **fixed at construction time** to enable CUDA graph capture.

| FOL Concept | Tensor Representation | Shape | Notes |
|-------------|----------------------|-------|-------|
| Atom | Integer triple | `[3]` | `(pred_idx, arg0_idx, arg1_idx)` |
| Proof state | Goal tensor | `[G, 3]` | G goal atoms per state |
| Batch of states | State tensor | `[B, S, G, 3]` | B queries, S states each |
| Substitution | Variable replacement | N/A | Applied by rewriting tensor slots |
| Fact set | FactIndex | indexed structure | Supports O(1) or O(log F) lookup |
| Rule set | RuleIndex | `[R, M+1, 3]` | Head + M body atoms per rule |
| Knowledge base | CompiledKB | composite | FactIndex + RuleIndex + metadata |
| Provable set | Hash tensor | `[I_max]` | Sorted int64 hashes for binary search |

### Encoding conventions

- **Entities** are mapped to consecutive integers `0, 1, ..., E-1`
- **Predicates** are mapped to consecutive integers `0, 1, ..., P-1`
- **Variables** use indices above the entity range (runtime-dependent)
- **Padding** uses a distinguished `padding_idx` value
- **Atom hashing**: `hash(p, a0, a1) = ((p * base) + a0) * base + a1` where `base = E + 2` (to avoid collisions with padding)

### Fixed shapes

All tensor dimensions are determined at construction time based on the knowledge base, depth, and capacity parameters. No dynamic reshaping occurs during grounding. This is a hard requirement for CUDA graph compatibility. See [tensors.md](tensors.md) for the full dimension table.
