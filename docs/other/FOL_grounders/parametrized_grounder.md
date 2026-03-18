# ParametrizedGrounder — BC-style vs. Fact-Anchored Approximation

Notation follows `notation.md`.

## Index

1. [§1 The Grounding Problem in SBR](#sec-1)
2. [§2 Traditional BC-style Grounding](#sec-2)
   - [§2.1 Algorithm](#sec-2-1)
   - [§2.2 Properties](#sec-2-2)
   - [§2.3 Complexity](#sec-2-3)
3. [§3 Fact-Anchored Approximation (Current Implementation)](#sec-3)
   - [§3.1 Algorithm](#sec-3-1)
   - [§3.2 Width and Depth Parameters](#sec-3-2)
   - [§3.3 PruneIncompleteProofs](#sec-3-3)
   - [§3.4 Complexity](#sec-3-4)
4. [§4 The Approximation Gap](#sec-4)
   - [§4.1 Validation is exact](#sec-4-1)
   - [§4.2 Enumeration is approximate](#sec-4-2)
   - [§4.3 Concrete example of missed groundings](#sec-4-3)
5. [§5 Why the Approximation Was Chosen](#sec-5)
6. [§6 TrueBCGrounder as the Fix](#sec-6)
7. [§7 Summary Table](#sec-7)

---

## §1 The Grounding Problem in SBR {#sec-1}

In Soft Belief Revision (SBR), the grounder is called per training batch. For
each ground query atom $q = h(a, b)$ and each rule $r$ with head predicate $h$,
it must return all valid ground substitutions of the rule body — the
**groundings** $\mathcal{G}_r(q)$.

A grounding $(g_1, \dots, g_m)$ is **valid** if every body atom $g_k$ is
provable from the knowledge base, i.e.\ $g_k \in I^*$ where
$I^* = \text{lfp}(T_P)$ is the least fixpoint.

The challenge: during training, this must be done for every query in the batch,
for every matching rule, efficiently enough to run inside a compiled forward
pass.

---

## §2 Traditional BC-style Grounding {#sec-2}

### §2.1 Algorithm {#sec-2-1}

Traditional backward chaining grounds a query by **carrying a goal stack per
query**. Starting from the query head, body atoms are pushed as sub-goals and
proved recursively.

For a rule $r: h(X,Y) \leftarrow b_1(X,Z),\ b_2(Z,Y)$ and query $q = h(a,?)$:

```
goal_stack ← [(h, a, ?)]

pop goal g = h(a, ?)
  unify g with head of r → bind X=a, Y=?
  push sub-goals: b1(a, ?) and b2(?, ?)

pop sub-goal b1(a, ?)
  look up fact_index(b1, a, ?) → {z1, z2, z3}   # base fact matches
  for each z_i:
    bind Z = z_i
    push sub-goal b2(z_i, ?)

    pop sub-goal b2(z_i, ?)
      look up fact_index(b2, z_i, ?) → {y1, y2}
      for each y_j:
        bind Y = y_j → grounding (b1(a,z_i), b2(z_i,y_j)) recorded

    OR: b2 is a derived predicate → push b2(z_i,?) as a new goal,
        find rules with head b2, expand recursively
```

At each step the goal stack is the **per-query proof state**. The algorithm
terminates when all sub-goals are either base facts or recursively proved.

### §2.2 Properties {#sec-2-2}

- **Sound**: only records groundings where all body atoms are provable.
- **Complete**: finds ALL valid groundings, including those where body atoms
  are themselves derived predicates (heads of other rules).
- **Carries goal state per query**: each query has its own goal stack evolving
  through the search.
- **Handles derived body predicates**: if $b_1$ is not a base predicate, it is
  expanded via the rules that can prove it.

### §2.3 Complexity {#sec-2-3}

Per query, per rule $r$ with $m$ body atoms and body chain length $n$:

$$O\!\left(D^m \cdot m\right) \quad \text{where } D = \text{avg degree per predicate}$$

For a 2-body chain rule: $O(D^2 \cdot m)$ per query.

---

## §3 Fact-Anchored Approximation (Current Implementation) {#sec-3}

### §3.1 Algorithm {#sec-3-1}

`ParametrizedGrounder` does **not** carry goal state per query. Instead it uses
**fact-anchored enumeration**: one body atom is used as an anchor to look up
base facts, and the remaining free variables are filled by enumeration or
lookup.

For the same rule $r: h(X,Y) \leftarrow b_1(X,Z),\ b_2(Z,Y)$ and query $q =
h(a,?)$:

```
for anchor ∈ {b1, b2}:

    # Anchor on b1 (direction A):
    look up fact_index(b1, a, ?) → {z1, z2, z3}   # D matches
    for each z_i:
        candidate grounding: (b1(a,z_i), b2(z_i,?))
        — b2(z_i, ?) is a FREE slot (Y is still unknown)
        — look up fact_index(b2, z_i, ?) → {y1, y2}
        — or, for width w>0: accept y even if b2(z_i,y) not a base fact
        → record (b1(a,z_i), b2(z_i,y_j))

    # Anchor on b2 (direction B, dual anchoring):
    look up fact_index(b2, ?, ?) — requires a bound argument from head
    ...
```

**There is no goal stack.** Each query is processed with a fixed sequence of
tensor lookups, producing a fixed-shape output `[B, R, G_max, M, 3]`.

### §3.2 Width and Depth Parameters {#sec-3-2}

The grounder is parameterized as `backward_W_D`:

- **W** (width): maximum number of body atoms allowed to be **not in the base
  facts**. For W=0, every body atom must exist in the fact index. For W=1, one
  atom may be absent from base facts (but still must be provable — see §3.3).
- **D** (depth): the number of forward-chaining steps used to pre-compute $I^*$.
  Controls how deep the provability check reaches.

| Setting | Meaning |
|---|---|
| `backward_0_1` | All body atoms must be base facts. Pure fact-lookup, no I*. (ParametrizedBCGrounder only) |
| `bcprune_2` | 1 body atom may be derived; validated by 2-step I*. |
| `backward_2_3` | 2 body atoms may be derived; validated by 3-step I*. (ParametrizedBCGrounder only) |

When W ≥ max_body_atoms, the grounder delegates to `FullBCGrounder`.

### §3.3 PruneIncompleteProofs {#sec-3-3}

For W > 0, the anchor produces bindings where some body atoms are not base
facts. These candidates must be validated: is the non-fact body atom actually
provable?

`PruneIncompleteProofs` does this via a binary search on the pre-computed
sorted tensor `provable_hashes` (which encodes $I^*$):

```python
# Inside forward() — fully static, fullgraph-compatible
positions = torch.searchsorted(self.provable_hashes, query_hashes)
match = self.provable_hashes[clamped] == query_hashes   # boolean mask
```

A candidate grounding is kept only if all its body atoms satisfy:

$$g_k \in F \quad \text{(base fact)} \quad \vee \quad g_k \in I^* \quad \text{(provable)}$$

This is the role of PruneIncompleteProofs: to make fact-anchored enumeration
**semantically equivalent to BC for the validation step**.

### §3.4 Complexity {#sec-3-4}

Per query, per rule $r$ with 2-body atoms:

$$O(D \cdot m) \quad \text{(fact-anchored, one anchor lookup + one validation)}$$

Compared to traditional BC's $O(D^2 \cdot m)$, this saves one $D$ factor. At
scale (IJCAI analysis):

| Dataset | C | D | BC: $D^2 \cdot m$ | Anchored: $D \cdot m$ | Speedup |
|---|---|---|---|---|---|
| kinship_family | 2,968 | ~5 | 25·m | 5·m | 5× |
| wn18rr | 40,559 | ~10 | 100·m | 10·m | 10× |
| FB15k237 | 14,505 | ~181 | 32,761·m | 181·m | 181× |

---

## §4 The Approximation Gap {#sec-4}

### §4.1 Validation is exact {#sec-4-1}

For groundings that the fact-anchored enumerator **does find**, PruneIncomplete
Proofs is **semantically exact**: checking $g_k \in I^*$ gives the same answer
as running BC recursively on $g_k$, because $I^*$ is precisely the set of all
atoms BC would prove. No approximation here.

For 2-body rules where the anchor predicate is a base predicate, the output of
fact-anchored + PruneIncompleteProofs is **identical** to traditional BC
(proven in IJCAI_grounding analysis §2.5).

### §4.2 Enumeration is approximate {#sec-4-2}

The gap arises in the **enumeration step**: the anchor lookup is
`fact_index.get(b_k, ...)` — it only finds entities where $b_k$ is a **base
fact**. If $b_k$ is a derived predicate (not directly observed in the KB, but
provable via rules), the lookup returns nothing and the grounder produces zero
groundings for that rule+query pair.

Traditional BC would push $b_k$ as a sub-goal, find rules that prove $b_k$,
and recursively find the substitutions. The ParametrizedGrounder does not carry
this sub-goal state, so it cannot discover these groundings.

**The approximation**: fact-anchored enumeration only reaches groundings
anchored on base-fact predicates. Groundings requiring a derived predicate as
the anchor are silently missed.

### §4.3 Concrete example of missed groundings {#sec-4-3}

```
Base facts:   parent(alice, carol),  sibling(carol, dave)
Rules:
  R1:  uncle(X,Y) :- parent(X,Z), sibling(Z,Y)
  R2:  family(X,Y) :- uncle(X,Y)
```

For query `family(alice, ?)`:

**Traditional BC**:
```
goal: family(alice, ?)
  → unify R2 head → sub-goal: uncle(alice, ?)
    → unify R1 head → sub-goals: parent(alice, Z), sibling(Z, ?)
      → fact_index(parent, alice, ?) = {carol}
      → fact_index(sibling, carol, ?) = {dave}
      → grounding found: uncle(alice, dave)
  → grounding for R2: family(alice, dave) ✓
```

**ParametrizedGrounder**:
```
anchor on uncle(alice, ?) in fact_index → [] (uncle is derived, not a base fact)
→ zero groundings for R2 ✗
```

PruneIncompleteProofs cannot help here: no grounding was enumerated to validate.
The grounder is blind to R2 for this query.

---

## §5 Why the Approximation Was Chosen {#sec-5}

Three reasons justify this design:

**1. Efficiency.** Fact-anchored enumeration replaces BC's per-query recursive
search with a fixed sequence of tensor lookups. The enumeration cost drops from
$O(D^m)$ to $O(D)$ per query (for 2-body rules). Over thousands of training
batches, this is the dominant factor.

**2. Static shapes.** Carrying a per-query goal stack requires dynamic control
flow (data-dependent depth, variable-size stacks). This is incompatible with
`torch.compile(fullgraph=True)`. The fact-anchored approach produces a
fixed-shape output tensor `[B, R, G_max, M, 3]` with no branching in the
forward pass.

**3. Practical irrelevance for most KGC datasets.** In standard KG completion
benchmarks (WN18RR, FB15k237, kinship), all rules are mined from the base facts
via AMIE. Their body predicates are always observed base predicates. The
enumeration approximation (§4.2) only matters when a body predicate is a
derived-only predicate — which does not occur in standard AMIE-mined rules.

**When the approximation matters:**

- Rules with derived predicates in the body (e.g., meta-rules like R2 above)
- Rule chains where intermediate predicates are not directly observed in the KB
- Datasets with partially observed predicates

---

## §6 Summary Table {#sec-6}

| Property | Traditional BC | ParametrizedGrounder |
|---|---|---|
| Goal state per query | Yes (dynamic stack) | No |
| Anchor on base facts | No (expands all) | Yes (enumeration step) |
| Handles derived body predicates | ✓ complete | ✗ misses them |
| Validation mechanism | Recursive sub-proof | I* binary search |
| PruneIncompleteProofs needed | No | Yes (for W>0) |
| Completeness | Full | Approx (enumeration) |
| torch.compile fullgraph | ✗ dynamic shapes | ✓ |
| Cost per query (2-body rule) | O(D²·m) | O(D·m) |
| Pre-computed I* required | No | Yes (for W>0) |
