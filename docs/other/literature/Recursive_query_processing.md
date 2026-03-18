# An Amateur's Introduction to Recursive Query Processing Strategies

**François Bancilhon** (1) · **Raghu Ramakrishnan** (1,2)

1. MCC, 9430 Research Blvd, Austin, Texas 78759
2. University of Texas at Austin, Austin, Texas 78712

*© 1986 ACM 0-89791-191-1/86/0500/0016*

---

## Abstract

This paper surveys and compares various strategies for processing logic queries in relational databases. The survey is limited to Horn Clauses with evaluable predicates but without function symbols. The paper is organized in three parts:

1. **Part 1** — Main concepts and definitions.
2. **Part 2** — Description of the various strategies: characteristics, application range, detailed algorithm, and worked example for each.
3. **Part 3** — Performance comparison: sample rules/queries, data characterization, analytical cost functions, and plotted cost curves.

---

## 1. Introduction

The database community has manifested strong interest in evaluating "logic queries" against relational databases, motivated by two converging trends:

1. Integrating database and AI technology to create "knowledge base systems."
2. Integrating logic programming and database technology to extend the database interface to the power of a general-purpose language.

Both share the mathematical foundation of first-order logic.

Database researchers already support non-recursive logic queries via the view mechanism. However, efficiently supporting **recursive queries** is still open.

Following pioneering work by Chang, Shapiro, McKay, Henschen, and Naqvi, numerous strategies have been proposed. The challenge is choosing among them — they are described at different levels of detail, application domains are unclear, and no performance evaluation exists. This paper evaluates all strategies with respect to:

- **Functionality** (application domain)
- **Performance**
- **Ease of implementation**

---

## 2. Logic Databases

### 2.1. An Example

```prolog
parent(cain, adam)
parent(abel, adam)
parent(cain, eve)
parent(abel, eve)
parent(sem, abel)

ancestor(X,Y) :- ancestor(X,Z), ancestor(Z,Y)
ancestor(X,Y) :- parent(X,Y)

generation(adam, 1)
generation(X,I) :- generation(Y,J), parent(X,Y), J=I-1
generation(X,I) :- generation(Y,J), parent(Y,X), J=I+1
```

- **Facts**: `parent(cain,adam)` — ground clauses
- **Rules**: `ancestor(X,Y) :- parent(X,Y)` — implications
- **EDB** (Extensional DB): the set of facts
- **IDB** (Intensional DB): the set of rules

The minimal model of this database gives:
```
ancestor = {(cain,adam), (cain,eve), (abel,adam), (abel,eve),
            (sem,abel),  (sem,adam), (sem,eve)}
generation = {(adam,1), (eve,1), (cain,2), (abel,2), (sem,3)}
```

**The core problem**: given a query like `generation(sem,?)` or `ancestor(?,adam)`, find the answer efficiently.

---

### 2.2. Syntax of a Logic Database

**Naming conventions** (Prolog):
- **Variables**: strings starting with uppercase (e.g., `X1`, `Father`, `Y`)
- **Constants**: strings starting with lowercase or integers (e.g., `john`, `345`)
- **Predicates/Relations**: lowercase identifiers

A **literal** is `p(t1, ..., tn)` where each `ti` is a constant or variable.
An **instantiated literal** contains no variables.

Evaluable literals may be written using standard notation:
- `Z = X+Y` ↔ `sum(X,Y,Z)`
- `I = J+1` ↔ `sum(J,1,I)`
- `X > 0`  ↔ `greater-than(X,0)`

A **rule** has the form `p :- q1, q2, ..., qn` where `p` is the **head** and `q1,...,qn` are the **body** (goals).

A **ground clause** is a rule with empty body. A **fact** is a variable-free ground clause.

---

### 2.3. Interpretation of a Logic Database

An **interpretation** maps each relation name to a set of instantiated tuples.

A **model** is an interpretation `I` satisfying:
1. For each evaluable predicate `p`: `I(p) = natural(p)`
2. For any rule `p(t) :- q1(t1), ..., qn(tn)`: if all body atoms hold under substitution `σ`, then `σ(p(t))` holds too

By the Van Emden–Kowalski theorem [76], Horn Clause databases have a unique **minimal model** (under set inclusion), which we take as *the* model.

---

### 2.4. Adornments, Queries, and Database Structure

An **adornment** of an `n`-ary predicate is a sequence of `b` (bound) and `f` (free) of length `n`.
Examples: `bbf` for a ternary predicate, `fbff` for a 4-ary predicate.

A **query form** is an adorned predicate (e.g., `father^bf`, `id^bffb`).
A **query** is a query form with bound positions instantiated to constants.

The **answer** to query `q(t)` is `{ σ(t) | σ(t) ∈ interpretation of q }`.

**Derived predicate**: appears only in the IDB.
**Base predicate**: appears only in the EDB or in rule bodies.

**Rule/goal graph**: two node types:
- **Square nodes** → predicates
- **Oval nodes** → rules

For rule `r: p :- p1, p2, ..., pn`: arc `r → p`, arcs `pi → r` for each `pi`.

---

### 2.5. Recursion

A rule is **recursive** if it contains the head predicate in the body.

A recursive rule is **linear** if the recursive predicate appears exactly once in the body (also called *regularity* [Chang 81]).

**Multi-rule recursion**: predicate `p` *derives* `q` (`p → q`) if `p` appears in the body of a rule with head `q`. Define `→+` as the transitive closure. A predicate is **recursive** if `p →+ p`. Two predicates are **mutually recursive** if `p →+ q` and `q →+ p`.

The set of recursive predicates decomposes into disjoint **blocks of mutually recursive predicates**. A set of rules is **linear** if every recursive rule is linear.

**Example of a linear system**:
```prolog
r1: p(X,Y) :- p1(X,Z), q(Z,Y)
r2: q(X,Y) :- p(X,Z), p2(Z,Y)
r3: p(X,Y) :- b3(X,Y)
r4: p1(X,Y) :- b1(X,Z), p1(Z,Y)
r5: p1(X,Y) :- b4(X,Y)
r6: p2(X,Y) :- b2(X,Z), p2(Z,Y)
r7: p2(X,Y) :- b5(X,Y)
```
Recursive predicates: `{p,q,p1,p2}`. Mutually recursive blocks: `{[p,q], [p1], [p2]}`.

---

### 2.6. Safety of Queries

A query `q` in database `D` is **safe** if its answer is finite. Unsafe queries are highly undesirable.

**Sources of unsafeness**:
1. Evaluable predicates have infinite natural interpretations (e.g., `greater-than(27,X)` is unsafe)
2. Rules with free variables in the head not appearing in the body

A rule is **range restricted** if every variable in the head appears somewhere in the body.

A rule is **strongly safe** if:
1. It is range restricted
2. Every variable in an evaluable predicate also appears in at least one base predicate

A set of strongly safe rules guarantees all queries are safe **and** safely computable.

**Safety dependencies**: for arithmetic predicates, a dependency `X → Y` means "fixing values of X yields finitely many values of Y." For `sum`:
```
{1,2} → {3},   {1,3} → {2},   {2,3} → {1}
```
For `greater-than`: only trivial dependencies.

---

### 2.7. Effective Computability

Safety ≠ effective computability. **Bottom-up evaluability** gives a sufficient condition.

A body variable is **secure** if it:
- appears in a non-evaluable predicate, OR
- appears in position `i` of an evaluable predicate `p` where a secure subset `I` satisfies `I → {i}`

A rule is **bottom-up evaluable** if it is range restricted AND every body variable is secure.

Any computation using only bottom-up evaluable rules can be carried out without materializing infinite intermediate results. However, there may be infinitely many steps.

---

## 3. The Strategies

### 3.1. Characteristics of the Strategies

#### 3.1.1. Query Evaluation vs. Query Optimization

**Query evaluation methods** (produce answers directly):
- Naive Evaluation, Semi-Naive Evaluation, QSQ (Iterative & Recursive), APEX, Prolog, Henschen-Naqvi

**Query optimization strategies** (rewrite rules to make evaluation more efficient; used as term rewriting systems on top of Naive/Semi-Naive):
- Aho-Ullman, Kifer-Lozinskii, Magic Sets, Counting, Reverse Counting

#### 3.1.2. Interpretation vs. Compilation

**Compiled**: two phases — (1) compilation phase accessing only the IDB generates an "object program"; (2) execution phase runs against facts only. All database query forms are generated during compilation, enabling DBMS precompilation.

**Interpreted**: no object code; a fixed interpreter runs against the query, rules, and facts.

#### 3.1.3. Recursion vs. Iteration

**Iterative**: target program uses loops; data is statically determined (finite number of temporary relations).

**Recursive**: interpreter uses a stack; unbounded number of temporary relations.

#### 3.1.4. Potentially Relevant Facts

A fact `p(a)` is **relevant** to query `q` iff there exists a derivation `p(a) →* q(b)` for some answer `b`.

A **sufficient set of relevant facts** gives the same answer as the full database. Since finding all relevant facts is generally as hard as answering the query, methods compute a **set of potentially relevant facts** (a superset). The trivial valid set is all facts, but smarter methods restrict this significantly.

#### 3.1.5. Top-Down vs. Bottom-Up

Viewing rules as grammar productions (base predicates = terminals, derived = non-terminals, query = start symbol):

**Bottom-up**: start from terminal symbols (base relations), assemble until reaching the query. Simple but computes many useless results.

**Top-down**: start from the query, expand using rules. More efficient (knows the query) but more complex.

All evaluation methods: (i) generate the language, (ii) evaluate sentences, (iii) check termination at each step.

---

### 3.2. The Methods

**Common example** used for all methods:

```prolog
% Intensional database:
r1: ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y)
r2: ancestor(X,Y) :- parent(X,Y)
r3: query(X)      :- ancestor(aa, X)

% Extensional database:
parent(a,   aa)
parent(a,   ab)
parent(aa,  aaa)
parent(aa,  aab)
parent(aaa, aaaa)
parent(c,   ca)
```

---

#### 3.2.1. Naive Evaluation

| Property | Value |
|----------|-------|
| Direction | Bottom-up |
| Style | Compiled, Iterative |
| Domain | Bottom-up evaluable rules |

Compile rules into an iterative fixpoint program. Assign a temporary relation to each derived predicate. Apply recursive rules in a loop until no new tuples are generated.

**Object program**:
```
begin
  ancestor := {}
  -- Apply r2: ancestor(X,Y) :- parent(X,Y)
  insert parent into ancestor
  while new tuples generated do
    -- Apply r1: ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y)
    compute and insert into ancestor
  -- Apply r3: query(X) :- ancestor(aa,X)
  compute and insert into query
end
```

**Execution trace** on example data:

| Step | New tuples added to `ancestor` |
|------|-------------------------------|
| 1 (r2) | (a,aa),(a,ab),(aa,aaa),(aa,aab),(aaa,aaaa),(c,ca) |
| 2 (r1) | (a,aaa),(a,aab),(aa,aaaa) |
| 3 (r1) | (a,aaaa) |
| 4 (r1) | none — loop ends |
| 5 (r3) | query = {(aa,aaa),(aa,aaaa)} |

**Problems**: (1) evaluates the entire relation (no use of query bindings); (2) each step completely duplicates the previous step's work.

---

#### 3.2.2. Semi-Naive Evaluation

| Property | Value |
|----------|-------|
| Direction | Bottom-up |
| Style | Compiled, Iterative |
| Domain | Bottom-up evaluable rules |

**Key idea**: compute only the *differential* `dφ` at each step instead of recomputing all of `φ`. For each recursive predicate `p`, maintain `p_before`, `p_after`, `dp_before`, `dp_after`.

The guarantee needed (where `+` denotes union):
```
φ(p1, ...) - φ(p1, ...) ⊆ dφ(p1, dp1, ...) ⊆ φ(p1+dp1, ...)
```

**Rewrite rules** (examples):
- If `φ(p,q) = p(X,Y), q(Y,Z)` → `dφ(p,dp,q) = dp(X,Y), q(Y,Z)`
- If `φ(p1,p2) = p1(X,Y), p2(Y,Z)` → `dφ = p1(X,Y),dp2(Y,Z) + dp1(X,Y),p2(Y,Z)`

For **linear rules**: `dφ(p) = φ(dp)` — replace `p` by `dp`.

> **Warning**: For non-linear rules (e.g., `ancestor(X,Y) :- ancestor(X,Z), ancestor(Z,Y)`), applying naive evaluation only to "new tuples" is **incorrect** — it produces only ancestors at power-of-two distances.

**Object program** (loop body):
```
while state changes do
  for each mutually recursive predicate p do
    dp_after := {}
    p_after  := p_before
  for each mutually recursive rule do
    evaluate dφ(p1, dp1, ..., pn, dpn, q1, ..., qm)
    add results to dp_after and p_after
```

---

#### 3.2.3. Iterative Query/Subquery (QSQI)

| Property | Value |
|----------|-------|
| Direction | Top-down |
| Style | Interpreted, Iterative |
| Domain | Range restricted, no arithmetic |

Maintains state `<Q, R>`:
- `Q` = set of **generalized queries** (batching, e.g., `ancestor({aa,aaa},X)`)
- `R` = derived relations with current values

**Interpreter**:
```
initial state: <{query(X)}, {}>
while state changes do
  for all generalized queries q in Q:
    for all rules whose head matches q:
      unify rule with q (propagate constants)
      generate new generalized queries for derived predicates in body
      generate new tuples from base relations
      add tuples to R, queries to Q
```

**Execution trace** (abbreviated):

| Step | Q | ancestor |
|------|---|----------|
| 1 | {query(X), ancestor({aa},X)} | {} |
| 2 | {query(X), ancestor({aa},X)} | {(aa,aaa),(aa,aab)} |
| 3 | {query(X), ancestor({aa,aaa,aab},X)} | {(aa,aaa),(aa,aab),(aaa,aaaa)} |
| 4 | {query(X), ancestor({aa,aaa,aab,aaaa},X)} | same + query={(aa,aaa),(aa,aaaa)} |

**Compared to Naive**: better potentially-relevant facts (close to optimal), but same duplication problem.

---

#### 3.2.4. Recursive Query/Subquery (QSQR) / Extension Tables

| Property | Value |
|----------|-------|
| Direction | Top-down |
| Style | Interpreted, Recursive |
| Domain | Range restricted, no arithmetic |

Recursive version of QSQI. Uses a **selection function** to pick the next derived predicate to solve. Uses dynamic programming: stores and reuses intermediate results. **Handles cycles** in data (Prolog loops infinitely; QSQR detects fixpoint and stops). Complete over its application domain.

**Recursive interpreter**:
```
procedure evaluate(q):
  while new tuples are generated do
    for all rules matching q:
      unify rule with q
      for each derived predicate p_i in body:
        generate generalized query q' for p_i
        remove queries already in Q from q'
        add q' to Q
        evaluate(q')                    ← recursive call
      evaluate full body using current values
      add results to R
      return results
```

**Key differences from Prolog**:
1. Set-at-a-time (generalized queries), not tuple-at-a-time
2. Dynamic programming — memoizes sub-query results, handles cycles

---

#### 3.2.5. Henschen-Naqvi

| Property | Value |
|----------|-------|
| Direction | Top-down |
| Style | Compiled, Iterative |
| Domain | Linear range restricted rules |

Designed for the canonical pattern:
```prolog
p(X,Y) :- up(X,XU), p(XU,YU), down(YU,Y)
p(X,Y) :- flat(X,Y)
query(X) :- p(a, X)
```

Using set-to-set mapping notation (`A·r = {y | r(x,y), x ∈ A}`), the full answer is:
```
{a}·flat  +  {a}·up·flat·down  +  {a}·up²·flat·down²  + ...
```

State `<V, E>` where `V` is the current node set and `E` is the accumulated "down-path" expression:

**Iterative program**:
```
V := {a}
E := λ            ← empty string
while new tuples generated in V do
  answer := answer + V·flat·E
  V := V·up
  E := E | down   ← cons 'down' onto E
```

At step `i`: `V = {a}·upⁱ`, `E = downⁱ`, so produced tuples are `{a}·upⁱ·flat·downⁱ`.

---

#### 3.2.6. Prolog

| Property | Value |
|----------|-------|
| Direction | Top-down |
| Style | Interpreted, Recursive |
| Domain | Data-dependent (requires acyclic EDB); no simple syntactic characterization |

Tuple-at-a-time, depth-first, no memoization. Mentioned for completeness and performance comparison. Key limitations:
- Does not terminate on cyclic data
- Performance degrades sharply with duplication in data structure
- Cannot propagate constants in certain binding patterns

---

#### 3.2.7. APEX

| Property | Value |
|----------|-------|
| Direction | Mixed |
| Style | Mixed (partly compiled), Recursive |
| Domain | Range restricted, no constants, no arithmetic |

Uses a predicate connection graph for preprocessing. Selects **relevant facts** from base predicates, then iterates: for each rule, instantiates body predicates with relevant/useful facts, recursively solves sub-queries (the recursion step), and adds results.

**Interpreter sketch**:
```
procedure solve(q, answer):
  if q is on a base relation: evaluate directly
  else:
    relevant := select relevant base facts for q
    while new tuples generated do
      for each rule:
        instantiate right predicates with relevant facts
        for each matching fact, plug into rule and propagate constants
        for each new sub-query q': solve(q', answer(q'))
        instantiate right predicates with useful facts
        produce tuples for left predicate
```

**Key limitation**: APEX does not distinguish which literals a fact is relevant *to*. This causes incorrect sub-queries when certain arguments are bound (e.g., for `ancestor(?,john)` it may compute `ancestor(john,?)` instead).

---

### 3.3. Optimization Strategies

The main drawbacks of naive evaluation:
1. Potentially relevant facts set is too large (ignores query bindings)
2. Generates duplicate computation

#### 3.3.1. Aho-Ullman

Optimizes recursive queries by **commuting selections with the Least Fixpoint (LFP) operator**.

Input: `σ_F(LFP(r = f(r)))` where `f(r)` is monotonic and contains at most one occurrence of `r`.

**Procedure**: Construct a series of equivalent expressions by repeatedly replacing `r` with `f(r)`, pushing the selection inside as far as possible. Succeeds when finding `h(LFP(s=g(s)))` where a previous expression was `h(LFP(s))`.

**Example** — transitive closure with `a1=john`:
```
σ_{a1=john}(a)
→ σ_{a1=john}(a·p ∪ p)           ← replace a by f(a)
→ σ_{a1=john}(a)·p ∪ σ_{a1=john}(p)   ← distribute
→ LFP(E = E·p ∪ σ_{a1=john}(p))   ← recognize fixpoint
```

Equivalent Horn clauses:
```prolog
a(john,Y) :- a(john,Z), p(Z,Y)
a(john,Y) :- p(john,Y)
```

**Limitation**: requires `f(r)` to contain at most one occurrence of `r`.

---

#### 3.3.2. Kifer-Lozinskii

Extension of Aho-Ullman using **filters on arcs** of the rule/goal graph. Filters are selections applied to tuples flowing through arcs.

**Procedure**: Starting with the filter representing the query constant, repeatedly **push filters through nodes**:
- Through a **relation node**: place disjunction of incoming filters on each incoming arc
- Through an **axiom node**: place the strongest consequence of the disjunction expressible in terms of the arc's literal variables

**Advantages over Aho-Ullman**:
- Handles expressions with more than one occurrence of the defined predicate, e.g.:
  ```
  σ_{a1=john}(LFP(a = a·p ∪ a·q ∪ p))
  ```
  → optimizes to `LFP((σ_{a1=john}(a)·p) ∪ (σ_{a1=john}(a)·q) ∪ σ_{a1=john}(p))`
- Can work directly on certain mutually recursive rules

**Failures**: Cannot optimize:
- `σ_{a1=john}(LFP(a = a·a ∪ p))`
- `σ_{a1=john}(LFP(a = a·p ∪ p·a ∪ p))`

---

#### 3.3.3. Magic Sets

| Property | Value |
|----------|-------|
| Direction | Bottom-up |
| Style | Compiled, Iterative |
| Domain | Bottom-up evaluable rules |

**Key idea**: Simulate sideways passing of bindings (à la Prolog) by introducing new rules, reducing the set of potentially relevant facts.

**Transformation** (three steps):

**Step 1 — Generate adorned rules**: Adorn each predicate with `b`/`f`. An argument is *distinguished* if it is bound in the head adornment, is a constant, or appears in a base predicate with a distinguished argument. Propagate bindings through base predicates only.

**Step 2 — Generate magic rules**: For each adorned derived literal on the right of an adorned rule:
1. Erase non-distinguished body variables and other derived literals
2. Replace predicate name `p^a` with `magic_p^a`
3. Exchange head and body magic predicates

**Step 3 — Generate modified rules**: For each adorned rule with head `p^a`, add `magic_p^a(X)` to the right-hand side (X = distinguished variables in that occurrence).

**Example** (same-generation rule `sg(X,Y) :- p(X,XP), p(Y,YP), sg(YP,XP)`, adornment `bf`):

Complete modified rule set:
```prolog
magic^{fb}(XP) :- p(X,XP), magic^{bf}(X)
magic^{bf}(YP) :- p(Y,YP), magic^{fb}(Y)
magic^{bf}(a)

sg^{bf}(X,Y)  :- p(X,XP), p(Y,YP), magic^{bf}(X), sg^{fb}(YP,XP)
sg^{fb}(X,Y)  :- p(X,XP), p(Y,YP), magic^{fb}(Y), sg^{bf}(YP,XP)
sg^{bf}(X,X)  :- magic^{bf}(X)
sg^{fb}(X,X)  :- magic^{fb}(X)
query^f(X)    :- sg^{bf}(a,X)
```

Magic rules simulate backward chaining for binding propagation.

---

#### 3.3.4. Counting and Reverse Counting

Derived from Magic Sets. Apply when:
1. Data is **acyclic**
2. At most one recursive rule per predicate, and it is **linear**

**Counting** introduces *counting sets* — magic sets where elements are numbered by distance from the query constant `a`:

```prolog
counting(a, 0)
counting(X,I) :- counting(Y,J), up(Y,X), I=J+1
p'(X,Y,I)    :- counting(X,I), flat(X,Y)
p'(X,Y,I)    :- counting(X,I), up(X,XU), p'(XU,YU,J), down(YU,Y), I=J-1
query(X)     :- p'(a, X, 0)
```

Further simplified (eliminating first attribute — simulates a stack):
```prolog
counting(a, 0)
counting(X,I) :- counting(Y,J), up(Y,X), I=J+1
p''(Y,I)     :- counting(X,I), flat(X,Y)
p''(Y,I)     :- p''(YU,J), down(YU,Y), I=J-1, J>0
query(Y)     :- p''(Y, 0)
```

**Reverse Counting**: (1) compute the magic set; (2) for each element `b` in the magic set, number all its `down` and `up` descendants; (3) answer = `down` descendants with same distance as `a` in the `up` chain.

---

### 3.4. Summary of Strategy Characteristics

| Method | Application Range | Direction | Compiled? | Iterative? |
|--------|------------------|-----------|-----------|-----------|
| Naive Evaluation | Bottom-up Evaluable | Bottom-Up | Yes | Yes |
| Semi-Naive Evaluation | Bottom-up Evaluable | Bottom-Up | Yes | Yes |
| QSQ Iterative | Range Restricted, No Arithmetic | Top-Down | No | Yes |
| QSQ Recursive | Range Restricted, No Arithmetic | Top-Down | No | No |
| APEX | Range Restricted, No Arithmetic, Constant-Free | Mixed | Mixed | No |
| Prolog | User responsible | Top-Down | No | No |
| Henschen-Naqvi | Linear | Top-Down | Yes | Yes |
| Aho-Ullman | Strongly Linear | Bottom-Up | Yes | Yes |
| Kifer-Lozinskii | Range Restricted, No Arithmetic | Bottom-Up | Yes | Yes |
| Counting | Strongly Linear | Bottom-Up | Yes | Yes |
| Magic Sets | Bottom-up Evaluable | Bottom-Up | Yes | Yes |

---

## 4. Performance Comparisons

### 4.1. Workload

**Query 1** — Ancestor, first argument bound (bf):
```prolog
a(X,Y) :- p(X,Y)
a(X,Y) :- p(X,Z), a(Z,Y)
query(X) :- a(john, X)
```

**Query 2** — Ancestor, second argument bound (fb):
```prolog
a(X,Y) :- p(X,Y)
a(X,Y) :- p(X,Z), a(Z,Y)
query(X) :- a(X, john)
```

**Query 3** — Ancestor, non-linear (recursive doubling):
```prolog
a(X,Y) :- p(X,Y)
a(X,Y) :- a(X,Z), a(Z,Y)
query(X) :- a(john, X)
```

**Query 4** — Same generation, first argument bound:
```prolog
p(X,Y) :- flat(X,Y)
p(X,Y) :- up(X,XU), p(XU,YU), down(YU,Y)
query(X) :- p(john, X)
```

---

### 4.2. Data Characterization

Every binary relation is characterized by four parameters:

| Symbol | Name | Definition |
|--------|------|-----------|
| `F_R` | Fan-out | Given set A of n nodes: `|A·R| = n·F_R` (before dedup) |
| `D_R` | Duplication | Ratio of size before/after duplicate elimination |
| `h_R` | Height | Length of longest chain in R |
| `b_R` | Base | Number of nodes with no antecedents |

**Expansion factor**: `E_R = F_R / D_R`

**Typical structures** (all with 100,000 tuples):

| Structure | F | D | Shape varied by |
|-----------|---|---|----------------|
| Tree | 2 | 1 | Fan-out F |
| Inverted Tree | 1 | 2 | Duplication D |
| Cylinder | 2 | 2 | Breadth/height ratio b/h |

**Transfer ratio** `T_{AB}`: given n nodes in A, `|A ∩ B|` after dedup = `n·T_{AB}`.

---

### 4.3. Cost Metric

**Cost = number of successful inferences = size of intermediate results before duplicate elimination.**

Complexity by operation:
- Join, Cartesian product, intersection, selection → size of result
- Union → sum of argument sizes
- Projection → size of argument

---

### 4.4. Cost Function Notation

- `n_R(i)` = nodes at level `i` in relation R
- `A_R` = total arcs (tuples) in R
- `gsum(E,h) = 1 + E + E² + ... + Eʰ`
- `a(l) = n(l)·gsum(E, h-l)` = arcs of length exactly `l` in transitive closure R*
- `h' = h - ⌊(Σᵢ i·n(i)) / N⌋` = distance of mean level from highest level

---

### 4.5. Cost Functions

#### Query 1 (Ancestor.bf)

| Strategy | Cost |
|----------|------|
| **Naive** | `D·Σᵢ(h-i+1)·a(i) + E·gsum(E,h'-1)` |
| **Semi-Naive** | `D·Σᵢ a(i) + E·gsum(E,h'-1)` |
| **QSQ Iterative** | `E·gsum(E,h'-1) + F·Σᵢ(h'-i+1)·i·Eⁱ⁻¹` |
| **QSQ Recursive** | `(F+E)·gsum(E,h'-1) + D·Σᵢ Eⁱ·gsum(E,h'-i)` |
| **Henschen-Naqvi** | `(F+E)·gsum(E,h'-1)` |
| **Prolog** | `gsum(F,h') + E·gsum(E,h'-1) + Σᵢ Fⁱ·gsum(F,h'-i)` |
| **APEX** | `(F+E)·gsum(E,h'-1) + D·Σᵢ Eⁱ·gsum(E,h'-i)` |
| **Kifer-Lozinskii** | `D·Σᵢ a(i) + E·gsum(E,h'-1)` |
| **Magic Sets** | `(F+E)·gsum(E,h'-1) + D·Σᵢ Eⁱ·gsum(E,h'-i)` |
| **Counting** | `(F+E)·gsum(E,h'-1)` |

#### Query 3 (Ancestor.bf, Non-Linear)

| Strategy | Cost |
|----------|------|
| **Naive** | `E·gsum(E,h'-1) + D·Σᵢ(log(h/i)+1)·(i-1)·a(i)` |
| **Semi-Naive** | `E·gsum(E,h'-1) + D·Σᵢ(i-1)·a(i)` |
| **QSQ Iterative** | `E·gsum(E,h'-1) + F·Σᵢ(h'-i+1)·i·Eⁱ⁻¹` |
| **QSQ Recursive** | `F + E·gsum(E,h'-1) + D·Σᵢ(i-1)·Eⁱ` |
| **Henschen-Naqvi** | *Does not apply* |
| **Prolog** | *Does not terminate* |
| **Kifer-Lozinskii** | `E·gsum(E,h'-1) + D·Σᵢ(i-1)·a(i)` |
| **Magic Sets** | `E·gsum(E,h'-1) + D·Σᵢ(i-1)·a(i)` |
| **Counting** | *Does not apply* |

---

### 4.6. Summary of Cost Orderings

Using `<<` to denote an order-of-magnitude or greater difference. Strategies in parentheses perform identically.

#### Query 1 (Ancestor.bf)

| Data | Ordering |
|------|----------|
| Tree | (HN,C) << (MS,QSQR,APEX) = P << QSQI << (SN,KL) << N |
| Inverted Tree | (HN,C) << (MS,QSQR,APEX) << P << QSQI << (SN,KL) << N |
| Cylinder | (HN,C) << (MS,QSQR,APEX) << QSQI << (SN,KL) << N << P |

#### Query 2 (Ancestor.fb)

| Data | Ordering |
|------|----------|
| All | (HN,C) << (MS,QSQR,KL) << QSQI << APEX << SN << N ≈ P |

#### Query 3 (Ancestor.bf, Non-Linear)

| Data | Ordering |
|------|----------|
| All | QSQR << QSQI << APEX << (SN,MS,KL) << N |
| Note | HN, Counting, Prolog: not applicable |

#### Query 4 (Same Generation.bf)

| Data | Ordering |
|------|----------|
| Tree | C << HN ≈ (MS,QSQR,APEX) = P << QSQI << (SN,KL) << N |
| Inverted Tree | C << HN ≈ (MS,QSQR,APEX) << P << QSQI << (SN,KL) << N |
| Cylinder | C << HN ≈ (MS,QSQR,APEX) << QSQI << (SN,KL) << N << P |

**General ancestor ordering**:
```
(HN, C)  <<  (MS, QSQR)  <<  QSQI  <<  APEX  <<  SN  <<  N
```

---

### 4.7. Interpreting the Results

Three factors greatly influence performance:

#### Factor 1: Duplication of Work

| Cause | Examples |
|-------|---------|
| Multiple derivation paths in data | Prolog (re-derives same fact along every path) |
| Iterative control without memory | QSQI, Naive (recomputes previous steps) |

QSQR avoids duplication via recursive memoization.
Semi-Naive avoids it by computing only differentials.

#### Factor 2: Size of Potentially Relevant Facts

| Strategy | Relevant facts set |
|----------|--------------------|
| HN, Counting, Magic Sets | Nodes reachable from query node |
| Semi-Naive, KL (when opt fails) | All nodes in the relation |
| APEX | Subset, but misidentifies which literals they apply to |

In the **non-linear case**:
- Magic Sets degenerates to Semi-Naive (cannot determine relevant facts)
- HN and Counting do not apply
- Prolog does not terminate
- **QSQR is the only strategy that both restricts relevant facts AND avoids duplicates**

#### Factor 3: Arity of Intermediate Relations

| Arity | Strategies | Properties |
|-------|-----------|------------|
| **Unary** (node sets) | Henschen-Naqvi, Counting | Process each node ≤D times; fail on cycles; don't handle non-linear rules |
| **Binary** (arc sets) | All others | Compute transitive closure arcs; more expensive but more general |

**QSQR insight**: Uses binary intermediate relations but via recursive sub-queries, effectively computing "arcs in the transitive closure of the subgraph rooted at the query node." This matches the cost of Magic Sets while uniquely handling both relevant-facts restriction and duplication avoidance.

**APEX** distinguishes between *relevant facts* and *useful facts* but fails to track which literals a relevant fact is useful for — a distinction that Magic Sets correctly makes.

---

### 4.8. Summary and Caveats

**Main conclusions**:

1. For a given query, there is a clear ordering of strategies — robust across data configurations
2. More specialized strategies perform significantly better (differences of **orders of magnitude**)
3. Recursion is a powerful control structure: reduces relevant facts and eliminates duplicate work
4. Choosing the right strategy is critical
5. Three performance factors: **(i) duplication**, **(ii) relevant facts**, **(iii) arity of intermediate relations**

**Caveats**:
- Cost function: Join cost is linear in result size; disk access and recursive control overhead are ignored
- Data: No cycles or shortcuts — favors specialized strategies (e.g., there are cases with shortcuts where Counting performs worse than Magic Sets)
- Benchmark: Favors large-data/small-answer scenarios; Semi-Naive matches any strategy for computing the full transitive closure

---

## 5. Conclusions

This paper has described and comparatively evaluated the major strategies for processing recursive logic queries without function symbols.

**Identified characteristics**: method vs. optimization, top-down vs. bottom-up, recursive vs. iterative, compiled vs. interpreted. The taxonomy of strategies is still open — some characterizations are somewhat arbitrary (e.g., SNIP is an interpreted version of Naive; a compiled iterative QSQ is also reasonable).

**Performance results** explained by three factors:
1. **Duplication** — well-known
2. **Relevant facts** — well-known
3. **Unary vs. binary intermediate relations** — surprising result (though likely understood in [Sacca and Zaniolo 86])

The cost of the optimal strategy is less than 10,000 on all queries tested with 100,000-tuple datasets, demonstrating that recursive queries can be implemented efficiently with the right choice of strategy.

---

## References

- **[Aho and Ullman 79]** "Universality of Data Retrieval Languages," *Proc. 6th ACM POPL*, 1979.
- **[Bancilhon 85]** "Naive Evaluation of Recursively Defined Relations," in *On Knowledge Base Management Systems*, Springer-Verlag.
- **[Bancilhon 85a]** "A Note on the Performance of Rule Based Systems," MCC Technical Report DB-022-85.
- **[Bancilhon et al. 86]** "Magic Sets and Other Strange Ways to Implement Logic Programs," *Proc. 5th ACM PODS*, 1986.
- **[Bancilhon et al. 86a]** "Magic Sets Algorithms and Examples," Unpublished Manuscript, 1986.
- **[Bancilhon and Ramakrishnan 86]** "Performance Evaluation of Data Intensive Logic Programs," Unpublished Manuscript, March 1986.
- **[Chang 81]** "On the Evaluation of Queries Containing Derived Relations in Relational Databases," in *Advances in Data Base Theory*, Vol. 1, Plenum Press.
- **[Dietrich and Warren 85]** "Dynamic Programming Strategies for the Evaluation of Recursive Queries," Unpublished Report.
- **[Gallaire et al. 84]** "Logic and Data Bases: A Deductive Approach," *Computing Surveys*, Vol. 16, No. 2, June 1984.
- **[Han and Lu 86]** "Some Performance Results on Recursive Query Processing in Relational Database Systems," *Proc. Data Engineering Conference*, Los Angeles, February 1986.
- **[Henschen and Naqvi 84]** "On Compiling Queries in Recursive First-Order Databases," *JACM*, Vol. 31, January 1984, pp. 47–85.
- **[Kifer and Lozinskii 85]** "Query Optimization in Logic Databases," Technical Report, SUNY at Stonybrook.
- **[Lozinskii 85]** "Evaluating Queries in Deductive Databases by Generating," *Proc. 11th IJCAI*, 1985.
- **[McKay and Shapiro 81]** "Using Active Connection Graphs for Reasoning with Recursive Rules," *Proc. 7th IJCAI*, 1981.
- **[Morris et al. 86]** "Design Overview of the NAIL! System," *Proc. 3rd Int. Conf. on Logic Programming*, 1986.
- **[Reiter 78]** "Deductive Question Answering on Relational Data Base," in *Logic and Data Bases*, Plenum Press.
- **[Rohmer and Lescoeur 85]** "La Méthode Alexandre," *Colloque RFIA*, Grenoble, November 1985.
- **[Sacca and Zaniolo 86a]** "On the Implementation of a Simple Class of Logic Queries for Databases," *Proc. 5th ACM PODS*, 1986.
- **[Sacca and Zaniolo 86b]** "Implementing Recursive Logic Queries with Function Symbols," Unpublished Manuscript, April 1986.
- **[Tarski 55]** "A Lattice Theoretical Fixpoint Theorem and its Applications," *Pacific Journal of Mathematics* 5, pp. 285–309.
- **[Ullman 85]** "Implementation of Logical Query Languages for Databases," *TODS*, Vol. 10, No. 3, pp. 289–321.
- **[Van Emden and Kowalski 76]** "The Semantics of Predicate Logic as a Programming Language," *JACM*, Vol. 23, No. 4, October 1976, pp. 733–742.
- **[Vieille 85/86]** "Recursive Axioms in Deductive Databases: The Query/Subquery Approach," *Proc. First Int. Conf. on Expert Database Systems*, Charleston, 1986.
- **[Zaniolo 86]** "Safety and Compilation of Non-Recursive Horn Clauses," *Proc. First Int. Conf. on Expert Database Systems*, Charleston, 1986.
