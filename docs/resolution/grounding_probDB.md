# Grounding in Probabilistic Databases and Neuro-Symbolic Systems

How the grounding strategies from `grounding_basics.md` extend to settings where
facts carry probabilities and the goal is not just "is Q derivable?" but
"what is P(Q)?" or "what is score(Q)?".

---

## 1. Probabilistic databases

### 1.1 Tuple-independent databases (TI-PDBs)

The simplest probabilistic data model (Suciu et al., 2011).  Each tuple
(ground atom) t exists independently with probability p(t):

```
0.9 :: parent(alice, bob).
0.8 :: parent(bob, charlie).
0.7 :: parent(bob, diana).
```

A TI-PDB defines a distribution over **possible worlds**: each world is a
classical database obtained by independently including or excluding each tuple.
With n tuples, there are 2ⁿ possible worlds.

### 1.2 Query semantics: possible worlds

The probability of a query answer Q is the sum over all worlds W where Q is true:

```
P(Q) = Σ_{W : Q is true in W}  P(W)

where P(W) = ∏_{t ∈ W} p(t) · ∏_{t ∉ W} (1 - p(t))
```

Computing this exactly requires enumerating all relevant worlds — exponential
in general.

### 1.3 Lineage (provenance)

The **lineage** of a query answer is a Boolean formula over the tuple variables
that is true exactly when the query is true.  For `ancestor(alice, charlie)`:

```
lineage = parent(alice,bob) ∧ parent(bob,charlie)          [via R1→R2]
        ∨ parent(alice,charlie)                              [via R1, if it existed]
```

Each proof path gives a conjunction (all body atoms must be true).  Multiple
proof paths give a disjunction (any proof suffices).  The probability of the
query is the probability that this Boolean formula evaluates to true:

```
P(Q) = P(lineage(Q))
```

This reduces query evaluation to **weighted model counting (WMC)** on the
lineage formula — a well-studied problem.

### 1.4 The role of grounding

Grounding is what produces the lineage.  Each ground rule instance contributes
one conjunction to the DNF:

```
Rule grounding:    ancestor(a,c) :- parent(a,b), ancestor(b,c).
Conjunction:       parent(a,b) ∧ ancestor(b,c)
```

The grounder must find **all** valid ground instances to construct the complete
lineage formula.  Missing a ground instance means missing a proof path, which
underestimates P(Q).

---

## 2. The ProbLog pipeline

ProbLog (De Raedt et al., 2007) is the canonical system for probabilistic logic
programming.  Its inference pipeline illustrates how grounding connects to
probability computation:

```
                                                    ┌─────────────┐
Query ──→ [1. Ground] ──→ [2. Convert] ──→ [3. Compile] ──→ [4. WMC] ──→ P(Q)
                                                    └─────────────┘
```

### Step 1: Grounding

Use SLD-based backward chaining (Prolog-style) to find all ground program
clauses relevant to the query.  This is goal-directed: only clauses reachable
from the query are grounded.

The result is a **relevant ground program** — a set of ground rules and the
probabilistic facts they depend on.

### Step 2: Convert to Boolean formula

The ground program is converted to a propositional formula.  Each ground atom
becomes a Boolean variable.  Rules become implications.  The formula encodes:
"the query is true iff its lineage is satisfiable."

### Step 3: Compile to tractable form

The Boolean formula is compiled into a knowledge compilation target:
- **BDD** (Binary Decision Diagram)
- **SDD** (Sentential Decision Diagram)
- **d-DNNF** (deterministic Decomposable Negation Normal Form)

These representations allow efficient WMC.

### Step 4: Weighted model counting

Traverse the compiled structure, multiplying probabilities along branches.
This gives P(Q) exactly.

### The grounding bottleneck

Step 1 (grounding) is often the bottleneck: for recursive programs with many
entities, the relevant ground program can be enormous.  Fierens et al. (2015)
showed that the ground program can be exponential in the proof depth.

Vlasselaer et al. (2020) addressed this with **Datalog techniques for
ProbLog**: replacing SLD-based grounding with semi-naive evaluation and magic
sets to avoid redundant computation during grounding.  This "grounding
bottleneck" paper showed orders-of-magnitude speedups on recursive programs.

---

## 3. Neuro-symbolic extensions

### 3.1 DeepProbLog (Manhaeve et al., 2018)

Extends ProbLog by introducing **neural predicates**: probabilistic facts whose
probabilities are parameterised by neural networks.

```
nn(mnist_net, [Image], Digit, [0,1,...,9]) :: digit(Image, Digit).
```

The grounding pipeline is identical to ProbLog.  The only change: after
grounding, neural predicates are evaluated by forward-passing through the
network.  Gradients flow back through the WMC computation to train the network.

### 3.2 DeepStochLog (Winters et al., 2022)

Uses stochastic definite clause grammars instead of ProbLog's distribution
semantics.  Grounding is based on **SLD-like derivation trees** restricted to a
grammar structure, which avoids the grounding bottleneck for many programs.

### 3.3 The generate-score-aggregate pattern

Most neuro-symbolic systems for knowledge graphs follow a common pattern:

```
Query → [Ground] → {ground rule instances} → [Score] → [Aggregate] → P(Q)
```

1. **Ground**: find all relevant ground rule instances for the query.
2. **Score**: assign a score to each body atom (via a neural model like a
   knowledge graph embedding).
3. **Aggregate**: combine scores across body atoms (within a proof) and across
   proofs (for the query).

The grounding step is independent of the scoring model.  Different systems
differ mainly in how they score and aggregate:

| System | Scoring | Aggregation |
|--------|---------|-------------|
| ProbLog | Exact probabilities | WMC (exact) |
| DeepProbLog | Neural network outputs | WMC (exact) |
| pLogicNet (Qu & Tang, 2019) | KGE embeddings | MLN energy function |
| NTP (Rocktäschel & Riedel, 2017) | Soft unification scores | Product of scores per proof |

---

## 4. TensorLog: rules as matrix operations

TensorLog (Cohen, 2016) takes a different approach: instead of grounding and
then scoring, it **compiles rules into differentiable matrix operations**.

### 4.1 Representation

Each binary predicate r is represented as a sparse matrix **M**_r where:
```
M_r[i, j] = θ_r(i,j)   if r(i,j) ∈ DB
             0           otherwise
```

θ_r(i,j) is a confidence parameter (probability) for the fact.  Unary
predicates become row vectors.  Constants become one-hot vectors.

### 4.2 Rule compilation

A rule like:
```
p(X, Z) :- q(X, Y), r(Y, Z).
```

compiles into a matrix multiplication chain:
```
f_p(u_X) = u_X · M_q · M_r
```

where u_X is the one-hot vector for the query constant X.  The result is a
vector over all entities, where entry Z gives the score for p(X, Z).

For a query `p(alice, ?)`:
1. Start with one-hot vector u_alice.
2. Multiply by M_q: result[Y] = 1 iff q(alice, Y) ∈ DB.  This is the
   "constant-anchored enumeration" — using alice as the anchor into the q table.
3. Multiply by M_r: result[Z] = Σ_Y M_q[alice,Y] · M_r[Y,Z].  This is the
   join on Y.

Multiple rules for the same head are summed before normalisation.  Recursion is
handled by **iterative deepening**: each recursive call increments a depth
counter, and computation stops at a fixed depth.

### 4.3 Connection to grounding

TensorLog does NOT produce explicit ground rule instances.  Instead, it
**implicitly** evaluates all groundings simultaneously via the matrix
multiplication.  The matrix product M_q · M_r effectively enumerates all
(X, Y, Z) triples where q(X,Y) and r(Y,Z) are both facts — the same join
that a grounder computes, but in one matrix operation.

This is efficient when the matrices are sparse (which they are for knowledge
graphs), but does not give explicit proof structures.

### 4.4 Relationship to grounding strategies

| Concept | Grounding view | TensorLog view |
|---------|---------------|----------------|
| Constant anchor (X=alice) | Index probe into fact table | One-hot vector u_alice |
| Join on Y | Nested-loop or hash join | Matrix multiplication |
| All groundings | Set of ground rule instances | Non-zero entries in result vector |
| Recursion depth D | D iterations of grounding | D matrix multiplications |

---

## 5. Soundness in the probabilistic setting

In a classical database, soundness means "every returned ground instance is
part of a valid proof."  In a probabilistic database, the stakes are higher:

### 5.1 Unsound groundings corrupt probabilities

If the grounder returns a ground rule instance where a body atom is NOT
provable, the lineage formula contains a spurious conjunction.  This can either:
- **Inflate** P(Q): the spurious proof path adds probability mass.
- **Distort gradients**: in a neuro-symbolic system, the model receives training
  signal from invalid proofs.

### 5.2 Incomplete groundings underestimate probabilities

If the grounder misses a valid ground instance, the lineage formula lacks a
disjunct.  P(Q) is underestimated, and valid proof paths are invisible to the
learning process.

### 5.3 The generate-and-filter pattern

Some systems deliberately over-generate ground instances (for efficiency or
simplicity) and then apply a **soundness filter**:

1. **Generate**: produce candidate ground rule instances, including some where
   body atoms may not be provable.
2. **Filter**: verify each candidate by checking that all body atoms are
   derivable.  This is typically done via a T_P fixpoint computation (see
   `grounding_basics.md` §3.1): iterate over the candidates, marking atoms as
   "proved" when another candidate derives them from facts, until convergence.

This separation works because:
- The **generation** step can use fast, GPU-friendly operations (enumeration,
  tensor gathers) that over-approximate the set of valid groundings.
- The **filter** step is a simple fixpoint iteration that restores soundness.

The T_P filter is equivalent to re-running naive evaluation on the collected
candidates.  If a body atom is not proved after convergence, the candidate is
discarded.

### 5.4 Scope of the filter

The T_P filter can operate at different scopes:

- **Global**: pre-compute the full provable set I* for the entire database.
  Any body atom in I* is provable.  Expensive to compute (O(P · E² · depth)),
  but complete and batch-independent.

- **Per-batch**: only use the ground instances collected within the current
  batch of queries.  Cheaper, but a body atom provable via a ground instance
  NOT in the batch will be missed.  Still sound (never marks an unprovable atom
  as proved), but incomplete.

---

## 6. Summary: the grounding problem across settings

| Setting | What the grounder produces | Why all proofs matter |
|---------|--------------------------|----------------------|
| Classical DB | Set of answers | Enumerate all solutions |
| Probabilistic DB | Lineage formula (DNF) | Compute P(Q) via WMC |
| Neuro-symbolic | Ground rule instances + body atoms | Score each proof, aggregate for P(Q) |

| Grounding strategy | Classical | Probabilistic | Neuro-symbolic |
|-------------------|-----------|---------------|----------------|
| Bottom-up (T_P) | Compute minimal model | Compute full lineage | Impractical (too many atoms) |
| Top-down (SLD) | Standard Prolog | ProbLog grounding | Proof structures, but hard to tensorise |
| Magic Sets | Query-directed bottom-up | Efficient ProbLog (Vlasselaer 2020) | Tensorisable query-directed grounding |
| Matrix compilation | N/A | TensorLog | Implicit grounding via matrix ops |

---

## References

### Probabilistic databases
- Suciu, D., Olteanu, D., Ré, C. & Koch, C. (2011). *Probabilistic Databases*. Morgan & Claypool. — Comprehensive textbook.
- Dalvi, N. & Suciu, D. (2004). *Efficient query evaluation on probabilistic databases*. VLDB Journal. — Dichotomy theorem, safe plans.

### Probabilistic logic programming
- De Raedt, L., Kimmig, A. & Toivonen, H. (2007). *ProbLog: A probabilistic Prolog and its application in link discovery*. IJCAI. — ProbLog.
- Fierens, D. et al. (2015). *Inference and learning in probabilistic logic programs using weighted Boolean formulas*. TPLP. — ProbLog2 pipeline.
- Vlasselaer, J. et al. (2020). *Beyond the grounding bottleneck: Datalog techniques for inference in probabilistic logic programs*. AAAI. — Magic sets for ProbLog.
- Manhaeve, R. et al. (2018). *DeepProbLog: Neural probabilistic logic programming*. NeurIPS. — Neural extension of ProbLog.
- Winters, T. et al. (2022). *DeepStochLog: Neural stochastic logic programming*. AAAI. — Grammar-based neural PLP.

### Differentiable reasoning
- Cohen, W. (2016). *TensorLog: A differentiable deductive database*. arXiv:1605.06523. — Rules as matrix multiplications.
- Rocktäschel, T. & Riedel, S. (2017). *End-to-end differentiable proving*. NeurIPS. — Neural Theorem Prover (NTP).
- Yang, F., Yang, Z. & Cohen, W. (2017). *Differentiable learning of logical rules for knowledge base reasoning*. NeurIPS. — NeuralLP.
- Sadeghian, A. et al. (2019). *DRUM: End-to-end differentiable rule mining on knowledge graphs*. NeurIPS.
- Qu, M. & Tang, J. (2019). *Probabilistic logic neural networks for reasoning*. NeurIPS. — pLogicNet.
