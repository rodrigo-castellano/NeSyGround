# notation.md — Shared Setting and Notation

This file defines shared notation used across grounders.

<a id="sec-0"></a>
## 0) Setting, Notation, and “Grounding as an Operator”

### 0.1 Datalog-style fragment (what your code assumes most of the time)
Most of the implementations you shared operate in a Datalog-like fragment:

- **Binary predicates**: atoms have shape `p(s,o)` (stored as triples `[pred, arg0, arg1]`).
- **Horn rules**: a single head atom and a conjunction of body atoms.
- Practical restriction in several optimized kernels: **body length 1–2** for provability precomputation (see `provable_set.py`).

We’ll write rules as:

$$
r:~ h(\vec{X}) \leftarrow b_1(\vec{Y_1}), \dots, b_m(\vec{Y_m})
$$

with variables among head vars and (possibly) existentials.

### 0.2 Core objects

- Entities/constants: $\lvert\mathcal{E}\rvert = E$
- Predicates: $\lvert\mathcal{P}\rvert = P$
- Facts (KG edges): $F \subseteq \mathcal{P}\times \mathcal{E}\times \mathcal{E}$, with $\lvert F\rvert=N_F$.
- Rules: $\mathcal{R}$, with $\lvert\mathcal{R}\rvert=N_R$, body length $m(r)$.

Useful **degree statistics** for binary relations:
- $deg_{p,s} = \lvert\{o: (p,s,o)\in F\}\rvert$
- $deg_{p,o} = \lvert\{s: (p,s,o)\in F\}\rvert$
- $deg_p$ = typical/avg degree per bound argument for predicate $p$.

### 0.3 What is “grounding” here?

Given a **ground query atom** $q = h(a,b)$ and a rule $r$ whose head predicate matches $h$, a grounding method returns **candidate grounded bodies**:

$$
\mathcal{G}_r(q) \subseteq (\text{Atoms})^{m(r)}
$$

Each element is a tuple $(g_1,\dots,g_{m(r)})$ of **ground body atoms** consistent with substituting head vars to match $q$, and filling any remaining vars (existentials) via some strategy.

### 0.4 Operator perspective

We’ll reuse the pattern of `operators.md`:

- Forward chaining: an operator on **interpretations** $I$ (sets of true ground atoms).
- Backward chaining: an operator on **goal states** (sets / sequences of atoms to prove).
- Approximate grounders: bounded / sampled / pruned versions of these operators.

To be concrete, define a **goal state** $S$ as a finite multiset/sequence of atoms to prove (your RL engine uses a padded tensor of such atoms).

We define a **one-step resolution operator**:

$$
\mathbf{U}(S) = \{S' : S \Rightarrow S' \text{ by resolving one atom using either a fact or a rule}\}
$$

The various systems differ in:
- how they generate candidates in $\mathbf{U}(S)$,
- whether they enumerate all of them vs cap/top-k/sample,
- whether they require “body facts” (proven-only) or allow unknown subgoals,
- whether they do multi-step pruning / provability checks.
