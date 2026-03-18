# Backward Chaining Grounder: Overview

---

## 1. The Prolog Strategy (5-Stage Pipeline)

Each proof step resolves the **first open goal** in every active state through 5 stages:

```
1. SELECT      -- pick first unresolved atom from each state
2. RESOLVE     -- unify it against facts (K_f matches) and rules (K_r matches)
3. PACK        -- compact K_f + K_r children into S output slots
4. POSTPROCESS -- prune atoms that are known facts, collect completed proofs
5. (repeat for next depth)
```

All operations are **vectorized tensors** over batches of queries. No Python loops over individual queries.

### Example 1: 1-body rule (1 depth)

```
KB:   father(john, mary), father(john, bob), mother(mary, sue)
Rule: parent(X, Y) :- father(X, Y)     [r1]
      parent(X, Y) :- mother(X, Y)     [r2]

Query: parent(john, mary)

Proof tree:

         parent(john, mary)                     SELECT
                |
      +---------+---------+                     RESOLVE
      |                   |
  [r1] father         [r2] mother
   (john,mary)         (john,mary)

```

### Example 2: 2-body rule with free variables (3 depths)

```
KB:   father(john, mary), mother(mary, sue)
Rules: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)    [r1]
       parent(X, Y) :- father(X, Y)                        [r2]
       parent(X, Y) :- mother(X, Y)                        [r3]

Query: grandparent(john, sue)

Proof tree:

                    grandparent(john, sue)                          D0: SELECT
                             |
                    [r1] parent(john,v2),                           D0: RESOLVE (v2 FREE)
                         parent(v2,sue)
                             |                                      D0: PACK -> 1 state
                             |                                      D0: POSTPROCESS (v2 not ground, skip)
                 +-----------+-----------+
                 |                       |
          [r2] father(john,v2)    [r3] mother(john,v2)              D1: RESOLVE first goal
                 |                       |                          D1: PACK -> 2 states
          fact lookup:             fact lookup:
          father(john,?) ->        mother(john,?) ->
          father(john,mary)        no match
          v2=mary                       |
                 |                  dead end
          parent(mary,sue)                                          D1: POSTPROCESS (apply v2=mary)
                 |
          [r3] mother(mary,sue)                                     D2: RESOLVE
                 |                                                  D2: PACK -> 1 state
          fact EXISTS -> PRUNE                                      D2: POSTPROCESS
                 |
              PROOF!
```

### Example 3: 3-body rule — multi-atom states (3 depths)

```
KB:   parent(john, mary), parent(mary, sue), parent(sue, tom)
Rules: great_gp(X, Y) :- parent(X, Z), parent(Z, W), parent(W, Y)    [r1]

Query: great_gp(john, tom)

Proof tree (state = list of open goals):

D0: SELECT   goal = great_gp(john, tom)

D0: RESOLVE  [r1] → state = [parent(john,v1), parent(v1,v2), parent(v2,tom)]
                              ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
                              3 atoms in one state (G=3)

D0: POSTPROCESS  v1,v2 not ground, nothing to prune

D1: SELECT   goal = parent(john,v1)       (first atom picked; 2 remain)

D1: RESOLVE  fact parent(john,mary) → v1=mary
             state = [parent(mary,v2), parent(v2,tom)]
                      ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
                      2 atoms remain (G=2)

D1: POSTPROCESS  apply v1=mary, v2 still free

D2: SELECT   goal = parent(mary,v2)       (first atom picked; 1 remains)

D2: RESOLVE  fact parent(mary,sue) → v2=sue
             state = [parent(sue,tom)]
                      ^^^^^^^^^^^^^
                      1 atom remains (G=1)

D2: POSTPROCESS  apply v2=sue, parent(sue,tom) is a fact → PRUNE
             state = []  →  PROOF!
```

This shows how a single state carries **multiple open goals** (G=3 after the 3-body rule fires), and SELECT always resolves just the **first** goal while the rest wait.

See [report_grounding.md](report_grounding.md) for the full step-by-step walkthrough of all 5 stages at each depth.

**Key: MGU and free variables**

The query is always grounded, but rule application introduces variables (like `v2` above) for body atoms not bound by the head. These free variables get resolved at subsequent depths when matched against facts. This avoids the cartesian product over entity domains.

---

## 3. Depth = 1 (RL / batched-env) vs Depth = D (torch-ns)

### Engine I/O

```
Input:  [B, S, G, 3]          -- current proof states
  → RESOLVE: [B, S, K, G, 3]  -- K children per state (intermediate)
  → PACK:    [B, S', G, 3]    -- flattened back (S' = min(S*K, S_max))
Output: [B, S', G, 3]         -- same shape as input, S grows
```

- **B** — batch size (queries processed in parallel)
- **S** — states per query (concurrent proof branches)
- **M** — max body atoms per rule
- **G** — goals per state (max unresolved atoms in a proof state). Fixed at construction time to $1 + D \times (M - 1)$ (worst case: each of D steps removes 1 goal and adds $M$ body atoms). The tensor dimension is constant across all steps; unused slots are padding
- **3** — each atom is a triple `(predicate, arg0, arg1)`
- **K** = $K_f + K_r$ — children per state ($K_f$ fact matches + $K_r$ rule matches, data-dependent)

The key difference between RL and torch-ns is **S** — G is fixed in both:

| | Input | Output | S |
|---|---|---|---|
| **RL (1-depth)** | `[B, 1, G, 3]` | `[B, K, G, 3]` | Always 1 — agent picks 1 child externally |
| **torch-ns (D-depth)** | `[B, S, G, 3]` | `[B, S', G, 3]` | Grows: 1 → K → K² → ... → min(K^D, S_max) |

### RL case: one depth at a time

```
          RL Agent
             |
       grounder.step()          [B, 1, G, 3] -> [B, K, G, 3]
             |
       agent selects 1           policy picks best child
             |
       grounder.step()          [B, 1, G, 3] -> [B, K, G, 3]
             |
            ...
```

- S is always 1 — the agent picks 1 child externally after each step
- G is fixed at construction — same every step
- Memory per step: **O(B x K x G x 3)** — constant

### torch-ns case: all depths at once

```
       grounder.forward(queries)
             |
         step() depth 0:       [B, 1, G, 3]   -> [B, K, G, 3]
         step() depth 1:       [B, K, G, 3]   -> [B, K², G, 3]
            ...                     S grows, G fixed
         step() depth D:       [B, S, G, 3]   -> [B, S', G, 3]
             |
       collect groundings
```

- S grows each step: 1 → K → K² → ... → min(K^D, S_max)
- G is fixed (same as RL)
- Memory per step: **O(B x S x K x G x 3)** — S is the problem

### Why the RL approach is more scalable

The memory cost is dominated by S (concurrent states). G is fixed in both cases, but S explodes only in torch-ns:

| | S (states) | Memory per step |
|---|---|---|
| **RL** | 1 (constant) | O(B x K x G x 3) |
| **torch-ns** | min(K^D, S_max) | O(B x S x K x G x 3) |

The combinatorial explosion of S on real datasets:

| Dataset | K | D=1 | D=2 | D=3 |
|---------|---|-----|-----|-----|
| family  | 50 | 50 states | 2.5K states | 125K states |
| wn18rr  | 481 | 481 | 231K | **impossible** |
| fb15k237| 3,642 | 3,642 | **13.3M** | -- |

The RL agent sidesteps this: S=1 regardless of dataset or depth

---

## 4. Backward Chaining as an Operator

Analogous to the $T_P$ operator in forward chaining, we define a **backward chaining step operator** $\text{BC}$ that maps a set of proof states to their successors. This gives us an algebraic specification of the grounding engine: a depth-$D$ proof is just $D$ applications of $\text{BC}(s)$, and the RL vs torch-ns difference reduces to what happens to the output (keep all children vs policy-select one).

Given a proof state $s = [g_1, g_2, \dots]$ (a list of goal atoms to resolve), a fact base $\mathcal{F}$, and a rule set $\mathcal{R}$:

$$\text{BC}(s, \mathcal{F}, \mathcal{R}) = \text{PRUNE}\Big(\text{PACK}\big(\text{RESOLVE}(g_1, \mathcal{F}, \mathcal{R}),\ [g_2, \dots]\big),\ \mathcal{F}\Big)$$

Where the sub-operators are:

| Operator | Definition |
|----------|-----------|
| **SELECT** | Pick first goal: $g_1 = s[0]$, remaining $= s[1:]$ |
| **MATCH** | $\text{MATCH}(g, \mathcal{F}) = \{f \in \mathcal{F} \mid \text{pred}(f) = \text{pred}(g) \wedge \text{args match}\}$, analogously for $\mathcal{R}$ by head predicate |
| **UNIFY** | $\text{MGU}(g, t) = \sigma$ s.t. $g\sigma = t\sigma$, or $\bot$ if no unifier exists |
| **APPLY** | $\text{APPLY}(\text{atoms}, \sigma) = \text{atoms}[v \mapsto \sigma(v)]$ |
| **PACK** | Flatten $K_f + K_r$ children into $S'$ output states (scatter-based, compile-safe) |
| **PRUNE** | Remove goals that are known ground facts: $\{a \in \text{goals} \mid a \text{ is ground} \wedge a \in \mathcal{F}\}$ |

A proof is found when some state $s \in S$ has no remaining goals ($s = \emptyset$).

### Depth 1 (RL) vs Depth D (torch-ns)

Let $S_d$ be the set of active proof states at depth $d$. Initially $S_0 = \{[\text{query}]\}$ (one state, one goal). Applying BC to a set means applying it to every state and collecting all children: $\text{BC}(S) = \bigcup_{s \in S} \text{BC}(s)$.

**RL — applies BC once, policy selects one child:**

$$S_d \xrightarrow{\text{BC}} S'_d \xrightarrow{\pi} S_{d+1} \quad \text{where } |S_{d+1}| = 1$$

BC produces $K$ children. The RL agent's policy $\pi$ picks **one** to keep. Memory stays constant.

**torch-ns — applies BC iteratively, keeping all children:**

$$S_0 \xrightarrow{\text{BC}} S_1 \xrightarrow{\text{BC}} S_2 \xrightarrow{\text{BC}} \dots \xrightarrow{\text{BC}} S_D$$

Each step keeps **all** children. The frontier grows: $|S_d| \leq K^d$. Complete up to depth $D$, but memory grows exponentially.

| | Active states after step d | Memory |
|---|---|---|
| **RL** | 1 (policy selects one child) | O(K) — constant |
| **torch-ns** | up to K^d (keeps all children) | O(K^d) — exponential |

See [report_grounding.md](report_grounding.md) for implementation details of each operator.
