# Walkthrough: Full Tensor Trace

A complete step-by-step trace through the BCGrounder pipeline with a family/kinship knowledge base. Every tensor is shown with its shape, values, and validity mask.

---

## Knowledge Base

### Facts

```
father(john, mary)
father(john, bob)
mother(mary, sue)
```

### Rules

```
r0: parent(X, Y) :- father(X, Y)
r1: parent(X, Y) :- mother(X, Y)
r2: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
```

### Query

```
grandparent(john, sue)
```

---

## Encoding

### Entities

| Entity | Index |
|--------|-------|
| john | 0 |
| mary | 1 |
| bob | 2 |
| sue | 3 |

`E = 4`, `padding_idx = 4`, `pack_base = 6`

### Predicates

| Predicate | Index |
|-----------|-------|
| father | 0 |
| mother | 1 |
| parent | 2 |
| grandparent | 3 |

`P = 4`

### Rules (compiled)

| Rule | head_pred | body_preds | num_body |
|------|-----------|------------|----------|
| r0 | 2 (parent) | [0 (father)] | 1 |
| r1 | 2 (parent) | [1 (mother)] | 1 |
| r2 | 3 (grandparent) | [2 (parent), 2 (parent)] | 2 |

`R = 3`, `M = 2` (max body atoms)

### Facts (tensor)

```
facts_idx: [3, 3]
  [[0, 0, 1],    # father(john, mary)
   [0, 0, 2],    # father(john, bob)
   [1, 1, 3]]    # mother(mary, sue)
```

### Fact hashes (sorted)

```
pack_triples_64(facts_idx, base=6):
  father(john, mary): ((0*6)+0)*6+1 = 1
  father(john, bob):  ((0*6)+0)*6+2 = 2
  mother(mary, sue):  ((1*6)+1)*6+3 = 15

fact_hashes: [1, 2, 15]   # sorted
```

---

## Configuration

```
BCGrounder(depth=2)

D = 2
G = 1 + 2*(2-1) = 3   (goals per state)
B = 1                   (single query)
S = 4                   (states per query, for illustration)
K_f = 3                 (max fact matches)
K_r = 3                 (max rule matches — one per rule)
```

---

## Initialization

Query encoding:

```
queries: [1, 3]
  [[3, 0, 3]]     # grandparent(john, sue)

query_mask: [1]
  [True]
```

Initial proof state:

```
proof_goals: [1, S=4, G=3, 3]
  State 0: [[3, 0, 3], [4, 4, 4], [4, 4, 4]]   # grandparent(john,sue) + padding
  State 1: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # empty
  State 2: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # empty
  State 3: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # empty

state_valid: [1, 4]
  [True, False, False, False]

goal_valid: [1, 4, 3]
  [[True, False, False],
   [False, False, False],
   [False, False, False],
   [False, False, False]]
```

---

## Depth 0: First Resolution Step

### SELECT

Select first valid goal from each active state.

```
goal: [1, 4, 3]
  [[3, 0, 3],        # grandparent(john, sue) — selected from state 0
   [4, 4, 4],        # padding (state 1 inactive)
   [4, 4, 4],        # padding
   [4, 4, 4]]        # padding

remaining: [1, 4, 3, 3]
  State 0: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # no remaining goals
  ...

active_mask: [1, 4]
  [True, False, False, False]
```

### RESOLVE — Facts

Look up `grandparent(john, sue)` in FactIndex:

```
No facts match predicate "grandparent" → no fact children

fact_children: [1, 4, K_f=3, G=3, 3]   # all padding
fact_valid: [1, 4, 3]                    # all False
```

### RESOLVE — Rules

Find rules with head predicate = 3 (grandparent):

```
Matching rules: [r2]   # grandparent(X,Y) :- parent(X,Z), parent(Z,Y)
```

Unify `grandparent(X, Y)` with `grandparent(john, sue)`:
- MGU: `{X -> john, Y -> sue}`

Apply MGU to r2's body:
- `parent(X, Z)` → `parent(john, Z)` — Z remains free (will be a goal)
- `parent(Z, Y)` → `parent(Z, sue)` — Z remains free

Since Z is a free variable, we need to enumerate candidates. For now, Z stays as a variable in the goal state (to be resolved at the next depth).

```
rule_children: [1, 4, K_r=3, G=3, 3]
  State 0, child 0 (r2 match):
    [[2, 0, Z],       # parent(john, Z) — goal 0
     [2, Z, 3],       # parent(Z, sue) — goal 1
     [4, 4, 4]]       # padding
  (remaining children: padding)

rule_valid: [1, 4, 3]
  [[True, False, False],     # State 0: 1 rule matched
   [False, False, False],    # States 1-3: inactive
   [False, False, False],
   [False, False, False]]

rule_idx: [1, 4, 3]
  [[2, -1, -1], ...]         # r2 matched in state 0, child 0
```

**Note on free variable Z**: In practice, Z is represented as a variable index above the entity range. The actual value of Z is determined when it gets resolved against facts in subsequent steps. The grounder tracks which slots contain variables vs constants.

### PACK

Merge fact and rule children, compact to S states:

```
proof_goals (after pack): [1, 4, 3, 3]
  State 0: [[2, 0, Z], [2, Z, 3], [4, 4, 4]]   # parent(john,Z), parent(Z,sue)
  State 1: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # empty
  State 2: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # empty
  State 3: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]   # empty

state_valid: [1, 4]
  [True, False, False, False]
```

---

## Depth 1: Second Resolution Step

### SELECT

```
goal: [1, 4, 3]
  [[2, 0, Z],        # parent(john, Z) — selected from state 0
   ...]

remaining: [1, 4, 3, 3]
  State 0: [[2, Z, 3], [4, 4, 4], [4, 4, 4]]   # parent(Z, sue) remains
```

### RESOLVE — Facts

Look up `parent(john, Z)` — no facts with predicate "parent" exist directly.

```
fact_valid: all False
```

### RESOLVE — Rules

Find rules with head predicate = 2 (parent):

```
Matching rules: [r0, r1]
  r0: parent(X,Y) :- father(X,Y)
  r1: parent(X,Y) :- mother(X,Y)
```

**r0 match**: unify `parent(X,Y)` with `parent(john, Z)` → `{X->john, Y->Z}`
- Body: `father(john, Z)` — still has free variable Z

**r1 match**: unify `parent(X,Y)` with `parent(john, Z)` → `{X->john, Y->Z}`
- Body: `mother(john, Z)` — still has free variable Z

```
rule_children: [1, 4, K_r=3, G=3, 3]
  State 0, child 0 (r0):
    [[0, 0, Z],       # father(john, Z) — new goal
     [2, Z, 3],       # parent(Z, sue) — carried forward
     [4, 4, 4]]

  State 0, child 1 (r1):
    [[1, 0, Z],       # mother(john, Z) — new goal
     [2, Z, 3],       # parent(Z, sue) — carried forward
     [4, 4, 4]]

rule_valid: [1, 4, 3]
  [[True, True, False], ...]
```

### PACK

```
proof_goals (after pack): [1, 4, 3, 3]
  State 0: [[0, 0, Z], [2, Z, 3], [4, 4, 4]]   # father(john,Z), parent(Z,sue)
  State 1: [[1, 0, Z], [2, Z, 3], [4, 4, 4]]   # mother(john,Z), parent(Z,sue)
  State 2: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]
  State 3: [[4, 4, 4], [4, 4, 4], [4, 4, 4]]

state_valid: [True, True, False, False]
```

---

## Depth 2: Third Resolution Step (final, D=2)

### SELECT

```
goal: [[0, 0, Z], [1, 0, Z], ...]   # father(john,Z) and mother(john,Z)
```

### RESOLVE — Facts

**State 0**: look up `father(john, Z)` → matches `father(john, mary)` and `father(john, bob)`
- Z=1 (mary): remaining goal becomes `parent(1, sue)` = `parent(mary, sue)`
- Z=2 (bob): remaining goal becomes `parent(2, sue)` = `parent(bob, sue)`

**State 1**: look up `mother(john, Z)` → no matches (no `mother(john, ...)` facts)

```
fact_children:
  State 0, child 0: [[2, 1, 3], [4, 4, 4], [4, 4, 4]]   # parent(mary,sue) remains
  State 0, child 1: [[2, 2, 3], [4, 4, 4], [4, 4, 4]]   # parent(bob,sue) remains

fact_valid: [[True, True, False], [False, False, False], ...]
```

### RESOLVE — Rules

Rule matches for `father(john, Z)` and `mother(john, Z)` would produce more children, but at D=2 these won't be resolved further.

### PACK

After depth 2, we have states with remaining goals. The postprocessing step identifies:

- State with `parent(mary, sue)`: this goal can be checked — does `parent(mary, sue)` resolve?
  - r1: `parent(X,Y) :- mother(X,Y)` → check `mother(mary, sue)` → **fact exists!**
  - This grounding is complete: `grandparent(john, sue) :- parent(john, mary), parent(mary, sue)`

- State with `parent(bob, sue)`: `parent(bob, sue)` → no matching facts or provable derivation → **incomplete**, pruned

---

## Output

```
body: [1, tG, M=2, 3]
  Grounding 0: [[2, 0, 1], [2, 1, 3]]    # parent(john,mary), parent(mary,sue)
  (remaining slots: padding)

mask: [1, tG]
  [True, False, False, ...]                # 1 valid grounding

count: [1]
  [1]

rule_idx: [1, tG]
  [2, -1, -1, ...]                         # Rule r2 (grandparent :- parent, parent)
```

### Interpretation

The grounder found one valid grounding for `grandparent(john, sue)`:

```
grandparent(john, sue) :- parent(john, mary), parent(mary, sue)
```

This grounding is backed by the proof chain:
1. `parent(john, mary)` via r0 from `father(john, mary)` (fact)
2. `parent(mary, sue)` via r1 from `mother(mary, sue)` (fact)

The grounding `parent(john, bob), parent(bob, sue)` was pruned because `parent(bob, sue)` is unprovable.

---

## Tensor Shape Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| `queries` | `[1, 3]` | Input query |
| `proof_goals` | `[1, 4, 3, 3]` | `[B, S, G, 3]` proof states |
| `goal` | `[1, 4, 3]` | `[B, S, 3]` selected goals |
| `fact_children` | `[1, 4, 3, 3, 3]` | `[B, S, K_f, G, 3]` fact resolution |
| `rule_children` | `[1, 4, 3, 3, 3]` | `[B, S, K_r, G, 3]` rule resolution |
| `state_valid` | `[1, 4]` | `[B, S]` active states |
| `body` (output) | `[1, tG, 2, 3]` | `[B, tG, M, 3]` groundings |
| `mask` (output) | `[1, tG]` | `[B, tG]` validity |
| `fact_hashes` | `[3]` | `[F]` sorted hashes |

All shapes remain constant throughout the computation — no dynamic resizing at any step.
