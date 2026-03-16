# Fact Index — Design & Internals

Fact indexing for membership checks, targeted MGU lookup, and free-variable enumeration.

## Hierarchy

```
FactIndex(nn.Module)          base: sort facts by hash, exists() via binary search
├── ArgKeyFactIndex           MGU: O(1) targeted lookup via (pred, arg) composite keys
├── InvertedFactIndex         Enum: O(1) enumeration via (pred*E + bound) offset tables
└── BlockSparseFactIndex      Enum: dense [P, E, K] blocks, offset fallback
```

Factory: `FactIndex.create(facts_idx, type='arg_key' | 'inverted' | 'block_sparse', ...)`

## Running Example

All examples below use this knowledge graph:

```
Facts:                        Encoded as [pred, arg0, arg1]:
parent(alice, bob)            [0, 1, 2]
parent(alice, carol)          [0, 1, 3]
parent(bob, dave)             [0, 2, 4]
parent(carol, eve)            [0, 3, 5]
likes(alice, dave)            [1, 1, 4]
likes(bob, carol)             [1, 2, 3]

Predicates: parent=0, likes=1     (P=2)
Entities:   alice=1, bob=2, carol=3, dave=4, eve=5  (E=5)
padding_idx = 0
Variables are indices > constant_no (e.g. X=6, Y=7)
```

---

## Base: `FactIndex` — Sorted Hashes + Binary Search

Every subclass inherits this. It packs each fact into a single `int64` hash and sorts them.

### Hash function

`((pred * base) + arg0) * base + arg1` where `base = max(E, padding) + 2 = 7`

```
parent(alice, bob)   -> ((0*7)+1)*7+2 =  9
parent(alice, carol) -> ((0*7)+1)*7+3 = 10
parent(bob, dave)    -> ((0*7)+2)*7+4 = 18
parent(carol, eve)   -> ((0*7)+3)*7+5 = 26
likes(alice, dave)   -> ((1*7)+1)*7+4 = 60
likes(bob, carol)    -> ((1*7)+2)*7+3 = 66
```

Stored sorted: `fact_hashes = [9, 10, 18, 26, 60, 66]`

### `exists(atoms)` — binary search, O(log F)

```
Query: parent(bob, dave)?  -> hash = 18
searchsorted([9, 10, 18, 26, 60, 66], 18) -> index 2
fact_hashes[2] == 18 -> True

Query: parent(bob, eve)?   -> hash = ((0*7)+2)*7+5 = 19
searchsorted([9, 10, 18, 26, 60, 66], 19) -> index 3
fact_hashes[3] = 26 != 19 -> False
```

All three subclasses inherit this. `BlockSparseFactIndex` overrides it with a faster O(K) dense scan.

---

## 1. `ArgKeyFactIndex` — Targeted Lookup for MGU

Used by Prolog-style MGU resolution (`bcprolog`). Given a partially-bound goal atom, returns **indices into the facts array** (whole triples) that could unify with it.

### What it builds

Three segment indices, each mapping a composite key to a contiguous range of fact positions. `_key_scale (ks) = 7`.

**Index by (pred, arg0):** key = `pred * ks + arg0`

```
parent(alice, bob)   -> 0*7+1 = 1   -> fact position 0
parent(alice, carol) -> 0*7+1 = 1   -> fact position 1
parent(bob, dave)    -> 0*7+2 = 2   -> fact position 2
parent(carol, eve)   -> 0*7+3 = 3   -> fact position 3
likes(alice, dave)   -> 1*7+1 = 8   -> fact position 4
likes(bob, carol)    -> 1*7+2 = 9   -> fact position 5
```

Sort by key, then build `starts` and `lens` arrays:

```
key 1 -> starts=0, lens=2   (facts 0,1: parent(alice,bob), parent(alice,carol))
key 2 -> starts=2, lens=1   (fact 2: parent(bob,dave))
key 3 -> starts=3, lens=1   (fact 3: parent(carol,eve))
key 8 -> starts=4, lens=1   (fact 4: likes(alice,dave))
key 9 -> starts=5, lens=1   (fact 5: likes(bob,carol))
```

**Index by (pred, arg1):** key = `pred * ks + arg1`

```
key 2  -> starts=0, lens=1   (parent(?,bob))
key 3  -> starts=1, lens=1   (parent(?,carol))
key 4  -> starts=2, lens=1   (parent(?,dave))
key 5  -> starts=3, lens=1   (parent(?,eve))
key 10 -> starts=4, lens=1   (likes(?,carol))
key 11 -> starts=5, lens=1   (likes(?,dave))
```

**Index by (pred) only:** for when both args are variables

```
pred 0 -> starts=0, lens=4   (all 4 parent facts)
pred 1 -> starts=4, lens=2   (all 2 likes facts)
```

### `targeted_lookup()` examples

The logic: check which arguments are constants (bound) vs variables (free), and pick the right index.

**Query: `parent(X, bob)` — "who is bob's parent?"**

```
atoms = [0, 6, 2]    (pred=parent, arg0=X(variable), arg1=bob(constant))
is_c0 = False  (6 > constant_no=5 -> variable)
is_c1 = True   (2 <= 5 -> constant)

-> Use (pred, arg1) index: key = 0*7+2 = 2
-> starts[2]=0, lens[2]=1
-> Returns fact indices [0], valid [True]
-> facts_idx[0] = [0,1,2] = parent(alice, bob)

The grounder then unifies X=alice from this fact.
```

**Query: `parent(alice, X)` — "who are alice's children?"**

```
atoms = [0, 1, 6]    (pred=parent, arg0=alice, arg1=X)
is_c0 = True   (1 <= 5)
is_c1 = False  (6 > 5)

-> Use (pred, arg0) index: key = 0*7+1 = 1
-> starts[1]=0, lens[1]=2
-> Returns fact indices [0, 1], valid [True, True]
-> facts_idx[0] = parent(alice, bob)
-> facts_idx[1] = parent(alice, carol)

The grounder unifies X=bob and X=carol.
```

**Query: `parent(X, Y)` — both variables**

```
is_c0 = False, is_c1 = False

-> Use (pred) index: key = 0
-> starts[0]=0, lens[0]=4
-> Returns all 4 parent facts
```

### Key property

Returns **fact indices** (pointers into the facts array). The grounder reads the full triple and performs unification to extract variable bindings. This is what Prolog-style MGU resolution needs: "give me all facts that could match this goal pattern."

---

## 2. `InvertedFactIndex` — Offset-Table Enumeration

Used by enumeration-based grounders (`bcprune`, `bcsld`, etc.). Given a predicate and one bound argument, directly returns **binding values** for the free variable.

### What it builds

Two offset tables (sorted value arrays + cumulative offset arrays). Composite key = `pred * E + bound_arg`.

**PS table (pred+subject -> objects):** "given pred and subject, what objects exist?"

```
Facts sorted by key (pred*5 + subject):
key=0*5+1=1: parent(alice,bob)   -> object=2(bob)
key=0*5+1=1: parent(alice,carol) -> object=3(carol)
key=0*5+2=2: parent(bob,dave)    -> object=4(dave)
key=0*5+3=3: parent(carol,eve)   -> object=5(eve)
key=1*5+1=6: likes(alice,dave)   -> object=4(dave)
key=1*5+2=7: likes(bob,carol)    -> object=3(carol)

_ps_values  = [2, 3, 4, 5, 4, 3]   (the objects, sorted by key)
_ps_offsets = [0, 0, 2, 3, 4, 4, 4, 5, 6, ...]
              ^0 ^1 ^2 ^3 ^4 ^5 ^6 ^7
```

The offsets array is cumulative: `offsets[k+1] - offsets[k]` = number of results for key `k`.

```
key 1: offsets[2]-offsets[1] = 2-0 = 2 results -> values[0:2] = [2, 3] (bob, carol)
key 2: offsets[3]-offsets[2] = 3-2 = 1 result  -> values[2:3] = [4]   (dave)
key 6: offsets[7]-offsets[6] = 5-4 = 1 result  -> values[4:5] = [4]   (dave)
```

**PO table (pred+object -> subjects):** "given pred and object, what subjects exist?"

```
key=0*5+2=2: parent(?,bob)   -> subject=1(alice)
key=0*5+3=3: parent(?,carol) -> subject=1(alice)
key=0*5+4=4: parent(?,dave)  -> subject=2(bob)
key=0*5+5=5: parent(?,eve)   -> subject=3(carol)
key=1*5+3=8: likes(?,carol)  -> subject=2(bob)
key=1*5+4=9: likes(?,dave)   -> subject=1(alice)

_po_values  = [1, 1, 2, 3, 2, 1]
_po_offsets = [0, 0, 0, 1, 2, 3, 4, 4, 4, 5, 6, ...]
```

### `enumerate()` examples

**Query: `parent(alice, ?X)` — enumerate objects where subject=alice**

```
pred=0, bound_arg=1(alice), direction=0 (enumerate objects -> use PS table)

key = 0*5+1 = 1
start = ps_offsets[1] = 0
count = ps_offsets[2] - ps_offsets[1] = 2-0 = 2

candidates = ps_values[0:2] = [2, 3]      -> [bob, carol]
valid      = [True, True, False, ...]       (padded to max_facts_per_query)
```

**Query: `parent(?X, dave)` — enumerate subjects where object=dave**

```
pred=0, bound_arg=4(dave), direction=1 (enumerate subjects -> use PO table)

key = 0*5+4 = 4
start = po_offsets[4] = 2
count = po_offsets[5] - po_offsets[4] = 3-2 = 1

candidates = po_values[2:3] = [2]          -> [bob]
valid      = [True, False, False, ...]
```

### Key property

Returns **entity values** (bindings for the free variable) directly, not whole fact triples. More efficient when you already know which argument is bound and just need the other side.

### Difference from ArgKey

| | ArgKey | Inverted |
|---|---|---|
| **Returns** | Fact indices (whole triples) | Entity values (bindings) |
| **Input** | Full atom `[pred, arg0, arg1]` with variables | pred + one bound arg + direction |
| **Dispatches on** | Which arg is constant (checked at query time) | `direction` parameter (caller decides) |
| **Both-vars-free** | Supported (pred-only index) | Not supported (needs one bound arg) |

---

## 3. `BlockSparseFactIndex` — Dense Blocks

Extends `InvertedFactIndex` with dense `[P, E, K]` tensors for pure tensor indexing — no offset arithmetic. Falls back to `InvertedFactIndex` when memory exceeds `max_memory_mb`.

### What it builds

First compute K = max facts per (pred, entity) pair:

```
parent+alice (as subject): 2 facts (->bob, ->carol)
parent+bob   (as subject): 1 fact  (->dave)
parent+carol (as subject): 1 fact  (->eve)
likes+alice  (as subject): 1 fact  (->dave)
likes+bob    (as subject): 1 fact  (->carol)
-> max subject-side count = 2

parent+bob   (as object): 1 fact  (alice->)
parent+carol (as object): 1 fact  (alice->)
parent+dave  (as object): 1 fact  (bob->)
parent+eve   (as object): 1 fact  (carol->)
-> max object-side count = 1

K = max(2, 1) = 2
```

**Memory check:** `2 * P * E * K * 8 = 2 * 2 * 5 * 2 * 8 = 320 bytes` — under 256 MB, so `_use_dense = True`.

Dense tensors:

```
ps_blocks[P=2, E=5, K=2]:            ps_counts[P=2, E=5]:
  [0, 1, :] = [2, 3]  (bob,carol)      [0, 1] = 2
  [0, 2, :] = [4, 0]  (dave,pad)       [0, 2] = 1
  [0, 3, :] = [5, 0]  (eve,pad)        [0, 3] = 1
  [1, 1, :] = [4, 0]  (dave,pad)       [1, 1] = 1
  [1, 2, :] = [3, 0]  (carol,pad)      [1, 2] = 1

po_blocks[P=2, E=5, K=2]:            po_counts[P=2, E=5]:
  [0, 2, :] = [1, 0]  (alice,pad)      [0, 2] = 1
  [0, 3, :] = [1, 0]  (alice,pad)      [0, 3] = 1
  [0, 4, :] = [2, 0]  (bob,pad)        [0, 4] = 1
  [0, 5, :] = [3, 0]  (carol,pad)      [0, 5] = 1
  [1, 3, :] = [2, 0]  (bob,pad)        [1, 3] = 1
  [1, 4, :] = [1, 0]  (alice,pad)      [1, 4] = 1
```

### `enumerate()` example

**Query: `parent(alice, ?X)`**

```
pred=0, bound_arg=1(alice), direction=0 (objects -> PS)

cands  = ps_blocks[0, 1]  = [2, 3]     -> one tensor index operation
counts = ps_counts[0, 1]  = 2
valid  = [0 < 2, 1 < 2]   = [True, True]

Result: [bob, carol]
```

Compare with `InvertedFactIndex` for the same query:

```
key = 0*5+1 = 1
start = ps_offsets[1] = 0                    <- offset arithmetic
count = ps_offsets[2] - ps_offsets[1] = 2    <- subtraction
candidates = ps_values[start : start+count]   <- indirect gather
```

BlockSparse replaces the offset math with a single `tensor[pred, entity]` indexing.

### `exists()` override — O(K) instead of O(log F)

```
Query: parent(alice, bob)?
block  = ps_blocks[0, 1] = [2, 3]
counts = ps_counts[0, 1] = 2
valid positions: [0, 1] (since K=2, count=2)

Check: any([2, 3][:2] == 2) -> True
```

### When it falls back

```
Large KG: P=1000, E=50000, K=100
Memory = 2 * 1000 * 50000 * 100 * 8 bytes = 80 GB  <- exceeds 256 MB

-> _use_dense = False
-> enumerate() and exists() fall back to InvertedFactIndex's offset tables
```

---

## Comparison

### Space complexity

| Index | Storage | F=100K, P=50, E=40K |
|---|---|---|
| **ArgKey** | 3F + 4PE + 2P (3 order arrays + starts/lens per index) | ~8.3M entries |
| **Inverted** | 2F + 2PE (2 value arrays + 2 offset arrays) | ~4.2M entries |
| **BlockSparse** | Inverted + 2PEK + 2PE (dense blocks + counts) | depends on K, can explode |

ArgKey uses roughly 2x the memory of Inverted because it maintains 3 indices (by arg0, by arg1, by pred-only) with separate starts+lens arrays each, while Inverted needs only 2 directions with a single offsets array each.

### Time complexity for enumeration

Using `parent(alice, ?X)` as the query:

**ArgKey (repurposed):**
```
1. Build query atom: [pred, bound_arg, VARIABLE]
2. targeted_lookup -> fact_indices [N, K]           <- O(1) + gather
3. facts_idx[fact_indices] -> full triples [N, K, 3] <- extra gather
4. Select column 2 -> bindings [N, K]               <- extra column select
```

**Inverted:**
```
1. enumerate(pred, bound_arg, direction) -> bindings [N, K]
   (offset lookup + single gather)
```

**BlockSparse:**
```
1. blocks[pred, bound_arg] -> bindings [K]
   (single tensor index, no arithmetic)
```

Steps 3-4 in the ArgKey path are extra GPU operations. Individually cheap, but in the inner loop of grounding thousands of queries per batch, they add up.

### Time complexity for MGU

ArgKey wins here. MGU needs the full triple to unify against the goal. Using Inverted for MGU would require reconstructing triples from the enumerated values. Additionally, Inverted cannot handle the both-variables-free case (`parent(X, Y)`) since it always requires one bound argument.

### Summary

| | ArgKey | Inverted | BlockSparse |
|---|---|---|---|
| **MGU (targeted)** | Native, best | Can't do both-vars-free | Can't do both-vars-free |
| **Enumeration** | Works but +1 indirection | Native, efficient | Native, fastest |
| **exists()** | O(log F) base | O(log F) base | O(K) dense scan |
| **Memory** | ~2x Inverted | Baseline | Can explode for large KGs |
| **Generality** | Handles all query patterns | Needs one bound arg | Needs one bound arg |

### Why separate subclasses?

ArgKey **could** serve all cases — it is the most general. But the enumeration grounders (`bcprune`, `bcsld`, etc.) never need MGU and always know which argument is bound. For them, ArgKey would pay:

1. **~2x memory** for the third index and extra starts/lens arrays they never use
2. **One extra gather** per query in the hot loop to go from fact indices to binding values

No grounder needs both `targeted_lookup()` and `enumerate()` simultaneously, so the separation avoids paying for structures that are never queried.

### Which grounder uses which

```
Grounder needs MGU resolution?          -> ArgKeyFactIndex
  (bcprolog: "find facts matching this goal, I'll unify")

Grounder needs enumeration?
  |-- KG fits in dense [P,E,K] blocks?  -> BlockSparseFactIndex  (default)
  |    (pure indexing, fastest)
  '-- Too large for dense?              -> InvertedFactIndex
       (offset tables, always fits)
```

The factory in `factory.py` defaults to `block_sparse`, which automatically falls back to inverted-style offset tables if memory is too high.
