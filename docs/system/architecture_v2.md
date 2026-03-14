# Architecture

NeSyGround is a compiled FOL grounding library. It takes a knowledge base of facts and Horn clause rules, and produces ground rule instantiations as fixed-shape tensors on GPU under `torch.compile(fullgraph=True, mode='reduce-overhead')`.

---

## 1. Grounding

A knowledge base `KB = (F, R)` consists of ground facts `F = {p(a,b), ...}` and Horn clause rules `R = {h(X,Y) :- b₁(X,Z), b₂(Z,Y), ...}`. All predicates are binary.

Given a ground query `q = p(a,b)` and depth bound `D`:

```
Ground(q, KB, D) = { (r, body(r)θ) | head(r)θ = q, θ grounds body(r), reachable within D steps }
```

The grounder produces these instantiations. A downstream consumer scores them. These concerns are strictly separated.

```
KB (F, R) ──→ Compile ──→ Grounder ──→ GroundingResult ──→ Consumer
               once        per batch    [B, tG, M, 3]      (SBR, R2N, DCR)
```

| Stage | What it does | When |
|-------|-------------|------|
| **Compile** | KB → indexed tensor structures (FactIndex, RuleIndex, metadata) | Once per KB |
| **Ground** | Queries → ground rule instantiations | Per batch, under CUDA graph |
| **Score** | Instantiations → truth values (t-norms, KGE) | Downstream, not our scope |

---

## 2. Resolution

Grounding is **resolution**: given a goal atom, find substitutions that ground a rule's body. Everything else — filtering, scoring, budget management — is policy layered on top.

Resolution decomposes into three FOL primitives:

| Primitive | What it does |
|-----------|-------------|
| **Unify** | Compute the Most General Unifier of two atoms |
| **Substitute** | Apply a substitution to a set of atoms |
| **Enumerate** | Find entity values for free variables |

These three are irreducible. All resolution strategies compose from them.

### Two strategies

For Horn clauses with binary predicates, there are exactly two ways to implement resolution:

**Strategy A — MGU resolution (runtime unification)**

Unify the goal with rule heads at runtime. Carry variables through proof states. Resolve body atoms recursively against facts and rules.

```
grandparent(john, sue)?
  → unify with grandparent(X,Y) :- parent(X,Z), parent(Z,Y)
  → θ = {X→john, Y→sue}
  → resolve parent(john, Z): unify against facts → Z = mary
  → resolve parent(mary, sue): unify against facts → success
```

Variable-carrying. Step-by-step. The standard SLD resolution strategy.

**Strategy B — Compiled enumeration (pre-compiled bindings)**

Pre-compute the binding pattern at compilation time into a lookup table (`body_arg_sources`). At runtime, fill body templates directly from query args and enumerated candidates. No unification needed.

```
Compile: grandparent(X,Y) :- parent(X,Z), parent(Z,Y)
  → body_arg_sources = [(HEAD_VAR0, FREE_0), (FREE_0, HEAD_VAR1)]

Runtime: query = grandparent(john, sue)
  → HEAD_VAR0 = john, HEAD_VAR1 = sue
  → enumerate FREE_0 from fact_index where parent(john, ?) → {mary, bob}
  → fill: [(parent, john, mary), (parent, mary, sue)]
```

**Why this works**: in binary predicates, the rule head has exactly two arguments. Matching a ground query fully determines the head substitution. Every body variable either appears in the head (value known from query) or doesn't (free — requires enumeration). This binary structure is static per rule, so it can be pre-compiled into `body_arg_sources`.

**Compilation replaces runtime unification with table-driven template filling.** This is what makes enumeration resolution fully tensorizable with static shapes.

### What is orthogonal to resolution

These are **policy choices**, not grounding operations. Any grounder can combine any resolution strategy with any policy. A new grounder can introduce a novel resolution strategy, or reuse an existing one with different policies.

| Concern | What it decides | Examples |
|---------|----------------|----------|
| **Search** | How to explore the proof space | Step × D loop, BFS, single-pass |
| **Soundness filter** | Which groundings to keep | PruneIncompleteProofs, FC provable set, none |
| **Scoring** | How to rank groundings | KGE, learned attention, soft provability |
| **Budget** | How many groundings to output | Top-k truncation, deduplication |

---

## 3. CUDA Graph Constraint

The entire forward pass runs under `torch.compile(fullgraph=True, mode='reduce-overhead')`. Every tensor operation is captured into a CUDA graph and replayed. This is not optional — there is no fallback mode.

This single constraint shapes every design decision:

| Rule | Why | Alternative that breaks |
|------|-----|------------------------|
| **Fixed tensor shapes** with validity masks | CUDA graphs record shapes at trace time | Dynamic indexing, variable-length lists |
| **No `.item()` or `.tolist()`** | GPU→CPU sync breaks graph capture | Reading a scalar to decide control flow |
| **No data-dependent branching** | Python `if` on tensor values creates graph breaks | `if count > 0:`, `while not done:` |
| **Cumsum + scatter for compaction** | Deterministic, no sync | `torch.topk` (non-deterministic order), `torch.nonzero` (dynamic shape) |
| **Masks everywhere** | Validity encoded in boolean tensors, never in shape | Filtering by removing elements |
| **Registered `nn.Module` buffers** | Tensors must be on the correct device at trace time | Manual `.to(device)` in forward |
| **Padding sentinel = `E`** | Distinguishable from all valid entities `0..E-1` | -1 (breaks unsigned indexing), 0 (collides with entity 0) |

When adding a new grounder: if any operation in `forward()` triggers a graph break, compilation fails. There is no dynamic fallback — fix the operation.

---

## 4. Dimensions

| Symbol | Meaning | Typical |
|--------|---------|---------|
| `B` | batch (queries per forward call) | 64–512 |
| `tG` | total groundings per query (output budget) | 50–200 |
| `M` | max body atoms per rule | 2–4 |
| `D` | proof depth | 1–3 |
| `R` | total rules | 10–200 |
| `R_eff` | max rules sharing one head predicate | 2–20 |
| `E` | entities (constants) | 100–40k |
| `P` | predicates | 10–240 |
| `F` | facts | 100–300k |
| `K_f` | max fact matches per goal | 32–128 |
| `K_r` | max rule matches per goal | ≤ R_eff |
| `S` | max concurrent proof states (MGU only) | 64 |
| `G` | max goals per state (MGU only) | `1 + D·(M−1)` |

Padding: `padding_idx = E`. Hash base: `pack_base = E + 2`.

---

## 5. Data Structures

Two dataclasses. Everything else is raw tensors.

### GroundingResult

The universal output contract between grounder and consumer.

```python
@dataclass
class GroundingResult:
    body: Tensor       # [B, tG, M, 3]   grounded body atoms (pred, subj, obj)
    mask: Tensor       # [B, tG]          which groundings are valid
    count: Tensor      # [B]              valid groundings per query
    rule_idx: Tensor   # [B, tG]          which rule produced each grounding
```

### CompiledRule

Per-rule metadata extracted once during compilation. This is the structure that makes compiled enumeration possible.

```python
@dataclass
class CompiledRule:
    rule_idx: int                            # original index in rule tensor
    head_pred: int                           # head predicate index
    head_bindings: Tuple[int, int]           # which vars the head args are
    num_body: int                            # body atom count
    body_preds: List[int]                    # predicate per body atom
    body_arg_sources: List[Tuple[int, int]]  # per body atom: source of each arg
    free_vars: List[int]                     # body vars absent from head
    body_order: List[int]                    # topological processing order
```

`body_arg_sources` encodes where each argument comes from:
- `HEAD_VAR0 = 0` — first head variable (bound by query subject)
- `HEAD_VAR1 = 1` — second head variable (bound by query object)
- `FREE_VAR(i) = 2 + i` — i-th free variable (filled by enumeration)

`body_order` is a topological sort: each body atom is processed only after at least one of its arguments is already known. This enables sequential enumeration without backtracking.

---

## 6. Modules

### Directory layout

```
grounder/
├── __init__.py               # public API exports
├── types.py                  # GroundingResult
├── primitives.py             # unify, substitute, hash_atoms, hash_contains
├── fact_index.py             # FactIndex protocol + ArgKey, Inverted, BlockSparse
├── rule_index.py             # RuleIndex (segment + table access)
├── compilation.py            # CompiledRule + compile_rules, tensorize, build_metadata
├── forward_chaining.py       # compute_provable_set (CPU semi-naive T_P)
├── grounders/
│   ├── __init__.py           # exports all grounders
│   ├── base.py               # Grounder base class (owns compiled KB)
│   ├── common.py             # shared search/output utilities
│   ├── prolog.py             # PrologGrounder + MGU resolution functions
│   ├── enum.py               # EnumGrounder + compiled enumeration functions
│   ├── kge.py                # KGEGrounder
│   ├── neural.py             # NeuralGrounder
│   ├── soft.py               # SoftGrounder
│   ├── sampler.py            # SamplerGrounder
│   └── lazy.py               # LazyGrounder
└── factory.py                # create_grounder, parse_grounder_type
```

Top level: infrastructure (primitives, indices, compilation, FC).
`grounders/`: all grounder implementations. Each file is self-contained — grounder-specific resolution logic lives with the grounder, not in a shared module.

### `types.py` — Output contract

| Export | Type | Purpose |
|--------|------|---------|
| `GroundingResult` | dataclass | Universal grounder → consumer interface |

### `primitives.py` — FOL primitives

The three resolution primitives plus one hash utility. Zero internal dependencies.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `unify` | `(a [L,3], b [L,3], constant_no, pad_idx)` → `(mask [L], θ [L,2,2])` | Pairwise MGU: predicate match, constant conflicts, variable binding, same-var consistency |
| `substitute` | `(atoms [N,M,3], θ [N,S,2,2], pad_idx)` → `[N,M,3]` | Apply variable→constant substitutions. Loop-unrolled for S=2 |
| `hash_atoms` | `(atoms [...,3], base)` → `[...] int64` | `((p × base) + a₀) × base + a₁` — atom → int64 for dedup and membership |
| `hash_contains` | `(atoms [...,3], sorted_hashes [F], base)` → `[...] bool` | O(log F) membership via `torch.searchsorted` on sorted hashes |

All functions are CUDA-graph-safe: static shapes, no sync, no branching.

### `fact_index.py` — Fact storage and retrieval

One protocol, three implementations. Each optimizes for a different access pattern.

**Protocol — shared by all:**

| Method / Property | Signature | Purpose |
|-------------------|-----------|---------|
| `exists` | `(atoms [...,3])` → `[...] bool` | Membership via binary search on `fact_hashes` |
| `facts_idx` | `Tensor [F, 3]` | Raw fact triples |
| `fact_hashes` | `Tensor [F] int64` | Sorted atom hashes |
| `pack_base` | `int` | Hash multiplier |

**Implementations:**

| Class | Extra method | Access pattern | Best for |
|-------|-------------|----------------|----------|
| `ArgKeyFactIndex` | `lookup(query [B,3], max_k)` → `(idx [B,K], valid [B,K])` | O(1) targeted by (pred, arg) composite key | MGU resolution |
| `InvertedFactIndex` | `enumerate(pred, bound, dir)` → `(cands [N,K], valid [N,K])` | O(1) enumerate via (pred, bound) offset tables | Compiled enumeration |
| `BlockSparseFactIndex` | Both | Dense `[P, E, K]` blocks; falls back to inverted | Either |

All methods return fixed-shape padded tensors with validity masks (CUDA-graph-safe).

### `rule_index.py` — Rule-to-predicate mapping

```python
class RuleIndex(nn.Module):
    """Maps predicate → rules with that head. Two access modes."""

    # Registered buffers:
    rules_heads: Tensor    # [R, 3]   sorted by head predicate
    rules_bodies: Tensor   # [R, M, 3]
    rule_lens: Tensor      # [R]
    R_eff: int             # max rules sharing one head predicate
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `lookup_by_segments` | `(preds [B], max_k)` → `(idx [B,K], valid [B,K], qidx [B,K])` | Sequential segment access. For MGU resolution. |
| `lookup_by_table` | `(preds [N])` → `(idx [N, R_eff], mask [N, R_eff])` | Parallel gather. For compiled enumeration. |

### `compilation.py` — KB analysis (once)

`CompiledRule` dataclass lives here (tightly coupled with compilation logic).

| Function | Output | Purpose |
|----------|--------|---------|
| `compile_rules(heads, bodies, lens, constant_no)` | `List[CompiledRule]` | Extract bindings, free vars, topological body order per rule |
| `tensorize_rules(compiled, M, device)` | `head_preds [R]`, `body_preds [R,M]`, `num_body [R]` | Rule structure as tensors |
| `build_enum_metadata(compiled, M, device)` | see below | Per-rule binding tables for compiled enumeration |
| `build_rule_clustering(compiled, P, device)` | `pred_rule_indices [P, R_eff]`, `pred_rule_mask [P, R_eff]`, `R_eff` | Rules grouped by head predicate |

**`build_enum_metadata` outputs:**

| Tensor | Shape | Content |
|--------|-------|---------|
| `has_free` | `[R]` | Whether rule has free variables |
| `enum_pred` | `[R]` | Predicate of first free-var enumeration atom |
| `enum_bound` | `[R]` | Binding source for enumeration anchor arg |
| `enum_dir` | `[R]` | Enumeration direction (subject→objects or object→subjects) |
| `check_arg_source` | `[R, M, 2]` | Per body atom, per arg: which slot provides the value |

### `forward_chaining.py` — Provable set computation (CPU, once)

Semi-naive T_P iteration computing all atoms derivable from `F ∪ R` within depth `D`:

```
I₀ = F;   Iₙ₊₁ = Iₙ ∪ { head(r)θ | body(r)θ ⊆ Iₙ, ≥1 body atom from Δₙ }
```

| Export | Signature | Purpose |
|--------|-----------|---------|
| `compute_provable_set` | `(compiled_rules, facts_idx, E, P, depth, device)` → `(sorted_hashes [I], n_provable)` | Entry point. Returns sorted int64 hashes of all provable atoms. |

Internally uses `FCDynamic` class — staged ragged join with greedy join ordering, PS/PO offset tables, combined base+provable lookups. Runs once per KB, not per batch. Not under CUDA graph — this is pure CPU precomputation.

### `grounders/base.py` — Grounder base class

```python
class Grounder(nn.Module):
    """Owns all compiled KB state. Every grounder inherits from this."""

    def __init__(self,
                 facts_idx: Tensor,           # [F, 3]
                 rules_heads_idx: Tensor,      # [R, 3]
                 rules_bodies_idx: Tensor,     # [R, M_raw, 3]
                 rule_lens: Tensor,            # [R]
                 constant_no: int,
                 padding_idx: int,
                 fact_index_type: str = 'inverted',
                 num_entities: int = ...,
                 num_predicates: int = ...,
                 max_facts_per_query: int = 64,
                 **kwargs):
        ...
```

**Registered buffers (available to all subclasses):**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `fact_index` | `FactIndex` | Chosen implementation |
| `rule_index` | `RuleIndex` | Segment + table access |
| `compiled_rules` | `List[CompiledRule]` | Per-rule metadata |
| `fact_hashes` | `Tensor [F]` | Sorted int64 hashes |
| `head_preds` | `Tensor [R]` | Head predicate per rule |
| `body_preds` | `Tensor [R, M]` | Body predicates per rule |
| `num_body` | `Tensor [R]` | Body atom count per rule |
| `has_free` | `Tensor [R]` | Free-variable flag per rule |
| `check_arg_source` | `Tensor [R, M, 2]` | Binding table for compiled enumeration |
| `enum_pred` | `Tensor [R]` | Enumeration predicate |
| `enum_bound` | `Tensor [R]` | Enumeration anchor binding |
| `enum_dir` | `Tensor [R]` | Enumeration direction |
| `pred_rule_indices` | `Tensor [P, R_eff]` | Rule clustering |
| `pred_rule_mask` | `Tensor [P, R_eff]` | Rule clustering validity |

**Properties:**

| Property | Type | Value |
|----------|------|-------|
| `M` | `int` | max body atoms per rule |
| `K_f` | `int` | max fact matches per query |
| `K_r` | `int` | max rule matches per query |
| `R_eff` | `int` | max rules per predicate |
| `pack_base` | `int` | `E + 2` |

```python
    def forward(self, queries: Tensor, query_mask: Tensor) -> GroundingResult:
        raise NotImplementedError
```

All subclasses inherit from `Grounder` directly. No intermediate `BCGrounder` / `VectorizedGrounder` layers.

### `grounders/common.py` — Shared utilities

Functions used by multiple grounders. All CUDA-graph-safe.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compact` | `(tensor [...,N,*], mask [...,N])` → same shape, valid first | Cumsum + scatter compaction. Maintains fixed shapes. |
| `dedup` | `(groundings [...,M,3], mask [...])` → updated mask | Hash-sort-adjacent_diff deduplication. |
| `exclude_query` | `(body [...,M,3], queries [B,3], mask [...])` → updated mask | Prevent trivial self-groundings. |
| `check_provable` | `(hashes [...], provable_hashes [I])` → `[...] bool` | Binary search on sorted provable set. |

### `grounders/prolog.py` — MGU resolution grounder

Contains `PrologGrounder` and all MGU-specific resolution logic.

```python
class PrologGrounder(Grounder):
    """SLD resolution with runtime unification. Step × D proof loop.

    Carries variables through proof states. Each step selects a goal,
    resolves it against facts and rules via MGU, and compacts children.
    """

    def __init__(self, ...,
                 depth: int,                        # proof steps (D)
                 mode: str = 'additive',            # 'additive' | 'cascade'
                 max_states: int = 64,              # concurrent proof states (S)
                 max_total_groundings: int = 100,   # output budget (tG)
                 compile_mode: str = 'reduce-overhead',
                 fact_index_type: str = 'arg_key'):
        ...
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward` | `(queries [B,3], query_mask [B])` → `GroundingResult` | Full D-step proof loop. Collects terminal states (all goals resolved). |
| `step` | `(proof_goals, grounding_body, state_valid, top_ridx, depth)` → same | One SELECT → RESOLVE → COMPACT cycle. **RL agent API.** |

**Modes:**
- `additive`: resolve facts and rules independently → `K = K_f + K_r` children
- `cascade`: resolve rules first, then resolve each rule child's body against facts → `K = K_f × K_r` children

**Module-level functions (MGU-specific, co-located here):**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `resolve_facts` | `(goals [B,3], remaining [B,G,3], fact_index, ...)` → `(children [B,K_f,G,3], success [B,K_f])` | Targeted lookup → unify → substitute. Goal proven. |
| `resolve_rules` | `(goals [B,3], remaining [B,G,3], rule_index, next_var [B], ...)` → `(children [B,K_r,G+M,3], success [B,K_r])` | Segment lookup → standardize apart → unify → substitute. Goal replaced by body. |
| `standardize_apart` | `(rule_bodies [B,K,M,3], next_var [B])` → `(renamed [B,K,M,3], new_next_var [B])` | Rename template variables to fresh indices. Prevents variable collision across rules. |
| `merge_children` | `(fact_children, rule_children)` → `(merged, merged_valid)` | Interleave fact + rule children into fixed-shape output. |

### `grounders/enum.py` — Compiled enumeration grounder

Contains `EnumGrounder` and all enumeration-specific resolution logic.

```python
class EnumGrounder(Grounder):
    """Compiled enumeration with configurable search and policy.

    Resolution uses pre-compiled binding tables — no runtime unification.
    Enumerates candidates from fact index, fills body templates.
    """

    def __init__(self, ...,
                 depth: int = 2,                    # BFS depth (D)
                 width: int = 1,                    # max unproven body atoms (W)
                 filter: str = 'prune',             # 'prune' | 'provset' | 'none'
                 dual_anchor: bool = False,         # enumerate from both directions
                 max_total_groundings: int = 100,   # output budget (tG)
                 compile_mode: str = 'reduce-overhead',
                 fact_index_type: str = 'inverted'):
        ...
        # If filter='provset': calls compute_provable_set() at __init__
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward` | `(queries [B,3], query_mask [B])` → `GroundingResult` | BFS over D depths → apply filter → top tG |

**Parameters replace four current classes:**

| Configuration | Replaces |
|---------------|----------|
| `filter='prune', depth=2` | BCPruneGrounder |
| `filter='provset', depth=2` | BCProvsetGrounder |
| `filter='prune', width=1, dual_anchor=True` | ParametrizedBCGrounder |
| `filter='none', width=∞` | FullBCGrounder |

**Soundness filters:**
- `prune` — PruneIncompleteProofs: iterative fixed-point removal of groundings whose body atoms have no supporting proof
- `provset` — FC oracle: keep only groundings whose body atoms appear in the forward-chaining provable set
- `none` — no filtering (all enumerated groundings pass through)

**Module-level functions (enumeration-specific, co-located here):**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `enumerate_and_resolve` | `(queries [B,3], pred_rule_indices, metadata, fact_index, ...)` → `(body [...,M,3], mask [...], ridx [...])` | Gather per-rule metadata → enumerate free-var candidates → fill body templates from binding table. No unification. |
| `gather_rule_metadata` | `(pred_rule_indices, pred_rule_mask, query_preds, buffers)` → `(active metadata tensors)` | Collect per-rule enum metadata for the active rules matching each query. |
| `fill_body_templates` | `(query_subjs, query_objs, candidates, check_arg_source, body_preds, M)` → `(body [...,M,3])` | Populate body atoms from the binding table and enumerated candidates. |
| `apply_prune_filter` | `(groundings, body, mask, fact_hashes)` → updated mask | PruneIncompleteProofs fixed-point. |
| `apply_provset_filter` | `(groundings, body, mask, provable_hashes)` → updated mask | Check body atoms against FC provable set. |

### `grounders/kge.py` — KGE scoring grounder

```python
class KGEGrounder(EnumGrounder):
    """Score by min(KGE(body_atom)), select top-k by score."""
    kge_model: nn.Module
```

### `grounders/neural.py` — Learned attention grounder

```python
class NeuralGrounder(EnumGrounder):
    """Score via learned GroundingAttention MLP, select top-k."""
    attention: nn.Module
```

### `grounders/soft.py` — Soft provability grounder

```python
class SoftGrounder(EnumGrounder):
    """Soft provability: known=1.0, unknown=sigmoid(score). Rank by product."""
    provability_mlp: nn.Module
```

### `grounders/sampler.py` — Sampling grounder

```python
class SamplerGrounder(EnumGrounder):
    """4× oversample, then random (train) or deterministic (eval) top-k."""
    sample_ratio: float
```

### `grounders/lazy.py` — Lazy evaluation grounder

```python
class LazyGrounder(EnumGrounder):
    """Predicate reachability BFS at init → reduced rule set → smaller search."""
    reachability_depth: int
```

### `factory.py` — Dispatch

| Function | Signature | Purpose |
|----------|-----------|---------|
| `parse_grounder_type` | `(name: str)` → `(cls, kwargs)` | `"bcprune_2"` → `(EnumGrounder, {depth=2, filter='prune'})` |
| `create_grounder` | `(name, facts, rules, ...)` → `Grounder` | Build and return fully configured grounder |

**Naming convention:** `{type}_{depth}` or `{type}_{width}_{depth}`.

---

## 7. Grounder Hierarchy

```
Grounder (nn.Module)                          grounders/base.py
  │
  │  Owns: FactIndex, RuleIndex, CompiledRules, all metadata buffers
  │  forward(queries, query_mask) → GroundingResult
  │
  ├── PrologGrounder                          grounders/prolog.py
  │     Resolution: MGU (runtime unification)
  │     Search: step × D proof loop with compound states
  │     Modes: additive (K_f + K_r) │ cascade (K_f × K_r)
  │     Soundness: inherent — only terminal states reach output
  │     API: step() for RL agents
  │
  └── EnumGrounder                            grounders/enum.py
        Resolution: compiled enumeration (pre-compiled bindings)
        Search: BFS over D depths
        Policy: configurable filter, width, dual anchoring
        │
        │  Parameters:
        │    depth D     — BFS depth
        │    width W     — max unproven body atoms
        │    filter      — 'prune' │ 'provset' │ 'none'
        │    dual_anchor — enumerate from both arg directions
        │
        ├── KGEGrounder      grounders/kge.py       + min(KGE(body)) scoring
        ├── NeuralGrounder   grounders/neural.py     + learned attention
        ├── SoftGrounder     grounders/soft.py       + soft provability
        ├── SamplerGrounder  grounders/sampler.py    + random subsampling
        └── LazyGrounder     grounders/lazy.py       + predicate reachability
```

### Algorithms

**PrologGrounder** — step-by-step proof search:

```
states = init(queries)                            [B, S, G, 3]
for d in range(D):
    goal = select(states)                         first non-padding atom
    cf = resolve_facts(goal, fact_index)           [B, K_f, G, 3]
    cr = resolve_rules(goal, rule_index)           [B, K_r, G+M, 3]
    states = compact(merge_children(cf, cr))      [B, S, G', 3]
return collect(terminal_states)                   goals exhausted → output
```

**EnumGrounder** — BFS template filling:

```
for d in range(D):
    rules = match(queries)                        by head predicate
    bodies = enumerate_and_resolve(
        queries, rules, metadata, fact_index)      fill templates from binding table
    collect complete groundings
    advance goal queue from proved body atoms
apply filter(groundings)                          prune │ provset │ none
return top tG groundings
```

### When to use what

| Grounder | Soundness | Speed | Best for |
|----------|-----------|-------|----------|
| Prolog (additive) | exact (compound state) | slower | Exact proofs, RL agents, small KBs |
| Prolog (cascade) | exact (compound state) | slower, more groundings | Dense rule sets |
| Enum (prune) | sound (fixed-point) | **fast** | **Default choice** |
| Enum (provset) | sound (FC oracle) | fast + FC init | Small provable sets |
| Enum (none) | no guarantee | fastest | Speed-critical experiments |
| KGE | inherited | + scoring | Quality-ranked groundings |
| Neural | inherited | + scoring | Learned ranking |
| Soft | inherited | + scoring | Differentiable provability |
| Sampler | inherited | + sampling | Diversity / exploration |
| Lazy | inherited | faster (fewer rules) | Large KBs, many unreachable predicates |

---

## 8. Dependency Graph

```
types.py                          ← no deps
    │
primitives.py                     ← no deps
    │
    ├── fact_index.py             ← primitives
    │
    ├── rule_index.py             ← no deps
    │
    └── compilation.py            ← no deps
            │
            └── forward_chaining.py ← compilation
                    │
grounders/                        │
    │                             │
    ├── base.py ──────────────────┤← fact_index, rule_index, compilation,
    │                             │  forward_chaining, types
    │
    ├── common.py                 ← primitives
    │
    ├── prolog.py                 ← base, common, primitives
    │
    ├── enum.py                   ← base, common, forward_chaining
    │
    ├── kge.py                    ← enum
    ├── neural.py                 ← enum
    ├── soft.py                   ← enum
    ├── sampler.py                ← enum
    └── lazy.py                   ← enum

factory.py                        ← all grounders (lazy imports)
```

No cycles. Each module depends only on modules above it or at its level.

---

## 9. Invariants

1. **Each operation once.** If two grounders need the same logic, it lives in `common.py` or the shared parent. Never copy-pasted.
2. **Full CUDA graph.** Every forward-pass operation runs under `torch.compile(fullgraph=True, mode='reduce-overhead')`. No fallback modes.
3. **Fixed tensor shapes.** All forward-pass outputs are statically shaped with validity masks. No dynamic allocation.
4. **No `.item()` in forward.** No GPU→CPU synchronization during grounding.
5. **No data-dependent branching.** Goal selection, compaction, and filtering use masked tensor operations only.
6. **Registered buffers.** All KB tensors are `nn.Module` buffers — automatic device movement.
7. **FC on CPU.** Forward chaining is a one-time precomputation, not in the compiled graph.
8. **Grounder-specific logic co-located.** Resolution functions that belong to one grounder live in that grounder's file, not in a shared module.

---

## 10. Extending NeSyGround

### Adding a scored grounder

Subclass `EnumGrounder`. Override `forward()` to add scoring after base enumeration. Place in `grounders/{name}.py`:

```python
class MyGrounder(EnumGrounder):
    def __init__(self, ..., my_param: int = 10):
        super().__init__(...)
        self.my_scorer = MyScorer(my_param)

    def forward(self, queries, query_mask):
        result = super().forward(queries, query_mask)
        scores = self.my_scorer(result.body, result.mask)
        # select top-k by score using masked operations (no .item()!)
        return GroundingResult(...)
```

Register in `factory.py`. Naming: `mygrounder_{width}_{depth}`.

### Adding a new resolution strategy

1. Create `grounders/{name}.py` with your grounder class and its resolution functions
2. Subclass `Grounder` directly (not `EnumGrounder` or `PrologGrounder`)
3. Use shared utilities from `common.py` (`compact`, `dedup`, `exclude_query`)
4. Override `forward()` — all operations must be CUDA-graph-safe
5. Register in `factory.py`
6. Document soundness properties in [soundness.md](soundness.md)

### CUDA graph compliance checklist

Before submitting a new grounder, verify:

- [ ] `torch.compile(fullgraph=True, mode='reduce-overhead')` succeeds
- [ ] No `.item()`, `.tolist()`, or `bool(tensor)` in `forward()`
- [ ] No `if`/`while` conditioned on tensor values
- [ ] All output shapes are fixed (same for any input of the same batch size)
- [ ] Uses cumsum + scatter for compaction, not `topk` or `nonzero`
- [ ] All tensors are registered buffers or function-local
