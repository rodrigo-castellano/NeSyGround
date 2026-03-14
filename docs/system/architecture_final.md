# Architecture

NeSyGround is a compiled FOL grounding library. It takes a knowledge base of facts and Horn clause rules, and produces ground rule instantiations as fixed-shape tensors on GPU under `torch.compile(fullgraph=True, mode='reduce-overhead')`.

---

## 1. Scope

NeSyGround is a **grounding library** — it computes which rule instantiations are relevant for a set of queries or a knowledge base. It is not a model, not a reasoner, and not a training framework.

```
KB (F, R) ──→ Compile ──→ Grounder ──→ GroundingResult ──→ Consumer
               once        per batch    [B, tG, M, 3]
```

| Stage | What it does | When |
|-------|-------------|------|
| **Compile** | KB → indexed tensor structures (FactIndex, RuleIndex, metadata) | Once per KB |
| **Ground** | Queries → ground rule instantiations | Per batch, under CUDA graph |
| **Consume** | Use instantiations downstream (score with t-norms, extract proofs, check provability, ...) | Not our scope |

The grounder produces instantiations. What happens next — scoring, proof extraction, provability checking — is the consumer's concern. These are strictly separated.

---

## 2. Grounding

A knowledge base `KB = (F, R)` consists of ground facts `F = {p(a,b), ...}` and Horn clause rules `R = {h(X,Y) :- b₁(X,Z), b₂(Z,Y), ...}`. All predicates are binary.

Given a ground query `q = p(a,b)` and depth bound `D`:

```
Ground(q, KB, D) = { (r, body(r)θ) | head(r)θ = q, θ grounds body(r), reachable within D steps }
```

The grounder produces these instantiations. A downstream consumer uses them. These concerns are strictly separated.

---

## 3. Resolution

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

## 4. CUDA Graph Constraint

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

## 5. Dimensions

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

## 6. Directory Layout

```
grounder/
├── __init__.py                    # public API exports
├── base.py                        # Grounder — abstract root (owns compiled KB)
├── types.py                       # GroundingResult, ProvableSet
├── primitives.py                  # unify, substitute, hash_atoms, hash_contains
├── fact_index.py                  # FactIndex protocol + ArgKey, Inverted, BlockSparse
├── rule_index.py                  # RuleIndex (segment + table access)
├── compilation.py                 # CompiledRule + compile_rules, tensorize, build_metadata
├── bc/
│   ├── __init__.py                # exports BCGrounder, LazyGrounder
│   ├── bc.py                      # BCGrounder(depth, width, filter, resolution)
│   ├── common.py                  # compact, dedup, exclude_query
│   └── lazy.py                    # LazyGrounder (predicate reachability → reduced rules)
├── fc/
│   ├── __init__.py                # exports FCGrounder
│   └── fc.py                      # FCGrounder — MATCH → JOIN → MERGE (SemiNaive, SpMM)
├── resolution/
│   ├── __init__.py                # exports resolution strategies
│   ├── mgu.py                     # resolve_facts, resolve_rules (shared MGU primitives)
│   ├── sld.py                     # SLD: compose mgu in parallel (K_f + K_r)
│   ├── rtf.py                     # RTF: compose mgu in cascade (K_f × K_r)
│   ├── enum.py                    # Compiled enumeration (no MGU)
│   └── standardization.py         # standardize_apart (variable renaming, MGU only)
├── filters/
│   ├── __init__.py                # exports filters
│   ├── prune.py                   # PruneIncompleteProofs fixed-point filter
│   └── provset.py                 # FC provable set filter (uses fc/)
├── nesy/
│   ├── __init__.py                # exports all NeSy hooks
│   ├── hooks.py                   # Hook protocols for each pipeline stage
│   ├── kge.py                     # KGE scoring (post-resolution + provability)
│   ├── neural.py                  # Learned attention (post-resolution)
│   ├── soft.py                    # Soft provability (provability hook)
│   └── sampler.py                 # Sampling strategy
└── factory.py                     # create_grounder, parse_grounder_type
```

**Top level** — `base.py` (abstract Grounder root) and infrastructure shared by all grounders: primitives, indices, compilation. Zero internal dependencies between these modules.

**`bc/`** — backward chaining. `BCGrounder` is the parametrized `BC_{w,d}` class. It uses a resolution strategy from `resolution/` and a filter from `filters/`. `LazyGrounder` extends BCGrounder with predicate reachability.

**`fc/`** — forward chaining. `FCGrounder` implements the T_P operator with MATCH → JOIN → MERGE phases. Includes SemiNaive and SpMM implementations.

**`resolution/`** — resolution strategies that plug into BCGrounder's `step()`. Pure FOL logic. MGU primitives (`mgu.py`) are shared by SLD and RTF. Compiled enumeration (`enum.py`) is independent.

**`filters/`** — soundness filters applied during/after resolution. Prune uses fixed-point analysis. Provset checks against FC provable set.

**`nesy/`** — neural-symbolic hooks. This is where you plug your neural strategy: scoring, ranking, soft provability. Hooks can be applied during resolution, post-resolution, or as provability replacements. Orthogonal to the FOL grounding logic.

### Hierarchy

```
Grounder (nn.Module)                          base.py
  │
  │  Owns: FactIndex, RuleIndex, CompiledRules, metadata buffers
  │  Contract: ground(queries, query_mask) → GroundingResult
  │            is_provable(atoms) → Tensor[bool]
  │
  ├── BCGrounder                              bc/bc.py
  │     │
  │     │  Parametrized: BC_{w,d} (Castellano et al., IJCAI 2025)
  │     │  Parameters: depth (d), width (w), filter, resolution
  │     │
  │     │  Canonical loop:
  │     │    states = init_states(queries)
  │     │    for d in range(D):
  │     │        states = step(states, d)          ← resolution-specific
  │     │    return filter_terminal(states)         ← keep only fully resolved
  │     │
  │     │  forward(queries, return_all_states=False):
  │     │    return_all_states=False → only completed groundings (training)
  │     │    return_all_states=True  → all states including unresolved (RL)
  │     │  RL agent: calls forward(depth=1, return_all_states=True) per decision
  │     │
  │     │  Abstract: step() — each resolution strategy implements this
  │     │  Shared: init_states, filter_terminal, check_known
  │     │
  │     │  Resolution strategies (pluggable via step):      resolution/
  │     │  ├── SLD            sld.py       mgu parallel (K_f + K_r)
  │     │  ├── RTF            rtf.py       mgu cascade (K_f × K_r)
  │     │  ├── Enumerate      enum.py      compiled enumeration (no MGU)
  │     │  └── (future: MagicSets, Tabled, ...)
  │     │
  │     │  Width: integrated into step() — prune states with #unknown > w
  │     │  Filters: prune | provset | none                  filters/
  │     │
  │     └── LazyGrounder      bc/lazy.py   predicate reachability → reduced rules
  │
  └── FCGrounder                              fc/fc.py
        │
        │  Data-driven, D iterations or fixpoint
        │  Abstract phases: match, join, merge
        │  Shared: initialize, is_fixpoint, compute_provable_set
        │  Contract: compute_provable_set() → ProvableSet
        │            is_provable(atoms) → Tensor[bool]
        │
        ├── SemiNaiveFC                       (in fc.py or separate)
        └── SpMMFC                            (in fc.py or separate)
```

### Configuration replaces classes

Every grounder variant is a configuration of `BCGrounder`, not a separate class:

| Configuration | Paper notation | Current class |
|---|---|---|
| `BCGrounder(depth=2, resolution='sld', filter='prune')` | — | BCPruneGrounder |
| `BCGrounder(depth=2, resolution='sld', filter='provset')` | — | BCProvsetGrounder |
| `BCGrounder(depth=2, width=1, resolution='enum', filter='prune')` | BC_{1,2} | ParametrizedBCGrounder |
| `BCGrounder(depth=1, width=None, resolution='enum', filter='none')` | BC^u_{∞,1} | FullBCGrounder |
| `BCGrounder(depth=2, width=None, resolution='sld', filter='none')` | BC_{∞,2} | PrologGrounder |
| `BCGrounder(depth=1, width=0, resolution='enum', filter='none')` | BC_{0,1} | Known body grounder |
| `BCGrounder(depth=2, width=1, resolution='enum', filter='none', hooks=[SoftHook(mode='kge')])` | — | SoftGrounder |
| `BCGrounder(depth=2, width=1, resolution='enum', filter='none', hooks=[SoftHook(mode='neural')])` | — | SoftGrounder (neural) |
| `BCGrounder(depth=2, width=1, resolution='enum', filter='prune', hooks=[KGEHook])` | — | KGEGrounder |
| `BCGrounder(depth=2, width=1, resolution='enum', filter='prune', hooks=[SamplerHook])` | — | SamplerGrounder |

### What lives where

| Concern | Location | Rationale |
|---------|----------|-----------|
| BCGrounder base + common utilities | `bc/` | BC loop, init_states, filter_terminal, compact, dedup |
| FCGrounder + SemiNaive, SpMM | `fc/` | T_P operator, provable set computation |
| MGU primitives (resolve_facts, resolve_rules) | `resolution/mgu.py` | Shared by SLD and RTF |
| SLD composition (parallel resolve) | `resolution/sld.py` | Thin: composes mgu in parallel |
| RTF composition (cascade resolve) | `resolution/rtf.py` | Thin: composes mgu in cascade |
| Compiled enumeration | `resolution/enum.py` | No MGU, pre-compiled bindings |
| Variable standardization | `resolution/standardization.py` | Used by MGU resolution only |
| Prune filter (fixed-point) | `filters/prune.py` | Pure FOL, no FC dependency |
| Provset filter (FC oracle) | `filters/provset.py` | Uses fc/ for provable set |
| NeSy hooks (KGE, neural, soft, sampler) | `nesy/` | Orthogonal to FOL grounding |

---

## 7. Modules

### `base.py` — Grounder abstract root

```python
class Grounder(nn.Module):
    """Abstract root. Owns compiled KB. All grounders inherit from this."""

    # Registered buffers (available to all subclasses)
    fact_index: FactIndex
    rule_index: RuleIndex
    compiled_rules: List[CompiledRule]
    fact_hashes: Tensor              # [F] sorted int64
    head_preds: Tensor               # [R]
    body_preds: Tensor               # [R, M]
    num_body: Tensor                 # [R]
    pred_rule_indices: Tensor        # [P, R_eff]
    pred_rule_mask: Tensor           # [P, R_eff]

    # Properties
    M: int                           # max body atoms per rule
    R_eff: int                       # max rules per predicate
    pack_base: int                   # E + 2

    def __init__(self, facts_idx, rules_heads_idx, rules_bodies_idx,
                 rule_lens, constant_no, padding_idx, **kwargs): ...

    def ground(self, queries: Tensor, query_mask: Tensor) -> GroundingResult:
        raise NotImplementedError

    def is_provable(self, atoms: Tensor) -> Tensor:
        raise NotImplementedError
```

### `types.py` — Output contracts

```python
@dataclass
class GroundingResult:
    body: Tensor       # [B, tG, M, 3]  grounded body atoms
    mask: Tensor       # [B, tG]         which groundings are valid
    count: Tensor      # [B]             valid groundings per query
    rule_idx: Tensor   # [B, tG]         which rule produced each grounding

@dataclass
class ProvableSet:
    hashes: Tensor     # [I_max] sorted int64 hashes of provable atoms
    n_provable: int    # number of valid entries
```

### `primitives.py` — FOL primitives

Three resolution primitives plus hash utilities. Zero internal dependencies. All CUDA-graph-safe.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `unify` | `(a [L,3], b [L,3], constant_no, pad_idx)` → `(mask [L], θ [L,2,2])` | Pairwise MGU: predicate match, constant conflicts, variable binding |
| `substitute` | `(atoms [N,M,3], θ [N,S,2,2], pad_idx)` → `[N,M,3]` | Apply substitutions. Loop-unrolled for S=2 |
| `hash_atoms` | `(atoms [...,3], base)` → `[...] int64` | `((p × base) + a₀) × base + a₁` |
| `hash_contains` | `(atoms [...,3], sorted_hashes [F], base)` → `[...] bool` | O(log F) membership via `torch.searchsorted` |

### `fact_index.py` — Fact storage and retrieval

One protocol, three implementations.

**Protocol:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `exists` | `(atoms [...,3])` → `[...] bool` | Membership via binary search on `fact_hashes` |
| `facts_idx` | `Tensor [F, 3]` | Raw fact triples |
| `fact_hashes` | `Tensor [F] int64` | Sorted atom hashes |
| `pack_base` | `int` | Hash multiplier |

**Implementations:**

| Class | Extra method | Best for |
|-------|-------------|----------|
| `ArgKeyFactIndex` | `lookup(query [B,3], max_k)` → `(idx [B,K], valid [B,K])` | MGU resolution (targeted lookup) |
| `InvertedFactIndex` | `enumerate(pred, bound, dir)` → `(cands [N,K], valid [N,K])` | Compiled enumeration (candidate listing) |
| `BlockSparseFactIndex` | Both | Either (dense blocks, higher memory) |

### `rule_index.py` — Rule-to-predicate mapping

```python
class RuleIndex(nn.Module):
    rules_heads: Tensor    # [R, 3]
    rules_bodies: Tensor   # [R, M, 3]
    rule_lens: Tensor      # [R]
    R_eff: int             # max rules sharing one head predicate
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `lookup_by_segments` | `(preds [B], max_k)` → `(idx, valid, qidx)` | Sequential segment access (MGU) |
| `lookup_by_table` | `(preds [N])` → `(idx [N, R_eff], mask)` | Parallel gather (compiled enumeration) |

### `compilation.py` — KB compilation (once)

```python
@dataclass
class CompiledRule:
    rule_idx: int
    head_pred: int
    head_bindings: Tuple[int, int]
    num_body: int
    body_preds: List[int]
    body_arg_sources: List[Tuple[int, int]]   # HEAD_VAR0=0, HEAD_VAR1=1, FREE_VAR(i)=2+i
    free_vars: List[int]
    body_order: List[int]                     # topological processing order
```

| Function | Output | Purpose |
|----------|--------|---------|
| `compile_rules` | `List[CompiledRule]` | Extract bindings, free vars, body order per rule |
| `tensorize_rules` | `head_preds [R]`, `body_preds [R,M]`, `num_body [R]` | Rule structure as tensors |
| `build_enum_metadata` | `has_free [R]`, `enum_pred [R]`, `enum_bound [R]`, `enum_dir [R]`, `check_arg_source [R,M,2]` | Binding tables for compiled enumeration |
| `build_rule_clustering` | `pred_rule_indices [P, R_eff]`, `pred_rule_mask [P, R_eff]` | Rules grouped by head predicate |

---

### `bc/bc.py` — BCGrounder

```python
class BCGrounder(Grounder):
    """Parametrized BC_{w,d} grounder (Castellano et al., IJCAI 2025).

    Configurable with orthogonal choices:
      depth (d), width (w), resolution strategy, soundness filter.
    """

    def __init__(self, ...,
                 depth: int = 2,
                 width: int | None = 1,
                 resolution: str = 'enum',        # 'sld' | 'rtf' | 'enum'
                 filter: str = 'prune',            # 'prune' | 'provset' | 'none'
                 max_total_groundings: int = 100,
                 compile_mode: str = 'reduce-overhead',
                 **kwargs): ...

    def ground(self, queries: Tensor, query_mask: Tensor,
               return_all_states: bool = False) -> GroundingResult:
        """Full D-step proof loop.
        return_all_states=False: only completed groundings (training).
        return_all_states=True:  all states including unresolved (RL).
        """
        states = self.init_states(queries)
        for d in range(self.depth):
            states = self.step(states, d)
        if return_all_states:
            return self._pack_all_states(states)
        return self.filter_terminal(states)

    def step(self, states, depth) -> states:
        """One resolution step. Delegates to resolution strategy."""

    def init_states(self, queries) -> states:
        """Query → initial proof states."""

    def filter_terminal(self, states) -> GroundingResult:
        """Keep only states where all goals are resolved."""

    def check_known(self, atoms: Tensor) -> Tensor:
        """Is atom known? Depends on filter (fact-only, provset, etc.)."""

    def is_provable(self, atoms: Tensor) -> Tensor:
        """Check if at least one grounding exists per atom."""
```

### `bc/common.py` — BC utilities

All CUDA-graph-safe. Used by BCGrounder and resolution strategies.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compact` | `(tensor [...,N,*], mask [...,N])` → same shape, valid first | Cumsum + scatter compaction |
| `dedup` | `(groundings [...,M,3], mask [...])` → updated mask | Hash-sort-adjacent_diff dedup |
| `exclude_query` | `(body [...,M,3], queries [B,3], mask)` → updated mask | Prevent trivial self-groundings |

### `bc/lazy.py` — LazyGrounder

```python
class LazyGrounder(BCGrounder):
    """Predicate reachability BFS at init → reduced rule set → smaller search."""

    def __init__(self, ..., query_predicates: set[str] | None = None): ...

    @staticmethod
    def compute_reachable_predicates(rules, query_predicates) -> set[str]:
        """BFS on head→body predicate graph."""
```

---

### `fc/fc.py` — FCGrounder

```python
class FCGrounder(Grounder):
    """Forward chaining via T_P operator. Semi-naive evaluation."""

    def __init__(self, ...,
                 max_iterations: int = 10,
                 method: str = 'semi_naive',     # 'semi_naive' | 'spmm'
                 **kwargs): ...

    def compute_provable_set(self) -> ProvableSet:
        """Iterate T_P until fixpoint or max_iterations.
        I₀ = F;  Iₙ₊₁ = Iₙ ∪ { head(r)θ | body(r)θ ⊆ Iₙ, ≥1 body atom from Δₙ }
        """
        I, delta = self.initialize()
        for d in range(self.max_iterations):
            applicable = self.match(delta, I)
            new_atoms = self.join(applicable, I)
            I, delta, is_fixpoint = self.merge(I, new_atoms)
            if is_fixpoint:
                break
        return ProvableSet(hashes=I, n_provable=len(I))

    def is_provable(self, atoms: Tensor) -> Tensor:
        """O(log I) membership via searchsorted on provable hashes."""

    # Abstract phases — implementations override these
    def match(self, delta, I) -> applicable: ...
    def join(self, applicable, I) -> new_atoms: ...
    def merge(self, I, new_atoms) -> (I, delta, is_fixpoint): ...

    def initialize(self) -> (I, delta):
        """I₀ = facts, Δ₀ = facts."""
```

FC runs on CPU at init time — not under CUDA graph. Not per-batch.

---

### `resolution/mgu.py` — MGU primitives

Shared by SLD and RTF. All CUDA-graph-safe.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `resolve_facts` | `(goals [B,S,3], remaining [B,S,G,3], fact_index, ...)` → `(children [B,S,K_f,G,3], success [B,S,K_f])` | Targeted lookup → unify → substitute. Goal proven. |
| `resolve_rules` | `(goals [B,S,3], remaining [B,S,G,3], rule_index, ...)` → `(children [B,S,K_r,G+M,3], success [B,S,K_r], rule_idx [B,S,K_r])` | Lookup matching rules → standardize apart → unify head → substitute body. Goal replaced by body. |

### `resolution/sld.py` — SLD resolution

```python
def sld_step(states, fact_index, rule_index, **kwargs) -> states:
    """One SLD resolution step: resolve facts and rules in parallel.
    K = K_f + K_r children per state.

    Composes:
      fact_children = mgu.resolve_facts(goal, ...)
      rule_children = mgu.resolve_rules(goal, ...)
      merged = interleave(fact_children, rule_children)
      return compact(merged)
    """
```

### `resolution/rtf.py` — RTF resolution

```python
def rtf_step(states, fact_index, rule_index, **kwargs) -> states:
    """One RTF resolution step: resolve rules first, then facts on body.
    K = K_f × K_r children per state.

    Composes:
      rule_children = mgu.resolve_rules(goal, ...)
      for each rule_child:
          fact_children = mgu.resolve_facts(body_atom, ...)
      return compact(all_children)
    """
```

### `resolution/enum.py` — Compiled enumeration

```python
def enum_step(queries, pred_rule_indices, metadata, fact_index,
              **kwargs) -> states:
    """One compiled enumeration step. No MGU — uses pre-compiled bindings.

    1. gather_rule_metadata: collect per-rule binding tables for active rules
    2. enumerate candidates from fact_index (InvertedFactIndex.enumerate)
    3. fill_body_templates: populate body atoms from binding table + candidates
    """

def gather_rule_metadata(pred_rule_indices, pred_rule_mask,
                         query_preds, buffers) -> active_metadata: ...

def fill_body_templates(query_subjs, query_objs, candidates,
                        check_arg_source, body_preds, M) -> body: ...
```

### `resolution/standardization.py` — Variable renaming

Two modes for preventing variable collision across rules. Used by MGU resolution only — compiled enumeration has no variables.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `standardize_canonical` | `(rule_bodies [B,K,M,3], rule_idx [B,K], canonical_offsets [R])` → `[B,K,M,3]` | Pre-assigned fixed variable indices per rule. Offsets computed at compilation. No runtime state. CUDA-graph-friendly. |
| `standardize_offset` | `(rule_bodies [B,K,M,3], next_var [B])` → `(renamed [B,K,M,3], new_next_var [B])` | Runtime counter. Rename variables starting from `next_var`. More flexible, requires mutable state. |

**Canonical** is preferred for `torch.compile` — no mutable `next_var` tracking. **Offset** is needed when the same rule may appear multiple times in a proof state with different variable bindings.

---

### `filters/prune.py` — PruneIncompleteProofs

| Function | Signature | Purpose |
|----------|-----------|---------|
| `apply_prune` | `(body [B,tG,M,3], mask [B,tG], fact_hashes, pack_base)` → updated mask | Iterative fixed-point: remove groundings whose body atoms have no supporting proof. No FC dependency. |

### `filters/provset.py` — FC provable set filter

| Function | Signature | Purpose |
|----------|-----------|---------|
| `apply_provset` | `(body [B,tG,M,3], mask [B,tG], provable_hashes, pack_base)` → updated mask | Check body atoms against FC provable set via `hash_contains`. |
| `build_provset` | `(fc_grounder: FCGrounder)` → `ProvableSet` | Convenience: run FC and return sorted hashes. |

---

### `nesy/hooks.py` — NeSy hook protocols

```python
class ResolutionHook(Protocol):
    """Applied during resolution — scores/filters entity candidates."""
    def score_candidates(self, candidates: Tensor, context: Tensor) -> Tensor: ...

class PostResolutionHook(Protocol):
    """Applied after resolution — scores/ranks final groundings."""
    def score_groundings(self, result: GroundingResult) -> Tensor: ...

class ProvabilityHook(Protocol):
    """Replaces hard provability with soft scores."""
    def provability_score(self, atoms: Tensor) -> Tensor: ...
```

### `nesy/kge.py` — KGE scoring

```python
class KGEHook(PostResolutionHook):
    """Score groundings by min(KGE(body_atom)). Select top-k."""
    kge_model: nn.Module   # must have .score_atoms(preds, subjs, objs)
```

### `nesy/neural.py` — Learned attention

```python
class NeuralHook(PostResolutionHook):
    """Score groundings via learned GroundingAttention MLP."""
    attention: nn.Module   # MLP: M*E → 1
```

### `nesy/soft.py` — Soft provability

```python
class SoftHook(ProvabilityHook):
    """known=1.0, unknown=sigmoid(score). Rank by product across body."""
    kge_model: nn.Module
    mode: str              # 'kge' | 'neural'
```

### `nesy/sampler.py` — Sampling

```python
class SamplerHook(PostResolutionHook):
    """Oversample, then random (train) or deterministic (eval) top-k."""
    sample_ratio: float
```

---

### `factory.py` — Dispatch

| Function | Signature | Purpose |
|----------|-----------|---------|
| `create_grounder` | `(name, facts, rules, ...)` → `Grounder` | Build and return fully configured grounder |
| `parse_grounder_type` | `(name: str)` → `(cls, kwargs)` | `"bcprune_2"` → `(BCGrounder, {depth=2, resolution='sld', filter='prune'})` |

**Naming convention:** `{type}_{depth}` or `{type}_{width}_{depth}`.

| Pattern | Configuration |
|---------|---------------|
| `bcprune_2` | `BCGrounder(depth=2, resolution='sld', filter='prune')` |
| `bcprovset_2` | `BCGrounder(depth=2, resolution='sld', filter='provset')` |
| `bcprolog_2` | `BCGrounder(depth=2, resolution='sld', filter='none')` |
| `rtf_2` | `BCGrounder(depth=2, resolution='rtf', filter='none')` |
| `backward_1_2` | `BCGrounder(depth=2, width=1, resolution='enum', filter='prune')` |
| `full` | `BCGrounder(depth=1, width=None, resolution='enum', filter='none')` |
| `lazy_0_1` | `LazyGrounder(depth=1, width=0, resolution='enum')` |
| `soft_1_2` | `BCGrounder(depth=2, width=1, resolution='enum', hooks=[SoftHook(mode='kge')])` |
| `softneural_1_2` | `BCGrounder(depth=2, width=1, resolution='enum', hooks=[SoftHook(mode='neural')])` |
| `kge_1_2` | `BCGrounder(depth=2, width=1, resolution='enum', filter='prune', hooks=[KGEHook])` |
| `sampler_1_2` | `BCGrounder(depth=2, width=1, resolution='enum', filter='prune', hooks=[SamplerHook])` |

---

## 8. Dependency Graph

Import relationships between modules. An arrow `A → B` means A imports from B. No cycles exist.

```
                    ┌──────────────────────────────────────────────────┐
                    │                  factory.py                      │
                    │  (imports everything to dispatch)                │
                    └──┬────────┬────────┬────────┬────────┬──────────┘
                       │        │        │        │        │
                       ▼        ▼        ▼        ▼        ▼
                   bc/bc.py  bc/lazy.py fc/fc.py nesy/*  filters/*
                       │        │        │
         ┌─────────────┼────────┘        │
         │             │                 │
         ▼             ▼                 ▼
    ┌─────────┐   ┌──────────┐     ┌──────────┐
    │ bc/     │   │resolution│     │ filters/ │
    │common.py│   │  /*.py   │     │  *.py    │
    └────┬────┘   └─────┬────┘     └────┬─────┘
         │              │               │
         │    ┌─────────┼───────────────┘
         │    │         │
         ▼    ▼         ▼
    ┌──────────────────────────┐
    │       base.py            │
    │  (Grounder, owns indices │
    │   and compiled KB)       │
    └──────┬───────────────────┘
           │
     ┌─────┼──────────┬──────────────┐
     │     │          │              │
     ▼     ▼          ▼              ▼
 types.py  primitives.py  fact_index.py  rule_index.py  compilation.py
     │                        │              │              │
     └────────────────────────┴──────────────┴──────────────┘
                            (leaf modules — no internal imports)
```

### Layer-by-layer

| Layer | Modules | Imports from |
|-------|---------|-------------|
| **0 — Leaf** | `types.py`, `primitives.py` | Nothing internal |
| **1 — Indices** | `fact_index.py`, `rule_index.py`, `compilation.py` | `primitives` (for hashing), `types` |
| **2 — Base** | `base.py` | Layer 0 + Layer 1 |
| **3a — Resolution** | `resolution/mgu.py` | `primitives`, `fact_index`, `rule_index` |
| **3b — Resolution** | `resolution/sld.py`, `resolution/rtf.py` | `resolution/mgu`, `bc/common` |
| **3c — Resolution** | `resolution/enum.py` | `fact_index`, `rule_index`, `compilation` |
| **3d — Resolution** | `resolution/standardization.py` | `primitives` |
| **3e — BC utilities** | `bc/common.py` | `primitives` |
| **4a — Grounders** | `bc/bc.py` | `base`, `bc/common`, `resolution/*`, `filters/*` |
| **4b — Grounders** | `bc/lazy.py` | `bc/bc` |
| **4c — Grounders** | `fc/fc.py` | `base`, `primitives`, `fact_index`, `rule_index` |
| **4d — Filters** | `filters/prune.py` | `primitives` |
| **4e — Filters** | `filters/provset.py` | `primitives`, `fc/fc` (for ProvableSet) |
| **5 — NeSy** | `nesy/hooks.py` | `types` (GroundingResult only) |
| **5 — NeSy** | `nesy/kge.py`, `neural.py`, `soft.py`, `sampler.py` | `nesy/hooks`, `types` |
| **6 — Dispatch** | `factory.py` | All of the above |

### Key constraints

1. **No cycles.** If module A imports B, then B never imports A (directly or transitively).
2. **Primitives and types are leaf.** They import nothing from the grounder package. Any module can import them.
3. **Resolution does not import grounders.** Resolution strategies are stateless functions that receive tensors — they do not know about BCGrounder or FCGrounder.
4. **Filters do not import grounders.** `provset.py` imports from `fc/` for `ProvableSet` but never from `bc/`. `prune.py` imports only primitives.
5. **NeSy hooks depend only on `types.py`.** Hook protocols reference `GroundingResult` and `Tensor`, nothing else. Concrete hooks (kge, neural, soft, sampler) depend on their protocol and `types`.
6. **`fc/` and `bc/` are independent.** They share only the common base class (`base.py`) and leaf modules. The only cross-reference is `filters/provset.py` importing from `fc/` — this is intentional (provset filter needs FC provable set).
7. **`factory.py` is the only module that imports everything.** It is never imported by any other module in the package.

---

## 9. Invariants

Properties that must hold at all times. Every module, every grounder, every extension. Violating any of these is a bug.

### 9.1 Tensor shape invariants

All tensors passed through `forward()` have **shapes fixed at trace time**. No operation may produce a tensor whose shape depends on data values.

| Invariant | Holds for |
|-----------|-----------|
| Queries are `[B, 3]` with `query_mask [B]` | All grounders |
| Output body is `[B, tG, M, 3]` | All grounders |
| Output mask is `[B, tG]`, count is `[B]`, rule_idx is `[B, tG]` | All grounders |
| Proof states are `[B, S, G, 3]` (MGU) or `[B, R_eff, K, M, 3]` (enum) | BC grounders |
| Provable set is `[I_max]` (pre-allocated, `n_provable` tracks valid entries) | FC grounders |

**Consequence**: variable-length results are represented as padded tensors + validity masks. Never as Python lists or dynamically-sized tensors.

### 9.2 Padding and sentinel

| Invariant | Value |
|-----------|-------|
| Padding index for entities | `E` (one past the last valid entity `0..E-1`) |
| Padding index for predicates | `P` (one past the last valid predicate `0..P-1`) |
| A padded atom is `(P, E, E)` | All three fields set to their respective padding values |
| `hash_atoms(padded_atom) ≠ hash_atoms(valid_atom)` for all valid atoms | Guaranteed by `pack_base = E + 2` |
| Masks are the sole indicator of validity | Never infer validity from entity/predicate values directly |

**Consequence**: code must never use `-1` as a sentinel (breaks unsigned indexing) or `0` (collides with entity 0).

### 9.3 Hash invariants

| Invariant | Enforced by |
|-----------|-------------|
| `hash_atoms` is injective over valid atoms: `h(a) = h(b) ⟹ a = b` | `pack_base = E + 2` ensures no collision for entities in `0..E` |
| `fact_hashes` is sorted ascending | `compilation.py` sorts once at build time |
| `hash_contains` returns correct membership | Binary search on sorted `fact_hashes` via `torch.searchsorted` |
| Hash of padded atom does not collide with any valid atom hash | `pack_base` spacing guarantees separation |

### 9.4 Compilation invariants

These hold after `compile_rules` and remain true for the lifetime of the grounder:

| Invariant | Meaning |
|-----------|---------|
| `len(compiled_rules) == R` | One CompiledRule per rule |
| `body_arg_sources[i]` is consistent with rule `i`'s head-body variable sharing | Pre-computed binding pattern matches the symbolic rule |
| `pred_rule_indices[p]` contains exactly the rules whose head predicate is `p` | Rule clustering is complete and correct |
| `pred_rule_mask[p]` is `True` for valid entries, `False` for padding | Padding in rule clustering is masked |
| `canonical_offsets` are globally unique across rules | No two rules share variable index space (canonical mode) |
| All compiled tensors are registered as `nn.Module` buffers | Automatic device placement |

### 9.5 Resolution soundness

Every grounding in the output is a valid instantiation of some rule in the KB:

| Invariant | Meaning |
|-----------|---------|
| If `body[b, g]` is valid (mask is True), then there exists rule `r = rule_idx[b, g]` and substitution `θ` such that `body[b, g] = body(r)θ` | Groundings are real rule instantiations, not fabricated |
| `head(r)θ` is reachable from `queries[b]` within `D` steps | Depth bound is respected |
| For MGU resolution: `θ` is the composition of all unifiers along the proof path | Unification is correct and complete at each step |
| For compiled enumeration: `body_arg_sources` correctly reconstructs `body(r)θ` from query args + enumerated candidates | Template filling matches what unification would produce |

### 9.6 BC invariants

| Invariant | Meaning |
|-----------|---------|
| Exactly `D` calls to `step()` | Proof depth is bounded. No early termination of the loop (terminated states are carried as padding). |
| After `filter_terminal`: every valid grounding has zero unresolved goals | Only complete proofs appear in the output |
| If `width = w`: every valid state has at most `w` body atoms not in the known set | Width bound is enforced during resolution, not post-hoc |
| If `width = None`: no width filtering is applied | Unbounded width |
| `init_states` produces exactly one initial state per query with the query as the sole goal | Starting point is well-defined |
| `step()` is pure: `step(states, d)` depends only on `states` and compiled KB, not on `d`'s value | Depth index is metadata, not control flow. CUDA graph replays the same operations. |

### 9.7 FC invariants

| Invariant | Meaning |
|-----------|---------|
| `I₀ = F` (initial provable set equals facts) | Base case of T_P |
| `Iₙ ⊆ Iₙ₊₁` (monotonic growth) | T_P is monotone for Horn clauses |
| Semi-naive: every new derivation uses at least one atom from `Δₙ` | No redundant re-derivation |
| Fixpoint: `Iₙ₊₁ = Iₙ` ⟹ loop terminates | Correctness of early stopping |
| `is_provable(a)` ⟺ `hash_contains(hash(a), provable_hashes)` | Provability check is consistent with computed set |

### 9.8 Mask discipline

| Invariant | Meaning |
|-----------|---------|
| `mask[i] = False` ⟹ `tensor[i]` is padding and must not be read | Invalid entries may contain garbage |
| After `compact()`: all valid entries precede all padding entries | `valid, valid, ..., pad, pad, ...` — no gaps |
| After `dedup()`: no two valid entries have the same hash | Duplicates are masked out |
| `count[b] = mask[b].sum()` | Count is consistent with mask |
| Masks are never used for control flow (`if mask.any()`) | CUDA graph constraint — masks are data, not branches |

### 9.9 CUDA graph contract

| Invariant | Meaning |
|-----------|---------|
| `forward()` contains zero graph breaks | Verified by `torch.compile(fullgraph=True)` — compilation fails otherwise |
| No `.item()`, `.tolist()`, or CPU readback in `forward()` | GPU→CPU sync breaks capture |
| No Python `if`/`while` on tensor values in `forward()` | Data-dependent branching breaks capture |
| All buffers are registered on the module | Device placement is automatic at trace time |
| FC runs at init time, not inside `forward()` | FC is not under CUDA graph |

---

## 10. Extending NeSyGround

How to add new components without modifying existing code. Each extension point has a clear contract, a single file to create, and a single registration point.

### 10.1 New resolution strategy

**When**: you have a new way to resolve goals against facts and rules (e.g., magic sets, tabled resolution).

**What to do**:

1. Create `resolution/magic.py` (or similar)
2. Implement a step function with this signature:

```python
def magic_step(states, fact_index, rule_index, **kwargs) -> states:
    """One resolution step using magic sets.

    Must return states with the same tensor shapes as input.
    Must be CUDA-graph-safe: no .item(), no dynamic shapes, no data-dependent branching.
    """
```

3. Register in `bc/bc.py` — add `'magic'` to the resolution dispatch in `step()`
4. Register in `factory.py` — add a naming pattern (e.g., `bcmagic_2`)

**Constraints**:
- The step function receives and returns fixed-shape tensors. No exceptions.
- It may import from `primitives.py`, `fact_index.py`, `rule_index.py`, `bc/common.py`. It must not import from `bc/bc.py` or `fc/`.
- If it needs new compiled metadata, add it to `compilation.py` and `build_*_metadata()`.

### 10.2 New soundness filter

**When**: you have a new criterion for pruning unsound or irrelevant groundings (e.g., type checking, confidence thresholds).

**What to do**:

1. Create `filters/typecheck.py` (or similar)
2. Implement:

```python
def apply_typecheck(body: Tensor, mask: Tensor, ...) -> Tensor:
    """Return updated mask. Must not change tensor shapes.

    body: [B, tG, M, 3]
    mask: [B, tG]
    Returns: [B, tG] updated mask (can only set True → False, never False → True)
    """
```

3. Register in `bc/bc.py` — add `'typecheck'` to the filter dispatch
4. Register in `factory.py`

**Constraints**:
- Filters can only tighten the mask (set True → False). Never inflate.
- If the filter needs pre-computed data (like provset needs FC), compute it in `__init__`, not in `forward()`.

### 10.3 New NeSy hook

**When**: you want to plug a neural component into the grounding pipeline (e.g., a GNN-based candidate scorer, a transformer-based provability model).

**What to do**:

1. Identify which hook protocol to implement:

| Protocol | When it fires | What it does |
|----------|--------------|-------------|
| `ResolutionHook` | During resolution, per candidate | Score/filter entity candidates before they become groundings |
| `PostResolutionHook` | After `step()` produces groundings | Score/rank/re-order final groundings |
| `ProvabilityHook` | When `check_known()` is called | Replace hard provability with soft scores |

2. Create `nesy/gnn.py` (or similar)
3. Implement the protocol:

```python
class GNNHook(ResolutionHook):
    def __init__(self, gnn_model: nn.Module): ...

    def score_candidates(self, candidates: Tensor, context: Tensor) -> Tensor:
        """Return scores [N, K]. Must be CUDA-graph-safe."""
```

4. Pass the hook to `BCGrounder` at construction time

**Constraints**:
- Hooks must be `nn.Module` subclasses (parameters participate in training).
- Hook methods must be CUDA-graph-safe — they run inside `forward()`.
- Hooks depend only on `types.py` (for `GroundingResult`) and `torch`. Never import from `bc/`, `fc/`, or `resolution/`.

### 10.4 New FC implementation

**When**: you have a new forward chaining strategy (e.g., parallel datalog, stratified negation).

**What to do**:

1. Subclass `FCGrounder` in `fc/fc.py` (or create `fc/stratified.py` for large implementations)
2. Override the three phases:

```python
class StratifiedFC(FCGrounder):
    def match(self, delta, I) -> applicable: ...
    def join(self, applicable, I) -> new_atoms: ...
    def merge(self, I, new_atoms) -> (I, delta, is_fixpoint): ...
```

3. Register in `factory.py`

**Constraints**:
- FC runs at init time (CPU), not under CUDA graph. Dynamic shapes and Python control flow are allowed.
- Must maintain FC invariants: monotonicity (`Iₙ ⊆ Iₙ₊₁`), semi-naive (use delta), fixpoint detection.
- Output must be a `ProvableSet` with sorted hashes.

### 10.5 New fact index

**When**: you need a different storage/retrieval trade-off (e.g., LSH-based approximate lookup, GPU hash table).

**What to do**:

1. Create a new class in `fact_index.py` implementing the `FactIndex` protocol:

```python
class LSHFactIndex(nn.Module):
    def exists(self, atoms: Tensor) -> Tensor: ...    # [...] bool

    # Plus at least one of:
    def lookup(self, query, max_k) -> (idx, valid):     # for MGU resolution
    def enumerate(self, pred, bound, dir) -> (cands, valid):  # for compiled enumeration
```

2. Register in `factory.py` or pass directly to grounder constructor

**Constraints**:
- Must be an `nn.Module` (buffers for device placement).
- `exists()` must be consistent with the stored facts — no false negatives.
- All methods called during `forward()` must be CUDA-graph-safe.

### 10.6 New BCGrounder subclass

**When**: you need to modify the BC loop itself, not just swap resolution/filter/hooks (e.g., `LazyGrounder` modifies the rule set at init time).

**What to do**:

1. Create `bc/mygrounder.py`
2. Subclass `BCGrounder`:

```python
class MyGrounder(BCGrounder):
    def __init__(self, ..., my_param, **kwargs):
        super().__init__(**kwargs)
        # modify compiled KB, add buffers, etc.
```

3. Override only what changes. The canonical loop (`ground → init_states → step × D → filter_terminal`) should rarely need overriding.

**Constraints**:
- Prefer configuration over subclassing. If your variant is just a combination of existing resolution + filter + hooks, use `BCGrounder(...)` directly.
- If you override `ground()`, you must still call `step()` exactly `D` times (invariant 9.6).
- Register in `factory.py`.

### Extension checklist

Before merging any extension:

- [ ] No new imports from modules at the same or higher layer (section 8)
- [ ] All tensors in `forward()` have fixed shapes
- [ ] Zero `.item()` / `.tolist()` / CPU readback in `forward()`
- [ ] No Python `if`/`while` on tensor values in `forward()`
- [ ] `torch.compile(fullgraph=True)` succeeds
- [ ] All invariants from section 9 still hold
- [ ] Registered in `factory.py` with a naming pattern
- [ ] Unit test covering the new component in isolation
