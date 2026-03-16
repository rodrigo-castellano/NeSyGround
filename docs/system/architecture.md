# Architecture

NeSyGround is a compiled FOL grounding library. It takes a knowledge base of facts and Horn clause rules, and produces ground rule instantiations as fixed-shape tensors on GPU under `torch.compile(fullgraph=True, mode='reduce-overhead')`.

---

## 1. Scope

NeSyGround is a **grounding library** — it computes which rule instantiations are relevant for a set of queries. It is not a model, not a reasoner, and not a training framework.

```
KB (F, R) ──→ Index ──→ BCGrounder ──→ GroundingResult ──→ Consumer
               once      per batch     [B, tG, M, 3]
```

| Stage | What it does | When |
|-------|-------------|------|
| **Index** | KB → indexed tensor structures (FactIndex, RuleIndex) | Once per KB |
| **Ground** | Queries → ground rule instantiations | Per batch, under CUDA graph |
| **Consume** | Use instantiations downstream (score, extract proofs, check provability) | Not our scope |

---

## 2. Grounding

A knowledge base `KB = (F, R)` consists of ground facts `F = {p(a,b), ...}` and Horn clause rules `R = {h(X,Y) :- b₁(X,Z), b₂(Z,Y), ...}`. All predicates are binary.

Given a ground query `q = p(a,b)` and depth bound `D`:

```
Ground(q, KB, D) = { (r, body(r)θ) | head(r)θ = q, θ grounds body(r), reachable within D steps }
```

---

## 3. Resolution

Three resolution strategies, all producing the same 7-tensor output format:

### Strategy A: MGU resolution (SLD, RTF)

Runtime unification via Most General Unifier. Carries variables through proof states.

- **SLD** (`resolution/sld.py`): parallel fact + rule resolution. `K = K_f + K_r`.
  ```
  resolve_sld = resolve_facts ∥ resolve_rules
  ```
- **RTF** (`resolution/rtf.py`): cascade rule-then-fact. `K = K_r × K_f`.
  ```
  resolve_rtf = resolve_rules → resolve_facts(body[0])
  ```

Both call shared primitives in `resolution/mgu.py`: `resolve_facts()` and `resolve_rules()`.

### Strategy B: Compiled enumeration (Enum)

Pre-compiled binding patterns from rule structure. Direct template filling at runtime — no unification.

- **Enum** (`resolution/enum.py`): enumerate entity candidates from fact index using pre-compiled binding tables. Iterative across depth (like keras-ns approximate BC).

### Unified output

All three strategies return the same 7-tensor tuple consumed by `_pack`:
```
(fact_goals, fact_gbody, fact_success, rule_goals, rule_gbody, rule_success, sub_rule_idx)
```

---

## 4. CUDA Graph Constraint

Everything runs under `torch.compile(fullgraph=True, mode='reduce-overhead')`:

- **Fixed tensor shapes** — dimensions known at trace time, validity via masks
- **No `.item()`** — no CPU readback in forward pass
- **No data-dependent branching** — no Python `if` on tensor values
- **Masks everywhere** — padding sentinel = `padding_idx`
- **Registered buffers** — all persistent tensors via `register_buffer()`

---

## 5. Dimensions

| Symbol | Meaning | Typical |
|--------|---------|---------|
| B | batch (queries) | 512 |
| S | proof states per query | 64–256 |
| G | max goals per state | 3–7 |
| M | max body atoms per rule | 2–3 |
| K | children per state per step | K_f+K_r or K_r×K_f |
| K_f | max fact matches | 64 |
| K_r | max rule matches | 10–20 |
| D | depth (proof steps) | 1–3 |
| tG | total groundings per query | 32–128 |
| R | number of rules | 10–100 |
| E | number of entities | 100–40000 |
| P | number of predicates | 10–500 |
| F | number of facts | 1000–200000 |

---

## 6. Directory Layout

```
grounder/
├── __init__.py          # public API
├── base.py              # Grounder(nn.Module) — owns KB state
├── types.py             # GroundingResult, ResolveResult, StepResult, PackResult
├── primitives.py        # apply_substitutions, unify_one_to_one
├── fact_index.py        # FactIndex → ArgKeyFactIndex | InvertedFactIndex → BlockSparseFactIndex
├── rule_index.py        # RuleIndex → RuleIndexEnum; RulePattern, compile_rules
├── factory.py           # create_grounder() dispatcher
├── data_loader.py       # KGDataset
│
├── bc/                  # backward chaining
│   ├── bc.py            # BCGrounder — unified, configurable
│   ├── common.py        # pack_states, compact_atoms, prune_ground_facts, collect_groundings
│   └── lazy.py          # LazyGrounder — predicate-filtered wrapper
│
├── fc/                  # forward chaining
│   └── fc.py            # run_forward_chaining (semi-naive FC for provable set)
│
├── resolution/          # resolution strategies (pluggable)
│   ├── mgu.py           # resolve_facts, resolve_rules, init_mgu (shared MGU primitives)
│   ├── sld.py           # resolve_sld (parallel fact + rule)
│   ├── rtf.py           # resolve_rtf (cascade rule-then-fact)
│   ├── enum.py          # resolve_enum, resolve_enum_step, init_enum
│   └── standardization.py  # variable standardization (MGU only)
│
├── filters/             # soundness filters
│   ├── __init__.py      # check_in_provable, cap_ground_children, prune_dead_nonground_rules
│   ├── prune.py         # apply_prune (PruneIncompleteProofs fixed-point)
│   └── provset.py       # apply_provset (FC provable set filter)
│
└── nesy/                # neural-symbolic hooks
    ├── hooks.py         # ResolutionHook, StepHook, GroundingHook (protocols)
    ├── kge.py           # KGEScorer (min-conjunction scoring)
    ├── neural.py        # NeuralScorer (learned attention)
    ├── soft.py          # SoftScorer (soft provability)
    └── sampler.py       # RandomSampler (random subsampling)
```

---

## 7. Modules

### base.py — `Grounder(nn.Module)`

Abstract root. Owns KB state: facts, rules, indices.

```python
class Grounder(nn.Module):
    def __init__(self, facts_idx, rules_heads_idx, rules_bodies_idx, rule_lens,
                 constant_no, padding_idx, device, **kwargs)
    # Builds: fact_index (ArgKey | Inverted | BlockSparse), rule_index (RuleIndex)
    # Registers: facts_idx, fact_hashes as buffers
    # Properties: num_facts, M, K_f, K_r, pack_base, constant_no, padding_idx
```

### types.py

```python
@dataclass
class GroundingResult:
    body: Tensor       # [B, tG, M, 3]
    mask: Tensor       # [B, tG]
    count: Tensor      # [B]
    rule_idx: Tensor   # [B, tG]
```

### fact_index.py

```
FactIndex(nn.Module)              — base: sort, hash, exists()
├── ArgKeyFactIndex               — MGU: O(1) targeted lookup via (pred, arg) tables
└── InvertedFactIndex             — Enum: O(1) enumeration via offset tables
    └── BlockSparseFactIndex      — Enum: dense [P, E, K] blocks (fallback to offset)
```

### rule_index.py

```
RuleIndex(nn.Module)              — base: sorted storage + segment lookup (for MGU)
└── RuleIndexEnum                 — adds binding analysis + enum metadata tensors

RulePattern                       — per-rule variable binding pattern (internal)
compile_rules()                   — raw tensors → List[RulePattern] (for FC)
```

### bc/bc.py — `BCGrounder(Grounder)`

Unified backward-chaining grounder. One class, configured by:

| Axis | Options |
|------|---------|
| resolution | `'sld'` \| `'rtf'` \| `'enum'` |
| filter | `'fp_batch'` \| `'fp_global'` \| `'none'` |
| hooks | `List[GroundingHook]` |
| depth | int (proof steps) |
| width | int \| None (enum: max unknown body atoms) |

**Canonical loop** (same code path for all resolutions):

```python
def ground(queries, query_mask) -> GroundingResult:
    states = init_states(queries, query_mask)
    for d in range(D):
        states = step(states, d)
    return filter_terminal(states)
```

**Step decomposition** (4 shared phases, only RESOLVE dispatches):

```python
def step(states, d):
    queries, remaining, active = _select(states)        # SELECT
    resolved = _resolve(queries, remaining, ..., d)      # RESOLVE (→ sld/rtf/enum)
    states = _pack(resolved, states)                     # PACK
    states = _postprocess(states)                        # POSTPROCESS
    return states
```

**State representation** (shared by all resolutions):

```python
states = {
    "queries":          Tensor,  # [B, 3]     original queries
    "query_mask":       Tensor,  # [B]
    "proof_goals":      Tensor,  # [B, S, G, 3]
    "grounding_body":   Tensor,  # [B, S, M, 3]
    "top_ridx":         Tensor,  # [B, S]
    "state_valid":      Tensor,  # [B, S]
    "next_var_indices": Tensor,  # [B]
    "collected_body":   Tensor,  # [B, tG, M, 3]
    "collected_mask":   Tensor,  # [B, tG]
    "collected_ridx":   Tensor,  # [B, tG]
}
```

**Hooks** applied in `forward()` after `ground()`:

```python
def forward(queries, query_mask):
    result = ground(queries, query_mask)
    for hook in hooks:
        body, mask, ridx = hook.apply(result.body, result.mask, result.rule_idx)
        result = GroundingResult(body=body, mask=mask, count=mask.sum(1), rule_idx=ridx)
    return result
```

### bc/common.py

```python
pack_states()          — flatten S×K children → propagate gbody/ridx → scatter-compact to S
compact_atoms()        — left-align goals after fact pruning (G dimension)
prune_ground_facts()   — remove known facts from proof goals
collect_groundings()   — collect completed proofs into output buffer
```

### bc/lazy.py — `LazyGrounder(nn.Module)`

Predicate-filtered wrapper. BFS on head→body predicate graph, filters rules to reachable predicates, wraps a BCGrounder with the filtered rule set.

### resolution/mgu.py

Shared MGU primitives + parameter computation:

```python
resolve_facts()    — targeted lookup or enumerate → unify → substitute
resolve_rules()    — segment lookup → standardize apart → unify head → substitute body
init_mgu()         — compute K, S, K_f, max_vars_per_rule, effective_total_G
```

### resolution/sld.py

```python
resolve_sld(queries, remaining, grounding_body, state_valid, active_mask, *,
            next_var_indices, fact_index, ...) -> 7-tuple
```

### resolution/rtf.py

```python
resolve_rtf(queries, remaining, grounding_body, state_valid, active_mask, *,
            next_var_indices, fact_index, ...) -> 7-tuple
```

### resolution/enum.py

```python
init_enum()            — build RuleIndexEnum + enum metadata + budgets
resolve_enum()         — core enumeration (12-tuple, compile-safe)
resolve_enum_step()    — adapter: flatten B×S, call resolve_enum, convert to 7-tuple
```

### filters/

```python
apply_prune(body, mask, queries, fact_index, pack_base, padding_idx, depth)
    # PruneIncompleteProofs fixed-point filter

apply_provset(body, mask, fact_index, pack_base, padding_idx, provable_hashes)
    # FC provable set filter

check_in_provable(atom_hashes, provable_hashes)
    # Binary search membership in sorted hash tensor
```

### nesy/hooks.py — Hook protocols

Three injection points in the pipeline:

| Hook | When | Protocol |
|------|------|----------|
| `ResolutionHook` | During RESOLVE | `score_candidates(candidates, context) → scores` |
| `StepHook` | After each STEP | `on_step(body, mask, rule_idx, d) → (body, mask, rule_idx)` |
| `GroundingHook` | After grounding | `apply(body, mask, rule_idx) → (body, mask, rule_idx)` |

Concrete `GroundingHook` implementations:

| Hook | Scoring method |
|------|---------------|
| `KGEScorer` | min(KGE body atom scores) + topk |
| `NeuralScorer` | learned attention MLP + topk |
| `SoftScorer` | soft provability (KGE sigmoid or MLP) + topk |
| `RandomSampler` | random (train) or valid-first (eval) + topk |

### factory.py

```python
create_grounder(grounder_type, *, facts_idx, rule_heads, ...) -> BCGrounder | LazyGrounder
```

All type strings map to `BCGrounder` with appropriate config:

| Type string | Resolution | Filter | Width |
|------------|------------|--------|-------|
| `bcprune_D` | sld | prune | — |
| `bcprovset_D` | sld | provset | — |
| `bcsld_D` | sld | none | — |
| `prolog_D` | sld | none | — |
| `rtf_D` | rtf | none | — |
| `backward_W_D` | enum | prune | W |
| `lazy_W_D` | enum (lazy) | prune | W |
| `full` | enum | prune | None (∞) |
| `kge_W_D` | enum | prune | W |
| `soft_W_D` | enum | prune | W |
| `sampler_W_D` | enum | prune | W |

---

## 8. Dependency Graph

```
factory.py
    ↓
bc/bc.py, bc/lazy.py
    ↓
resolution/sld.py, resolution/rtf.py, resolution/enum.py
    ↓
resolution/mgu.py, bc/common.py, filters/
    ↓
base.py (owns indices)
    ↓
types.py, primitives.py, fact_index.py, rule_index.py
```

**Constraints:**
- Acyclic — no circular imports
- Resolution modules import primitives and indices, never grounders
- Filters import primitives only (+ `check_in_provable`)
- NeSy hooks are leaf modules (import nothing from grounder except types)

---

## 9. Invariants

### Tensor shapes
- All shapes fixed at trace time
- Validity via boolean masks, never inferred from values
- Padding sentinel = `padding_idx` (registered at Grounder init)

### Hashing
- `pack_triples_64(atoms, base)` → injective int64 keys
- `fact_hashes` sorted for `searchsorted` membership
- `check_in_provable` uses binary search on sorted provable set

### Resolution soundness
- Every returned grounding `(r, bodyθ)` satisfies `head(r)θ = q`
- SLD: `K = K_f + K_r` (parallel)
- RTF: `K = K_r × K_f` (cascade)
- Enum: pre-compiled bindings, width-filtered

### BC depth bounding
- Exactly D calls to `step()`
- `init_states` creates one state per query
- `filter_terminal` collects completed groundings + applies soundness filter

### CUDA graph contract
- No graph breaks in compiled path
- No GPU→CPU readback
- No data-dependent Python branching in `forward()`
- `torch.compile(fullgraph=True, mode='reduce-overhead')`

---

## 10. Extending NeSyGround

### New resolution strategy

1. Add `resolution/new.py` with `resolve_new(queries, remaining, ...) -> 7-tuple`
2. Add dispatch in `BCGrounder._resolve()`
3. Optionally add `init_new()` for resolution-specific setup

### New soundness filter

1. Add `filters/new.py` with `apply_new(body, mask, ...) -> mask`
2. Add dispatch in `BCGrounder.filter_terminal()`

### New hook

1. Implement `GroundingHook` protocol (or `ResolutionHook`/`StepHook`)
2. Pass to `BCGrounder(hooks=[MyHook()])`

### New fact index

1. Subclass `FactIndex` in `fact_index.py`
2. Implement `exists()` and optionally `targeted_lookup()` or `enumerate()`
3. Add to `Grounder.__init__` dispatch

### Pre-merge checklist

- [ ] All tensors have documented shapes
- [ ] No `.item()` in forward path
- [ ] `torch.compile(fullgraph=True)` traces without graph break
- [ ] Unit tests pass: `python -u tests/run.py test unit`
- [ ] Speed regression: `python -u tests/run.py test speed`
- [ ] Grounding regression: `python -u tests/run.py test groundings`
