# GROUNDER LIBRARY — CLEAN-SHEET REDESIGN (canonical)

Repo: `/home/castellanoontiv/repos/grounder-swarm/main`

This is the final, canonical redesign. It is a **proper redesign**: the current tree is referenced only to learn responsibilities and the tensor/grounding domain. Nothing is framed as "migration"; every module is designed for what it *should* be.

The single most important domain correction (verified at `bc/finalize.py:60-89`, `bc/considered.py`, `bc/forward.py:50-75/320-340/455-465`): the rule-grounding output is built from a **considered firing accumulator that is PRIMARY**, with evidence-derived groundings as a **chunk-merge-only fallback**. Evidence-only undercounts ~3× (ablation_d3 BC13: 80 vs keras 252) because evidence sees firings only inside *completed* proof trees. This redesign preserves that semantics and — critically — fixes the four contracts the prior draft asserted but could not satisfy:

1. **The considered accumulator spans chunks.** It is reset once per `forward()`, accumulates across all chunks (lifting batch-local `b_idx` to a global index), and is finalized once at merge (verified `bc/forward.py:56-71`, `:329-335`). It therefore lives on a per-call **`RunState`** threaded through `iter_chunks`/`merge`, **not** purely by value inside a per-chunk `StepState`. (Fixes the impossible "threaded-by-value + built-once-at-merge" triangle.)
2. **Capture is split into a graph-safe in-step scatter and an eager post-loop finalize.** The compiled inner step *cannot* run the considered capture as written today (`bc/step.py:107-110` skips it under `torch.compiler.is_compiling()` because of the Python `list.append`). So capture writes into a **pre-allocated static survivor buffer inside the step** (no `.item()`, no append), and dedup/validation/`variant_to_orig` collapse run **eager after the loop**. This is the *only* design that makes compiled-dense produce the considered count instead of silently degrading to the ~3× undercount.
3. **`d` and `is_last` are an int/bool on the eager path and 0-dim tensors on the compiled path, and the duality spans every phase that consumes them** (resolve width-gate, enum, sync write-slot), funneled through one helper. Verified: `bc/compiled_step.py:69-70` builds `d_t`/`is_last_t` as 0-dim tensors; `resolution/enum.py:222-226` consumes them as tensors; `bc/step.py:400/447` branches on `isinstance(d, Tensor)`. The duality is **not** confinable to `sync.py`.
4. **Forward chaining preserves the old `fp_global` soundness use-case via `KB.with_closure()`.** The old `augment_kb_with_closure` mutated the KB in place so backward SLD body-matching could hit derived atoms as facts (`fc/fp_global.py:119-128`). The redesign keeps KB immutable but lets a backward grounder consume a `Closure` by composing a *new* KB: `bg = create_grounder(kb.with_closure(fg.closure(d)), ...)`. The capability is relocated, not lost.

Shape symbols (the standard set): `D` depth, `W` width, `B` batch, `N=B*S`, `G` goals, `M` body atoms, `A=D*M`, `S` states, `C` collected budget, `K` children, `K_f` facts/pred, `K_r` rules/pred, `G_r` groundings/rule, `K_v` cands/free-var, `V` free vars, `K_max` cap, `pad` padding, `E` entities, `T` flat survivors.

---

## 1. Design principles

- **Two grounder classes, full stop.** `BackwardGrounder` (resolution ∈ {sld, rtf, enum} is a *config axis*) and `ForwardGrounder` (forward chaining *is* a grounding method). Subclassing `BackwardGrounder` (override `build_resolver()`/`build_filter()`/declare hooks) is the only extension point. There is never an `SldGrounder`/`EnumGrounder`/`ClosureGrounder` subclass.
- **Forward chaining is a grounder, not a filter.** The only filter is `fp_batch` (Kleene `T_P`). There is no `fp_global` filter anywhere; that capability is `ForwardGrounder` + `KB.with_closure()`.
- **`RunPlan` + `RunState` are the sole engine I/O.** The engine reads an immutable `RunPlan` snapshot and threads a per-call `RunState`; it never reads a `grounder.*` attribute. Backward grounding is reentrant and thread-safe by construction.
- **One execution-strategy owner.** `Compiler.wrap` is the only `torch.compile` site; `cudagraph.detach_from_pool` is the only clone-out-of-pool seam (and it is a pytree deep-clone covering both the step-output tuple and the `GrounderOutput` tree). `validate()` rejects illegal combos with exceptions, never silent fallback. `auto_select` keys on static budget `(K_f, K_r, G_r, M, B)`.
- **Compile a single STEP, never the loop.** Static shapes inside compiled regions; no `.item()`, no data-dependent Python branching. The eager/compiled `d` duality is funneled through one `DepthSelector` helper.
- **Enum is L3 join-consistent and query-directed.** Candidate sets are intersected on the free variable (Yannakakis order for acyclic multi-var rules; magic-set demand for depth>1), capped *during* the join so the in-flight tuple count is bounded at every step. Never cartesian-then-filter. `flat` and `dense` are two materialization layouts behind one seam, and the join has **two materialization-coupled variants** (ragged-eager → flat; padded-static → compiled-dense).
- **No silent fallbacks.** Bugs propagate as exceptions (the lib-wide Technical Rule).
- **Small files, one responsibility each.** The four real monoliths are eliminated: `resolution/enum.py` (1661), `groundings.py` (859), `data/fact_index.py` (665), `data/rule_index.py` (635). The two real FC engines (leapfrog-triejoin in `fc/fc.py` 1211; spmm with three pluggable strategies in `fc/spmm/`) get a faithful two-engine layout, not a single mis-named `seminaive.py`.
- **One owner per cross-cutting responsibility.** Rule-grounding dedup has exactly one owner (`engine/dedup.py`); binding analysis has exactly one owner (`RulePattern` → `EnumRuleIndex.binding_tables()` as pure tensorization); chunk mechanics have exactly one owner (`ExecStrategy`).

## 2. Contracts we must respect (the canonical top-level list)

1. **Exactly two grounder classes**, named `BackwardGrounder` and `ForwardGrounder`. Resolution is config; subclassing is the only extension point.
2. **Forward chaining is a grounder, not a filter.** Only filter = `fp_batch`. No `fp_global` anywhere. The soundness use-case is preserved by `KB.with_closure()`.
3. **`RunPlan` in, `GrounderOutput` out; `RunState` threaded.** Engine never reads a module attr. Reentrant + thread-safe.
4. **One execution-strategy owner.** Single `torch.compile` site; single clone seam (a pytree deep-clone); `validate()` raises on illegal combos; `auto_select` on static budget; grounders only DECLARE a sparse capability set.
5. **Compile a single STEP.** Static shapes in compiled regions; no `.item()`/data-dependent branching. `d`/`is_last` are int/bool eager and 0-dim tensors compiled, funneled through one `DepthSelector`.
6. **Enum is L3 join-consistent and query-directed.** Cap applied DURING the join; never cartesian-then-filter; flat/dense are two layouts behind one seam; the join has two materialization variants (ragged-eager, padded-static).
7. **Considered accumulator is PRIMARY for rule groundings**, evidence is the chunk-merge fallback only. It spans chunks via `RunState`. Capture = graph-safe static scatter in-step; dedup/validate/`variant_to_orig` collapse = eager post-loop. Compiled and eager produce the SAME RuleGroundings. Built once at merge.
8. **The per-query cap (`G_r`) default is unchanged** — an opt-in safety valve; overflow flagged (`JoinResult.overflowed`), top-k clamped by default (baseline-identical), `CapMissError` raised only in opt-in strict mode.
9. **KB is immutable.** Indexes never mutated by grounding. FC returns a new `Closure`; backward grounding consumes one via `KB.with_closure()` → a new KB.
10. **Forced defaults preserved:** `all_anchors=True` (enum invariant, not a knob), `u=0`→`fp_batch`, width filter uniform on 1-body rules (torch-strict; the correct `u=0` behavior).
11. **No silent fallbacks** — bugs propagate as exceptions.
12. **Small files, one responsibility.** No monoliths; one owner per cross-cutting concern (dedup, binding analysis, chunk mechanics, depth duality).
13. **Encoding has one source of truth** (`data/encoding.py::Encoding`), which probfol's `grounder_adapter` Encoding contract and `RulePattern` must agree with.

---

## 3. File tree (small files, one responsibility each)

```
grounder/
├── __init__.py                      # re-exports only, zero logic
├── shapes.py                        # Shapes: frozen static-symbol struct, computed once
├── config.py                        # GrounderConfig + EnumConfig (designed surface)
├── plan.py                          # RunPlan: immutable shell→engine snapshot
├── run_state.py                     # RunState (spans chunks) + StepState + CollectedBudget
├── types.py                         # frozen output dataclasses + compile-safe NamedTuple seams
├── glossary.py                      # one-line vocab anchors (forward=chaining; driver=run_backward)
├── errors.py                        # typed exceptions (no silent fallback anywhere)
├── factory.py                       # create_grounder() the ONE impl; parse_type_string/make_bcwd shims
│
├── grounder/                        # ── SHELL (exactly two classes) ──
│   ├── __init__.py
│   ├── base.py                      # Grounder ABC: declare_strategies(), forward(), build_plan()
│   ├── backward.py                  # BackwardGrounder (resolution ∈ {sld,rtf,enum})
│   └── forward.py                   # ForwardGrounder (forward chaining as a grounding method)
│
├── execution/                       # ── EXECUTION-STRATEGY (renamed from exec/: no builtin shadow) ──
│   ├── __init__.py
│   ├── strategy.py                  # ExecStrategy: resolved policy; OWNS all chunk mechanics
│   ├── capability.py                # Cell, CapabilityRow, validate() on illegal combos
│   ├── auto_select.py               # auto_select(): static budget → Cell
│   ├── compiler.py                  # Compiler.wrap(): THE ONLY torch.compile call site
│   ├── cudagraph.py                 # detach_from_pool(): THE ONLY clone seam (pytree deep-clone)
│   ├── chunk_policy.py              # ChunkPolicy: pure value object (batch_size + static mem math)
│   └── depth.py                     # DepthSelector: the one int/0-dim-tensor d duality helper
│
├── data/                            # ── ADAPTERS: KB + indexing ──
│   ├── __init__.py
│   ├── kb.py                        # KB (read-only) + KB.with_closure()
│   ├── dataset.py                   # KGDataset → build_kb()
│   ├── loader.py                    # parse triples/rules → tensors
│   ├── encoding.py                  # Encoding: the SINGLE entity/var id source of truth
│   ├── fact_index/
│   │   ├── __init__.py              #   FactIndex ABC (membership) + create() dispatcher
│   │   ├── arg_key.py               #   ArgKeyFactIndex (composite-keyed CSR; the join primitive)
│   │   ├── inverted.py              #   InvertedFactIndex (owns enumerate)
│   │   └── block_sparse.py          #   BlockSparseFactIndex
│   └── rule_index/
│       ├── __init__.py              #   RuleIndex ABC + create()
│       ├── pattern.py               #   RulePattern (SOLE binding-analysis owner)
│       ├── sld.py                   #   SldRuleIndex
│       └── enum.py                  #   EnumRuleIndex (free-var tables, variants, variant_to_orig, binding_tables)
│
├── resolution/                      # ── RESOLUTION (the WHAT axis) ──
│   ├── __init__.py
│   ├── api.py                       # Resolver Protocol + ResolveRequest contract + dedup_index()
│   ├── primitives.py                # unify_one_to_one, apply_substitutions
│   ├── standardize.py               # variable renaming for ungrounded states
│   ├── mgu.py                       # resolve_facts / resolve_rules (shared by sld/rtf)
│   ├── sld.py                       # SldResolver
│   ├── rtf.py                       # RtfResolver
│   ├── magic_set.py                 # SHARED magic-set adornment (one owner; imported by enum + forward)
│   └── enum/                        # the join-based BC_{w,d,u} core
│       ├── __init__.py
│       ├── resolver.py              # EnumResolver: plan → join → width → layout
│       ├── plan.py                  # JoinPlan: Yannakakis order + demand seeds (built once/rule)
│       ├── join.py                  # L3 join (cap DURING join); ragged + padded-static variants
│       ├── candidates.py            # per-free-var candidate sets (the L2 substrate)
│       ├── join_demand.py           # magic-set demand for the join (imports resolution/magic_set)
│       ├── width.py                 # (w,u) unknown-body-atom filter (NOT fp_batch)
│       └── layout.py                # materialize_dense / materialize_flat over one JoinResult
│
├── engine/                          # ── ENGINE (shared backward runtime) ──
│   ├── __init__.py
│   ├── loop.py                      # run_backward(): chunk→depth driver
│   ├── step.py                      # step(): SELECT(inlined)→RESOLVE→PACK→POST for one depth
│   ├── pack.py                      # PACK phase (dense + flat; dispatch on Layout value)
│   ├── postprocess.py               # POST phase (prune-facts → sync → collect)
│   ├── sync.py                      # sync_accumulated (depth-d write; uses DepthSelector)
│   ├── considered.py               # FiringAccumulator: graph-safe in-step capture (static scatter)
│   ├── evidence.py                  # CompletedTreeFirings assembly (in-tree firings)
│   ├── dedup.py                     # dedup_firings(): the SOLE rule-grounding dedup owner
│   ├── convert.py                   # 40-line dispatcher: considered→RG (primary), evidence→RG (fallback)
│   ├── finalize.py                  # build the three output views; apply fp_batch
│   └── prune/
│       ├── __init__.py
│       ├── facts.py                 # search-time ground-fact pruning
│       └── dead.py                  # search-time dead-branch pruning (sld/rtf)
│
├── filters/                         # ── THE ONLY FILTER ──
│   ├── __init__.py
│   └── fp_batch.py                  # Kleene T_P soundness prune over the firing set
│
├── forward/                         # ── FORWARD CHAINING (a grounder; two real engines) ──
│   ├── __init__.py
│   ├── api.py                       # Closure, Witnesses, ForwardStrategy enum
│   ├── relation_matrix.py           # build sparse per-relation adjacency
│   ├── query_demand.py              # query-directed (magic-set) FC — DEFAULT for answering
│   ├── witnesses.py                 # per-atom (rule, body-grounding) witnesses → output
│   ├── triejoin/                    # engine A: leapfrog-triejoin / anchored stages (from fc/fc.py)
│   │   ├── __init__.py
│   │   ├── lftj.py                  # leapfrog triejoin
│   │   ├── anchored.py             # anchored semi-naive stages
│   │   ├── join_order.py           # _compute_join_order
│   │   └── frontiers.py            # _compute_frontiers
│   └── spmm/                        # engine B: sparse-matmul fixpoint with strategy axis
│       ├── __init__.py
│       ├── fixpoint.py             # the fixed-point runner (was mis-named seminaive.py)
│       ├── strategies.py           # IterationStrategy ABC + Seminaive | TrueIStar | Hybrid
│       ├── kernels.py              # spgemm / spmv
│       ├── matrices.py
│       ├── ops.py
│       └── state.py                # FcState
│
├── memo/                            # ── OPTIONAL, v1 PLACEHOLDER (see contract) ──
│   ├── __init__.py
│   └── store.py                     # MemoStore Protocol + NullMemo default
│
└── nesy/                            # ── ADAPTERS: neural/KGE hooks (Protocol-typed) ──
    ├── __init__.py
    ├── api.py                       # four hook Protocols (fact/rule/step/grounding)
    ├── kge.py
    ├── scoring.py
    └── sampler.py
```

**Eliminated** (not migrated): `bc/{init_resolution,cache_common,cache,common,state,terminal,grounding_convert,compile,compiled_step,bc}.py`, the `groundings.py` god-module, `factory.make_bcwd`/`parse_grounder_type` *triplicated derivation* (the shim functions stay; the `u→filter` logic moves to one owner), `fc/fp_global.py` + `filters/{soundness,search}/fp_global.py` (all three) + `filters/search/fp_global.py` — the *filter* is gone; the **closure-set builder algorithm is relocated into `forward/`** (see §memo/forward contracts), `resolution/closure.py`-as-resolution (its `_resolve_closure_tensors` demand-building is subsumed by `forward/query_demand.py`), `_PerRuleDictView`, the `GroundingResult` alias, `_cache_prop`/`getattr` toggles, `pad_outputs`/`_next_pow2`, every mutable `grounder.*` scratch attr (~50), `grounder._winning_subs_noop`, the `D==0` legacy flat layout, the `resolution="closure"` 4th axis.

**Tabling/subgoal are explicitly DESCOPED from v1** (see §memo contract). The real `bc/tabling.py::writeback_and_replay` (lines 153-209) and `bc/subgoal.py` *rewrite the considered accumulator's `[marker:]` suffix in place and splice replayed HIT rows back in* — this is incompatible with a frozen append-only accumulator. v1 ships the `MemoStore` Protocol + `NullMemo` only; porting tabling requires the memo to own firing *production* (see fix in §memo).

---

## 4. Per-file API (every class/function: args in/out + shapes)

### Top level

**`grounder/shapes.py`** — one frozen STATIC-symbol registry; computed once in factory, carried on `RunPlan`, never recomputed by a phase. `S_out` (per-step output state count) is NOT here: it is data-dependent and lives on `PackedStates`; on the compiled/dense path `pack` MUST pad it to `Shapes.S`.

```python
@dataclass(frozen=True)
class Shapes:
    """This module owns the canonical STATIC shape symbols for one run. Data-dependent
    counts (S_out, T) are NOT owned here — they live on the per-step NamedTuples."""
    B: int; S: int; G: int; M: int; D: int; A: int; C: int
    K_f: int; K_r: int; G_r: int; K_v: int; V: int; E: int; N: int; pad: int
    def n(self) -> int: ...                          # -> N = B*S
    def a(self) -> int: ...                          # -> A = D*M
    def with_batch(self, B: int) -> "Shapes": ...    # -> immutable rebatched copy
```

**`grounder/config.py`** — designed surface; no forced/computed/test-only knobs leak. The `u→filter` derivation lives HERE and nowhere else.

```python
class Resolution(StrEnum): ...   # sld | rtf | enum    (NO "closure")
class FilterMode(StrEnum): ...   # none | fp_batch     (NO fp_global/provset/prune aliases)

@dataclass(frozen=True)
class EnumConfig:
    """enum-only knobs. all_anchors is an INVARIANT (forced in EnumResolver), not exposed."""
    w: int                                          # max_unknown_fact_count (intermediate)
    d: int                                          # num_steps (== depth)
    u: int = 0                                       # max_unknown_fact_count_last_step (paper: 0)
    flat_intermediate: bool = True
    max_groundings_per_query: Optional[int] = None  # G_r cap; DEFAULT UNCHANGED so baselines hold
    strict_cap: bool = False                         # opt-in: raise CapMissError on overflow

@dataclass(frozen=True)
class GrounderConfig:
    """This module owns the user-facing knob surface AND the SOLE u→filter derivation.
    Validated at construction; frozen after."""
    resolution: Resolution
    depth: int                                      # D
    max_total_groundings: int                       # C
    filter: Optional[FilterMode] = None             # None → resolved_filter() derives from u
    enum: Optional[EnumConfig] = None               # required iff resolution == enum
    collect_rule_groundings: bool = True
    standardize: bool = False
    def resolved_filter(self) -> FilterMode: ...     # -> the ONLY u=0→fp_batch derivation site
    def validate(self) -> None: ...                  # raises ConfigError; no silent coercion
```

**`grounder/plan.py`** — the shell→engine boundary contract. `enum_index` is GONE from here; dedup metadata is owned by the resolver.

```python
@dataclass(frozen=True)
class RunPlan:
    """This module owns the immutable per-run snapshot handed shell→engine. The engine
    reads ONLY this (plus the threaded RunState), NEVER the grounder object. Reentrant +
    thread-safe by construction. NOTE: there is NO enum-specific field here — dedup metadata
    is reached via plan.resolver.dedup_index() so RunPlan stays resolution-agnostic."""
    shapes: Shapes
    config: GrounderConfig
    resolver: "Resolver"                 # bound resolution-layer object (owns dedup_index())
    strategy: "ExecStrategy"             # bound execution-layer object
    fp_batch: Optional["FpBatch"]        # the only filter, or None
    kb: "KB"
    closure: Optional["Closure"] = None  # read-only; backward grounding consults it iff present
    fact_hooks: Tuple["ResolutionFactHook", ...] = ()
    rule_hooks: Tuple["ResolutionRuleHook", ...] = ()
    step_hooks: Tuple["StepHook", ...] = ()
    grounding_hooks: Tuple["GroundingHook", ...] = ()
    memo: "MemoStore" = ...              # NullMemo() default; explicit, never getattr'd
    def for_chunk(self, B: int) -> "RunPlan": ...  # -> new immutable plan with rebatched Shapes
```

**`grounder/run_state.py`** — the threaded per-call accumulators. `RunState` SPANS chunks (it carries the considered `FiringAccumulator` + the chunk-query offset); `StepState` is per-chunk-per-step. This replaces the ~50 mutable `grounder.*` attrs.

```python
@dataclass(frozen=True)
class CollectedBudget:
    """The C-budget terminal accumulator, threaded by value within a chunk.
    append() takes ONLY finished per-depth structured tensors — never SyncParams
    (collection must not know what subs_noop is)."""
    body: Tensor          # [B, C, D, M, 3]
    rule_idx: Tensor      # [B, C, D]
    head: Tensor          # [B, C, D, 3]
    body_count: Tensor    # [B, C, D]
    count: Tensor         # [B]
    def append(self, body_d: Tensor, ridx_d: Tensor, head_d: Tensor,
               bcount_d: Tensor) -> "CollectedBudget": ...   # immutable; tensors already synced

@dataclass(frozen=True)
class FiringAccumulator:
    """The CONSIDERED firing accumulator (firings OUTSIDE completed proof trees). It SPANS
    CHUNKS: reset once per forward(), one fixed-capacity static buffer per (chunk, step) is
    scattered into in-step (graph-safe), then concatenated here across chunks. Threaded on
    RunState, NOT inside StepState (the cross-chunk identity is the whole point).

    WHY THIS EXISTS (verified, do not remove): rule_groundings from completed-tree evidence
    ALONE undercounts ~3× (ablation_d3 BC13: 80 vs keras 252). This records every fired rule
    application; fp_batch then drops apps whose body atoms aren't transitively groundable."""
    rule: Tuple[Tensor, ...]   # per (chunk,step) [T] orig rule idx (UNcollapsed variant id)
    head: Tuple[Tensor, ...]   # per (chunk,step) [T, 3]   (the SELECTED-goal head)
    body: Tuple[Tensor, ...]   # per (chunk,step) [T, M, 3] (NOT pre-substituted; see capture)
    gidx: Tuple[Tensor, ...]   # per (chunk,step) [T] GLOBAL query index (chunk offset applied)
    @staticmethod
    def empty() -> "FiringAccumulator": ...
    def extend(self, buf: "FiringBuffer", chunk_query_offset: int) -> "FiringAccumulator": ...
        # immutable: trim buf to its valid prefix, lift gidx by offset, append -> new acc

@dataclass(frozen=True)
class FiringBuffer:
    """Fixed-capacity STATIC scatter target for ONE compiled step (no list.append, no .item()).
    capacity = S * K_r (the static upper bound on firings a step can propose)."""
    rule: Tensor          # [cap]
    head: Tensor          # [cap, 3]
    body: Tensor          # [cap, M, 3]
    bidx: Tensor          # [cap] batch-local
    n_valid: Tensor       # [] 0-dim count written (read EAGERLY post-loop, never in-graph)

@dataclass(frozen=True)
class RunState:
    """This module owns the PER-CALL state that spans chunks. Threaded through
    ExecStrategy.iter_chunks/merge. The engine never stores any of this on the module."""
    considered: FiringAccumulator
    chunk_query_offset: int          # lifts batch-local bidx → global query idx across chunks
    def absorb_chunk(self, buf_per_step: Sequence[FiringBuffer], n_chunk: int) -> "RunState": ...
        # extend considered with this chunk's per-step buffers; advance offset -> new RunState

@dataclass(frozen=True)
class StepState:
    """Immutable per-step carrier threaded through phases by value within one chunk; the engine
    is reentrant. `depth` is the eager Python int; the compiled path carries a DepthSelector."""
    proof_goals: Tensor          # [B, S, G, 3]
    grounding_body: Tensor       # [B, S, M, 3]
    accumulated_body: Tensor     # [B, S, D, M, 3]
    state_valid: Tensor          # [B, S]
    top_ridx: Tensor             # [B, S]
    body_count: Tensor           # [B, S]
    selected_goal: Tensor        # [B, S, 3]  (head source for the firing capture; all resolutions)
    collected: CollectedBudget
    firing_buf: FiringBuffer     # this step's static scatter target
    next_var: Optional[Tensor] = None   # [B]
    def replace(self, **kw) -> "StepState": ...
```

**`grounder/types.py`** — frozen output structs + compile-safe NamedTuple seams. The `D==0` legacy flat layout and `_PerRuleDictView` are gone. `Layout` is carried as a FIELD on the resolved children (a real value tag, not type-sniffing). `subs_noop` lives only where it is actually determined (post-pack `SyncParams`, and constant-True on `FlatResolvedChildren`) — NOT on the dense resolution output for sld/rtf where subs are real.

```python
class Layout(StrEnum): ...  # dense | flat | sparse

# ── Output views (consumer-facing, frozen) ──
@dataclass(frozen=True)
class ProofState:
    proof_goals: Tensor          # [B, S, G, 3]
    state_valid: Tensor          # [B, S]
    top_ridx: Tensor             # [B, S]
    next_var: Optional[Tensor] = None   # [B]

@dataclass(frozen=True)
class CompletedTreeFirings:
    """In-completed-proof-tree firings ONLY (renamed from ProofEvidence for clarity;
    the considered/evidence primary-vs-fallback pair undercounts here by ~3×). Carries
    Shapes by construction so every consumer (incl. hooks) preserves D/M."""
    body: Tensor                 # [B, C, D, M, 3]   (structured only; no D==0 legacy)
    mask: Tensor                 # [B, C]
    count: Tensor                # [B]
    rule_idx: Tensor             # [B, C, D]
    body_count: Tensor           # [B, C, D]
    head: Tensor                 # [B, C, D, 3]
    shapes: Shapes
    @property
    def body_flat(self) -> Tensor: ...        # -> [B, C, D*M, 3]   (PROPERTY: in-tree caller reads it bare)
    @property
    def body_atom_mask_flat(self) -> Tensor: ...  # -> [B, C, D*M]
    @property
    def rule_idx_top(self) -> Tensor: ...     # -> [B, C]  (kept for the GroundingHook path + SBR)
    @property
    def body_count_total(self) -> Tensor: ... # -> [B, C]

@dataclass(frozen=True)
class RuleGroundings:
    """Flat CSR-style rule firings, keras-compatible. NO _PerRuleDictView. fp_batch DROPS rows
    and rebuilds rule_offsets (firing_valid is NOT the pruning channel — see fp_batch contract)."""
    atom_table: Tensor           # [num_atoms, 3]
    body_pool_idx: Tensor        # [N_firings, M] long
    body_atom_valid: Tensor      # [N_firings, M] bool
    head_pool_idx: Tensor        # [N_firings] long
    rule_idx: Tensor             # [N_firings] long, sorted ascending
    rule_offsets: Tensor         # [num_rules + 1] long, cumsum
    firing_valid: Tensor         # [N_firings] bool (post-prune: all True; pruning drops rows)
    num_atoms: int; num_rules: int; M_max: int
    query_pool_idx: Optional[Tensor] = None  # [B]
    @staticmethod
    def empty(num_rules: int, M: int, device, dtype) -> "RuleGroundings": ...
    def rule_slice(self, r: int) -> Tuple[int, int]: ...   # -> (start, end) via rule_offsets

@dataclass(frozen=True)
class GrounderOutput:
    state: ProofState
    evidence: Optional[CompletedTreeFirings] = None
    rule_groundings: Optional[RuleGroundings] = None

# ── Internal pipeline NamedTuples (torch.compile-safe) ──
class ResolvedChildren(NamedTuple):     # dense resolution→pack contract
    layout: Layout          # VALUE tag carried on the contract; pack reads children.layout
    fact_goals: Tensor      # [B, S, K_f, G, 3]
    fact_gbody: Tensor      # [B, S, K_f, M, 3]
    fact_success: Tensor    # [B, S, K_f]
    rule_goals: Tensor      # [B, S, K_r, G, 3]
    rule_gbody: Tensor      # [B, S, K_r, M, 3]
    rule_success: Tensor    # [B, S, K_r]
    sub_rule_idx: Tensor    # [B, S, K_r]
    fact_subs: Tensor       # [B, S, K_f, 2, 2]
    rule_subs: Tensor       # [B, S, K_r, 2, 2]
    # NO subs_noop here: for sld/rtf it is a POST-pack property, not known at resolve time.

class FlatResolvedChildren(NamedTuple): # flat resolution→pack contract
    layout: Layout          # == Layout.flat
    flat_goals: Tensor      # [T, G, 3]
    flat_gbody: Tensor      # [T, A, 3]
    flat_rule_idx: Tensor   # [T]
    flat_b_idx: Tensor      # [T]
    flat_s_idx: Tensor      # [T]
    flat_subs: Tensor       # [T, 2, 2]
    B: int
    S: int
    subs_noop: bool         # constant True for enum-flat (identity subs); a documented optimization

class PackedStates(NamedTuple):
    grounding_body: Tensor  # [B, S_out, M, 3]   (S_out data-dependent; padded to Shapes.S on dense)
    proof_goals: Tensor     # [B, S_out, G, 3]
    top_ridx: Tensor        # [B, S_out]
    state_valid: Tensor     # [B, S_out]
    body_count: Tensor      # [B, S_out]
    has_new_body: Tensor    # [B, S_out]
    current_ridx: Tensor    # [B, S_out]
    selected_goal: Tensor   # [B, S_out, 3]  (threaded so the head cross-check survives)

class SyncParams(NamedTuple):
    parent_map: Tensor      # [B, S_out]
    winning_subs: Tensor    # [B, S_out, 2, 2]
    has_new_body: Tensor    # [B, S_out]
    parent_bcount: Tensor   # [B, S_out, D]
    current_ridx: Tensor    # [B, S_out]
    current_head: Tensor    # [B, S_out, 3]
    subs_noop: bool         # determined HERE (post-pack); sync skips substitution passes when True
```

**`grounder/glossary.py`** — vocabulary anchor (referenced by the grounder CLAUDE.md). Reserves "forward" for chaining; the proof-search driver is `run_backward` with zero "forward" lineage.

```python
GLOSSARY: Mapping[str, str] = ...   # {"forward": "forward chaining only",
                                    #  "run_backward": "the proof-search driver",
                                    #  "considered": "firings outside completed trees (PRIMARY)",
                                    #  "evidence": "in-completed-tree firings (fallback)", ...}
```

**`grounder/errors.py`**

```python
class GrounderError(Exception): ...
class ConfigError(GrounderError): ...            # bad/contradictory config
class StrategyError(GrounderError): ...          # illegal grounder×strategy combo
class CapMissError(GrounderError): ...           # G_r overflow when strict_cap=True (opt-in only)
class ShapeContractError(GrounderError): ...     # a phase produced an off-contract tensor (debug)
```

**`grounder/factory.py`** — `create_grounder` is the ONE implementation; the other two are zero-logic shims that build a `GrounderConfig` and delegate. None derives the filter (that is `GrounderConfig.resolved_filter()`).

```python
def create_grounder(
    kb: "KB", config: GrounderConfig, *,
    strategy: Optional["ExecStrategy"] = None,      # None → auto_select
    fact_hooks: Sequence["ResolutionFactHook"] = (),
    rule_hooks: Sequence["ResolutionRuleHook"] = (),
    step_hooks: Sequence["StepHook"] = (),
    grounding_hooks: Sequence["GroundingHook"] = (),
    memo: Optional["MemoStore"] = None,             # None → NullMemo
    closure: Optional["Closure"] = None,            # forwarded to RunPlan.closure
) -> "Grounder": ...                                # -> BackwardGrounder | ForwardGrounder

def parse_type_string(s: str, kb: "KB") -> GrounderConfig: ...
    # "bc{w}{d}" / "bc{w}{d}u{u}" / "sld" / "rtf" / "forward" -> GrounderConfig.
    # ZERO construction logic beyond building the config; no filter/index work here.

def make_bcwd(kb: "KB", w: int, d: int, u: int = 0, **kw) -> "BackwardGrounder": ...
    # thin paper-notation shim: builds GrounderConfig(+EnumConfig) and calls create_grounder.
    # Does NOT derive the filter (resolved_filter does) — enforced by test_factory_no_config_logic.
```

**`grounder/__init__.py`** — re-exports only: `BackwardGrounder, ForwardGrounder, GrounderConfig, EnumConfig, Resolution, FilterMode, ExecStrategy, create_grounder, parse_type_string, make_bcwd, KB, KGDataset, GrounderOutput, ProofState, CompletedTreeFirings, RuleGroundings, Shapes, Closure` + nesy hook protocols.

### Shell layer — `grounder/grounder/`

**`base.py`**

```python
class Grounder(nn.Module, ABC):
    """This module owns the shell contract: build a RunPlan, hand it to the engine, return
    GrounderOutput. SUBCLASSING is the extension point for custom resolution/filters/hooks.
    The shell holds NO per-call mutable state (all per-call state is RunState/StepState)."""
    def __init__(self, kb: "KB", config: GrounderConfig, **plan_parts): ...
    @abstractmethod
    def declare_strategies(self) -> "CapabilityRow": ...    # -> sparse set of supported Cells
    @abstractmethod
    def forward(self, queries: Tensor, mask: Optional[Tensor] = None) -> GrounderOutput: ...
        # queries [B,3], mask [B] -> GrounderOutput
    def build_plan(self, B: int) -> RunPlan: ...            # pure; resolves strategy + rebatches Shapes
```

**`backward.py`** — ONE class; resolution is a config axis.

```python
class BackwardGrounder(Grounder):
    """This module owns batched proof-search query answering (SELECT→RESOLVE→PACK→POST).
    Resolution ∈ {sld,rtf,enum} is a CONFIG AXIS, never a subclass. Subclass this only to
    inject a custom Resolver/filter/hook via build_resolver()/build_filter(). If plan.closure
    is present, body-matching may treat its derived atoms as facts (the old fp_global
    soundness use-case, now via KB.with_closure()).

    Dependency chain:
      BackwardGrounder.forward
        └── engine.loop.run_backward(queries, plan)
              ├── plan.strategy.iter_chunks → RunState (spans chunks)
              ├── engine.step.step (per depth)
              │     ├── plan.resolver.resolve(req)        # resolution layer
              │     └── plan.strategy.wrap_step(step_fn)  # execution layer
              └── plan.strategy.merge(parts, run_state, finalize_fn) → finalize ONCE
    """
    def declare_strategies(self) -> "CapabilityRow": ...
        # sld → {(dense,eager), (dense,outer_reduce_overhead)}
        # rtf → {(dense,eager), (flat,eager)}
        # enum → {(flat,eager), (dense,eager[debug]), (dense,compiled_step)}
    def forward(self, queries: Tensor, mask: Optional[Tensor] = None) -> GrounderOutput: ...
        # queries [B,3], mask [B] -> GrounderOutput
    def build_resolver(self) -> "Resolver": ...             # override to swap resolution
    def build_filter(self) -> Optional["FpBatch"]: ...      # override to swap soundness filter
```

**`forward.py`** — ONE class; FC is a grounding method.

```python
class ForwardGrounder(Grounder):
    """This module owns provable-atom grounding via sparse forward chaining (the old
    closure/fp_global capability — now a first-class GROUNDER). Default answering mode is
    QUERY-DIRECTED (demand + witness gather); full closure is opt-in and unbounded on dense
    KGs (fb15k: 272k facts → 43.8M edges @ d1).

    Dependency chain:
      ForwardGrounder.forward
        ├── forward.query_demand.run_demand(kb, queries, depth, strategy)   # DEFAULT
        │     OR forward.spmm.fixpoint.run_closure(kb, depth, strategy)       # opt-in, UNBOUNDED
        │           └── forward.spmm.strategies.{Seminaive|TrueIStar|Hybrid}
        │           OR forward.triejoin.lftj (leapfrog) / anchored
        └── forward.witnesses.gather → GrounderOutput
    """
    def declare_strategies(self) -> "CapabilityRow": ...     # forward → {(sparse, eager)} only
    def forward(self, queries: Tensor, mask: Optional[Tensor] = None) -> GrounderOutput: ...
        # queries [B,3], mask [B] -> GrounderOutput (evidence + rule_groundings from witnesses)
    def closure(self, depth: int) -> "Closure": ...          # explicit opt-in full closure
```

### Execution-strategy layer — `grounder/execution/`

**`capability.py`**

```python
class CompileMode(StrEnum): ...  # eager | compiled_step | outer_reduce_overhead
# `dynamic` is NOT a free field: it is implied by CompileMode (outer_reduce_overhead ⇒ dynamic=False,
# compiled_step ⇒ dynamic=False). There is no dynamic=True cell anywhere (it is dominated).

@dataclass(frozen=True)
class Cell:
    layout: Layout              # dense | flat | sparse
    compile: CompileMode

@dataclass(frozen=True)
class CapabilityRow:
    """What ONE grounder declares it supports (a SPARSE set, not a full matrix)."""
    cells: FrozenSet[Cell]
    def supports(self, cell: Cell) -> bool: ...

def validate(row: CapabilityRow, cell: Cell) -> None: ...
    # raises StrategyError on any illegal combo:
    #   flat + compiled_step            (data-dependent nonzero/.item() ⇒ no fullgraph)
    #   sld  + compiled_step            (per-step compile not expressible for sld)
    #   sparse + (dense|flat)  OR  forward grounder asking for dense/flat
    #   cell ∉ row.cells                (grounder didn't declare it)
```

**`auto_select.py`** — keyed on STATIC budget only (no phantom per-query estimate).

```python
def auto_select(row: CapabilityRow, *, K_f: int, K_r: int, G_r: int, M: int, B: int) -> Cell: ...
    # Policy (keys on the STATIC budget S*K_r*G_r*M and fan-out K_f; NO data-dependent probe):
    #   forward                              -> (sparse, eager)
    #   sld                                  -> (dense, outer_reduce_overhead) if replayed else (dense, eager)
    #   enum, low fan-out (K_f small)        -> (flat, eager)                      # ~6× faster, ~1.5 GiB
    #   enum, high fan-out / cliff risk      -> (dense, compiled_step) + G_r cap   # the only bounded path
    # NEVER returns a dynamic mode or max-autotune (measured strictly worse).
```

**`strategy.py`** — the resolved execution policy object. It OWNS all chunk mechanics; the engine provides a `finalize_fn` so the strategy never imports grounding semantics (it stitches chunks; the engine builds RuleGroundings).

```python
@dataclass(frozen=True)
class ExecStrategy:
    """This module owns EVERY compile/layout/chunk/cudagraph decision AND all chunk mechanics
    (plan/pad/trim/iterate/merge) for one run. The engine names none of torch.compile / CUDA
    graphs / a layout. merge() stitches chunk parts + threads RunState; it calls the
    engine-provided finalize_fn to build RuleGroundings ONCE — it never imports convert/dedup."""
    cell: Cell
    chunk: "ChunkPolicy"
    @staticmethod
    def auto(row: CapabilityRow, *, K_f: int, K_r: int, G_r: int, M: int, B: int) -> "ExecStrategy": ...
    @staticmethod
    def explicit(row: CapabilityRow, cell: Cell, chunk: "ChunkPolicy") -> "ExecStrategy": ...  # validates
    def layout(self) -> Layout: ...
    def depth_selector(self, d: int, D: int) -> "DepthSelector": ...
        # eager Cell → int/bool selector; compiled Cell → 0-dim-tensor selector (THE duality source)
    def wrap_step(self, step_fn: Callable, shapes: Shapes) -> Callable: ...
        # step_fn -> possibly compiled+cudagraph-detached callable (compiles ONE step, never the loop)
    def iter_chunks(self, queries: Tensor, mask: Tensor, shapes: Shapes
                    ) -> Iterator[Tuple[Tensor, Tensor, Shapes, int]]: ...
        # yields (chunk_queries [b,3], chunk_mask [b], chunk_shapes, chunk_query_offset)
    def merge(self, parts: Sequence[Tuple[GrounderOutput, Sequence[FiringBuffer]]],
              run_state: RunState, shapes: Shapes,
              finalize_fn: Callable[[Sequence[GrounderOutput], RunState], GrounderOutput]
              ) -> GrounderOutput: ...
        # trims padding, concatenates opaque parts, folds per-chunk FiringBuffers into run_state,
        # then DELEGATES rule-grounding assembly to finalize_fn (engine owns "build"; strategy owns "stitch")
```

**`compiler.py`** — THE ONLY `torch.compile` call site.

```python
class Compiler:
    """This module owns the SINGLE torch.compile invocation. No other file may call
    torch.compile. mode/dynamic/fullgraph are fixed by the Cell."""
    @staticmethod
    def wrap(fn: Callable, *, mode: CompileMode, fullgraph: bool, shapes: Shapes) -> Callable: ...
        # fn -> compiled fn (or fn unchanged when mode==eager). THE ONLY torch.compile appearance.
```

**`cudagraph.py`** — the single clone seam, generalized to a pytree so BOTH the step-output tuple and the nested `GrounderOutput` tree go through it.

```python
def mark_step_begin() -> None: ...                       # torch.compiler.cudagraph_mark_step_begin
def detach_from_pool(obj: T) -> T: ...
    # structural deep-clone over an arbitrary pytree (tuple of tensors OR a nested dataclass tree).
    # THE ONLY clone-to-detach-from-CUDA-graph-pool seam in the library.
```

**`chunk_policy.py`** — a pure value object (no chunk mechanics here; those live on `ExecStrategy`).

```python
@dataclass(frozen=True)
class ChunkPolicy:
    batch_size: int          # padded bucket size (stable shape for compile)
    @staticmethod
    def auto(*, K_f: int, peak_budget_bytes: int, shapes: Shapes) -> "ChunkPolicy": ...
        # static memory math only; ExecStrategy consumes this to drive iter_chunks/merge
```

**`depth.py`** — the ONE place the eager-int/compiled-tensor `d` duality is expressed.

```python
@dataclass(frozen=True)
class DepthSelector:
    """This module owns the d / is_last representation duality. Eager: d is int, is_last is bool.
    Compiled: d is a 0-dim long tensor, is_last a 0-dim bool tensor. EVERY phase that consumes
    depth (enum width-gate, width filter, sync write-slot) takes a DepthSelector — the duality
    is NOT scattered as isinstance(d, Tensor) and is NOT confined to sync.py."""
    d: Union[int, Tensor]              # int (eager) | [] long (compiled)
    is_last: Union[bool, Tensor]       # bool (eager) | [] bool (compiled)
    compiled: bool
    def write_slot(self, accumulated: Tensor) -> Callable: ...  # index-write (eager) | one-hot scatter (compiled)
    def width_for(self, w: int, u: int) -> Union[int, Tensor]: ...  # branch (eager) | torch.where (compiled)
```

### Adapters: KB + indexing — `grounder/data/`

**`encoding.py`** — the single id source of truth (anchors the cross-repo encoding contract probfol cites).

```python
@dataclass(frozen=True)
class Encoding:
    """This module owns the entity/variable id convention. The canonical counterpart that
    grounder.data.rule_index.RulePattern and probfol's grounder_adapter Encoding contract
    must agree with: entity ids 0..E-1 (constant_no=E-1); var ids E.. (head_var0=E,
    head_var1=E+1, free body vars E+2,E+3,… sorted-by-name); constant iff id<E."""
    E: int
    pad: int
    def is_var(self, ids: Tensor) -> Tensor: ...     # [...] -> [...] bool (ids >= E)
    def is_const(self, ids: Tensor) -> Tensor: ...   # [...] -> [...] bool
    def var_base(self) -> int: ...                    # -> E
```

**`kb.py`** — read-only; `with_closure` is the seam that preserves the old fp_global soundness use-case without mutation.

```python
@dataclass(frozen=True)
class KB:
    """This module owns the READ-ONLY fact+rule container. Indexes are built once and NEVER
    mutated by grounding. Forward chaining returns a NEW Closure; a backward grounder consumes
    it via with_closure() which returns a NEW KB (the old augment_kb_with_closure mutation is gone)."""
    facts: Tensor                # [num_facts, 3]
    rule_index: "RuleIndex"
    fact_index: "FactIndex"
    encoding: Encoding
    E: int; M: int; pad: int; num_rules: int; K_f: int; K_r: int
    def fan_out(self) -> int: ...                 # -> K_f (drives auto_select)
    def with_closure(self, closure: "Closure") -> "KB": ...
        # -> a NEW immutable KB whose FactIndex additionally contains closure.provable atoms.
        # This is how backward SLD body-matching hits derived atoms as facts. No mutation.
```

**`dataset.py`**

```python
class KGDataset:
    def __init__(self, root: str, *, rules_file: str = "rules.txt"): ...
    def build_kb(self, fact_index: str = "arg_key") -> KB: ...   # -> KB
```

**`loader.py`**

```python
def parse_triples(path: str, ent_vocab, rel_vocab) -> Tensor: ...   # -> [num_facts, 3]
def parse_rules(path: str, enc: Encoding) -> Tuple[Tensor, Tensor, Tensor]: ...
    # -> (heads [R,3], bodies [R,M,3], lens [R])
```

**`fact_index/__init__.py`** — the ABC is the MEMBERSHIP contract only. `enumerate` is NOT a uniform ABC method (ArgKey cannot answer it — it is the join primitive that answers `candidate_set`/`targeted_lookup`). Layout-specific capabilities are advertised; `auto_select`/the resolver pick the index whose capability they need.

```python
class FactIndex(ABC):
    """This module owns the MEMBERSHIP contract every layout answers identically:
    exists(atoms) -> bool, plus k_f(). Candidate ENUMERATION is layout-specific and NOT on
    this ABC (Inverted/BlockSparse implement enumerate(); ArgKey implements candidate_set()/
    targeted_lookup() — the join primitive). Swapping a membership-equivalent layout MUST NOT
    change grounding counts."""
    @abstractmethod
    def exists(self, atoms: Tensor) -> Tensor: ...   # atoms [...,3] -> bool [...]
    @abstractmethod
    def k_f(self) -> int: ...                          # -> K_f
    @abstractmethod
    def capabilities(self) -> FrozenSet[str]: ...      # e.g. {"enumerate"} or {"candidate_set","targeted_lookup"}
    @staticmethod
    def create(layout: str, facts: Tensor, enc: Encoding) -> "FactIndex": ...
```

**`fact_index/arg_key.py`** — composite-keyed CSR; the join primitive. The CSR is keyed on `pred*ks+arg` and its `order`/`offsets` index into FACT ROWS, not a sorted entity-id list. `candidate_set` therefore returns `(offsets, fact_row_idx)`; projecting to the free-var column + sort/unique is an EXPLICIT step the join (or an index-build-time entity-keyed CSR) performs — it is new code, not a property the existing CSR already holds.

```python
class ArgKeyFactIndex(FactIndex):
    """Sorted (pred,arg,dir)-keyed CSR over FACT ROWS. THE join primitive. Does NOT implement
    enumerate() (use InvertedFactIndex for that)."""
    def __init__(self, facts: Tensor, enc: Encoding): ...
    def exists(self, atoms: Tensor) -> Tensor: ...                 # [...,3] -> bool [...]
    def k_f(self) -> int: ...
    def capabilities(self) -> FrozenSet[str]: ...                  # {"candidate_set","targeted_lookup"}
    def targeted_lookup(self, query_atoms: Tensor, max_results: int) -> Tuple[Tensor, Tensor]: ...
        # query_atoms [B,3] -> (fact_idx [B, max_results], valid [B, max_results])  (FACT ROW indices)
    def candidate_set(self, pred: Tensor, bound: Tensor, direction: Tensor
                      ) -> Tuple[Tensor, Tensor]: ...
        # -> (offsets [N+1] CSR row bounds, fact_row_idx [total]) — fact rows for (pred,bound,dir) keys.
        # The join PROJECTS fact_row_idx to the free-var column + sort/unique to get entity ids.
    def entity_csr(self, pred: Tensor, bound: Tensor, direction: Tensor
                   ) -> Tuple[Tensor, Tensor]: ...
        # -> (offsets [N+1], values [total] SORTED entity ids) — the projected, merge-ready set.
        # Built lazily from candidate_set; this is the new entity-keyed view merge_intersect needs.
```

`inverted.py`, `block_sparse.py`: each `class XFactIndex(FactIndex)` with `__init__(facts, enc)`, `exists`, `k_f`, `capabilities` (= `{"enumerate"}`), plus:

```python
def enumerate(self, pred: Tensor, bound_arg: Tensor, direction: Tensor) -> Tensor: ...
    # pred [N], bound_arg [N], direction [N] -> candidates [N, K_f] (pad-filled). Inverted/BlockSparse only.
# block_sparse.py also: def _max_facts_per_query(self) -> int: ...
```

**`rule_index/pattern.py`** — the SOLE binding-analysis owner.

```python
@dataclass(frozen=True)
class RulePattern:
    """This module owns rule binding/variable-slot analysis (the single source of truth).
    EnumRuleIndex.binding_tables() is a PURE TENSORIZATION of these results — no second
    derivation, no per-rule Python loop in the hot path. Encoding matches data.encoding.Encoding."""
    head: Tensor                 # [3]
    body: Tensor                 # [M, 3]
    body_pred_indices: Tensor    # [M]
    arg_source_dep: Tensor       # [M, 2]  per-arg binding source
    free_vars: Tensor            # [V]
    canon_src: Tensor            # [V] canonical source slot per free var (shared-var consistency)
    body_len: int
```

**`rule_index/__init__.py`**

```python
class RuleIndex(ABC):
    """This module owns generic rule lookup + the rules_heads/rules_bodies/rule_lens tensors
    that the firing binding-table validation reads (for ALL resolutions, not just enum)."""
    @abstractmethod
    def rules_for(self, pred: Tensor) -> Tensor: ...   # pred [N] -> rule ids [N, K_r] (pad)
    @abstractmethod
    def pattern(self, rule_idx: int) -> RulePattern: ...
    @property
    @abstractmethod
    def rules_heads(self) -> Tensor: ...               # [R, 3]
    @property
    @abstractmethod
    def rules_bodies(self) -> Tensor: ...              # [R, M, 3]
    @property
    @abstractmethod
    def rule_lens(self) -> Tensor: ...                 # [R]
    def k_r(self) -> int: ...                           # -> K_r
    @staticmethod
    def create(resolution: str, raw, enc: Encoding) -> "RuleIndex": ...
```

`rule_index/sld.py`: `class SldRuleIndex(RuleIndex)`.

**`rule_index/enum.py`**

```python
class EnumRuleIndex(RuleIndex):
    """This module owns enum free-var enumeration tables + anchor variants + per-rule join
    plans + variant_to_orig. all_anchors is an INVARIANT; the K_r anchor variants of one
    logical rule collapse to a single key at dedup. binding_tables() is a PURE tensorization
    of RulePattern (built once at __init__, no Python rule-loop in the hot path)."""
    def __init__(self, raw, enc: Encoding): ...
    def free_var_table(self, rule_idx: int) -> Tensor: ...     # -> [V, K_v] candidate slots
    def anchor_variants(self, rule_idx: int) -> Tensor: ...     # -> [K_r] variant rule ids
    def variant_to_orig(self) -> Tensor: ...                     # -> [num_variants] logical rule id
    def join_plan(self, rule_idx: int) -> "JoinPlan": ...        # -> Yannakakis order for this rule
    def binding_tables(self) -> "BindingTables": ...             # tensorized RulePattern (validity check)
    def dedup_index(self) -> "DedupIndex": ...                   # variant_to_orig + binding_tables bundle
```

### Resolution layer — `grounder/resolution/`

**`api.py`** — the engine↔resolution boundary. `ResolveRequest` carries a `DepthSelector` (NOT a bare int), so the compiled path is expressible. The resolver owns its dedup metadata (`dedup_index()`), so `RunPlan` stays resolution-agnostic and `convert` works uniformly.

```python
@dataclass(frozen=True)
class ResolveRequest:
    """This module owns the engine→resolution contract. depth is a DepthSelector (int/bool eager;
    0-dim tensors compiled) — a resolver on the compiled path receives tensor d/is_last and MUST
    use torch.where, never a Python branch."""
    goal: Tensor          # [B, S, 3]
    remaining: Tensor     # [B, S, G, 3]
    active: Tensor        # [B, S]
    depth: "DepthSelector"
    kb: KB
    shapes: Shapes
    closure: Optional["Closure"]                 # read-only; consulted iff present
    fact_hooks: Tuple["ResolutionFactHook", ...]
    rule_hooks: Tuple["ResolutionRuleHook", ...]

class DedupIndex(Protocol):
    """variant_to_orig + binding tables for the rule-grounding dedup. Identity/NullDedup for
    sld/rtf/forward; the real enum bundle for enum."""
    def variant_to_orig(self) -> Optional[Tensor]: ...   # [num_variants] or None (identity)
    def binding_tables(self) -> "BindingTables": ...      # from the GENERIC rule index for all resolutions

class Resolver(Protocol):
    """All three resolutions implement this; they return ResolvedChildren (dense) or
    FlatResolvedChildren (flat) — the SAME contract regardless of algorithm. Stateless: no
    per-call attribute is stored on self. dedup_index() lets convert read dedup metadata without
    a hoisted enum field on RunPlan."""
    def layout(self) -> Layout: ...
    def resolve(self, req: ResolveRequest) -> Union[ResolvedChildren, FlatResolvedChildren]: ...
    def dedup_index(self) -> DedupIndex: ...
```

**`primitives.py`**

```python
def unify_one_to_one(a: Tensor, b: Tensor, enc: Encoding) -> Tuple[Tensor, Tensor]: ...
    # a [...,3], b [...,3] -> (success [...], subs [...,2,2])
def apply_substitutions(atoms: Tensor, subs: Tensor, enc: Encoding) -> Tensor: ...
    # atoms [...,K,3], subs [...,2,2] -> atoms [...,K,3]
```

**`standardize.py`**

```python
def standardize(goals: Tensor, next_var: Tensor, enc: Encoding) -> Tuple[Tensor, Tensor]: ...
    # goals [B,S,G,3], next_var [B] -> (renamed [B,S,G,3], next_var [B])
```

**`mgu.py`**

```python
def resolve_facts(goal: Tensor, kb: KB, closure: Optional["Closure"]) -> Tuple[Tensor, Tensor, Tensor]: ...
    # goal [B,S,3] -> (fact_goals [B,S,K_f,G,3], fact_subs [B,S,K_f,2,2], fact_success [B,S,K_f])
    # consults closure (if present) so derived atoms match as facts — the with_closure() use-case.
def resolve_rules(goal: Tensor, kb: KB) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...
    # goal [B,S,3] -> (rule_goals [B,S,K_r,G,3], rule_subs [B,S,K_r,2,2], rule_success [B,S,K_r], sub_rule_idx [B,S,K_r])
```

**`sld.py`**

```python
class SldResolver(Resolver):
    def __init__(self, kb: KB, shapes: Shapes): ...
    def layout(self) -> Layout: ...                    # -> dense
    def resolve(self, req: ResolveRequest) -> ResolvedChildren: ...   # subs_noop NOT set here (post-pack)
    def dedup_index(self) -> DedupIndex: ...           # NullDedup (identity variant_to_orig)
```

**`rtf.py`**

```python
class RtfResolver(Resolver):
    def __init__(self, kb: KB, shapes: Shapes, layout: Layout): ...
    def layout(self) -> Layout: ...                    # -> dense | flat
    def resolve(self, req: ResolveRequest) -> Union[ResolvedChildren, FlatResolvedChildren]: ...
    def dedup_index(self) -> DedupIndex: ...           # NullDedup
```

**`magic_set.py`** — the SHARED magic-set adornment owner (one owner; imported by both `enum/join_demand.py` and `forward/query_demand.py`).

```python
@dataclass(frozen=True)
class AdornedRules:
    adornment: Tensor       # [R, M, 2]  bound/free per arg per body atom
    demanded: Tensor        # [num_pred] bool
def adorn(rules: RuleIndex, goal_pred: Tensor) -> AdornedRules: ...   # the one magic-set implementation
```

#### Enum core — `grounder/resolution/enum/`

**`resolver.py`**

```python
class EnumResolver(Resolver):
    """This module owns the BC_{w,d,u} resolver: assemble plan → join → width → layout. NO
    cartesian-product-then-filter. all_anchors is forced True here (invariant, not a knob).
    Emits the layout the strategy asked for; it does NOT choose the layout itself. On the
    compiled-dense path it calls join.join_padded; on eager-flat it calls join.join_ragged."""
    def __init__(self, kb: KB, enum: EnumConfig, shapes: Shapes, layout: Layout): ...
    def layout(self) -> Layout: ...                    # -> flat | dense (set by strategy)
    def resolve(self, req: ResolveRequest) -> Union[ResolvedChildren, FlatResolvedChildren]: ...
    def dedup_index(self) -> DedupIndex: ...           # the EnumRuleIndex bundle (variant_to_orig + binding_tables)
```

**`plan.py`** — per-rule join order, built once.

```python
@dataclass(frozen=True)
class JoinPlan:
    """This module owns the per-rule join order, built ONCE per rule from its RulePattern.
    Width pressure is folded into the join (see join.py): an atom that would exceed w unknown
    leaves is pruned DURING the join, not after."""
    order: Tensor          # [body_len] atom visit order (Yannakakis for acyclic)
    shared_var: Tensor     # [body_len] free var atom i joins on with the partial tuple so far
    anchor: int
def build_join_plan(p: RulePattern, demand: bool) -> JoinPlan: ...
```

**`join.py`** — the L3 fix, pinned to a concrete tensor recipe, with TWO materialization-coupled variants. This is NEW code (the current path is L2 `_enumerate_cartesian` + `torch.nonzero`); it is gated by a MEASURED peak-memory test, not a symbol grep.

```python
@dataclass(frozen=True)
class JoinResult:
    """Join-consistent variable bindings, ALREADY capped to K_max=G_r during the join."""
    bindings: Tensor       # ragged variant: [total, V] + offsets;  padded variant: [N, K_max, V]
    valid: Tensor          # ragged: [total];  padded: [N, K_max]
    offsets: Optional[Tensor]   # [N+1] for the ragged variant; None for padded
    overflowed: Tensor     # [N] bool — cap was hit (flagged; default top-k clamp, strict ⇒ CapMissError)

def join_ragged(plan: JoinPlan, rule: RulePattern, anchor_bind: Tensor,
                fact_index: ArgKeyFactIndex, shapes: Shapes, w: int, cap: Optional[int]
                ) -> JoinResult: ...
    """EAGER-FLAT variant. Concrete recipe (acyclic, Yannakakis order):
      1. Seed: partial = anchor_bind [N, V_seed], valid via offsets.
      2. For each atom i in plan.order:
           v = plan.shared_var[i]
           entity_csr(...) gives each partial tuple's candidate entity list (sorted) on v;
           merge_intersect the running tuple set against it ON v — extends each partial by one var.
           WIDTH FOLD: an atom adding an unknown (non-fact) leaf increments the unknown count;
           tuples exceeding w are dropped HERE (width pressure bounds intermediates, not post-hoc).
      3. INVARIANT (the OOM fix): after EACH atom, cap each row's survivors to K_max via a per-row
         stable top-k BEFORE the next product. K_max bounds the in-flight tuple count at EVERY step.
    Uses torch.nonzero/data-dependent shapes ⇒ EAGER ONLY. Peak: O(total_survivors * deg_max)."""

def join_padded(plan: JoinPlan, rule: RulePattern, anchor_bind: Tensor,
                fact_index: ArgKeyFactIndex, shapes: Shapes, w: int, cap: int
                ) -> JoinResult: ...
    """COMPILED-DENSE variant. Same join semantics but via a FIXED-WIDTH gather-and-mask into
    [N, K_r, K_v, M] (no merge_intersect, no nonzero, no .item()), capped by a STATIC top-k to
    [N, K_max, V]. Expressible under fullgraph=True. This is the variant compiled-dense consumes;
    join_ragged feeds eager-flat. One signature does NOT serve both layouts."""

def merge_intersect(a_off: Tensor, a_val: Tensor, b_off: Tensor, b_val: Tensor, cap: int
                    ) -> Tuple[Tensor, Tensor]: ...
    # two per-row SORTED CSR candidate lists -> (out_off [Np+1], out_val) merge-intersection,
    # each row capped to `cap`. No dense [V_a × V_b] mask is ever formed. (ragged variant only)
def yannakakis_order(rule: RulePattern) -> Tensor: ...   # -> [M] semi-join-reducible order
```

**`candidates.py`** — the L2 substrate the join reduces.

```python
def free_var_candidates(rule: RulePattern, var: int, bound: Tensor, kb: KB) -> Tuple[Tensor, Tensor]: ...
    # var index, bound [N,2] -> per-row CSR (offsets [N+1], values) candidate entities (fact-restricted, L2)
```

**`join_demand.py`** — magic-set demand for the join (imports `resolution/magic_set.py`).

```python
def demand_restrict(rule: RulePattern, goal: Tensor, kb: KB) -> Tensor: ...
    # goal [B,S,3] -> demanded-pred mask [num_pred] (only derive atoms the query needs); uses magic_set.adorn
```

**`width.py`** — the (w,u) leaf filter. Width PRESSURE is folded into the join; this module only does the final `is_last`/`u` leaf check, and it takes a `DepthSelector` so the last-step `w→u` switch is a `torch.where` in compiled mode.

```python
def width_filter(join: JoinResult, rule: RulePattern, kb: KB, w: int, u: int,
                 depth: "DepthSelector") -> Tensor: ...
    # -> keep mask (ragged [total] or padded [N,K_max]): final ≤u unknown-leaf check on the last step.
    # Needs kb.fact_index.exists to decide ground-ness. depth.is_last selects w vs u (torch.where compiled).
```

**`layout.py`** — the single materialization seam; both layouts consume one `JoinResult`, NEVER filter.

```python
def materialize(join: JoinResult, rule: RulePattern, req: ResolveRequest, layout: Layout
                ) -> Union[ResolvedChildren, FlatResolvedChildren]: ...
    """STAGE materialize: lay the (already capped/width-checked) JoinResult into the requested
    layout. NEVER filters — only fills templates. dense → static [N,K_r,G_r,M,3]; flat → compact."""
def materialize_dense(join: JoinResult, rule: RulePattern, req: ResolveRequest) -> ResolvedChildren: ...
    # -> static [N,K_r,G_r,M,3] padded; layout=Layout.dense. (per-rule; engine stacks/caps across rules to K_max)
def materialize_flat(join: JoinResult, rule: RulePattern, req: ResolveRequest) -> FlatResolvedChildren: ...
    # -> compact [T,…]; T = #valid survivors (bindings ARE the survivors — no nonzero over a product);
    #    layout=Layout.flat; subs_noop=True (enum has no MGU).
```

### Engine layer — `grounder/engine/`

**`loop.py`** — the BACKWARD driver. The word "forward" never appears here (it means chaining; this is `run_backward`). It threads `RunState` across chunks and builds RuleGroundings exactly once via `strategy.merge(..., finalize_fn=...)`.

```python
def run_backward(queries: Tensor, mask: Tensor, plan: RunPlan) -> GrounderOutput: ...
    """This module owns the BACKWARD batch runtime: chunk → per-depth step → finalize. NO
    grounder.* reads. The considered FiringAccumulator lives on RunState and spans chunks.

    Dependency chain:
      run_backward
        ├── run_state = RunState(considered=FiringAccumulator.empty(), chunk_query_offset=0)
        ├── for (cq, cm, cshapes, offset) in plan.strategy.iter_chunks(...):
        │     part, bufs = _run_chunk(cq, cm, plan.for_chunk(...))   # bufs: per-step FiringBuffer
        │     run_state = run_state.absorb_chunk(bufs, n_chunk)
        └── plan.strategy.merge(parts, run_state, shapes, finalize_fn=engine.finalize.finalize)
    """
    # queries [B,3], mask [B] -> GrounderOutput
def _run_chunk(queries: Tensor, mask: Tensor, plan: RunPlan
               ) -> Tuple[GrounderOutput, Sequence[FiringBuffer]]: ...
    # one padded chunk -> (partial GrounderOutput with rule_groundings=None, per-step FiringBuffers)
```

**`step.py`** — one depth step; SELECT is inlined (no standalone `select.py`). The considered firing is captured into the step's static `FiringBuffer` AFTER resolve+hooks, BEFORE pack — graph-safe (scatter, no append, no `.item()`).

```python
def step(state: StepState, depth: "DepthSelector", plan: RunPlan) -> StepState: ...
    """SELECT(inlined)→RESOLVE→PACK→POST for one depth. depth is a DepthSelector.
    After resolve + hooks, BEFORE pack: scatter the considered firings into state.firing_buf
    (static, graph-safe) so firings outside completed trees are recorded for ALL resolutions.
    The head is gathered from state.selected_goal (threaded), preserving the head==selected-goal
    cross-check the binding-table guard depends on. -> next StepState"""
def _select(state: StepState) -> Tuple[Tensor, Tensor, Tensor]: ...
    # inlined helper -> (goal [B,S,3], remaining [B,S,G,3], active [B,S])
```

**`pack.py`** — dense + flat compaction; dispatch on the `children.layout` VALUE (not `isinstance`).

```python
def pack(children: Union[ResolvedChildren, FlatResolvedChildren], state: StepState, shapes: Shapes
         ) -> Tuple[PackedStates, SyncParams]: ...
    # reads children.layout to route. On dense/compiled, S_out is padded to shapes.S (static).
    # SyncParams.subs_noop is DETERMINED HERE (post-pack), from the winning subs.
def pack_dense(c: ResolvedChildren, state, shapes) -> Tuple[PackedStates, SyncParams]: ...
def pack_flat(c: FlatResolvedChildren, state, shapes) -> Tuple[PackedStates, SyncParams]: ...
```

**`sync.py`** — depth-d accumulated-body write. The eager-index vs compiled-one-hot write is selected via `DepthSelector.write_slot` (the duality is owned by `execution/depth.py`; this module just calls it). Uses `sp.subs_noop` to skip substitution passes.

```python
def sync_accumulated(accumulated: Tensor, packed: PackedStates, sp: SyncParams,
                     depth: "DepthSelector", shapes: Shapes) -> Tensor: ...
    # accumulated [B,S,D,M,3]; parent gather BEFORE write; write depth d via depth.write_slot -> [B,S,D,M,3].
    # skips apply_substitutions passes when sp.subs_noop (value on the contract, not a module attr).
```

**`postprocess.py`**

```python
def postprocess(packed: PackedStates, sp: SyncParams, state: StepState, depth: "DepthSelector",
                plan: RunPlan) -> StepState: ...
    # prune_facts → sync_accumulated → collected.append(synced per-depth tensors only) → state.replace
```

**`considered.py`** — the graph-safe in-step capture (renamed firing accumulator concept; the eager finalize lives in `dedup.py`/`convert.py`).

```python
def capture_into(buf: FiringBuffer, children: Union[ResolvedChildren, FlatResolvedChildren],
                 selected_goal: Tensor, depth: "DepthSelector") -> FiringBuffer: ...
    """This module owns the GRAPH-SAFE firing capture: scatter every proposed rule application
    (UNcollapsed variant id; head from selected_goal; body from children's flat/dense goals,
    NOT pre-substituted) into the fixed-capacity static buffer. NO list.append, NO .item(), NO
    allocation, NO sync — runs INSIDE the compiled step. Returns the updated FiringBuffer.
    Contract: capacity = S*K_r; n_valid is written as a 0-dim count, read EAGERLY post-loop."""
```

**`evidence.py`**

```python
def build_evidence(collected: CollectedBudget, shapes: Shapes) -> CompletedTreeFirings: ...
    # CollectedBudget -> CompletedTreeFirings ([B,C,D,M,3] structured; carries shapes; no D==0 legacy)
```

**`dedup.py`** — the SOLE rule-grounding dedup owner (prevents `convert.py` regrowing into the `groundings.py` god-module).

```python
def dedup_firings(rule_idx: Tensor, head: Tensor, body: Tensor, body_valid: Tensor,
                  variant_to_orig: Optional[Tensor], num_rules: int, M: int, pad: int
                  ) -> RuleGroundings: ...
    """This module owns the ONE collision-free dedup pipeline (the atom_table / akey / row-pack /
    torch.unique collapse). variant_to_orig (None ⇒ identity) maps anchor variants to the logical
    rule so the K_r variants collapse to one key. Dedup key = (orig_rule_idx, head, sorted_body).
    Both considered_to_rule_groundings and evidence_to_rule_groundings call THIS — they only
    PRODUCE raw firings; they never re-implement the collapse. -> RuleGroundings (sorted CSR)."""
def validate_firings(rule_idx: Tensor, head: Tensor, body: Tensor, binding: "BindingTables"
                     ) -> Tensor: ...
    # -> keep mask: head-matches-entailment + shared-var consistency (catches the all_anchors leak).
    # binding tables come from the GENERIC kb.rule_index, so this runs for sld/rtf/enum alike.
```

**`convert.py`** — a thin (~40-line) dispatcher; considered-primary, evidence-fallback. It consumes `resolver.dedup_index()` (uniform across resolutions), validates with the GENERIC binding tables, and calls `dedup_firings`.

```python
def considered_to_rule_groundings(acc: FiringAccumulator, dedup: DedupIndex, shapes: Shapes
                                   ) -> Optional[RuleGroundings]: ...
    """PRIMARY path. Concatenate the cross-chunk considered firings; validate_firings against
    dedup.binding_tables() (generic rule index); dedup_firings with dedup.variant_to_orig().
    Returns None only when collect_rule_groundings is off (so the merge path can fall back)."""
def evidence_to_rule_groundings(ev: CompletedTreeFirings, dedup: DedupIndex, shapes: Shapes
                                 ) -> RuleGroundings: ...
    """FALLBACK path (chunk-merge only): in-completed-tree firings, used when considered yields
    None on the merge path (e.g. compiled chunks where capture was skipped historically — not
    in this design, where capture_into is graph-safe). Same dedup_firings call. Known to
    undercount vs considered — fallback only."""
```

**`finalize.py`** — build the three views; apply fp_batch. This is the `finalize_fn` the strategy calls at merge.

```python
def finalize(parts: Sequence[GrounderOutput], run_state: RunState, plan: RunPlan,
             *, evidence_fallback: bool) -> GrounderOutput: ...
    """This module owns assembly of ProofState + CompletedTreeFirings + RuleGroundings, applying
    plan.fp_batch when present. Rule-grounding path: considered_to_rule_groundings(run_state.
    considered, plan.resolver.dedup_index()) is PRIMARY; evidence_to_rule_groundings only when
    considered is None AND evidence_fallback (merge path). Built ONCE here.
    PRECONDITION asymmetry (load-bearing): single-batch path passes evidence_fallback=False
    (considered is always populated); merge path passes True. Failure: shape/vocab errors
    propagate as exceptions — no silent fallback. -> GrounderOutput"""
```

**`prune/facts.py`**, **`prune/dead.py`** — search-time reducers. POST-phase, over proof state; they read CSR tables from `kb.fact_index`/`kb.rule_index`, NEVER a module buffer. Ordering: `prune_dead` (sld/rtf) then `prune_facts`, both inside `postprocess`.

```python
def prune_facts(state: StepState, kb: KB) -> StepState: ...   # drop ground-fact goals (all resolutions)
def prune_dead(state: StepState) -> StepState: ...             # drop dead branches (sld/rtf only)
```

### The only filter — `grounder/filters/fp_batch.py`

```python
class FpBatch:
    """This module owns THE ONLY filter in the library: a cross-query Kleene T_P fixed point
    over the COLLECTED rule applications. Keep a firing iff every body atom is a fact or the
    head of another kept firing; iterate `depth` times (converges by construction). == keras
    prune_incomplete_proofs. There is NO fp_global filter; that capability is ForwardGrounder.

    Contract:
    - Input: a complete RuleGroundings set + KB facts. Post-hoc; NEVER mid-search.
    - Output: a NEW RuleGroundings with dropped firings REMOVED and rule_offsets RECOMPUTED
      (bincount+cumsum); firing_valid on the result is all-True. firing_valid is NOT the
      pruning channel — the row count SHRINKS. (Matches the verified behavior at bc/pruning.py.)
    - Default ON when enum + u==0 (paper); OFF otherwise (derived in GrounderConfig.resolved_filter).
    - The masked_select is the one sync-inducing op; runs `depth` iterations unconditionally
      (no torch.equal host sync).
    """
    def __init__(self, kb: KB, shapes: Shapes): ...
    def apply(self, rg: RuleGroundings) -> RuleGroundings: ...   # -> NEW RuleGroundings (rows dropped)
```

### Forward chaining as a grounder — `grounder/forward/`

This is genuinely TWO engines (leapfrog-triejoin and sparse-matmul-with-strategies), not one small semi-naive loop. The naming reserves "forward" for chaining; "seminaive" is one of three spmm strategies, not the file name.

**`api.py`**

```python
class ForwardStrategy(StrEnum): ...   # leapfrog | spmm   (engine choice)
class SpmmStrategy(StrEnum): ...      # seminaive | true_istar | hybrid   (within spmm)

@dataclass(frozen=True)
class Closure:
    """Derivable-atom set + witnesses. Returned by FC; consumed by KB.with_closure() (backward
    soundness) and by witness gather (output). NEVER mutates a KB."""
    provable: Tensor          # [num_provable, 3] sorted hash set
    witnesses: "Witnesses"
    def contains(self, atoms: Tensor) -> Tensor: ...   # [...,3] -> bool [...]

@dataclass(frozen=True)
class Witnesses:
    rule_idx: Tensor          # [num_provable]
    body_grounding: Tensor    # [num_provable, M, 3]
```

**`relation_matrix.py`**

```python
def build_relation_matrices(kb: KB) -> List[Tensor]: ...   # per-relation sparse [E,E] (pure build)
```

**`query_demand.py`** — the DEFAULT answering path (bounded). Imports `resolution/magic_set.py`.

```python
def run_demand(kb: KB, queries: Tensor, depth: int, strategy: ExecStrategy) -> Closure: ...
    """Query-directed (magic-set) FC — DEFAULT for answering. Seeds the fixed point from query
    bindings so only relevant atoms are derived (bounded). Returns a relevant Closure; NEVER
    mutates KB. Subsumes the old resolution/closure.py demand-building and fp_global.build_fp_global_set
    closure-set construction (relocated here, not deleted)."""
def check_membership(queries: Tensor, closure: Closure) -> Tensor: ...   # [B,3] -> [B] bool
```

**`witnesses.py`**

```python
def gather(atoms: Tensor, w: Witnesses, shapes: Shapes) -> GrounderOutput: ...
    # atoms [B,3] -> GrounderOutput (evidence + rule_groundings from witnesses)
```

**`triejoin/lftj.py`**, **`anchored.py`**, **`join_order.py`**, **`frontiers.py`** — engine A.

```python
# lftj.py
def leapfrog_triejoin(matrices: List[Tensor], rule: RulePattern, frontier: Tensor) -> Tensor: ...
    # -> new derived atoms [num_new, 3] for one rule (Leapfrog Triejoin, Veldhuizen 2014)
# anchored.py
def run_stages_anchored(matrices: List[Tensor], rule: RulePattern, partial: Tensor) -> Tensor: ...
    # -> derived atoms [num_new, 3] via anchored semi-naive stages
# join_order.py
def compute_join_order(body_pred_indices: Tensor, m: int) -> List[int]: ...     # -> visit order
# frontiers.py
def compute_frontiers(rule: RulePattern, ordered_bps: Optional[list]) -> List[set]: ...
```

**`spmm/fixpoint.py`**, **`strategies.py`**, **`kernels.py`**, **`matrices.py`**, **`ops.py`**, **`state.py`** — engine B.

```python
# fixpoint.py
def run_closure(kb: KB, depth: int, strategy: ExecStrategy) -> Closure: ...
    """Depth-bounded FULL closure via the spmm fixed point. Returns a NEW Closure; NEVER mutates
    KB. UNBOUNDED on dense KGs — caller must opt in. Dispatches to the chosen SpmmStrategy."""
def semi_naive_fixpoint(matrices: List[Tensor], rules, depth: int, strat: "IterationStrategy"
                        ) -> Tuple[Tensor, Witnesses]: ...
    # -> (derived atoms [num_derived,3] sorted+unique, witnesses)
# strategies.py
class IterationStrategy(ABC):
    @abstractmethod
    def fire(self, state: "FcState", rule) -> "FireSpec": ...
class SeminaiveStrategy(IterationStrategy): ...   # Bancilhon–Ramakrishnan, 2^n_slots − 1 fires
class TrueIStarStrategy(IterationStrategy): ...    # single accum @ accum fire per rule per step
class HybridStrategy(IterationStrategy): ...       # default: picks TrueIStar or Seminaive per rule
# kernels.py
def spgemm(a: Tensor, b: Tensor) -> Tensor: ...   # sparse @ sparse
def spmv(a: Tensor, x: Tensor) -> Tensor: ...     # sparse @ dense
# state.py
@dataclass
class FcState:
    frontier: Tensor; delta: Tensor               # frontier/delta accumulators
```

### Optional memoization — `grounder/memo/` (v1 PLACEHOLDER)

```python
# store.py
class MemoStore(Protocol):
    """v1 PLACEHOLDER. Tabling/subgoal are DESCOPED from v1: the real bc/tabling.py
    writeback_and_replay REWRITES the considered accumulator's suffix in place and splices HIT
    rows back in — incompatible with the frozen, append-only FiringAccumulator. When ported, the
    store must own firing PRODUCTION (resolve_with_memo returns a NEW FiringAccumulator), NOT a
    passive lookup. Until then, only NullMemo is wired; the engine reads plan.memo, never getattr.

    Planned v2 surface (NOT yet implemented):
      def resolve_with_memo(self, req, resolver) -> Tuple[FiringBuffer, "MemoStats"]: ...
    """
    def lookup(self, key: Tensor) -> Tuple[Optional[Tensor], int, int]: ...   # [N] -> (children|None, hits, misses)
    def store(self, key: Tensor, children) -> None: ...
class NullMemo:
    """The default no-op (the ONLY memo wired in v1). lookup → (None,0,0); store → no-op."""
```

### NeSy adapters — `grounder/nesy/`

The hook surface is FOUR Protocols on success masks (matching the real `nesy/hooks.py`), not a single per-atom score hook. The `GroundingHook` is the one the output assembly calls.

```python
# api.py
class ResolutionFactHook(Protocol):
    """This module owns neural/KGE injection at resolution. Orthogonal to the discrete set:
    hooks gate candidates, they do not change which groundings are structurally possible."""
    def filter_facts(self, fact_goals: Tensor, fact_success: Tensor, queries: Tensor) -> Tensor: ...
        # fact_goals [B,S,K_f,G,3], fact_success [B,S,K_f], queries [B,S,3] -> mask [B,S,K_f]
class ResolutionRuleHook(Protocol):
    def filter_rules(self, rule_goals: Tensor, rule_success: Tensor, queries: Tensor) -> Tensor: ...
        # rule_goals [B,S,K_r,G,3], rule_success [B,S,K_r], queries [B,S,3] -> mask [B,S,K_r]
class StepHook(Protocol):
    def on_step(self, body: Tensor, mask: Tensor, rule_idx: Tensor, d: int
                ) -> Tuple[Tensor, Tensor, Tensor]: ...   # per-step rewrite
class GroundingHook(Protocol):
    """The evidence-rewriting hook the output assembly calls (finalize). Takes/returns a
    fully-formed CompletedTreeFirings so the `shapes` field is preserved by construction."""
    def apply(self, evidence: CompletedTreeFirings) -> CompletedTreeFirings: ...
# kge.py / scoring.py / sampler.py implement these. Orthogonal; do not change the grounding set.
```

---

## 5. DESIGN CONTRACTS (three tiers, probfol style)

Tier-1 boundary contracts go in the grounder `CLAUDE.md` under "Adding or Changing Code", each ending in its enforcing test. Tier-2 = the "This module owns…" docstrings + `Dependency chain` blocks (inline above). Tier-3 = the in-code `Contract:` blocks (one precondition / one postcondition-and-where-written / one explicit failure mode each). READMEs defer to the in-code `Contract:` block as authoritative.

### Contracts for `plan.py` + `run_state.py` (the shell↔engine boundary)

- `RunPlan` MUST be frozen and contain everything a run needs; the engine MUST NEVER read attributes off the `Grounder`/`nn.Module`. *Enforced by `tests/test_plan_is_sole_engine_input.py`* (greps `engine/` for any `grounder.`/`self.` field read; must be zero).
- `RunPlan` MUST be resolution-agnostic: it carries NO enum-specific field. Dedup metadata is reached via `plan.resolver.dedup_index()`. *Enforced by `tests/test_plan_resolution_agnostic.py`* (asserts no `EnumRuleIndex`/`enum_index` attr on `RunPlan`).
- A `BackwardGrounder` MUST be reentrant: two concurrent `forward()` calls share no mutable state. Per-call state lives only in `RunState`/`StepState`/`CollectedBudget`/`FiringAccumulator`/`FiringBuffer`, never on the module. *Enforced by `tests/test_no_module_scratch.py`* (`set(grounder.__dict__)` after `forward()` == after `__init__`) and `tests/test_reentrant_backward.py` (interleaves two passes; identity with serial).
- The considered `FiringAccumulator` MUST span chunks: reset once per `forward()`, accumulated across chunks with a global query offset, finalized once at merge. *Enforced by `tests/test_considered_spans_chunks.py`* (a chunked run and an unchunked run produce identical rule groundings).
- `RunPlan.for_chunk(B)` / `RunState.absorb_chunk(...)` MUST return new immutable objects; they MUST NOT mutate `self`. *Enforced by `tests/test_immutable_threading.py`*.

### Contracts for `engine/` ↔ `resolution/` (the resolve boundary)

- Every `Resolver.resolve(req)` MUST return `ResolvedChildren` (dense) or `FlatResolvedChildren` (flat), shaped exactly per `types.py`, with `layout` carried as a field; no fourth shape. It MUST NOT mutate `req`, the `KB`, or any engine state. *Enforced by `tests/test_resolved_children_contract.py`*.
- `subs_noop` MUST be determined POST-pack (`SyncParams.subs_noop`) — and may additionally be a constant-True field on `FlatResolvedChildren` — but MUST NOT appear on the dense `ResolvedChildren` (where sld/rtf subs are real) and MUST NOT be a module attribute. *Enforced by `tests/test_subs_noop_placement.py`* (greps for `_winning_subs_noop`; absent; asserts `ResolvedChildren` has no `subs_noop`).
- `ResolveRequest.depth` MUST be a `DepthSelector` (int/bool eager; 0-dim tensors compiled). A resolver on the compiled path MUST NOT Python-branch on it. *Enforced by `tests/test_depth_selector_in_resolver.py`* + `tests/test_no_item_in_compiled_step.py`.
- Resolution is an AXIS: exactly ONE resolver class per algorithm; NEVER an `SldGrounder`/`EnumGrounder` subclass. *Enforced by `tests/test_one_class_per_grounder.py`* (`Grounder.__subclasses__()` == `{BackwardGrounder, ForwardGrounder}`).
- Search-time reducers (`prune_facts`, `prune_dead`, `width_filter`) MUST read CSR tables only from `kb.fact_index`/`kb.rule_index`, never a module buffer; width runs INSIDE the join (pressure) + a final leaf check; prune runs POST-phase in `postprocess` (order: `prune_dead` then `prune_facts`). *Enforced by `tests/test_reducers_no_module_buffer.py`*.

### Contracts for `engine/` ↔ `execution/` (the execute boundary)

- `execution/compiler.py::Compiler.wrap` is the ONLY `torch.compile` call site. *Enforced by `tests/test_single_compile_site.py`* (exactly one `torch.compile` in the tree).
- `execution/cudagraph.py::detach_from_pool` is the ONLY clone-to-detach seam, and it MUST be a pytree deep-clone (covers both the step-output tuple and the nested `GrounderOutput` tree). *Enforced by `tests/test_single_cudagraph_clone.py`*.
- `ExecStrategy` OWNS all chunk mechanics (plan/pad/trim/iterate/merge); `ChunkPolicy` is a pure value object. `merge` MUST call the engine-provided `finalize_fn` to build RuleGroundings; it MUST NOT import `convert`/`dedup`/`EnumRuleIndex`. *Enforced by `tests/test_strategy_no_grounding_imports.py`* (greps `execution/` for grounding-semantics imports; zero).
- `validate()` MUST reject illegal combos with `StrategyError` (`flat+compiled_step`, `sld+compiled_step`, `sparse`/`forward`+`dense`/`flat`, undeclared cell); no silent fallback to another cell. *Enforced by `tests/test_strategy_validate.py`* (parametrized over every illegal pair).
- `auto_select` MUST key only on the static budget `(K_f, K_r, G_r, M, B)`; it MUST NEVER probe data-dependent per-query statistics, return a dynamic mode, or `max-autotune`. *Enforced by `tests/test_auto_select_policy.py`*.
- The engine MUST build statically-shaped tensors in any compiled region: no `.item()`, no data-dependent Python branching; compile ONE step, never the loop. *Enforced by `tests/test_no_item_in_compiled_step.py`* (`fullgraph=True` on a step fixture; must not graph-break).

### Contracts for `engine/finalize.py` + `convert.py` + `dedup.py` (the rule-grounding output)

- The PRIMARY rule-grounding path is `considered_to_rule_groundings`; `evidence_to_rule_groundings` is a FALLBACK only (chunk-merge path, when considered is None). A pure evidence-derived producer is FORBIDDEN — it undercounts ~3× (ablation_d3 BC13: 80 vs 252). *Enforced by `tests/test_considered_not_evidence.py`* + `tests/test_keras_grounding_comparison.py`.
- The single-batch path MUST pass `evidence_fallback=False` (considered always populated); the merge path MUST pass `True`. *Enforced by `tests/test_finalize_fallback_asymmetry.py`*.
- The dedup pipeline has EXACTLY ONE owner (`engine/dedup.py::dedup_firings`). `convert.py` only PRODUCES raw firings and calls it; it MUST NOT re-implement the `torch.unique` collapse. Binding validation MUST use the GENERIC `kb.rule_index` binding tables (so it runs for sld/rtf/enum), not an enum-only structure. *Enforced by `tests/test_single_dedup_owner.py`* (the `unique`-based collapse appears once) + `tests/test_binding_validation_all_resolutions.py`.
- The dedup key is `(orig_rule_idx, head, sorted_body)` via `dedup_index().variant_to_orig()` (identity for non-enum), collapsing `all_anchors` variants. *Enforced by `tests/test_keras_grounding_comparison.py`*.
- Considered CAPTURE MUST be graph-safe (`capture_into`: static scatter, no `.item()`, no append, no alloc); dedup/validation/`variant_to_orig` collapse MUST be eager post-loop. The compiled and eager paths MUST therefore produce the SAME RuleGroundings. *Enforced by `tests/test_rule_groundings_path_parity.py`* (eager-flat vs compiled-dense; equal firings) + `tests/test_capture_into_graph_safe.py` (`fullgraph=True`).
- `RuleGroundings` is built ONCE at chunk merge (`strategy.merge` → `finalize`); per-chunk `_run_chunk` returns `rule_groundings=None`. *Enforced by `tests/test_merge_builds_groundings_once.py`*.
- The per-query cap `G_r` DEFAULT is UNCHANGED; on overflow `JoinResult.overflowed=True` and the row is top-k clamped (baseline-identical); `CapMissError` is raised ONLY when `EnumConfig.strict_cap=True`. *Enforced by `tests/test_grounding_count_baselines.py`* + `tests/test_strict_cap_raises.py`.

### Contracts for `resolution/enum/` (the join core)

- Candidate generation MUST be join-based and query-directed; it MUST NEVER materialize a cartesian product before the cap. The `G_r` cap MUST be applied DURING the join (after each atom, before the next product), so the in-flight tuple count is bounded by `K_max` at every step. The join is NEW code (replaces the L2 `_enumerate_cartesian`+`torch.nonzero` path). *Enforced by `tests/test_join_peak_memory.py`* (MEASURED peak transient ≤ `O(survivors*deg_max)` on a high-fan-out fb15k fixture that OOMs under L2 — a measured-memory gate, NOT a symbol grep).
- The join has TWO materialization-coupled variants: `join_ragged` (CSR merge, data-dependent T) feeds EAGER-flat; `join_padded` (fixed-width gather-and-mask + static top-k, no `nonzero`/`.item()`) feeds COMPILED-dense. One signature does NOT serve both. *Enforced by `tests/test_join_two_variants.py`* (`join_padded` is `fullgraph`-safe; `join_ragged` equals it on results).
- `flat` and `dense` are two materialization layouts behind ONE seam (`layout.py`), both consuming the same `JoinResult`; no duplicated enumeration body. *Enforced by `tests/test_layouts_share_join.py`*.
- `flat` is eager-only; `dense` is the compile/CUDA-graph path. The strategy enforces the pairing; enum MUST NOT pick a layout itself. *Enforced by `tests/test_strategy_validate.py`*.
- `all_anchors` is FORCED True internally and MUST NOT be a user knob; `u=0`⇒`fp_batch`, `u>0`⇒`none`. The (w,u) width filter applies uniformly to 1-body rules (torch-strict; correct under `u=0`). *Enforced by `tests/test_enum_forced_defaults.py`*.

### Contracts for `data/fact_index/` (the candidate adapter)

- The `FactIndex` ABC is MEMBERSHIP only (`exists`, `k_f`, `capabilities`). `enumerate` is NOT a uniform ABC method — ArgKey does not implement it (it answers `candidate_set`/`targeted_lookup`, the join primitive). Membership-equivalent layouts MUST give identical grounding counts. *Enforced by `tests/test_fact_index_membership_equivalence.py`* + `tests/test_arg_key_no_enumerate.py` (ArgKey advertises `{"candidate_set","targeted_lookup"}`, not `enumerate`).
- `ArgKeyFactIndex.candidate_set` returns `(offsets, fact_row_idx)` (FACT ROWS, composite-keyed CSR). `entity_csr` is the explicit projected, SORTED entity-id view `merge_intersect` consumes; its sortedness is a precondition. *Enforced by `tests/test_arg_key_csr_domain.py`* (values are fact rows) + `tests/test_entity_csr_sorted.py`.
- Indexes are immutable after `KB` construction; FC returns a new `Closure` and NEVER mutates `kb.fact_index`/`kb.K_f`. Backward grounding consumes a closure via `KB.with_closure()` → a NEW KB. *Enforced by `tests/test_kb_immutable.py`* (hashes index tensors before/after) + `tests/test_with_closure_new_kb.py`.

### Contracts for `forward/` (forward chaining IS a grounder)

- Forward chaining is exposed ONLY as `ForwardGrounder`. There is NO `fp_global` FILTER, NO `resolution="closure"` axis, NO closure special-case in the backward loop. *Enforced by `tests/test_no_fp_global_filter.py`* (greps for an `fp_global` filter; the closure-set BUILDER survives inside `forward/`, only the filter is banned).
- The old `fp_global` soundness capability (derived atoms usable as facts in backward SLD) is PRESERVED via `KB.with_closure()`, not deleted. *Enforced by `tests/test_closure_soundness_use_case.py`* (a goal provable only through a derived atom succeeds when `kb.with_closure(fg.closure(d))` is used).
- FC is TWO engines (leapfrog-triejoin + spmm-with-strategies); the spmm fixpoint file is `forward/spmm/fixpoint.py`, NOT a top-level `seminaive.py`. The strategy axis (`seminaive|true_istar|hybrid`) is explicit. *Enforced by `tests/test_forward_two_engines.py`*.
- `run_closure`/`run_demand` MUST return a new `Closure` and MUST NOT mutate the `KB`. Default answering is query-directed. *Enforced by `tests/test_forward_no_kb_mutation.py`*.

### Contracts for `filters/fp_batch.py` (the only filter)

- `fp_batch` is the ONLY filter. It runs post-hoc over a complete `RuleGroundings` set, NEVER mid-search, and returns a NEW `RuleGroundings` with dropped rows removed and `rule_offsets` recomputed (`firing_valid` is all-True; row count shrinks — NOT a mask channel). *Enforced by `tests/test_only_fp_batch.py`* (`filters/` contains exactly `fp_batch.py`) + `tests/test_fp_batch_drops_rows.py` (asserts shrink) + `tests/test_fp_batch_parity.py` (keras oracle).

### Contracts for `memo/` (descoped v1)

- v1 ships ONLY `NullMemo`. Tabling/subgoal are DESCOPED (they rewrite the firing accumulator in place, incompatible with the frozen `FiringAccumulator`). The engine reads `plan.memo`; it NEVER uses `getattr` to discover an off-by-default subsystem. *Enforced by `tests/test_memo_null_only_v1.py`* + `tests/test_memo_no_getattr.py`.

### Contracts for `execution/depth.py` (the d duality)

- `DepthSelector` is the ONE place the eager-int / compiled-0-dim-tensor `d`/`is_last` duality is expressed. No phase may use `isinstance(d, Tensor)`; every depth-consuming phase (enum, width, sync) takes a `DepthSelector`. *Enforced by `tests/test_no_isinstance_d_tensor.py`* (greps for `isinstance(d, ... Tensor)`; zero outside `depth.py`).

---

## 6. Capability matrix (grounder × execution strategy)

Each grounder DECLARES a SPARSE set of supported `Cell`s via `declare_strategies()`. `validate()` rejects undeclared cells and globally-illegal combos. `auto_select` picks within the declared set keyed on the static budget `(K_f, K_r, G_r, M, B)`.

Cells: layout ∈ {dense, flat, sparse} × compile ∈ {eager, compiled_step, outer_reduce_overhead}.

| grounder / resolution | eager-flat | eager-dense | compiled_step-dense (+G_r cap) | outer reduce_overhead (dense) | sparse (eager) |
|---|:---:|:---:|:---:|:---:|:---:|
| **BackwardGrounder / sld** | n/a | ✓ | ✗ (per-step compile impossible) | ✓ **default when replayed** (4× warm) | n/a |
| **BackwardGrounder / rtf** | ✓ | ✓ | ✗ | ✓ | n/a |
| **BackwardGrounder / enum** | ✓ **default low fan-out** | ✓ (debug only) | ✓ **default high fan-out** (only bounded path) | ✗ | n/a |
| **ForwardGrounder** | n/a | n/a | n/a | n/a | ✓ **only cell** |

Globally illegal combos `validate()` raises `StrategyError` on (independent of grounder): `flat`+`compiled_step` (data-dependent `nonzero`/`.item()` ⇒ no fullgraph); `sld`+`compiled_step` (per-step not expressible for sld); `forward`/`sparse`+`dense`/`flat`; any cell ∉ the grounder's declared `CapabilityRow`. There is no `dynamic=True` cell anywhere (dominated; `dynamic` is implied-False by `CompileMode`, not a free field).

`auto_select` decision (STUDY.md verdict): `ForwardGrounder` → `(sparse, eager)`. `sld` → `(dense, outer_reduce_overhead)` when the same shape is replayed many times (4× warm), else `(dense, eager)` to dodge the 20–33 s warmup. `enum`, low fan-out (family, `K_f≈28`) → `(flat, eager)` (~6× faster, ~1.5 GiB). `enum`, high fan-out / cliff risk (fb15k, `K_f≈3612`) → `(dense, compiled_step)` + `G_r` cap — the only bounded, non-OOM enum path (1024 q → ~2.1 GiB, linear); flat OOMs on single queries and chunking does not save it.

---

## 7. What each grounder DECLARES as supported

- **`BackwardGrounder`** declares a resolution-dependent sparse set (its `declare_strategies()` switches on `config.resolution`):
  - **sld** → `{(dense, eager), (dense, outer_reduce_overhead)}`. Auto-selects `outer_reduce_overhead` only when the same chunk shape is replayed (amortizes the 4× warm cost); else `eager`. Never `compiled_step` (a per-step compile is not expressible for sld's MGU recursion).
  - **rtf** → `{(dense, eager), (flat, eager)}`. Auto-selects `flat` for low fan-out, `dense` otherwise; both eager (rtf has no bounded compiled path that beats eager in the study).
  - **enum** → `{(flat, eager), (dense, eager), (dense, compiled_step)}`. `(dense, eager)` is debug-only. Auto-selects `(flat, eager)` for low fan-out and `(dense, compiled_step)` + `G_r` cap for high fan-out (the only bounded path). Declares neither `outer_reduce_overhead` nor any `dynamic` mode.
- **`ForwardGrounder`** declares exactly `{(sparse, eager)}`. It never supports dense/flat/compiled cells; FC is sparse-matmul / triejoin and runs eager. Its engine choice (`leapfrog` vs `spmm`) and spmm strategy (`seminaive|true_istar|hybrid`) are FC-internal axes, orthogonal to the execution `Cell`.

Any request for a cell a grounder did not declare raises `StrategyError` (no silent fallback). Explicit `ExecStrategy.explicit(row, cell, chunk)` validates against the declared row at construction; `ExecStrategy.auto(...)` only ever returns a declared cell.
