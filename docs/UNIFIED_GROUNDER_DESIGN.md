# Grounder Library — Unified Architecture (FINAL, canonical)

Lead-architect final pass. Every load-bearing claim re-verified against `_build/` ground truth (file:line cited inline). This document overrules v1 where the adversarial walk-through found v1 asserting code that does not exist; the decision log records each reversal. This is the project's single architecture reference.

Bar: concrete (real names/signatures/fields), coherent end-to-end, reachable from current code via byte-identity-gated increments, honestly minimal (no speculative seam ships with <2 impls), and provably extensible to the forcing-function grounders.

---

## 0. The one-paragraph model

A **grounder** is a function of a **KB**. There are exactly **two grounder families today** — `BackwardGrounder` (query-directed proof search) and `ForwardGrounder` (data-directed `T_P` closure). They are *config-parametrized, not subclassed*: a new backward algorithm is a new **Resolver**; a new closure algorithm is a new **ForwardMethod** (engine) or **JoinAlgo** (its sub-strategy). Both families implement one protocol, `Grounder.ground(GroundRequest) -> GroundResult`, where `GroundResult` is an honest **two-member** tagged union (`BackwardResult | Closure`) behind a marker protocol. The two results do **not** share a forced firings bridge — `RuleGroundings` is a **BackwardResult-only** tier today, because FC has no provenance machinery (verified). Orthogonal to the family is the **execution** axis (`Cell = Layout × CompileSpec`, declared via `CapabilityRow`, resolved by `ExecStrategy`) — genuinely clean, consumed by the backward engine; FC shares its *vocabulary* (`Cell`/`CapabilityRow`) and the `ChunkPolicy` value object, **not** its query-keyed `iter_chunks`. Above any grounder sits the **ProgramTransform** axis: a pure `(kb, request) -> (kb', request')` rewrite (magic-sets) composed by the single entry `make_grounder`. **Four axes — execution (clean), resolver, forward-method, transform — three of them promoted from `if/elif` to registry-backed protocols.**

## 1. Principles (the load-bearing decisions)

1. **One `Grounder` protocol; families differ by impl, not by a divergent `forward` signature.** Today `BackwardGrounder.forward(queries, query_mask) -> GrounderOutput` (`grounder/backward.py`) and `ForwardGrounder.forward() -> Tensor` (`forward/grounder.py:66`) are unrelated shapes. They become one `ground(GroundRequest) -> GroundResult`; `nn.Module.forward` survives as a thin back-compat shim.
2. **The run interface is honestly two-shaped, unified only at the type level.** BC is query-directed; FC is data-directed (`forward/grounder.py:66` takes no args). `GroundRequest.queries: Optional[Tensor]` is `None` for FC. Never force FC to take queries.
3. **`GroundResult` is a TWO-member tagged union (`BackwardResult | Closure`) — there is NO cross-family firings bridge today.** Verified: `forward/witnesses.py` is fp_global BACKWARD machinery (`build_witness_table(grounder,...)` mutates a BackwardGrounder, reads `grounder._enum_ri`/`fp_global_hashes`, `witnesses.py:31-204`), invoked only from `forward/closure.py:113` — both are dead fp_global code. FC's `run_forward_chaining` returns only `(sorted_hashes, n_provable)` (`forward/fc.py:1156`). So `Closure.as_rule_groundings()` returns `None`. FIRINGS is a BackwardResult-only tier. A future FC firing path is net-new plumbing (see §2c), not a rename.
4. **Generalize only what already works AND has ≥2 impls.** The `Resolver` protocol IS `pbc/resolve.py`'s `resolve_step(inp, plan, fact_index, materializer)` + the sld/rtf function pairs, lifted to one dispatch table — 3 real impls (sld/rtf/pbc) justify it. `Materializer` (dense/flat) stays pbc-internal. `ForwardMethod` (spmm/staged) has 2 real impls; its `JoinAlgo` sub-axis (staged/chunked) has 2 real impls — leapfrog is a documented placeholder (`fc.py:951-968` falls through to `_run_stages`) and is NOT enshrined as an extension point. `ProgramTransform` ships with `MagicSetTransform` + `IdentityTransform`.
5. **No speculative seam.** `LearnedGrounder`/`ScoredResult`/a SCORES tier are NOT introduced. Their real seam is the existing `nesy/hooks.py` scoring overlay, not a third family with its own engine (the entire per-run stack — `RunPlan` reads ~30 BC attrs `plan.py:79-121`, `Frontier` is a proof tape `state.py:43`, `run_backward` is the only non-FC driver — is BC-shaped). Adding a union member with zero impls is a guaranteed future breaking change; deferred.
6. **Transforms are a first-class pre-grounding phase composed by `make_grounder`**, not subclasses that secretly rebuild a KB. They require `KB.with_program` (rule-rewriteable KB); today only `with_closure` (facts-only union, re-runs `KB.__init__`, `kb.py:150-160`) exists. The per-call rebuild cost is real and the cache is **in scope**, not deferred (§2b).
7. **Execution stays the orthogonal axis; FC sharing is honestly scoped.** `auto_select` already returns sparse via `if row.supports(sparse)` (`auto_select.py:34`) and `iter_chunks(queries, mask, shapes)` slices the query dim (`strategy.py:80-83`) — FC has no query batch. So FC shares the `Cell`/`CapabilityRow` *vocabulary* and a `ChunkPolicy` value object only; FC keeps its own fixpoint chunking (`FCDynamic.join_chunk_size`). No claim that `ExecStrategy.iter_chunks` unifies the two.
8. **One construction vocabulary, honoring the existing typed configs.** `config.py` is a *deliberate* design ("There is no Resolution enum and no GrounderConfig wrapper", `config.py:3-4`) — it is **unwired** (`RunPlan.config = getattr(grounder,'config',None)` is unconditionally `None`, `plan.py:104`), not vestigial. We do NOT introduce a competing flat `GrounderConfig`. We WIRE the existing `SLDConfig|RTFConfig|PBCConfig|FCConfig` into `make_grounder` (dispatch on config TYPE, which already encodes resolution + `filter()`), and either populate `RunPlan.config` or delete the dead field — a one-line decision, not a rename.
9. **`RulePattern` is the SOLE binding-analysis owner** (`data/rule_index/pattern.py`), consumed by pbc and forward. Magic-set CONSUMES its analysis to compute adornments but PRODUCES raw rule tensors fed to `KB.with_program`, which re-derives fresh patterns. RulePattern is read, not rewritten (§2a).
10. **One agreed term per concept; synonyms deleted.** `enum` survives only as a factory parser alias (`factory.py:56`); `closure` is a *family*, never a resolution value. The glossary is extended to the seam nouns AND a new contract enforces GLOSSARY-vs-FIELDS token disjointness (today `evidence` is BOTH a concept `glossary.py:28` and a field `glossary.py:98`; `considered` `glossary.py:27` is an orphan concept colliding with the `firings` tier).
11. **"fp_batch is the only filter" is FALSE — there are two live filters.** `filters/prune_facts.py:prune_ground_facts` is wired in the hot path (`engine/step.py:21,167-173`, gated by `RunPlan.prune_facts`). The correct statement: fp_batch is the only *soundness* filter; prune_facts is a separate *search-time candidate* prune. Both stay.
12. **Every FC-touching change is gated by an FC byte-identity oracle that must be BUILT FIRST.** Verified: `tests/` contains only `fingerprint_new.py` (BC), `flat_ab.py` (BC sld/rtf), `nesy_ab.py` — `grep -rln` for any FC symbol in `tests/` returns nothing. The "closure-set A/B" the `forward/__init__.py:5` docstring references does not exist. It is refactor step 0b, a precondition for steps touching Closure/ForwardMethod/with_program.

## 2. Taxonomy + the 4-axis extension model

```
                         ┌─────────────────────────────────────────────┐
   AXIS 4 (above)        │  ProgramTransform  (kb,req) -> (kb',req')     │  magic_set, identity
                         │  composed by make_grounder; pure; ordered     │  → KB.with_program (+cache)
                         └─────────────────────────────────────────────┘
                                            │ composes with ANY family
   AXIS 0 (family)   ┌──────────────────────┴───────────────────────────┐
                     │  Grounder.ground(GroundRequest) -> GroundResult   │  closed 2-member set
                     ├───────────────────────┬───────────────────────────┤
                     │ Backward              │ Forward                    │
   AXIS 1 / AXIS 3   │  seam = RESOLVER      │  seam = FORWARD_METHOD      │
                     │  sld rtf pbc          │  spmm  staged              │
                     │   (+ join ← L3)       │                            │
                     ├───────────────────────┼───────────────────────────┤
   SUB-AXIS          │ within pbc/join:      │ within staged:             │
                     │   MATERIALIZER        │   JOIN_ALGO                 │
                     │   dense | flat | join │   staged | chunked         │
                     └───────────────────────┴───────────────────────────┘
   ORTHOGONAL EXEC   Cell = Layout × CompileSpec ; CapabilityRow          (BC consumes fully;
                     ExecStrategy.iter_chunks is BC-query-keyed            FC shares vocab + ChunkPolicy only)
```

**Why this shape (sharpened, corrected).** v1's "three flat seams" was wrong on two counts the critiques verified: (a) the forward axis is TWO sub-axes — `method ∈ {spmm,staged}` (the closure engine) AND `join_algo ∈ {staged,chunked,leapfrog}` (FCDynamic's per-rule join, `fc.py:1153-1155,662`) — exactly mirroring backward's Resolver × Materializer; v1 flattened them into one enum and minted a `staged`/`staged` synonym. (b) The three grounder-level seams are at different *altitudes*: Resolver/ForwardMethod are intra-family; Transform is supra-family. The only genuinely clean axis today is execution. This design promotes the two backward-symmetric sub-axes and adds the supra-family transform.

**Each forcing-function grounder plugs into a DIFFERENT axis:**

- **MAGIC-SET / demand → ProgramTransform (axis 4).** Query-dependent program rewrite, NOT a resolver. §2a.
- **JOIN-enum L3 → a sibling `JoinResolver` (axis 1), NOT a byte-identical Materializer swap.** §2c.
- **New FORWARD engine (GPU triejoin, spmm variant) → ForwardMethod (axis 3); a new per-rule join → JoinAlgo (sub-axis).** A new forward grounder must declare WHICH sub-seam it targets. §2d.
- **Plain Datalog / semi-naive → a JoinAlgo or a ForwardMethod**, not a new family.

### 2a. Worked example — MAGIC-SET (axis 4, ProgramTransform)

```python
# transform/magic_set.py
@register_transform("magic_set")
class MagicSetTransform:                          # implements ProgramTransform
    def __init__(self, base_kb: KB):
        # query-INDEPENDENT: the adorned RULE SKELETON, computed ONCE at construction.
        self._adorned_rules = build_adorned_rules(base_kb.rule_index.patterns)  # reads RulePattern
        self._skeleton_kb = base_kb.with_program(                              # rebuild indices ONCE
            heads=self._adorned_rules.heads, bodies=self._adorned_rules.bodies,
            lens=self._adorned_rules.lens, facts=base_kb.facts_idx)

    def apply(self, kb: KB, req: GroundRequest) -> tuple[KB, GroundRequest]:
        # query-DEPENDENT but CHEAP: only the magic SEED facts depend on the query.
        seeds = magic_seeds(self._adorned_rules, req.queries)                 # [Nseed,3]
        kb2 = self._skeleton_kb.with_program(facts=seeds)                     # FACTS-only union path
        return kb2, req
```

- **The cost split (this is the fix to v1's buried open-question).** Adornment GENERATES new rules; that is query-independent given the adorned skeleton, so it is done ONCE at construction (one `with_program(heads,bodies,lens)` → one PbcRuleIndex/binding-table rebuild). Per call, only magic SEED facts depend on the query, so the hot path is `with_program(facts=...)` — the existing cheap `with_closure` shape (union + re-index facts, no rule re-tensorize). This turns the per-call cost from "rebuild all pbc buffers" into "facts-only re-index", which `KB` already supports.
- **Where it interposes (corrected).** It CONSUMES `RulePattern` binding info to choose adornments; it PRODUCES raw rule tensors consumed by `KB.with_program`, which re-derives fresh `RulePattern`s. It sits between rule-tensors and KB construction, not "at RulePattern" (v1's claim was imprecise; `pattern.py` analyzes a fixed rule, it generates nothing).
- **Composition:** `make_grounder(kb, PBCConfig(...), transforms=[MagicSetTransform(kb)])` returns `Pipeline([t], BackwardGrounder(skeleton_kb, ...))`. `Pipeline.ground(req)` runs `t.apply` per call (so query-dependent seeds are honored — what build-once `RunPlan` cannot do), then `base.rebound(kb2).ground(req2)`. `rebound` re-snapshots `RunPlan`; because the rule skeleton is unchanged, the pbc binding tables are reused via the `KB.with_program` index cache (keyed on the rule-program signature, §3).
- **Reuses unchanged:** execution, `BackwardResult`, the 14/14 gate, the depth loop. Composes with `ForwardGrounder` too (query-directed FC for free).

### 2b. The enabling primitive — `KB.with_program` + index cache (in scope)

`KB.with_program(*, facts=None, heads=None, bodies=None, lens=None) -> KB` returns a NEW KB; `with_closure(f)` becomes `with_program(facts=f)`. Index build stays in `KB.__init__` (so a rewrite re-derives `SldRuleIndex`; pbc's `PbcRuleIndex` is built by the grounder on `rebound`). To stop magic-set re-tensorizing pbc buffers every call, `with_program` accepts the rule program's content signature and a module-level memo returns the cached `SldRuleIndex`/binding tables when only `facts` changed. This is part of v1 of the transform axis, not deferred — without it `MagicSetTransform` is correctness-neutral but performance-fatal on the tiny-batch case it targets.

### 2c. Why FC firings are net-new (not a bridge)

`Closure.as_rule_groundings()` returns `None` in v1. If FC firings are ever required, the honest path is a NEW `forward/firings.py` that records `(rule_idx, head, body-grounding, query_idx)` from FCDynamic's per-rule emissions (`fc.py` `_apply_rule`/`_run_stages` iterate rules and could log the instantiation) and then reuses the SINGLE dedup owner `engine/considered.finalize()` (`considered.py`) to emit a `RuleGroundings` — satisfying the "one dedup owner" principle. This is budgeted as plumbing in `fc.py` + `spmm/runner.py` behind its own A/B, NOT a rename of `witnesses.py` (which is deleted). It is out of scope for v1.

### 2d. Worked example — JOIN-enum L3 (axis 1, a sibling `JoinResolver` — NOT a Materializer)

v1 modeled this as a `JoinMaterializer` that "caps during the join (top-k each row to G_r)" claiming byte-identity. **That is the wrong operation.** Verified `resolution/pbc/width.py`: the BC width filter is `num_unknown = (body_active & ~exists).sum(-1) <= width` — a per-grounding UNKNOWN-ATOM-COUNT predicate (`width.py:71`) PLUS `head_pred_all_ok` (every active body atom is fact OR head-pred-provable, `width.py:44-50`) PLUS query-exclusion, all computed AFTER full candidate enumeration. A mid-join top-k by cardinality changes WHICH groundings exist before those count/head-pred predicates run, so it cannot reproduce the survivor set; `pbc/__init__.py:8` itself flags "NO L3 join … fingerprint-gated follow-up".

- **Correct placement:** `JoinResolver` registered as `RESOLVERS["join"]`, a SEMANTICS-PRESERVING optimization (AGM/WCO-bounded enumeration that must still reproduce the exact width + head-pred + query-exclusion survivor SET). The width predicates are pushed INTO the join as branch pruners that only drop branches *provably* exceeding `width` given the remaining atoms — never a top-k.
- **Gate (different from pbc):** NOT byte-identity. A new `tests/join_ab.py` asserting SET-EQUALITY of emitted groundings vs `pbc` per (dataset, w, d, u) + a measured peak-memory test. pbc stays byte-identical and untouched.
- **Selection:** `RESOLVERS["join"]` is chosen by config (`PBCConfig` gains an opt-in `materialization="cartesian"|"join"` or a sibling `JoinConfig`); it declares its own `Cell`s via `Resolver.declared_cells()`.

## 3. The abstractions (real signatures)

All protocols are structural (`runtime_checkable` where cheap). `nn.Module`-ness is an impl detail of concrete grounders, NOT part of the protocol.

```python
# core/grounder.py — THE minimal common abstraction
@runtime_checkable
class Grounder(Protocol):
    kb: KB
    def ground(self, request: GroundRequest) -> GroundResult: ...
    def capability_row(self) -> CapabilityRow: ...          # exec axis (BackwardGrounder already has it)
    def producible_tiers(self) -> frozenset["Tier"]: ...    # which tiers this family can fill
    def rebound(self, kb: KB) -> "Grounder": ...            # re-snapshot over a rewritten KB (transforms)

# core/request.py — honest unification of the two run shapes
@dataclass(frozen=True)
class GroundRequest:
    queries: Optional[Tensor] = None        # [B,3]; None => data-directed (FC)
    query_mask: Optional[Tensor] = None     # [B]
    output_spec: "OutputSpec" = OutputSpec() # BC tiers; FC reads only its own ClosureSpec fields
    excluded_queries: Optional[Tensor] = None
    closure_depth: Optional[int] = None     # FC-only depth override (None for BC)
    def requires_queries(self) -> bool: ...

# core/result.py — TWO-member tagged union behind a marker protocol
@runtime_checkable
class GroundResult(Protocol):
    kind: str                                       # "backward" | "closure"
    def as_rule_groundings(self) -> Optional["RuleGroundings"]: ...  # BC: the firings; FC: None (v1)

@dataclass(frozen=True)
class BackwardResult:                  # == today's GrounderOutput, renamed + tagged
    state: ProofState                                      # tier PROOF_STATE (always-on; DpRL)
    rule_groundings: Optional[RuleGroundings] = None       # tier FIRINGS (torch-ns run_bc)
    completed_tree_firings: Optional["CompletedTreeFirings"] = None  # tier TREES (probfol)
    kind: str = "backward"
    def as_rule_groundings(self): return self.rule_groundings

@dataclass(frozen=True)
class Closure:                         # FC's honest result (today: bare (hashes,n))
    hashes: Tensor                                        # [N] sorted, h = pred*E^2 + s*E + o
    n_provable: int
    num_entities: int                                     # E, for decode (ForwardGrounder._num_entities)
    kind: str = "closure"
    def facts(self) -> Tensor: ...                        # decode -> [N,3] (== old closure_facts())
    def contains(self, atoms: Tensor) -> Tensor: ...
    def as_rule_groundings(self): return None             # FC has NO provenance today (verified)

# resolution/api.py — AXIS 1 (replaces engine/step.py:resolve() if/elif)
@runtime_checkable
class Resolver(Protocol):
    name: str                                       # "sld" | "rtf" | "pbc" | "join"
    def declared_cells(self) -> FrozenSet[Cell]: ...           # contributes to CapabilityRow
    def resolve(self, req: "ResolveRequest") -> "ResolvedChildren | FlatResolvedChildren": ...

class ResolveRequest(NamedTuple):                   # generalize StepInputs to the seam; honest superset
    queries: Tensor; remaining: Tensor; grounding_body: Tensor
    state_valid: Tensor; active_mask: Tensor; padding_idx: int
    depth_selector: "DepthSelector"                 # NOT a bare int (compiled path expressible)
    width: object; w_last_depth: int; excluded_queries: Optional[Tensor]
    collect_evidence: bool
    fact_hook: object = None; rule_hook: object = None  # Optional: wired for sld/rtf; INERT for pbc today

# resolution/pbc/ — the Materializer SUB-axis stays pbc-internal (dense|flat|join); NOT on Resolver.
# sld/rtf use plain function pairs (resolve_sld vs resolve_sld_flat); not forced into Materializer objects.

# forward/api.py — AXIS 3 (replaces run_forward_chaining method= dispatch)
@runtime_checkable
class ForwardMethod(Protocol):
    name: str                                       # "spmm" | "staged"
    def supports(self, rules: list[RulePattern]) -> bool: ...   # spmm: all classify_rule != UNSUPPORTED
    def run(self, rules, facts_idx, num_entities, num_predicates,
            *, depth: int, device: str, join_algo: str, join_chunk_size: int) -> Closure: ...
# SUB-axis JoinAlgo {staged, chunked} lives INSIDE StagedMethod (FCDynamic.join_algo).
# leapfrog is a documented placeholder (fc.py:951) — NOT registered as an extension point until finished.
# run_forward_chaining stays a thin per-rule ROUTER: spmm for supported rules, fall-through to staged
# for any UNSUPPORTED rule (fc.py:1188-1211) — supports()->bool gates the whole-set spmm fast path.

# transform/api.py — AXIS 4 (NEW; result-preserving pre-stage)
@runtime_checkable
class ProgramTransform(Protocol):
    name: str
    def apply(self, kb: KB, req: GroundRequest) -> tuple[KB, GroundRequest]: ...   # pure; order-significant
class IdentityTransform:
    name = "identity"
    def apply(self, kb, req): return kb, req

# core/pipeline.py — Transform∘Grounder is itself a Grounder
class Pipeline:
    def __init__(self, transforms: Sequence[ProgramTransform], base: Grounder): ...
    def ground(self, req: GroundRequest) -> GroundResult:
        kb = self.base.kb
        for t in self.transforms: kb, req = t.apply(kb, req)
        return self.base.rebound(kb).ground(req)
    def capability_row(self): return self.base.capability_row()
    def producible_tiers(self): return self.base.producible_tiers()

# Registries — one decorator-keyed table per seam (each has ≥2 real impls)
RESOLVERS:       dict[str, Resolver]          # "sld","rtf","pbc" (+"join")
FORWARD_METHODS: dict[str, ForwardMethod]     # "spmm","staged"
TRANSFORMS:      dict[str, type]              # "magic_set","identity"

# data/kb.py — the enabling change (today only with_closure unions facts, kb.py:150)
class KB(nn.Module):
    def with_program(self, *, facts=None, heads=None, bodies=None, lens=None,
                     index_cache_key: Optional[Hashable]=None) -> "KB": ...
    def with_closure(self, closure_facts) -> "KB": return self.with_program(facts=closure_facts)
```

## 4. The vocabulary (canonical terms; one per concept)

| canonical | meaning |
|---|---|
| **grounder family** | one of `{BackwardGrounder, ForwardGrounder}`; differs by algorithm class, returns its own `GroundResult` variant. |
| **resolution / resolver** | AXIS 1: backward RESOLVE-step strategy `{sld, rtf, pbc, join}`; the `Resolver` protocol. Reserved for backward. |
| **pbc** | Parametrized Backward Chaining (paper `BC_{w,d,u}`); canonical name. `enum` is a factory parser-only alias (`factory.py:56`). |
| **materializer** | pbc-internal per-step LAYOUT strategy `{dense, flat, join}`. Sub-axis of pbc only; NOT on the `Resolver` protocol; sld/rtf don't use it. |
| **method / forward method** | AXIS 3: closure ENGINE `{spmm, staged}`; the `ForwardMethod` protocol. Distinct word from `resolution`. |
| **join_algo** | staged-method sub-axis: per-rule join `{staged, chunked}` (leapfrog = unfinished placeholder). Mirrors materializer. |
| **transform / program transform** | AXIS 4: pure `(kb,req)->(kb',req')` rewrite `{magic_set, identity}`; the `ProgramTransform` protocol. |
| **ground / GroundRequest / GroundResult** | the single runtime verb + its request (queries Optional) + its 2-member tagged-union result. |
| **firing** | one fired rule application `(rule_idx, head, body)`; the unit accumulated into `FiringSet`. Replaces the orphan concept **considered**. |
| **FiringSet** | run-scoped accumulator of all firings — PRIMARY for `RuleGroundings`. |
| **RuleGroundings** | the BC CSR firings tier (torch-ns run_bc). BACKWARD-ONLY today. KEEP name. |
| **completed-tree firings** | fired rule apps inside completed proof trees (~3× undercount); the TREES tier. Was the type `ProofEvidence` AND the field `evidence` AND the concept `evidence` — collapsed to one name. |
| **Closure** | the forward result type (provable triple hashes + n + E); a named dataclass, not a bare Tensor. |
| **Cell / CapabilityRow / ExecStrategy** | the orthogonal `(layout, compile)` execution axis. BC consumes fully; FC shares `Cell`/`CapabilityRow` + `ChunkPolicy` only. KEEP. |
| **layout** | `{dense, flat, sparse}`; `sparse` is forward-only and never produces `ResolvedChildren`. |
| **filter** | TWO live filters: **fp_batch** (the only SOUNDNESS filter, Kleene T_P) and **prune_facts** (search-time candidate prune, `step.py:167`). |
| **make_grounder** | the ONE construction entry; dispatches on the existing typed config (`SLDConfig|RTFConfig|PBCConfig|FCConfig`). |
| **run_backward / run_forward_chaining** | the two family drivers (`backward/loop.py`, `forward/fc.py`). |

### 4a. Current → canonical rename table (load-bearing diffs only)

| current (in `_build/`) | canonical | kind |
|---|---|---|
| `engine/` (package) | `backward/` | package — symmetry with `forward/`; matches `run_backward` |
| `ProofEvidence` (type) + `evidence` (field) + `evidence` (GLOSSARY concept) | `CompletedTreeFirings` (type) + `completed_tree_firings` (field) + one concept | type+field+concept — one name; field tracks type |
| `considered` (orphan GLOSSARY concept) | `firing` (cross-reference legacy "considered") | concept |
| `StepInputs` (`pbc/resolve.py`) | `ResolveRequest` (in `resolution/api.py`; carries `DepthSelector`, Optional hooks) | type |
| `engine/step.py:resolve()` if/elif | `RESOLVERS[plan.resolution].resolve(req)` | dispatch → registry |
| `run_forward_chaining` `method=` dispatch | `FORWARD_METHODS[name].run(...)` (router keeps per-rule spmm→staged fallback) | dispatch → registry |
| `ForwardGrounder.forward() -> Tensor` | `ForwardGrounder.ground(req) -> Closure` (+ `.closure_facts()` kept) | method/type |
| `BackwardGrounder.forward(q,mask) -> GrounderOutput` | `BackwardGrounder.ground(GroundRequest) -> BackwardResult` | method/type |
| `GrounderOutput` | `BackwardResult` (tagged) | type |
| `(no FC result type — bare (hashes,n))` | `Closure` | type |
| `create_grounder` / `make_bcwd` / `BackwardGrounder(kb,…)` / `ForwardGrounder(kb,…)` | shims over `make_grounder(kb, config, *, exec-knobs, transforms=())` | entry |
| `RunPlan.config` field (unconditionally None, `plan.py:104`) | DELETE the field, or wire the existing typed config in `make_grounder` | config |
| `config.py` typed configs (UNWIRED, not vestigial) | WIRE into `make_grounder`; KEEP the per-family typing; do NOT add a flat `GrounderConfig` | config |
| `FCConfig.method` comment "spmm \| triejoin" + missing `join_algo` | `method ∈ {spmm,staged}` + add `join_algo ∈ {staged,chunked}` | config |
| factory regex filter aliases `fp_global`/`prune`/`provset` (`factory.py:24,30`) | `prune`→`fp_batch` kept; `fp_global`/`provset`/`closure`→ route to `ForwardGrounder` family, not a BC filter | value |
| resolution value `"enum"` | `"pbc"` (parser-only alias) | value |
| resolution value `"closure"` (`factory.py:95`) | the `ForwardGrounder` family, not a resolution | value |
| `forward/closure.py` + `forward/witnesses.py` (dead fp_global; only self/closure refs) | DELETE (no live importers; no A/B needed) | code |
| `nesy/hooks.py` wired in sld/rtf only; `StepHook`/`GroundingHook` unwired | wired across all resolvers (default-None, inert) — own A/B | wiring |

> **NOT renamed (v1 errors corrected):** `T` is NOT added to `SHAPES` — `shapes.py:4` deliberately excludes data-dependent counts and `test_vocabulary.test_shape_symbols_match_vocab` requires exact `Shapes`↔`SHAPES` equality, so the v1 item would self-break the contract. `prune_facts` is NOT deleted (live filter). `FilterMode` in `config.py` already has only `NONE`/`FP_BATCH` (v1's "delete prune/provset/fp_global from config.py" was a phantom — those live in the factory regex, not the enum). `ScoredResult`/`LearnedGrounder`/SCORES are NOT introduced.

## 5. Type hierarchy + the output/result contract

Three strata by lifetime:

- **(A) DATA / SUBSTRATE** (shared, immutable, resolution-agnostic): `KB`, `Encoding`, `FactIndex`, `RuleIndex`, `RulePattern`, `Shapes`. A transform produces a NEW KB via `KB.with_program` (§2b).
- **(B) PER-RUN PLAN/STATE** (private, BC-only): `RunPlan` (immutable snapshot, `plan.py`), `Frontier` (chunk tape), `RunState` (run accumulators keyed by tier — the existing `FiringSet`/`ProofTrees` pattern, `state.py:121-145`), `DepthSelector`, `ExecStrategy`.
- **(C) RESULT / OUTPUT** (public, frozen): the 2-member union in §3.

KEEP field-for-field (consumer-locked; registered in `glossary.FIELDS`): `ProofState`, `RuleGroundings` (`atom_table/body_pool_idx/head_pool_idx/rule_offsets/rule_idx`, `.empty()`, `.rule_slice()`, `types.py:76-108`), and the compile-safe seams `ResolvedChildren`/`FlatResolvedChildren`/`PackedStates`/`SyncParams` (`types.py:121-168`). Renames: `GrounderOutput→BackwardResult`, `ProofEvidence→CompletedTreeFirings` (identical fields), the field `evidence→completed_tree_firings`, and the `ProofEvidence.top_rule_idx` per-tree @property → `tree_top_rule_idx` (resolving the [B,S]-field vs [B,C]-property overload, `types.py:35` vs `types.py:67`); the flat abbreviation `flat_top_ridx` → register as the one sanctioned flat spelling. Each rename keeps a one-window alias.

**OutputSpec generalizes from 3 bools to a typed tier set — but stays honestly BC-shaped.** Today `OutputSpec{groundings, firings, trees}` (`state.py:33-35`) is BC-only with `needs_provenance()` read in plan/step/loop/finalize.

```python
class Tier(StrEnum):  PROOF_STATE="proof_state"; FIRINGS="firings"; TREES="trees"
@dataclass(frozen=True)
class OutputSpec:
    tiers: FrozenSet[Tier] = frozenset({Tier.PROOF_STATE})   # field named 'tiers' (NOT 'cells' — collides
                                                             # with CapabilityRow.cells, capability.py:89)
    def needs_provenance(self): return bool(self.tiers & {Tier.FIRINGS, Tier.TREES})
    def wants(self, t: Tier): return t in self.tiers
    # back-compat @property groundings/firings/trees for ONE migration window
```

`Tier` is the BACKWARD tier set only (no overloaded `PROOF_STATE`-means-three-things, no FC `PROVABLE_SET`, no `SCORES`). FC does not use `Tier`; `Closure` is its single product, with `GroundRequest.closure_depth` its only knob. `Grounder.producible_tiers()` returns `{PROOF_STATE, FIRINGS, TREES}` for BC and `frozenset()` for FC (FC produces a `Closure`, not tiers); `ground()` raises `ConfigError` on a tier the family cannot fill — never a silent empty. This keeps the request honest the same way the result is: BC asks for tiers, FC asks for a closure depth, unified only by the `GroundRequest` envelope (mirroring `queries: Optional`).

**Why not one mega-dataclass:** it would carry `proof_goals` on FC results (always `None`) and `hashes` on BC results (always `None`) — a union lying about every grounder. Protocol + concrete types keeps each honest; `as_rule_groundings()` is the one optional bridge (BC: the firings; FC: `None`).

## 6. Final module tree

`backward/` (renamed from `engine/`) and `forward/` are peer algorithm packages; both consume `core/` + `data/` + `execution/`. `resolution/` is the AXIS-1 seam and stays TOP-LEVEL (it is the shared seam contract: the magic-set transform's adornment reads `RulePattern`, and a future `JoinResolver` registers here) — it is NOT buried under `backward/` (v1 contradicted itself; resolved here). `transform/` is the AXIS-4 seam (shared).

```
grounder/                              (promoted from _build/)
├── __init__.py                        re-exports; make_grounder is THE entry; both families eager
├── factory.py                         make_grounder(kb, config, *, layout, compile, chunk_size,
│                                         transforms=()) -> Grounder
│                                         + parse_type_string / make_bcwd / create_grounder shims
├── glossary.py                        extend GLOSSARY + FIELDS; collapse evidence/considered; add seam nouns
├── config.py                          KEEP the typed SLD/RTF/PBC/FCConfig; WIRE into make_grounder;
│                                         add FCConfig.join_algo
├── core/                              NEW — cross-family contracts (shared)
│   ├── grounder.py                    Grounder Protocol (ground/capability_row/producible_tiers/rebound)
│   ├── request.py                     GroundRequest, OutputSpec, Tier
│   ├── result.py                      GroundResult Protocol + BackwardResult
│   └── pipeline.py                    Pipeline (Transform∘Grounder is a Grounder)
├── grounder/                          family SHELLS
│   ├── backward.py                    BackwardGrounder (Resolver-parametrized; +ground/+rebound)
│   ├── forward.py                     ForwardGrounder (ForwardMethod-parametrized; +ground->Closure)
│   └── registry.py                    RESOLVERS / FORWARD_METHODS / TRANSFORMS + @register
├── transform/                         NEW — AXIS 4 (shared; composes with any family)
│   ├── api.py                         ProgramTransform Protocol + IdentityTransform + adorn (one owner)
│   └── magic_set.py                   MagicSetTransform (construction-time skeleton + per-call seeds)
├── resolution/                        AXIS 1 (TOP-LEVEL; the shared seam contract)
│   ├── api.py                         Resolver + ResolveRequest Protocols
│   ├── primitives.py mgu.py
│   ├── sld.py rtf.py                  SldResolver / RtfResolver (wrap existing fns)
│   └── pbc/                           PbcResolver + Dense/Flat/JoinMaterializer (materializer SUB-axis here)
├── backward/                          (renamed engine/) shared backward runtime; resolution-agnostic
│   ├── loop.py(run_backward) step.py pack.py sync.py considered.py postprocess.py finalize.py buffers.py
│   ├── state.py                       Frontier/FiringSet/ProofTrees/RunState   (BC-only)
│   └── plan.py                        RunPlan                                    (BC-only)
├── forward/                          AXIS 3 (closure engine)
│   ├── api.py                         ForwardMethod Protocol + Closure
│   ├── methods.py                     SpmmMethod/StagedMethod; run_forward_chaining = per-rule router
│   ├── fc.py spmm/                    (JoinAlgo staged/chunked inside StagedMethod; spmm IterationStrategy
│   │                                   stays a SUB-strategy in SpmmMethod)
│   └── (closure.py, witnesses.py DELETED — dead fp_global code)
├── execution/                        ORTHOGONAL axis; BC consumes fully, FC shares Cell/Row + ChunkPolicy
│   └── strategy.py capability.py auto_select.py compiler.py depth.py chunk_policy.py cudagraph.py
├── data/                             SHARED substrate; KB.with_program NEW (with_closure delegates)
│   └── kb.py encoding.py fact_index/ rule_index/(pattern.py = sole binding-analysis owner)
├── filters/                          TWO live filters
│   └── fp_batch.py(soundness) prune_facts.py(search-time)
├── nesy/hooks.py                     4 hook Protocols, wired across all resolvers (the learned-scoring seam)
└── contracts/                        test_vocabulary(+GLOSSARY/FIELDS disjointness) test_execution test_data
                                       + test_seams(NEW: registry entries impl their Protocol; no if/elif)
                                       + test_results(NEW: each GroundResult member impls as_rule_groundings)
```

**Single entry point.** `make_grounder(kb, config, *, layout="auto", compile="off", chunk_size=None, transforms=()) -> Grounder`: (1) build the family grounder over `kb` by dispatching on `type(config)` (`PBCConfig|SLDConfig|RTFConfig`→`BackwardGrounder`; `FCConfig`→`ForwardGrounder`) + exec knobs; (2) wrap in `Pipeline` iff `transforms` non-empty. `create_grounder(type_str, **raw_tensors)` = build-KB-from-raw + parse a config + `make_grounder`; `make_bcwd` builds a `PBCConfig` and delegates. This collapses the BC/FC string fork at `factory.py:95` and wires the previously-dead typed configs.

**Shared vs specific.** SHARED: `core/`, `data/`, `execution/`, `transform/`, `resolution/api.py`, `glossary`/`shapes`/`contracts`, `filters/`, `nesy/`. BACKWARD-SPECIFIC: `backward/*` (loop/step/pack/sync/plan/state), `ProofState`/`CompletedTreeFirings`/`BackwardResult`/`RuleGroundings`, `Resolved*Children`. FORWARD-SPECIFIC: `forward/*` (`Closure`, methods, spmm). COMPOSED: `Pipeline`.

## 7. Naming conventions

- **Families:** `<Direction>Grounder` — `BackwardGrounder`, `ForwardGrounder`. Never name by resolution/method.
- **Result types — ONE axis (domain noun), applied uniformly:** `BackwardResult` and `Closure` are both domain nouns (the proof-bundle vs the T_P closure). The union `GroundResult` is tagged by `.kind`. (v1's mixed family/domain scheme is rejected; a reader predicts the name from the domain.)
- **Seam protocols:** bare role nouns `Grounder`, `Resolver`, `ForwardMethod`, `ProgramTransform`, `Materializer`. Concrete impls add the algorithm prefix: `SldResolver`, `PbcResolver`, `JoinResolver`, `SpmmMethod`, `StagedMethod`, `MagicSetTransform`.
- **The verbs:** `ground` (grounder), `resolve` (resolver), `run` (forward method), `apply` (transform); builders `build_<thing>`/`init_<thing>`; the single compile site `Compiler.wrap`; the single cudagraph marker `cudagraph_mark_step_begin`.
- **Registries:** uppercase module-level dicts `RESOLVERS`, `FORWARD_METHODS`, `TRANSFORMS`, populated by `@register("name")`.
- **Construction:** `make_grounder(kb, config, ...)` canonical; `create_grounder`/`parse_type_string`/`make_bcwd` thin shims. Paper letters `w/d/u` survive as `PBCConfig` fields (a sanctioned glossary exception for `BC_{w,d,u}` reproducibility).
- **Packages:** name == concept owned == driver suffix. `backward/` hosts `run_backward`; `forward/` hosts `run_forward_chaining`; `transform/` hosts transforms. No `*_new`/`*_v2`/`*_copy`.
- **Glossary discipline:** every new field name (`queries`, `query_mask`, `kind`, `hashes`, `n_provable`, `closure_depth`, `tiers`, `completed_tree_firings`, `tree_top_rule_idx`) registered in `glossary.FIELDS`; every new concept registered in `GLOSSARY`. A NEW contract asserts GLOSSARY-keys and FIELDS-keys are token-disjoint (or, where a token legitimately spans both, that the two definitions cross-link) — catching the `evidence` concept/field collision and any future one. `T` is NOT registered (not a static shape symbol, not a named field).


---

# Appendix A — Decision log

- **GroundResult is a TWO-member union (BackwardResult | Closure); Closure.as_rule_groundings() returns None; FIRINGS is BACKWARD-ONLY.**
  - why: Verified blocker: forward/witnesses.py is fp_global BACKWARD machinery — build_witness_table(grounder,...) mutates a BackwardGrounder, reads grounder._enum_ri / fp_global_hashes (witnesses.py:31-204), and is invoked ONLY from forward/closure.py:113. It does NOT take a Closure and does NOT produce a RuleGroundings CSR. ForwardGrounder.closure_hashes returns only (sorted_hashes,n) (forward/grounder.py:47-53; forward/fc.py:1156). There is no FC->RuleGroundings adapter. v1 also deleted the very files it leaned on. So the bridge is fiction; FC has no provenance.
  - rejected: v1's 'RuleGroundings is the ONE cross-family bridge, produced by BC AND FC (witnesses.py)'. A future FC firings path is net-new plumbing (forward/firings.py emitting into the single considered.finalize dedup owner), behind its own A/B — out of scope for v1, NOT a rename of witnesses.py.
- **Build an FC byte-identity gate (tests/fc_fingerprint.py) as refactor step 0b, BEFORE any FC-touching change.**
  - why: Verified blocker: grep -rln for any FC symbol across tests/ returns nothing; tests/ holds only fingerprint_new.py (BC), flat_ab.py (BC sld/rtf), nesy_ab.py. The 'closure-set A/B oracle' the forward/__init__.py:5 docstring claims, and that v1 leaned on for the Closure/ForwardMethod/with_program steps, DOES NOT EXIST. Three FC-touching steps would be unverifiable.
  - rejected: v1 treating the FC A/B as already-existing and an FC fingerprint cell as an optional open-question.
- **Forward axis = ForwardMethod {spmm, staged} (engine) with a JoinAlgo sub-axis {staged, chunked} inside StagedMethod; run_forward_chaining stays a per-rule ROUTER (spmm fast path with fall-through to staged for UNSUPPORTED rules).**
  - why: Verified: run_forward_chaining has method=spmm|staged AND join_algo=staged|chunked|leapfrog (fc.py:1153-1155); method=='spmm' is a whole-set precheck all(classify_rule!=UNSUPPORTED) then per-rule fallback (fc.py:1188-1211). leapfrog is a placeholder that falls through to _run_stages (fc.py:951-968). So it is two sub-axes mirroring Resolver x Materializer, plus a per-rule router — not a flat method list, and supports(rules)->bool alone can't express per-rule routing.
  - rejected: v1's flat ForwardMethod {spmm,staged,datalog,triejoin}: it minted a staged/staged synonym, mislabeled leapfrog as 'triejoin', dropped chunked/leapfrog, and listed two unbuilt methods (datalog/triejoin) in the canonical glossary.
- **Join-enum L3 is a sibling JoinResolver (RESOLVERS['join']) gated by SET-EQUALITY (tests/join_ab.py) + peak-mem; it does NOT claim byte-identity and does NOT touch pbc.**
  - why: Verified: BC width is num_unknown=(body_active & ~exists).sum(-1)<=width (width.py:71) + head_pred_all_ok (width.py:44-50) + query-exclusion, all computed AFTER full enumeration. A mid-join top-k by cardinality changes which groundings exist before those predicates, so it cannot reproduce the survivor set byte-identically. pbc/__init__.py:8 itself flags 'NO L3 join — fingerprint-gated follow-up'. The width predicates are pushed into the join as branch pruners (only drop provably-over-width branches), preserving the count semantics.
  - rejected: v1's JoinMaterializer that 'caps during the join (top-k each row to G_r)' claiming byte-identity — wrong operation; would break the fingerprint and flat-vs-dense oracle.
- **Honor config.py's typed configs: WIRE SLD/RTF/PBC/FCConfig into make_grounder (dispatch on config TYPE); do NOT introduce a flat GrounderConfig; delete-or-populate the dead RunPlan.config field.**
  - why: config.py:3-4 explicitly states the design: 'There is no Resolution enum and no GrounderConfig wrapper: each grounder family carries exactly its own parameters, and the factory dispatches on config type.' RunPlan.config is typed Optional[BackwardConfig] (plan.py:45) but unconditionally None because grounder never sets self.config (plan.py:104). It is UNWIRED, not vestigial. The genuinely-missing step is wiring, not replacement; the leaf filter() owner (PBC->fp_batch, sld/rtf->none) is real and worth keeping.
  - rejected: v1's 'fold the vestigial config.py into a flat GrounderConfig and wire it into RunPlan' — reintroduces exactly the wrapper config.py consciously rejected, erases per-family typing, and adds a field to the byte-identity-critical RunPlan for no consumer. Either delete RunPlan.config or populate it from the existing typed config — a one-liner.
- **Do NOT register T in SHAPES/Shapes.**
  - why: shapes.py:4 deliberately excludes data-dependent counts ('Data-dependent counts (S_out, T) are NOT owned here'), and contracts/test_vocabulary.test_shape_symbols_match_vocab asserts exact Shapes<->SHAPES equality (test_vocabulary.py:86-96). Adding T to SHAPES without Shapes fails the contract; adding it to Shapes pollutes the static registry with a per-run count. T is also not a named field on any dataclass (FlatResolvedChildren uses no 'T' field).
  - rejected: v1's 'register T in SHAPES + Shapes' — a self-inflicted contract failure presented as a vocabulary fix.
- **Keep BOTH live filters; correct 'fp_batch is the only filter' to 'fp_batch is the only SOUNDNESS filter; prune_facts is a search-time candidate prune'.**
  - why: Verified: filters/prune_facts.py:prune_ground_facts is imported and called in the hot path (engine/step.py:21,167-173), gated by RunPlan.prune_facts (plan.py:65), exposed as a BackwardGrounder ctor kwarg (backward.py:63) and used by fingerprint_new.py:55/62 and flat_ab.py:40. Calling fp_batch 'the ONLY filter' is false and would mislead the deletion pass.
  - rejected: v1's repeated 'the ONLY filter is fp_batch' across principles/module-tree/refactor-gate.
- **Defer LearnedGrounder / ScoredResult / a SCORES tier entirely; keep GroundResult = BackwardResult | Closure.**
  - why: The whole per-run stack is BC-shaped (RunPlan reads ~30 BC attrs plan.py:79-121; Frontier is a proof tape state.py:43; run_backward is the only non-FC driver). A learned grounder is most honestly a scoring overlay on an existing family via the existing nesy/hooks.py protocols, not a third family with its own engine. Reserving a union member with zero impls is a guaranteed future breaking change (v1's own open-question #7).
  - rejected: v1 reserving ScoredResult/LearnedGrounder/Tier.SCORES as the justification for a tagged union — an unbuilt thing justifying the union's shape, with its real seam (nesy hooks) left unaddressed.
- **OutputSpec generalizes to FrozenSet[Tier] named 'tiers' (Tier = BACKWARD tiers only: PROOF_STATE/FIRINGS/TREES); FC uses GroundRequest.closure_depth, not Tier; producible_tiers() per family; ConfigError on an unproducible tier.**
  - why: Today OutputSpec is BC-only 3 bools (state.py:33-35). A typed BC tier set is honest; overloading PROOF_STATE across three families (v1) re-introduces on the request side the same dishonesty v1 rejects for a mega-result. The field MUST NOT be named 'cells' — CapabilityRow.cells already exists (capability.py:89), a same-codebase synonym the glossary forbids; use 'tiers'. Keep groundings/firings/trees @property for one window, then drop with the bools to satisfy test_no_dead_vocab_entries (test_vocabulary.py:55).
  - rejected: v1's overloaded Tier {PROOF_STATE='groundings',...,PROVABLE_SET,SCORES} with StrEnum values reusing the legacy bool field names, and the field named 'cells'.
- **Delete forward/closure.py AND forward/witnesses.py with NO A/B (dead code), not behind an oracle.**
  - why: Verified: closure.py has no live importers (only forward/__init__.py:5 mentions it in a docstring); witnesses.py is imported only by closure.py:113. Both are fp_global machinery; glossary.py:30 already declares fp_global GONE. Deleting unreachable code needs no oracle (and the FC A/B does not exist anyway).
  - rejected: v1/D8's 'closure.py is a correctness landmine that needs its own A/B before removal' — it is unreachable dead code.
- **resolution/ stays TOP-LEVEL (the shared seam contract); only engine/->backward/ moves, in a dedicated last commit with one-window aliases.**
  - why: v1 simultaneously called resolution/api.py 'the shared seam contract' AND moved resolution/ under backward/ — self-contradictory. The magic-set transform's adorn() reads RulePattern and a future JoinResolver registers in resolution/, so it is shared, not BC-only. engine/ IS BC-only (engine/__init__ imports nothing from forward/) so engine->backward is defensible. External consumers (probfol, torch-ns) import GrounderOutput/ProofEvidence, so renames keep aliases >=1 release and the SHA-pin cascade coordinates consumer bumps.
  - rejected: v1's 'move resolution/ + plan.py + state.py under backward/'.
- **Add a GLOSSARY-vs-FIELDS token-disjointness contract; collapse evidence (concept+type+field) to one name and the orphan concept 'considered' into 'firing'.**
  - why: Verified collisions test_vocabulary cannot currently catch (it checks FIELDS membership + Shapes==SHAPES + no-dup-literal-keys only): 'evidence' is both a GLOSSARY concept (glossary.py:28) and a FIELDS field (glossary.py:98) with divergent meanings; 'considered' (glossary.py:27) is an orphan concept = 'PRIMARY for RuleGroundings' that duplicates the 'firings' tier; ProofEvidence.top_rule_idx is a [B,C] @property (types.py:67) overloading the [B,S] field (types.py:35) and the flat_top_ridx abbreviation (types.py:144).
  - rejected: v1 renaming only the type ProofEvidence while leaving the field 'evidence', the concept 'evidence', and 'considered' untouched — three words for in-tree firings plus a new 'witness' near-synonym.
- **ResolveRequest carries Optional hooks, explicitly inert for pbc; do NOT claim it 'bundles exactly today's kwargs'.**
  - why: Verified asymmetry: sld/rtf resolvers take fact_hook/rule_hook (step.py:83,92) but pbc resolve_step takes none (step.py:118-123); StepInputs carries no hooks. A unified ResolveRequest with hook fields is a real interface unification (hooks Optional, inert for pbc in v1), not a pure rename.
  - rejected: v1's refactor-step-1 risk note 'ResolveRequest must bundle exactly today's kwargs' — the sld/rtf and pbc kwarg sets differ.


# Appendix B — Gap analysis

ALREADY ALIGNED (keep): (1) data/ is the genuinely shared, resolution-agnostic substrate — KB immutable with with_closure returning a new KB (kb.py:150-160), Encoding single id source, fact_index/rule_index, RulePattern the SOLE binding-analysis owner (pattern.py) consumed by pbc and forward. (2) execution/ is the one clean orthogonal axis — Cell/CapabilityRow/validate (capability.py), ExecStrategy single-owner of compile/layout/chunk/cudagraph (strategy.py), single Compiler.wrap site, DepthSelector duality owner. BackwardGrounder already declares capability_row() (backward.py) and the two-knob ctor surface is wired+validated at snapshot (plan.py:86-102). (3) glossary.py + contracts/test_vocabulary.py is a working controlled-vocab chokepoint (FIELDS membership + Shapes==SHAPES + no-dup-literal-keys). (4) BackwardGrounder is ALREADY a single config-parametrized class (backward.py:36, resolution kwarg) — subclassing was abandoned; the registry promotion fits the shipped reality. (5) The pbc materializer pattern (pbc/resolve.py resolve_step + Dense/Flat materializers) is the best-formed seam and the template for the Resolver registry. (6) RunPlan is a clean immutable shell->engine snapshot (plan.py); RunState/Frontier/FiringSet/ProofTrees tier-keyed accumulators (state.py:121-145) are the correct add-a-tier pattern. (7) Layout is a VALUE field on resolved-children tuples (types.py:121-122), pack dispatches on it (one isinstance in pack.py is a known nit). (8) Two live filters fp_batch + prune_facts both correct and wired.\n\nALREADY DEAD (delete, no gate): forward/closure.py (no live importers; references stale grounder._build_witness_table/fc_method) and forward/witnesses.py (imported only by closure.py:113; fp_global BC machinery). errors.py:CapMissError docstring references stale 'EnumConfig.strict_cap' (config.py calls it PBCConfig.strict_cap) — fix the comment. RunPlan.config is unconditionally None (plan.py:104) — delete the field or populate it from the wired typed config.\n\nFALSE PREMISES IN v1 TO DROP (verified): (a) FC produces RuleGroundings via witnesses.py — FALSE (witnesses.py is fp_global BC code; FC returns only (hashes,n)). (b) An FC closure-set A/B oracle exists — FALSE (no FC test in tests/). (c) config.py is vestigial — FALSE (it is a deliberate 'no wrapper' design, just unwired). (d) 'register T in SHAPES' — would BREAK test_shape_symbols_match_vocab. (e) 'fp_batch is the only filter' — FALSE (prune_facts is live at step.py:167). (f) FilterMode has prune/provset/fp_global aliases to delete from config.py — FALSE (config.py FilterMode is only NONE/FP_BATCH; those aliases live in the factory.py:24,30 regex). (g) execution axis is 'shared' with FC ending join_chunk_size — NOMINAL only (auto_select already special-cases sparse at auto_select.py:34; iter_chunks is query-keyed at strategy.py:80).\n\nMUST RENAME (behavior-neutral; consumer-coordination): GrounderOutput->BackwardResult (tagged); ProofEvidence->CompletedTreeFirings + field evidence->completed_tree_firings + ProofEvidence.top_rule_idx property->tree_top_rule_idx; GLOSSARY concept 'evidence'->one 'completed-tree firings', 'considered'->'firing'; StepInputs->ResolveRequest (carry DepthSelector, Optional hooks). engine/->backward/ (move plan.py+state.py under it; resolution/ STAYS top-level). resolution value 'enum'->parser-only; 'closure' as a resolution value->ForwardGrounder family. All with one-window aliases.\n\nMUST REFACTOR (dispatch->registry; byte-neutral where gated): engine/step.py:resolve() if/elif -> RESOLVERS[plan.resolution].resolve(req) returning IDENTICAL ResolvedChildren/FlatResolvedChildren (the 14/14 gate depends on byte-identical tuples). The materializer ternary (step.py:116) stays pbc-internal as PbcResolver picks dense/flat. run_forward_chaining method= dispatch -> FORWARD_METHODS[name].run keeping the per-rule spmm->staged router and the staged JoinAlgo sub-axis. Two divergent forward() -> ground(GroundRequest); add nn.Module.forward compat shim. OutputSpec 3 bools -> FrozenSet[Tier] named 'tiers' (keep bools as @property one window; touches state.py/plan.py/loop.py/finalize.py/backward.py). Wire the existing typed config into make_grounder.\n\nMUST ADD (new surface, ≥2 impls each): core/ (Grounder/GroundRequest/GroundResult Protocols + Pipeline + Tier/OutputSpec). transform/ (ProgramTransform + IdentityTransform + shared adorn + MagicSetTransform). KB.with_program (rule-rewriteable; with_closure delegates) + the index cache keyed on the rule-program signature (in scope, not deferred — magic-set's hot path is facts-only union). Closure named result type (replaces ForwardGrounder's bare Tensor; .facts() preserves the old return). ForwardGrounder.ground/capability_row (vocab-share only). FCConfig.join_algo. make_grounder single entry; create_grounder/parse_type_string/make_bcwd shims. JoinResolver + tests/join_ab.py are reserved for the L3 follow-up (cleanest forcing-function plug-in). contracts: test_seams (registry entries impl their Protocol; AST-grep no if/elif on resolution/method in dispatch sites), test_results (each GroundResult member impls as_rule_groundings), and a GLOSSARY-vs-FIELDS token-disjointness check in test_vocabulary.\n\nGATE EXPOSURE: the 14/14 fingerprint covers BackwardGrounder cells ONLY (fingerprint_new.py); flat_ab.py covers BC sld/rtf flat-vs-dense; nesy_ab.py covers hooks. There is NO FC gate. So: (1) the Resolver-registry, engine->backward rename, and OutputSpec generalization are byte-neutral IF the registry is a pure lookup and ResolveRequest bundles today's kwargs (the hook asymmetry means pbc hooks stay inert) — RE-RUN 14/14, never assume. (2) Every FC-touching change (Closure type, ForwardMethod registry, with_program) is blocked on building tests/fc_fingerprint.py FIRST (refactor step 0b). (3) JoinResolver gets a SET-EQUALITY gate, not byte-identity.


# Appendix C — Gated refactor plan (0a..8)

### [0] 0a. Add core/ contracts (additive, nothing wired)
- change: Create core/grounder.py (Grounder Protocol with ground/capability_row/producible_tiers/rebound), core/request.py (GroundRequest + Tier + generalized OutputSpec.tiers with back-compat groundings/firings/trees @property), core/result.py (GroundResult Protocol + BackwardResult=alias of GrounderOutput with as_rule_groundings), core/pipeline.py (Pipeline). Register new FIELDS (queries/query_mask/kind/tiers/closure_depth). Wire NOTHING.
- files: ['grounder/core/grounder.py', 'grounder/core/request.py', 'grounder/core/result.py', 'grounder/core/pipeline.py', 'grounder/glossary.py', 'grounder/contracts/test_vocabulary.py']
- gate: contracts green incl. extended test_vocabulary; fingerprint 14/14 unchanged (nothing wired). Assert OutputSpec default tiers == frozenset({PROOF_STATE}).
- risk: Low — additive. Only risk is OutputSpec default drift; asserted.
### [1] 0b. Build the FC byte-identity gate (PRECONDITION for all FC work)
- change: Create tests/fc_fingerprint.py: byte-identical closure set (sorted hashes + n_provable) for ForwardGrounder on spmm + staged across the datasets the closure currently runs (family/wn18rr/etc), frozen baseline, mirroring fingerprint_new.py. This oracle does not exist yet (verified: no FC symbol anywhere in tests/).
- files: ['grounder/tests/fc_fingerprint.py', 'grounder/tests/baselines/']
- gate: fc_fingerprint green on spmm + staged; baseline committed. No source change.
- risk: Low-code / high-leverage — without it steps 3/5 are unverifiable. Must precede any FC-touching step.
### [2] 0c. Delete dead fp_global code
- change: Delete forward/closure.py and forward/witnesses.py (verified: closure.py has no live importers; witnesses.py imported only by closure.py:113; both fp_global BC machinery). Fix errors.py CapMissError docstring (EnumConfig->PBCConfig). Update forward/__init__.py docstring (drop the closure.py/witnesses.py lines and the phantom 'closure-set A/B' claim).
- files: ['grounder/forward/closure.py', 'grounder/forward/witnesses.py', 'grounder/forward/__init__.py', 'grounder/errors.py']
- gate: grep shows no remaining importer of closure/witnesses; full BC contracts + fingerprint 14/14 unchanged (BC never imported these); fc_fingerprint unchanged.
- risk: Low — unreachable code; no A/B needed.
### [3] 1. Promote resolve() if/elif to a Resolver registry
- change: Add resolution/api.py (Resolver Protocol + ResolveRequest = generalized StepInputs carrying DepthSelector + Optional hooks INERT for pbc). Wrap existing resolve_sld/rtf + pbc resolve_step as SldResolver/RtfResolver/PbcResolver registered in RESOLVERS. engine/step.py:resolve() becomes RESOLVERS[plan.resolution].resolve(req) returning the IDENTICAL tuples. Materializer dense/flat selection stays inside PbcResolver (pbc-internal).
- files: ['grounder/resolution/api.py', 'grounder/resolution/sld.py', 'grounder/resolution/rtf.py', 'grounder/resolution/pbc/resolve.py', 'grounder/grounder/registry.py', 'grounder/engine/step.py', 'grounder/contracts/test_seams.py']
- gate: fingerprint 14/14 byte-identical (RE-RUN) + flat_ab + new test_seams (no if/elif on resolution in step.py; every RESOLVERS entry impls Resolver).
- risk: Medium — dispatch order / tuple contents must not shift; ResolveRequest bundles today's kwargs (hooks Optional, inert for pbc).
### [4] 2. ground(GroundRequest) on both families + nn.Module shim + Closure type
- change: Add forward/api.py:Closure. BackwardGrounder.ground/rebound; ForwardGrounder.ground returns Closure (.facts() == old closure_facts() exactly); both forward() become compat shims wrapping ground(). ForwardGrounder.capability_row() = {Cell(SPARSE,EAGER)} (vocab-share only; no iter_chunks). Wire OutputSpec.tiers through state.py/plan.py/loop.py/finalize.py/backward.py (bools kept as @property).
- files: ['grounder/grounder/backward.py', 'grounder/forward/grounder.py', 'grounder/forward/api.py', 'grounder/core/result.py', 'grounder/state.py', 'grounder/plan.py', 'grounder/engine/loop.py', 'grounder/engine/finalize.py']
- gate: BC fingerprint 14/14 (ground() reproduces forward()); fc_fingerprint asserts Closure.facts() == old closure_facts() exactly.
- risk: Medium — OutputSpec generalization threads hot fields; Closure must preserve the exact decode.
### [5] 3. ForwardMethod registry (engine) + JoinAlgo sub-axis
- change: Add forward/methods.py with SpmmMethod/StagedMethod in FORWARD_METHODS; run_forward_chaining becomes the per-rule ROUTER (spmm whole-set fast path via supports(), fall-through to staged for UNSUPPORTED rules). JoinAlgo {staged,chunked} stays inside StagedMethod; spmm IterationStrategy stays a sub-strategy inside SpmmMethod. Add FCConfig.join_algo. Leave leapfrog as the documented placeholder (NOT registered).
- files: ['grounder/forward/methods.py', 'grounder/forward/fc.py', 'grounder/forward/spmm/', 'grounder/config.py', 'grounder/contracts/test_seams.py']
- gate: fc_fingerprint green on spmm + staged across all datasets (the router must reproduce today's fall-through exactly).
- risk: Medium — FC has historically drifted; the gate from step 0b is the protection.
### [6] 4. Wire the existing typed config + make_grounder; collapse the BC/FC fork
- change: WIRE config.py's SLD/RTF/PBC/FCConfig into make_grounder(kb, config, *, exec-knobs, transforms=()) dispatching on config TYPE (collapses the create_grounder string fork at factory.py:95). Make create_grounder/parse_type_string/make_bcwd shims that build a typed config + delegate. Either populate RunPlan.config from the wired config or delete the dead field. Do NOT add a flat GrounderConfig.
- files: ['grounder/factory.py', 'grounder/config.py', 'grounder/plan.py', 'grounder/__init__.py', 'grounder/contracts/test_execution.py']
- gate: All construction paths (type-string, make_bcwd, direct config) produce byte-identical grounders -> fingerprint 14/14; fc_fingerprint unchanged.
- risk: Medium — dispatch must reproduce today's defaults exactly (u->filter, all_anchors, flat_intermediate).
### [7] 5. KB.with_program + index cache
- change: Add KB.with_program(*, facts, heads, bodies, lens, index_cache_key) -> KB (new KB; index build stays in __init__); with_closure delegates to with_program(facts=...). Add a module-level memo so a facts-only rewrite reuses SldRuleIndex/binding tables keyed on the rule-program signature (the magic-set hot path).
- files: ['grounder/data/kb.py', 'grounder/data/rule_index/', 'grounder/contracts/test_data.py']
- gate: with_closure path A/B (closure soundness) unchanged; new test_data case: with_program(heads,bodies,lens) re-derives indices, with_program(facts=...) hits the cache; fc_fingerprint unchanged.
- risk: Medium — widens the immutable-KB contract; the cache must key correctly so facts-only stays cheap and rule-changing invalidates.
### [8] 6. ProgramTransform axis + MagicSetTransform
- change: Add transform/api.py (ProgramTransform + IdentityTransform + shared adorn owner reading RulePattern) and transform/magic_set.py (construction-time adorned skeleton via with_program(heads,bodies,lens) ONCE; per-call magic seeds via with_program(facts=...)). make_grounder wraps in Pipeline when transforms set; Pipeline.ground runs apply per call then base.rebound(kb2).ground(req2).
- files: ['grounder/transform/api.py', 'grounder/transform/magic_set.py', 'grounder/factory.py', 'grounder/grounder/registry.py', 'grounder/contracts/test_seams.py', 'grounder/tests/magic_ab.py']
- gate: Identity transform path -> fingerprint 14/14 unchanged; new tests/magic_ab.py (demand-grounding correctness on a toy KB where magic-set must equal the plain grounder's provable set); test_seams asserts TRANSFORMS entries impl ProgramTransform.
- risk: High — magic-set is query-dependent vs build-once RunPlan; correctness must be proven on a toy KB AND the per-call cost must stay facts-only (skeleton built once). Prototype before declaring the seam extensible.
### [9] 7. JoinResolver (L3 follow-up; the cleanest forcing-function plug-in)
- change: Add resolution/pbc JoinResolver (or a sibling) registered as RESOLVERS['join'] that pushes width/head-pred/query-exclusion predicates INTO the join as branch pruners (only drop provably-over-width branches), preserving the exact survivor SET. Add a PBCConfig.materialization knob (or JoinConfig) + a declared Cell/Layout. pbc stays untouched.
- files: ['grounder/resolution/pbc/', 'grounder/resolution/api.py', 'grounder/config.py', 'grounder/execution/auto_select.py', 'grounder/tests/join_ab.py']
- gate: NEW tests/join_ab.py: SET-EQUALITY of emitted groundings vs pbc per (dataset,w,d,u) + peak-mem regression. pbc fingerprint 14/14 unchanged.
- risk: Medium — semantics-preserving join is the substantive algorithmic work; the gate is set-equality, NOT byte-identity (top-k would break it).
### [10] 8. Mechanical renames in a dedicated last commit
- change: engine/ -> backward/ (move plan.py + state.py under it; resolution/ STAYS top-level). GrounderOutput->BackwardResult (alias); ProofEvidence->CompletedTreeFirings + field evidence->completed_tree_firings + property top_rule_idx->tree_top_rule_idx (aliases); GLOSSARY concept evidence->completed-tree-firings, considered->firing; StepInputs->ResolveRequest. Add the GLOSSARY-vs-FIELDS token-disjointness check. Update contracts _MODULES to core/* + forward/api.py. Drop OutputSpec bool @property once consumers migrate.
- files: ['grounder/backward/', 'grounder/types.py', 'grounder/core/result.py', 'grounder/glossary.py', 'grounder/contracts/', 'grounder/__init__.py']
- gate: Full contracts (incl. new disjointness check) + fingerprint 14/14 + flat_ab + fc_fingerprint + magic_ab all green after the rename (pure churn must be behavior-neutral).
- risk: Low-runtime / high-diff — schedule LAST, standalone. External consumers (probfol, torch-ns) import GrounderOutput/ProofEvidence; keep aliases >=1 release and coordinate the SHA-pin cascade in the same PR set.


# Appendix D — Open questions

- KB.with_program index cache: keying a facts-only rewrite on the rule-program content signature reuses SldRuleIndex/binding tables, but pbc's PbcRuleIndex is built by the GROUNDER on rebound, not by KB. Confirm the cache lives where it can also short-circuit init_enum's per-rule binding-table tensorization (build_plan), or magic-set's per-call rebound still re-runs init_enum even when only seeds changed. Decide the cache owner (KB vs grounder) before MagicSetTransform lands (refactor step 5/6).
- Should RunPlan.config be deleted or populated? Deleting is the minimal move (the field is unconditionally None, plan.py:104) and avoids adding anything to the byte-identity-critical snapshot; populating it makes the wired typed config introspectable downstream. Pick during refactor step 4 — both are one-liners, neither changes grounding output.
- JoinResolver placement: a JoinResolver under resolution/pbc/ reuses the PbcPlan binding tables but must replace candidates.enumerate_cartesian_* + the width.apply_filters_* post-filter with in-join branch pruning. Confirm the PbcPlan interface is reusable for a join (it bundles cartesian-shaped buffers) or whether join needs its own compile/init path — decide once the cap-into-join pruner is prototyped against tests/join_ab.py set-equality.
- FC firings (forward/firings.py) feeding the single considered.finalize dedup owner: is recording (rule_idx,head,body,query_idx) feasible without perturbing the spmm hot path (which is set-valued, not firing-valued), or is it only feasible on the staged/FCDynamic path (which iterates rules explicitly)? Until decided, Closure.as_rule_groundings() stays None and FIRINGS is BC-only. Net-new, gated separately; out of scope for v1.
- ResolveRequest hooks are Optional and inert for pbc in v1 (sld/rtf take hooks, pbc resolve_step does not, step.py:118). Is wiring pbc hooks ever wanted (learned scoring inside enum), and if so does it ride the nesy-hooks-across-all-resolvers change with its own A/B, or stay permanently inert for pbc? Affects whether the ResolveRequest hook fields are a real seam or documentation-only for one resolver.
- Deprecation window for the consumer-facing renames: probfol and torch-ns import GrounderOutput and ProofEvidence directly. The aliases (BackwardResult=GrounderOutput, CompletedTreeFirings=ProofEvidence) must live >=1 release; confirm whether the SHA-pin cascade (workspace CLAUDE.md) forces bumping torch-ns + probfol pins in the same PR set as the grounder rename commit (step 8), or whether the aliases let the cascade lag a release.
- Should the engine->backward/ rename (step 8) also rename run_backward's package-internal imports across the ~8 engine modules in one commit, or is a thinner compat-shim package (backward/ re-exporting engine/) acceptable for one window to shrink the consumer blast radius? The diff size vs the alias surface is the tradeoff.