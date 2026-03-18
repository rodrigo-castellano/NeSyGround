# Grounder Restructure Plan

This document describes the planned restructure of the grounder library,
informed by the theoretical framework in `resolution_and_filters.md` and the
filter architecture in `filters.md`.

---

## 1. Goals

1. **Separate data, resolution, filters, orchestration, nesy** into clean
   directories with explicit dependencies.
2. **Hook-based filter pipeline**: 5 named hook points, logical then nesy at
   each, all disabled by default.
3. **Semantic purity**: with no filters enabled, SLD behaves as standard SLD
   (one resolution per step, no implicit pruning).
4. **Remove dead code**: LazyGrounder (unused), grounding_fundamentals.md
   (superseded).
5. **Rename fp_global consistently**: the provable set I_D is always referred
   to as `fp_global`, never `provset`. fp_global can be applied per-step
   (search filter) or terminally (soundness filter) — same provable set, same
   name, different hook point.

---

## 2. Current structure

```
grounder/
├── __init__.py
├── base.py                      # Grounder base class
├── types.py                     # Type definitions
├── primitives.py                # Unification primitives (unify, substitute)
├── kb.py                        # Knowledge base container
├── fact_index.py                # Fact indexing strategies
├── rule_index.py                # Rule indexing strategies
├── factory.py                   # Factory (name → grounder)
├── data_loader.py               # KG dataset loader
├── utils.py                     # Utilities
├── resolution/
│   ├── mgu.py                   # MGU primitives
│   ├── sld.py
│   ├── rtf.py
│   ├── enum.py
│   └── standardization.py
├── bc/
│   ├── bc.py                    # BCGrounder (parametrized grounder)
│   ├── lazy.py                  # LazyGrounder (unused)
│   └── common.py                # Pack, collect, prune utilities
├── fc/
│   └── fc.py                    # Forward chaining (builds I_D)
├── filters/
│   ├── _hash.py
│   ├── soundness/
│   │   ├── fp_batch.py
│   │   └── fp_global.py         # Terminal-only fp_global
│   └── search/
│       ├── width.py
│       └── prune_dead.py
├── nesy/
│   ├── hooks.py
│   ├── scoring.py
│   ├── kge.py
│   ├── neural.py
│   ├── sampler.py
│   └── soft.py
├── analysis/
├── tests/
└── docs/
```

---

## 3. Proposed structure

```
grounder/
│
│   ─── TOP LEVEL ───
│
├── __init__.py                     # Public API exports (updated imports)
├── types.py                        # Core types (unchanged)
├── factory.py                      # Factory: name → BCGrounder (updated)
├── utils.py                        # General utilities (unchanged)
│
│   ─── DATA LAYER ───
│
├── data/                           # NEW directory
│   ├── __init__.py
│   ├── kb.py                       # ← from grounder/kb.py
│   ├── fact_index.py               # ← from grounder/fact_index.py
│   ├── rule_index.py               # ← from grounder/rule_index.py
│   └── loader.py                   # ← from grounder/data_loader.py (renamed)
│
│   ─── RESOLUTION ───
│
├── resolution/
│   ├── __init__.py
│   ├── primitives.py               # ← from grounder/primitives.py
│   ├── standardization.py          #   (unchanged)
│   ├── mgu.py                      #   (unchanged)
│   ├── sld.py                      #   (unchanged)
│   ├── rtf.py                      #   (unchanged)
│   └── enum.py                     #   (unchanged)
│
│   ─── FILTERS ───
│
├── filters/
│   ├── __init__.py
│   ├── hooks.py                    # NEW: FilterHook protocol definition
│   ├── _hash.py                    #   (unchanged)
│   ├── search/                     # Per-step filters (hooks 1-4)
│   │   ├── __init__.py
│   │   ├── width.py                #   (unchanged)
│   │   ├── fp_global.py            # NEW: per-step check against I_D
│   │   ├── prune_facts.py          # NEW: extracted from bc/common.py postprocess
│   │   └── prune_dead.py           #   (unchanged, disabled by default)
│   └── soundness/                  # Terminal filters (hook 5)
│       ├── __init__.py
│       ├── fp_batch.py             #   (unchanged)
│       └── fp_global.py            # EXISTING: terminal check against I_D
│
│   ─── FORWARD CHAINING ───
│
├── fc/
│   ├── __init__.py
│   └── fc.py                       #   (unchanged: builds I_D for fp_global)
│
│   ─── BC ORCHESTRATION ───
│
├── bc/
│   ├── __init__.py                 # Exports BCGrounder only (LazyGrounder removed)
│   ├── bc.py                       # BCGrounder: depth loop + 5 hook points
│   └── common.py                   # Pack, collect, compact (prune_facts extracted)
│
│   ─── NESY INTERFACE ───
│
├── nesy/
│   ├── __init__.py
│   ├── hooks.py                    #   (unchanged: nesy hook protocols)
│   ├── _util.py                    #   (unchanged)
│   ├── scoring.py                  #   (unchanged)
│   ├── kge.py                      #   (unchanged)
│   ├── neural.py                   #   (unchanged)
│   ├── sampler.py                  #   (unchanged)
│   └── soft.py                     #   (unchanged)
│
│   ─── ANALYSIS, TESTS, DOCS ───
│
├── analysis/                       #   (unchanged)
├── tests/                          #   (unchanged, update imports)
└── docs/
    ├── grounding_basics.md         # Datalog fundamentals
    ├── grounding_probDB.md         # Probabilistic setting
    ├── resolution_and_filters.md   # Unified framework
    ├── filters.md                  # Filter hook architecture
    ├── restructure_plan.md         # This document
    ├── sld_vs_enum.md              # SLD vs ENUM comparison (existing)
    ├── fact_index.md               # Fact index docs (existing)
    ├── pipeline_tensors.md         # Tensor pipeline docs (existing)
    └── system/                     # System docs (existing)
```

---

## 4. Changes in detail

### 4.1 Create `data/` directory

Move data-layer files into a dedicated directory:

| From | To |
|------|-----|
| `grounder/kb.py` | `grounder/data/kb.py` |
| `grounder/fact_index.py` | `grounder/data/fact_index.py` |
| `grounder/rule_index.py` | `grounder/data/rule_index.py` |
| `grounder/data_loader.py` | `grounder/data/loader.py` |

Import updates:
```
grounder.kb           → grounder.data.kb
grounder.fact_index   → grounder.data.fact_index
grounder.rule_index   → grounder.data.rule_index
grounder.data_loader  → grounder.data.loader
```

### 4.2 Move primitives into resolution

| From | To |
|------|-----|
| `grounder/primitives.py` | `grounder/resolution/primitives.py` |

`unify_one_to_one()` and `apply_substitutions()` are unification primitives
used only by MGU-based resolution (SLD, RTF).  They belong with the resolution
strategies.

Import update:
```
grounder.primitives → grounder.resolution.primitives
```

### 4.3 Remove LazyGrounder

Delete `grounder/bc/lazy.py`.  Remove all references:
- `grounder/bc/__init__.py`: remove LazyGrounder import/export
- `grounder/__init__.py`: remove LazyGrounder import/export
- `grounder/factory.py`: remove `lazy.` prefix handling and LazyGrounder import

LazyGrounder is not used by any test, training script, or experiment.  The
predicate reachability it provides is a minor optimisation for large rule sets.
If needed in the future, the BFS reachability logic (~30 lines) can be a
utility function in `data/rule_index.py`.

### 4.4 Remove base.py

| From | To |
|------|-----|
| `grounder/base.py` | Deleted — fold into `bc/bc.py` or `data/kb.py` as needed |

The Grounder base class owns the KB.  If BCGrounder is the only grounder
(LazyGrounder removed), the base class is unnecessary indirection.  BCGrounder
can own the KB directly.

### 4.5 Extract prune_facts from postprocess

Currently, `bc/common.py` contains `prune_ground_facts()` which is called
unconditionally in POSTPROCESS.  Extract it into a filter:

| From | To |
|------|-----|
| `prune_ground_facts()` in `bc/common.py` | `grounder/filters/search/prune_facts.py` |

`bc/common.py` POSTPROCESS shrinks to: collect terminals + compact states.
`prune_facts` becomes a hook 4 filter, disabled by default.

### 4.6 fp_global: per-step and terminal

fp_global currently exists only as a terminal soundness filter
(`filters/soundness/fp_global.py`).  Add a per-step variant:

| File | Hook | Purpose |
|------|------|---------|
| `filters/search/fp_global.py` | 1, 2, 3 (per-step) | Reject candidates with body atoms ∉ I_D |
| `filters/soundness/fp_global.py` | 5 (terminal) | Verify collected groundings against I_D |

Both use the same provable set I_D (built by `fc/fc.py` at init).  The
per-step variant prunes early; the terminal variant does a final verification.
They can be used independently or together.

**Naming**: always `fp_global`.  The name `provset` is retired.  References to
`provset`, `provable_set`, or `_build_provable_set` in code and docs should be
renamed to `fp_global` or `fp_global_set` for the data structure.

### 4.7 Add FilterHook protocol

New file `filters/hooks.py`:

```python
@runtime_checkable
class FilterHook(Protocol):
    def __call__(
        self,
        candidates: Tensor,
        mask: Tensor,
        **context,
    ) -> Tensor:
        """Return updated mask. True = keep, False = reject."""
        ...
```

All logical filters and nesy filters implement this protocol.

### 4.8 BCGrounder: 5 hook points

Update `bc/bc.py` to accept filter lists for each hook point:

```python
class BCGrounder(nn.Module):
    def __init__(
        self,
        kb: KB,
        resolution: str,
        depth: int,
        filter_resolve_facts: List[FilterHook] = [],   # hook 1
        filter_resolve_rules: List[FilterHook] = [],   # hook 2
        filter_children: List[FilterHook] = [],         # hook 3
        filter_step: List[FilterHook] = [],             # hook 4
        filter_groundings: List[FilterHook] = [],       # hook 5
    ):
```

All empty by default → standard resolution, no filters.

### 4.9 Defaults: all filters disabled

| Filter | Default | Reason |
|--------|---------|--------|
| width | disabled | Not standard; heuristic |
| fp_global (per-step) | disabled | Not standard; requires pre-computed I_D |
| prune_dead | disabled | Not standard; heuristic |
| prune_facts | disabled | Changes depth semantics |
| fp_batch | disabled | Not standard; terminal soundness |
| All nesy hooks | none | No neural model by default |

With no filters, SLD at depth d behaves exactly as standard SLD: d resolution
steps (including trivial fact resolutions), no implicit pruning.

### 4.10 Clean up docs

| Action | File |
|--------|------|
| Delete | `docs/grounding_fundamentals.md` (superseded by grounding_basics.md) |
| Keep | `docs/grounding_basics.md` (new) |
| Keep | `docs/grounding_probDB.md` (new) |
| Keep | `docs/resolution_and_filters.md` (new) |
| Keep | `docs/filters.md` (new) |
| Keep | `docs/sld_vs_enum.md` (existing, codebase-specific) |
| Keep | `docs/fact_index.md` (existing) |
| Keep | `docs/pipeline_tensors.md` (existing) |
| Keep | `docs/system/` (existing) |

### 4.11 Rename fp_global everywhere

All references to `provset`, `provable_set`, `_build_provable_set`,
`check_in_provable` should be renamed to use `fp_global` consistently:

| Old name | New name |
|----------|----------|
| `provset` (in docs, comments) | `fp_global` |
| `provable_set` (variable name) | `fp_global_set` |
| `_build_provable_set()` | `_build_fp_global_set()` |
| `check_in_provable()` | `check_in_fp_global()` |
| `BCStaticProvset`, `BCDynamicProvset` (legacy) | `BCStaticFpGlobal`, `BCDynamicFpGlobal` |

---

## 5. Migration

### 5.1 Import updates (find-and-replace)

```
grounder.kb                       → grounder.data.kb
grounder.fact_index               → grounder.data.fact_index
grounder.rule_index               → grounder.data.rule_index
grounder.data_loader              → grounder.data.loader
grounder.primitives               → grounder.resolution.primitives
grounder.bc.lazy                  → (deleted)
grounder.base                     → (deleted, folded into bc.bc)
```

The public API in `__init__.py` re-exports everything, so downstream consumers
(ns_lib, experiments, tests) only need import updates if they import from
submodules directly.

### 5.2 Verification

After restructure, run all three test levels:

1. Grounder own tests:
   ```
   cd grounder && python -m pytest tests/ -v
   ```

2. torch-ns precommit:
   ```
   cd torch-ns && python -u tests/run.py precommit
   ```

3. DpRL-KGR parity (after syncing):
   ```
   rsync -av --exclude='__pycache__' --exclude='.git' grounder/ /path/to/DpRL/grounder/
   cd DpRL && python -m pytest kge_experiments/tests/parity/ -v
   ```

---

## 6. Summary of all changes

| # | Change | Files affected |
|---|--------|---------------|
| 1 | Create `data/` directory | kb.py, fact_index.py, rule_index.py, data_loader.py |
| 2 | Move primitives to resolution | primitives.py |
| 3 | Remove LazyGrounder | bc/lazy.py, factory.py, __init__.py |
| 4 | Remove base.py | base.py, bc/bc.py |
| 5 | Extract prune_facts filter | bc/common.py → filters/search/prune_facts.py |
| 6 | Add per-step fp_global | filters/search/fp_global.py (new) |
| 7 | Add FilterHook protocol | filters/hooks.py (new) |
| 8 | BCGrounder: 5 hook points | bc/bc.py |
| 9 | All filters disabled by default | bc/bc.py, factory.py |
| 10 | Rename provset → fp_global | all files referencing provset |
| 11 | Delete grounding_fundamentals.md | docs/ |
| 12 | Clean up docs | docs/ |

**Migration cost**: moderate.  5 files move, 1 file deleted, 2 new files
created, import updates across the codebase.  Public API re-exports insulate
most downstream consumers.
