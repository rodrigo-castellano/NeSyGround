# Filters: Hook-Based Pipeline Architecture

This document defines the filter architecture for the grounding pipeline.
Filters are optional, composable operations plugged into specific hook points
in the grounding loop.  At each hook point, **logical filters** run first,
then **neural (nesy) filters**.

Prerequisites: `resolution_and_filters.md` (resolution + filter framework),
`grounding_basics.md` (evaluation strategies).

---

## 1. The grounding pipeline

A single depth step has three phases: RESOLVE, PACK, POSTPROCESS.  The full
pipeline repeats these D times, then applies terminal filters on the collected
groundings.

Five **hook points** are defined — named locations in the pipeline where zero
or more filters can be plugged in.

```
RESOLVE
│
├─ fact_resolution()
│  └─→ HOOK 1: filter_resolve_facts
│
├─ rule_resolution()
│  └─→ HOOK 2: filter_resolve_rules
│
├─ combine(fact_children, rule_children) → all children
│  └─→ HOOK 3: filter_children
│
PACK (scatter-compact, topk cap)
│
POSTPROCESS (collect terminals, compact states)
│  └─→ HOOK 4: filter_step
│
═══════════ end of depth loop (repeat D times) ═══════════
│
collected groundings
│  └─→ HOOK 5: filter_groundings
│
▼
final proofs
```

---

## 2. Hook points

### Hook 1: filter_resolve_facts

**When**: after fact resolution produces K_f fact candidates per state.

**Input**: fact candidates `[B, S, K_f, ...]` with success mask.

**Purpose**: score or reject individual fact candidates before they are
combined with rule candidates and packed.

**Applies to**: SLD, RTF (which have a separate fact resolution step).
Skipped for ENUM (no separate fact resolution).

**Example filters**:
- Logical: provset check (reject fact candidates whose atoms ∉ I_D).
- Neural: KGEFactFilter (score candidates by KGE, keep top-k by score).

### Hook 2: filter_resolve_rules

**When**: after rule resolution produces K_r rule candidates per state.

**Input**: rule candidates `[B, S, K_r, ...]` with success mask.

**Purpose**: score or reject individual rule candidates before they are
combined with fact candidates and packed.

**Applies to**: SLD, RTF (which have a separate rule resolution step).
Skipped for ENUM (rules and enumeration are combined).

**Example filters**:
- Logical: provset check (reject rule candidates with body atoms ∉ I_D).
- Neural: KGERuleFilter (score candidates by KGE, keep top-k by score).

### Hook 3: filter_children

**When**: after all candidates are produced (fact + rule combined for SLD/RTF,
or enumeration output for ENUM), before PACK.

**Input**: combined children `[B, S, K, ...]` with success mask.

**Purpose**: filter or rerank the full set of candidates per state.  This is
the main per-step filter point — most logical filters plug in here.

**Applies to**: all resolutions.  For ENUM, this is the only per-resolution
hook point (hooks 1 and 2 are skipped).

**Note on pushdown**: for efficiency, ENUM may push filters from this hook
into its internal enumeration loop (avoiding materialisation of rejected
candidates).  The interface is still hook 3 — the pushdown is an
implementation optimisation.

**Example filters**:
- Logical: width (reject children with > W unknown body atoms).
- Logical: provset (reject children with body atoms ∉ I_D).
- Logical: prune_dead (reject children with atoms that have no matching facts
  and are not rule-head predicates).
- Neural: score-based top-k selection.

### Hook 4: filter_step

**When**: after POSTPROCESS (collect terminals, compact states), before the
next depth iteration.

**Input**: packed states `[B, S, G, 3]` with state mask.

**Purpose**: filter or rerank the surviving states.  Applied to the states
that will be carried into the next depth step.

**Applies to**: all resolutions.

**Example filters**:
- Logical: prune_facts (remove base-fact goals from the goal list, saving a
  depth step; disabled by default because it changes depth semantics — see §4).
- Neural: StepHook (rerank states by learned score, keep top-k).

### Hook 5: filter_groundings

**When**: after the depth loop completes, on the collected groundings.

**Input**: collected groundings `[B, tG, M, 3]` with grounding mask.

**Purpose**: verify soundness and/or score final groundings.  This is the
only point where cross-query information is available (all groundings in the
batch are collected).

**Applies to**: all resolutions.

**Example filters**:
- Logical: fp_batch (Kleene T_P fixpoint over collected groundings — marks
  groundings as proved when all body atoms are facts or heads of other proved
  groundings; requires all groundings collected before running).
- Neural: GroundingHook (score groundings by KGE/MLP, rerank, select top-k).
- Neural: SoftScorer (assign provability confidence to each body atom).
- Neural: RandomSampler (subsample groundings for training efficiency).

---

## 3. Filter execution order

At each hook point, filters execute in a fixed order:

1. **Logical filters** (in registration order).
2. **Neural filters** (in registration order, after all logical filters).

```python
# At each hook point:
for f in logical_filters:
    candidates, mask = f(candidates, mask, ...)
for f in nesy_filters:
    candidates, mask = f(candidates, mask, ...)
```

Logical filters run first because they are cheaper (no neural forward pass)
and may significantly reduce the number of candidates the neural filters need
to score.

---

## 4. Filter catalogue

### 4.1 Logical filters

| Filter | Hook | Purpose | Default |
|--------|------|---------|---------|
| **width** | 3 (children) | Reject children with > W unknown body atoms | Disabled |
| **provset** | 1, 2, or 3 | Reject candidates with body atoms ∉ I_D | Disabled |
| **prune_dead** | 1 or 3 | Reject candidates with unreachable atoms | Disabled |
| **prune_facts** | 4 (step) | Remove base-fact goals from goal list | Disabled |
| **fp_batch** | 5 (groundings) | Kleene T_P soundness fixpoint | Disabled |

**All disabled by default.**  With no filters, the grounder runs standard SLD
resolution — one goal resolved per step, no pruning, no soundness check.

#### width

Heuristic bound on proof complexity per state.  Rejects children where more
than W body atoms are ground but not base facts ("unknowns").

- Hook 3 (filter_children).
- Applies to ENUM (all body atoms ground) and SLD (only ground goals counted).
- For ENUM, typically pushed down into the enumeration loop for efficiency.
- Trades completeness for search space reduction.

#### provset

Per-step check against the pre-computed provable set I_D (built by
`fc/fc.py` at init time via semi-naive forward chaining to depth D).

- Hooks 1, 2, or 3 (any per-resolution hook).
- Rejects candidates whose unknown body atoms are not in I_D.
- Serves dual role: search filter (prune branches) and soundness check
  (reject unprovable body atoms).
- When used per-step, strictly more efficient than terminal-only checking.

#### prune_dead

Kills candidates containing body atoms whose predicate has no matching facts
AND is not a rule-head predicate.  Such atoms can never be proved.

- Hook 1 or 3.
- Not part of standard SLD or ENUM — it is a heuristic optimisation.
- Disabled by default.

#### prune_facts

Removes base-fact goals from the goal list, effectively resolving them "for
free" without consuming a depth step.

- Hook 4 (filter_step).
- **Changes depth semantics**: SLD depth d with prune_facts enabled effectively
  gets free fact resolutions, making it incomparable to standard SLD depth d.
  See `resolution_and_filters.md` §4.2.
- Disabled by default for semantic purity.
- Typically enabled for ENUM in practice (where all goals are ground, and fact
  goals would trivially re-resolve at the next step anyway).

#### fp_batch

Kleene T_P fixpoint over collected groundings.  Marks a grounding as "proved"
when all its body atoms are either base facts or heads of other proved
groundings in the batch.  Iterates until convergence.

- Hook 5 (filter_groundings) — **terminal only**.
- Cannot be per-step: requires all groundings collected before running (later
  groundings may prove atoms needed by earlier ones).
- Sound (snapshot mechanism prevents circular reasoning).
- Incomplete (depends on batch composition — body atoms provable only via
  groundings NOT in the batch are missed).

### 4.2 Neural (nesy) filters

| Filter | Hook | Purpose |
|--------|------|---------|
| **KGEFactFilter** | 1 (facts) | Score fact candidates by KGE, keep top-k |
| **KGERuleFilter** | 2 (rules) | Score rule candidates by KGE, keep top-k |
| **StepHook** | 4 (step) | Rerank states by learned score |
| **GroundingHook** | 5 (groundings) | Score/rerank final groundings |
| **NeuralScorer** | 5 (groundings) | Learned attention over body embeddings |
| **SoftScorer** | 5 (groundings) | Soft provability: known→1.0, unknown→sigmoid |
| **RandomSampler** | 5 (groundings) | Random subsampling (train: random, eval: valid-first) |

Neural filters always run **after** logical filters at the same hook point.

---

## 5. Resolution-specific hook applicability

| Hook | SLD | RTF | ENUM |
|------|-----|-----|------|
| **1. filter_resolve_facts** | Yes | Yes | Skipped |
| **2. filter_resolve_rules** | Yes | Yes | Skipped |
| **3. filter_children** | Yes | Yes | Yes (only per-resolution hook) |
| **4. filter_step** | Yes | Yes | Yes |
| **5. filter_groundings** | Yes | Yes | Yes |

For ENUM, hooks 1 and 2 are not applicable because ENUM does not have separate
fact and rule resolution phases.  All ENUM filtering goes through hook 3
(filter_children) or hooks 4–5.

---

## 6. Grounder configuration

The BCGrounder accepts a list of filters for each hook point:

```python
class BCGrounder(nn.Module):
    def __init__(
        self,
        kb: KB,
        resolution: str,                                  # 'sld', 'rtf', 'enum'
        depth: int,
        filter_resolve_facts: List[FilterHook] = [],      # hook 1
        filter_resolve_rules: List[FilterHook] = [],      # hook 2
        filter_children: List[FilterHook] = [],            # hook 3
        filter_step: List[FilterHook] = [],                # hook 4
        filter_groundings: List[FilterHook] = [],          # hook 5
    ):
```

All lists empty by default → pure resolution with no filtering.

### Example configurations

**Standard SLD, no filters (pure)**:
```python
grounder = BCGrounder(kb, resolution='sld', depth=3)
```

**ENUM + width + provset + fp_batch**:
```python
grounder = BCGrounder(kb, resolution='enum', depth=2,
    filter_children=[width(1), provset_check(provable_set)],
    filter_step=[prune_facts()],
    filter_groundings=[fp_batch()],
)
```

**SLD + KGE scoring at every level**:
```python
grounder = BCGrounder(kb, resolution='sld', depth=3,
    filter_resolve_facts=[KGEFactFilter(kge_model, top_k=50)],
    filter_resolve_rules=[KGERuleFilter(kge_model, top_k=20)],
    filter_step=[StepHook(learned_ranker)],
    filter_groundings=[GroundingHook(final_scorer)],
)
```

**ENUM + width + fp_batch + soft scoring (keras-ns style)**:
```python
grounder = BCGrounder(kb, resolution='enum', depth=2,
    filter_children=[width(1)],
    filter_step=[prune_facts()],
    filter_groundings=[fp_batch(), SoftScorer(kge_model)],
)
```

---

## 7. FilterHook protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class FilterHook(Protocol):
    """A filter plugged into one of the 5 hook points."""

    def __call__(
        self,
        candidates: Tensor,       # the data to filter (children, states, or groundings)
        mask: Tensor,              # validity mask (True = active)
        **context,                 # hook-specific context (fact_index, provable_set, etc.)
    ) -> Tensor:
        """Return an updated mask (same shape as input mask).

        True = keep, False = reject.
        The candidates tensor is NOT modified — only the mask changes.
        """
        ...
```

Filters compose by AND-ing masks: each filter can only reject (set True→False),
never resurrect (set False→True).  This guarantees monotonicity — adding a
filter can only reduce the output, never expand it.

---

## 8. Provable set construction

The provable set I_D used by the `provset` search filter is built at init time
by `fc/fc.py`:

```python
from grounder.fc.fc import run_forward_chaining

provable_set = run_forward_chaining(
    facts_idx=facts,
    rule_heads=heads,
    rule_bodies=bodies,
    rule_lens=lens,
    constant_no=C,
    depth=D,
)
```

This runs semi-naive T_P evaluation for D iterations from base facts, producing
I_D: the set of all atoms derivable in at most D forward steps.

I_D is independent of any query — it is computed once and shared across all
batches and queries.  This is in contrast to fp_batch, which is computed
per-batch from the collected groundings.

| | Provset (per-step) | fp_batch (terminal) |
|--|-------------------|-------------------|
| Built | Once at init | Once per batch |
| Scope | All atoms derivable in D FC steps | Atoms provable from batch groundings |
| Used at | Hooks 1, 2, 3 (per-step) | Hook 5 (terminal) |
| Role | Search + soundness | Soundness only |
| Cost | O(rules × E^arity × D) init | O(steps × batch_groundings) per batch |
