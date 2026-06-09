# NAMING — the unified vocabulary (proposed rewrite of `glossary.py`)

Draft for review. This is the *target* vocabulary; landing it in `glossary.py`
requires the code migration (renaming the `Shapes` fields, knobs, and `FIELDS`).
Until then this file is the canonical reference and the old names still ship.

---

## Naming laws

1. **goal ≡ state.** A *goal* is a **set of atoms still to be proved** — the MDP
   "state" under its logic name. We use **goal** everywhere. A goal is **not** an
   atom: it *contains* atoms.
2. **atom.** One `(predicate, arg0, arg1)` triple. A goal holds up to `L` atoms.
3. **One concept ⇄ one name.** No concept with two names; no name for two concepts.
4. **Tier is visible in the name.** Every sizing name is exactly one of:
   *data dim* · *capacity axis* · *derived scalar* · *runtime count* · *knob*.
5. **Family-rooted symbols.** Related quantities share a root letter from a closed
   set; a subscript is always the *scope* ("per what").
6. **One knob per tunable symbol**, named `max_<noun>[_per_<scope>]`; the `KNOBS`
   registry is the machine-checked knob⇄symbol index.

### Closed symbol families

| Family | Root(s) | What it counts |
|--------|---------|----------------|
| Data    | `B D M V E` (+ `pad`) | fixed by the KB / query batch |
| Frontier| `G L A`     | the goal tensor `[B, G, L, 3]` and accumulated body |
| Fan-out | `K`         | children of a goal's selected atom in one step |
| Yield   | `Y`         | groundings (the output) |
| Derived | `N`         | computed from the above |

---

## SHAPES — tensor axes (1:1 with the `Shapes` dataclass)

```python
SHAPES = {
    # ── Data (fixed by KB / query) ──
    "B":   "queries per batch (batch size)",
    "D":   "proof depth (number of steps)",
    "M":   "atoms per rule body (max over rules)",
    "V":   "free variables per rule (max over rules)",
    "E":   "entities (the all-entity candidate space)",
    "pad": "padding id — sentinel (neither entity nor variable)",
    # ── Frontier (the goal tensor [B, G, L, 3]) ──
    "G":   "goals per step — the frontier width (= states)",         # was S
    "L":   "atoms per goal — the goal length (= M + (M-1)*D)",        # was G
    "A":   "accumulated atoms per goal (= D*M)",
    # ── Fan-out (children of a goal's selected atom) ──
    "K_f": "fact children per atom (from the fact index)",
    "K_r": "rule children per atom (rules per predicate)",
    "K_v": "candidates per free variable (= min(K_f, Y_r))",
    # ── Yield (groundings = output) ──
    "Y_r": "groundings per rule (per query)",                        # was G_r
    "Y_q": "groundings per query — the collected budget",            # was C
    # ── Derived ──
    "N":   "flattened goal count (= B*G)",                          # was B*S
}
```

## SCALARS — derived quantities that are NOT tensor axes (concepts, not in SHAPES)

```python
SCALARS = {
    "K":     "total children per atom (SLD: K_f+K_r · RTF: K_f*K_r · PBC: K_r*Y_r); capped by max_children",
    "K_max": "the children ceiling constant (value of max_children, default 550)",
    "W":     "width — unknown-atom tolerance at intermediate steps (BC_{w,d,u})",
    "W_l":   "last-step width tolerance (paper convention 0: every leaf atom is a fact)",
    "P":     "predicate count",
    "R":     "rule count",
    "max_vars_per_rule": "variable-id spacing reserved per goal for standardization",
}
```

## DYNAMIC — runtime counts (NOT static; NOT in SHAPES; live on per-step tensors)

`SHAPES` holds only ints fixed at construction, so the dense path is
`torch.compile`/CUDA-graph safe. Counts known only at runtime are excluded by
design and appear as the leading dim of per-step NamedTuples:

```python
DYNAMIC = {
    "T":     "flat-path survivor count — real (nonzero) child rows after compaction this step (FlatResolvedChildren)",
    "G_out": "packed goals out — goals surviving pack this step (== G on dense; dynamic on flat)",  # was S_out
}
```

This boundary is exactly **why flat is eager-only**: `T`/`G_out` vary per step, so
flat tensors can't be CUDA-graph-captured; the dense path instead pads to the
static `G, K_r, Y_r`.

## KNOBS — user/ctor parameters → the symbol they set or cap (the new index)

```python
KNOBS = {
    # knob:                  (sets|caps, symbol, default)
    "depth":                 ("sets",  "D",   None),
    "max_goals":             ("caps",  "G",   256),          # was max_states
    "max_goal_length":       ("caps",  "L",   "M+(M-1)*D"),  # was max_goals (REPURPOSED)
    "max_facts_per_query":   ("caps",  "K_f", 64),
    "max_groundings_per_rule":  ("caps", "Y_r", 32),         # was max_groundings_per_query
    "max_groundings_per_query": ("caps", "Y_q", 64),         # was max_total_groundings
    "max_children":          ("caps",  "K",   550),          # was K_MAX + max_derived_per_state
    "width":                 ("sets",  "W",   1),            # PBC shorthand: w
    "last_step_width":       ("sets",  "W_l", 0),            # PBC shorthand: u
    # ── execution policy (not sizes) ──
    "chunk_size":     ("policy", "chunk",         None),
    "layout":         ("policy", "layout",        "auto"),
    "compile":        ("policy", "compile",       "off"),
    "materialization":("policy", "materialization","cartesian"),
}
```

## GLOSSARY — domain concepts

```python
GLOSSARY = {
    "goal":        "a SET of atoms still to be proved = the MDP state; the frontier unit (G of them per step). NOT an atom.",
    "atom":        "one (predicate, arg0, arg1) triple; a goal holds up to L atoms",
    "state":       "MDP-framing alias of goal; the code uses 'goal'",
    "frontier":    "the set of G goals carried at one proof step",
    "forward":     "forward chaining only (ForwardGrounder); never the BC batch driver",
    "run_backward":"the backward proof-search batch driver (backward/loop.py)",
    "backward":    "query-directed proof search (BackwardGrounder; sld/rtf/pbc)",
    "resolution":  "the WHAT axis of BackwardGrounder: sld | rtf | pbc (config type, not subclass)",
    "pbc":         "Parametrized Backward Chaining (IJCAI BC_{w,d,u}); pbc resolution + fp_batch filter",
    "width":       "W — how many atoms of a goal may stay unknown (not yet facts) at a step",
    "firing":      "one fired rule application (rule_idx, head, body); the FiringSet unit, PRIMARY for RuleGroundings",
    "completed-tree firings": "fired rule apps INSIDE completed proof trees (~3x undercount); the TREES tier",
    "filter":      "post-hoc soundness prune; the ONLY filter is fp_batch (Kleene T_P)",
    "layout":      "materialization of one step: flat (compact/eager) | dense (padded/compiled) | sparse (spmm)",
    "strategy":    "ExecStrategy — single owner of compile/layout/chunk/cudagraph policy",
    "cell":        "one (layout, compile-mode) point a grounder DECLARES it supports",
    "ground":      "the single runtime verb (Grounder.ground); request in, GroundResult out",
    "grounder":    "a family (BackwardGrounder | ForwardGrounder); ground + capability_row + producible_tiers + rebound",
    "tier":        "a BACKWARD output stratum (proof_state | firings | trees); the OutputSpec.tiers set",
    "closure":     "the ForwardGrounder result FAMILY (provable triples); never a resolution value",
}
```

## FIELDS — the goal/atom discipline applied to carried struct fields

**Rule:** a field naming a *set of atoms* (a goal) uses **goal**; a field naming a
*single* atom uses **atom**; the `G` axis is goals, the `L` axis is atoms.

Renames forced by `goal ≡ set-of-atoms` and `S → G`, `L = atoms/goal`:

| Old field | New field | Shape | Why |
|-----------|-----------|-------|-----|
| `proof_goals` | `goal_atoms` | `[B, G, L, 3]` | the atoms of each goal |
| `selected_goal` | `selected_atom` | `[.., 3]` | it's ONE atom picked from the goal, not a goal |
| `state_valid` | `goal_valid` | `[B, G]` | active-goal mask |
| `fact_goals` | `fact_child_goals` | `[B, G, K_f, L, 3]` | child goals (atom-sets), inner dim is atoms (L) |
| `rule_goals` | `rule_child_goals` | `[B, G, K_r, L, 3]` | child goals |
| `flat_goals` | `flat_child_goals` | `[T, L, 3]` | child goals (flat) |
| `S` (carried, e.g. FlatResolvedChildren) | `G` | scalar | goals per step |
| `ProofState` (type) | `GoalState` | — | the proof-state output tier = the final goals |

All other `FIELDS` entries keep their meaning; only `state→goal` and the
`atom` vs `goal` distinction apply.

---

## Old → new migration map

| Old | New | Kind | Note |
|-----|-----|------|------|
| `S` | `G` | axis | goals per step (= states) |
| `G` | `L` | axis | atoms per goal (goal length) |
| `C` | `Y_q` | axis | groundings per query (collected budget) |
| `G_r` | `Y_r` | axis | groundings per rule |
| `K` | `K` | scalar | total children/atom; now one cap (`max_children`) |
| `K_max` | `K_max` | scalar | = value of `max_children` |
| `max_states` | `max_goals` | knob | caps `G` |
| `max_goals` | `max_goal_length` | knob | **REPURPOSED** — caps `L`, not goal count |
| `max_total_groundings` | `max_groundings_per_query` | knob | caps `Y_q` |
| `max_groundings_per_query` | `max_groundings_per_rule` | knob | caps `Y_r` (fixes misnomer) |
| `K_MAX`, `max_derived_per_state` | `max_children` | knob | one cap for `K`, all resolutions |
| `w_last_depth`, `u` | `last_step_width`, `u` | knob | caps `W_l` |

### ⚠️ Highest-care item

`max_goals` is **repurposed** (old: caps atoms/goal `L`; new: caps goal count `G`)
— *opposite* meaning under the *same* name. Do **not** alias it. Deprecate/remove
the old `max_goals`, introduce the two new knobs fresh, so no consumer silently
gets the old behavior.

## Staging

- **Land now (fingerprint-neutral, no consumer break):** this `NAMING.md`, plus
  the `KNOBS`/`SCALARS` registries documenting *today's* wiring under the new names.
- **Gated (consumer-breaking → one-window aliases + SHA cascade):** renaming the
  `Shapes` fields, the public knobs, and `FIELDS`.
- **Fingerprint-gated (behavior):** the `max_children` unification (`K`'s single cap).
