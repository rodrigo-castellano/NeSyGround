# grounder: design

A library of grounding techniques for neural-symbolic reasoning: given a program (facts and rules) and queries, produce
the rule groundings a reasoner scores. Three techniques, one class each, sharing one data model, one output type and one
set of operations:

| class | technique | goals | runs |
|---|---|---|---|
| `PBC` | parametrized backward chaining, BC_{w,d} (IJCAI-25) | ground atoms | Triton kernels on the actual sizes, streamed |
| `SLD` | SLD resolution | proof states (conjunctions with variables) | eager (static shapes) |
| `Forward` | semi-naive forward chaining | none: data-directed | eager |

Consumers: torch-ns (`PBC`), probfol-llm (`SLD`, `Forward`). LOGIC2RL keeps its own engine; `SLD.derive` reproduces it
(checked side by side, LOGIC2RL unmodified).

## Why PBC and SLD are two engines

SLD's goals carry variables: a proof state `q1(X,Z), q2(Z,Y)` is a conjunction whose atoms share `Z`, so SLD carries
whole states `[S, L, W]`, unifies with substitutions, renames rule variables apart, packs children and harvests completed
proofs. PBC's goals are ground: a grounding's unknown body atoms are independent, so a state collapses into single
atoms — each step grounds a pool's unique goal atoms and passes on the unknown body atoms as the next goals (an AND-OR
graph over atoms). No unification, substitution, renaming or packing. One pipeline with swappable resolution (the old
general engine) paid SLD's state machinery for PBC; here each engine carries only what its goals need, and they share
operations, not a pipeline.

## Data model

**Atom**: int64 `(p, a1, …, a_n)`, width `W = n + 1` (`W = 3`: binary predicates; SLD takes any arity). Constants are
`< E`, variables `>= E` (SLD only), `pad` one reserved id. **Key**: one int64 per atom, `((pool·B + p)·B + a1)·B + a2`
with `B` above every id (`key`/`decode` in `ops`); every hash set, sort and membership test uses it.

```python
class KB:                      # the program; immutable
    facts: Facts
    rules: Rules
    E, P, pad: int

class Facts:                   # one index for every technique
    atoms: Tensor              # [F, W] sorted by key
    def contains(atoms) -> Tensor              # [..., W] -> [...]: hash set (GPU) / sorted keys (CPU)
    def lookup(pred, arg, side) -> (start, count)   # CSR slice of the other argument's values (stable order)
    values: Tensor                             # [2F] the CSR values: by subject, then by object
    def of_pred(pred) -> (start, count)        # every fact of a predicate (SLD goals with no constant)

class Rules:                   # canonical order: sorted by head predicate (stable); body in dependency order
    heads [R, W], bodies [R, M, W], lens [R]
    order: Tensor              # [R] input index of each canonical rule
    body_order: list           # input body position of each canonical column
    patterns: list[RulePattern]   # head variables, free variables, each argument's source
    variants                   # PBC: one per anchor body atom
```

**Outputs**:

```python
@dataclass(frozen=True)
class Groundings:              # every grounder's product (PBC now; SLD's proofs; Forward later)
    atoms: Tensor              # [A, 3] unique, sorted by (pool, key)
    rule: Tensor               # [N] canonical rule id, sorted within each pool
    head: Tensor               # [N] row of atoms
    body: Tensor               # [N, M] rows of atoms
    body_mask: Tensor          # [N, M]
    offsets: Tensor            # [S, R + 1] per pool, per rule: CSR into the N groundings
    atom_offsets: Tensor       # [S + 1] per pool: its slice of atoms
    query: Tensor              # [S, Q] row of each query in atoms

@dataclass(frozen=True)
class Proofs:                  # SLD's completed proof trees, per query
    head [Q, P, D, W], body [Q, P, D, M, W], rule [Q, P, D], body_len [Q, P, D], mask [Q, P], count [Q]

@dataclass(frozen=True)
class Closure:                 # forward chaining's derived atoms
    atoms: Tensor              # [N, 3] sorted by key
```

**Canonical order** (torch-ns sums in this order; changing it moves its metrics in the last bit): rules sorted by head
predicate (stable), body columns in dependency order, atoms sorted by (pool, key), groundings sorted by (pool, rule,
head row, body rows).

## The contract

```python
class Grounder(nn.Module):
    kb: KB
    def ground(self, queries: Tensor, mask: Tensor | None = None) -> Groundings
        # queries [S, Q, W]: S pools of Q queries (a pool is grounded and pruned as one set); [Q, W] = one pool
```

```python
class PBC(Grounder):
    def __init__(self, kb, *, depth, width=1, last_width=0,
                 prune="fp_batch",     # "fp_batch" | "keras" | None
                 rounds=None,          # keras proof-walk rounds (.r<N>)
                 bind="facts",         # "facts" | "all" (IJCAI's Full grounder) | a Guide (scored)
                 guide=None)           # select: top k groundings per goal
    sizing: Sizing                     # worst case per goal and step, bytes per slot, goals per call
    @classmethod
    def parse(cls, kb, type: str)      # "enum.keras.w1.d2.r2.flat" -> PBC(...)

class SLD(Grounder):
    def __init__(self, kb, *, depth, prune="fp_batch", drop_facts=True, bind=None, guide=None,
                 max_states=256, max_children=550, max_proofs=64)
    def derive(self, states, next_var, excluded=None) -> (children, counts, next_var, rule)   # one step
    def prove(self, queries) -> Proofs
    def ground(self, queries) -> Groundings      # the groundings inside the completed proofs

class Forward(Grounder):
    def __init__(self, kb, *, depth)             # spmm when every rule has a matrix form, else the join
    def closure(self) -> Closure
```

One step per class, uniform verbs:

| class | state | step | after the last step |
|---|---|---|---|
| PBC | unique goal atoms per pool | **match** (goal × its predicate's rule variants) → **bind** (free variables but the last) → **test** (fused kernel: last variable, body atoms, width, cycle, provability; writes only kept groundings) → **next** (their unknown atoms, unique) | **prune** (fp_batch or keras) → **canonical** |
| SLD | states `[S, L, W]` | **select** (leftmost atom) → **resolve** (facts ∥ rule heads, renamed apart) → **bind** (optional: open variables) → **pack** → **drop facts** → **compact** → **rename** | **prune** (fp_batch) → `Proofs` |
| Forward | derived atoms `I`, new atoms `Δ` | **join** (bodies with ≥ 1 atom in `Δ`: sparse matmul or staged join) → **new** (heads not in `I`) | `Closure` |

## Binding free variables

The one operation every engine shares. A free variable ranges over:

| mode | values | where |
|---|---|---|
| `facts` | what a fact lookup returns (hard unification against the facts) | PBC's lookup; SLD's fact resolution |
| `all` | every entity | IJCAI's Full grounder |
| scored, `Guide(k)` | the top k entities by a model's score over all entities | soft unification (k = 1), learned grounding |

## The model seam

```python
class Scorer(Protocol):                          # lives with the model (torch-ns, torch-kge, LOGIC2RL-KGE)
    def score(self, atoms: Tensor) -> Tensor: ...                       # [N, 3] -> [N] in (0, 1]
    def score_all(self, pred, arg, side) -> Tensor: ...                 # [N] -> [N, E]: one argument free

@dataclass(frozen=True)
class Guide:
    scorer: Scorer
    k: int | Tensor              # per group, or per query ([Q]: learned budgets)
    tnorm: str = "min"           # a candidate's score over its unknown atoms: min | product
    tau: float | None = None     # Gumbel top-k: Plackett-Luce sampling (training)
    capture: list | None = None  # each decision's atoms, scores and kept mask (REINFORCE)
```

Applied at named points: **select** keeps the top k candidates per group, all-fact candidates always kept (PBC: a goal's
groundings after `test`, and after each free variable when V ≥ 2; SLD: a state's children after `resolve`); **fill**
binds free variables to their top k entities by `score_all` (PBC's `bind=Guide`; SLD's open rule children: soft
unification). Replaces the old nesy hooks and the guided beam's setters; the models live in their own repositories.
The per-goal beam replaces the old per-proof-state beam (PBC has no states): same properties (a subset of exhaustive;
k = ∞ is exhaustive; monotone in depth), different outputs than the old engine's beam.

## Shared operations (`ops.py`)

| op | | PBC | SLD | Forward |
|---|---|---|---|---|
| `key`, `decode` | atom (+ pool) ↔ int64 | ✓ | ✓ | ✓ |
| `AtomSet` | `contains`, `insert`: hash table on GPU, sorted keys on CPU | ✓ | ✓ | ✓ |
| `unique`, `compact` | distinct rows; stream compaction | ✓ | ✓ | ✓ |
| `fixpoint` | least model of a ground program in `rounds` rounds | fp_batch | fp_batch | |
| `canonical` | → `Groundings` | ✓ | ✓ | later |

fp_batch is forward chaining over the grounded rules; PBC and SLD call one implementation with their own round counts.
Technique-owned: unification, substitution, renaming, packing (SLD); anchor variants, the fused kernel, the width test,
the keras walk, sizing (PBC); the matrix ops and joins (Forward).

## PBC: no caps, bounded memory per step, exact about explosion

- **No truncation.** No fact cap, no per-rule, per-query, per-step or children cap: every lookup is enumerated in full.
- **Actual sizes.** The Triton kernels take their sizes at run time; the glue is eager. A step allocates what its goals
  produce, never a worst-case buffer, so what fits is what the data needs.
- **Streamed steps** (`engine.py`): a step holds a bounded slice of its work at a time, and the result is the same as
  in one piece. Goals go in chunks of about `ROWS` (goal, variant) rows; the expansion before the last free variable
  in groups of `ROWS` rows; one launch of the fused kernel keeps at most `WALK` candidates (its output buffers); kept
  groundings become bodies `ROWS` at a time. Only the kept groundings outlive a chunk.
- **Dropping early, exactly.** fp_batch: the step before the last keeps only the groundings whose unknown atom can still
  be proved (a goal so far, or one a width-0 last step can ground). keras (width ≤ 1): a step writing more than
  `KERAS_STORE_ROWS` groundings, and the last step unless it is small, is grounded again in each Kleene round of the
  provable atoms' least fixed point, writing only what the keras prune can keep. On WN18RR a query side's step 2 has
  ~10^8 goals and keeps ~10^4 groundings.
- **Sizing** (`sizing.py`, from the rule variants and the fact index): the worst case, reported, not allocated — a
  multi-type branching model over the goals, as `docs/grounding_limits.md` in torch-ns, but over the grounder's own
  enumeration (every anchor variant; at a width-0 step one variant, the cheapest; next goals only of predicates some
  rule concludes). It states what no call can exceed; the measured distribution states what calls of a protocol need.
- **Calls** are split across pools, never within one (the prune is per pool). A pool that does not fit runs out of
  memory: the caller splits its calls, not a pool.
- With a `Guide(k)`, a goal's successors are bounded by k × width.

## SLD

Takes LOGIC2RL's engine shape: `derive` (one step) as the core verb, any arity, static shapes; `prove` loops over it
and keeps the proof trail and fp_batch for probfol-llm. Keeps its caps (`max_states`, `max_children`, `max_proofs`):
probfol's outputs depend on them; lifting them is part of probfol's own migration.

## Forward

The spmm engine (semi-naive, hybrid with full re-evaluation) when every rule has a matrix form, else the staged join
(any rule shape, constants). Returns a `Closure`. Firings (`ground`) come later, with magic sets.

## Naming

| concept | name | replaces |
|---|---|---|
| a ground rule instance | grounding | firing, rule application, considered row |
| queries grounded and pruned together | pool | batch, seg, chunk |
| an atom to ground at a step (PBC) | goal | query row, state |
| a conjunction of goals with a proof trail (SLD) | state | goal, frontier row |
| output fields | `atoms`, `head`, `body`, `body_mask`, `rule`, `offsets`, `query` | `atom_table`, `*_pool_idx`, `body_atom_valid`, `rule_idx`, `rule_offsets`, `query_pool_idx` |
| dimensions | S pools, Q queries per pool, G goals, N groundings, A atoms, R rules, M body atoms, V free variables, D depth, W width (unknown atoms; SLD: atom width), E entities, P predicates, F facts | |

## Layout

```
grounder/
├── __init__.py      KB, PBC, SLD, Forward, Guide, Groundings, Proofs, Closure, errors
├── types.py         Groundings, Proofs, Closure, Guide, Scorer, GroundingMemoryError, ConfigError
├── ops.py           key/decode, AtomSet, unique, compact, fixpoint, canonical
├── kb.py            KB, Facts, Rules, RulePattern, variants, parsing
├── pbc/             PBC, parse; kernels.py, step.py, prune.py (fp_batch, keras), guide.py, sizing.py
├── sld/             SLD; resolve.py (unify, substitute, rename apart), state.py (pack, drop facts, compact, trail)
└── forward/         Forward; spmm/, join/
```

## Verification

- `python -m pytest tests/ -q`.
- `python tests/gate.py` (GPU, < 90 s): PBC's counts and output hashes exact on Family, WN18RR and Countries S3
  (fp_batch and keras, depths 1-3), forward chaining's closures exact, and each cell's speed with a ratchet (a faster
  passing run becomes the baseline; never slower).
- `tests/probfol_record.py`: probfol-llm's forward-chaining and SLD calls, identical to the recording taken on main.
- LOGIC2RL: its `SLD.derive` and this `SLD.derive` on the same program and states, identical outputs.
- torch-ns: `python -m pytest tests/unit/ -q`, `python tests/gate.py` (metrics exact, no slower), and one paper cell
  per protocol, test metrics identical to `reproduce/ckpt/`.
- fp32 and deterministic throughout.
