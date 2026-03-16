# Grounders

Every grounder variant in NeSyGround, with algorithm, formal definition, full constructor signature, key methods, soundness properties, and usage guidance.

---

## Class Hierarchy

```
Grounder (nn.Module)                        # Base: owns a KB reference
├── BCGrounder                              # Unified backward chaining (configured, not subclassed)
└── LazyGrounder                            # Predicate-filtered wrapper around BCGrounder
```

BCGrounder is configured via three orthogonal axes (no subclasses):

| Axis | Options | Description |
|------|---------|-------------|
| **resolution** | `'sld'`, `'rtf'`, `'enum'` | How unification candidates are generated |
| **filter** | `'prune'`, `'provset'`, `'none'` | How incomplete proof branches are pruned |
| **hooks** | `ResolutionFactHook`, `ResolutionRuleHook`, `GroundingHook` | Optional neuro-symbolic callbacks |

Resolution strategies:

| Strategy | Config | K formula | How it works |
|----------|--------|-----------|--------------|
| Single-level SLD | `resolution='sld'` | `K = K_f + K_r` | Facts and rules resolved independently, children concatenated |
| Rule-Then-Fact | `resolution='rtf'` | `K = K_f * K_r` | Rules resolved first, then body atoms resolved against facts |
| Entity enumeration | `resolution='enum'` | `K = R_eff * E` | Full entity enumeration with width control |

---

## 1. BCGrounder — Unified Backward Chaining

The single backward-chaining grounder class. Configured at construction time via `resolution`, `filter`, `depth`, `width`, and optional hooks. There are no subclasses — all behaviour is composed from the configuration.

### Pipeline

SLD resolution with MGU-based unification, up to depth `D`. Each step:
1. **SELECT** the first unresolved goal atom
2. **RESOLVE** — dispatch to the configured resolution strategy (`sld`, `rtf`, or `enum`)
3. **PACK** children into fixed-size state tensor, deduplicate, truncate
4. **POSTPROCESS** — apply configured filter (`prune`, `provset`, or `none`), collect completed groundings

### Formal definition

```
Ground_BC(Q, KB, D) -> G
```

### Constructor

```python
BCGrounder(
    kb: KB,                                 # Knowledge base (facts + rules + indices)
    *,
    depth: int = 2,                         # D — max resolution steps
    width: Optional[int] = 1,              # W — max unproven body atoms (enum only; None=∞)
    resolution: str = "enum",              # 'sld' | 'rtf' | 'enum'
    filter: str = "prune",                 # 'prune' | 'provset' | 'none'
    max_total_groundings: int = 64,         # tG — output budget
    compile_mode: Optional[str] = None,     # torch.compile mode
    hooks: Optional[List] = None,           # GroundingHook list
    fact_hook = None,                       # ResolutionFactHook
    rule_hook = None,                       # ResolutionRuleHook
    max_goals: Optional[int] = None,        # G — auto-computed if None
    max_states: Optional[int] = None,       # S — auto-computed if None
)
```

### Key methods

```python
def forward(
    self,
    queries: Tensor,            # [B, 3]
    query_mask: Tensor,         # [B]
) -> GroundingResult:
    """Multi-depth proof loop."""

def _step_impl(self, proof_goals, state_valid, grounding_body, rule_idx, depth):
    """One SELECT -> RESOLVE -> PACK cycle."""
```

### Usage

```python
from grounder import KB, BCGrounder

kb = KB(facts, heads, bodies, lens,
        constant_no=C, predicate_no=P, padding_idx=pad, device=dev)

# SLD resolution with pruning (formerly PrologGrounder + BCPruneGrounder)
g = BCGrounder(kb, resolution='sld', filter='prune', depth=2)

# RTF resolution, no filter (formerly RTFGrounder)
g = BCGrounder(kb, resolution='rtf', filter='none', depth=2)

# Entity enumeration with provable-set filter (formerly ParametrizedBCGrounder + BCProvsetGrounder)
g = BCGrounder(kb, resolution='enum', filter='provset', depth=1, width=1)
```

---

## 2. BCPruneGrounder — Fixed-Point Pruning

### Algorithm

Standard BCGrounder with an additional **fixed-point pruning** pass. After grounding, iteratively removes groundings where any body atom is unprovable (not a fact and not the head of any other grounding). This is an internal consistency check — no FC required.

### Formal definition

```
Ground_Prune(Q, KB, D) -> G'  where  G' ⊆ Ground_BC(Q, KB, D)
```

### Constructor

Same as BCGrounder — no additional parameters.

```python
BCPruneGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    max_states: int = None,
    max_total_groundings: int = 64,
    max_goals: int = None,
    compile_mode: str = "reduce-overhead",
    device: str = "cuda",
)
```

### Overrides

- `_postprocess`: adds iterative fixed-point filtering after standard postprocessing

### Soundness

- **Sound**: Yes
- **Complete within D**: Yes minus pruned incomplete proofs (proofs where body atoms cannot be supported)
- **Bounds**: D, K_max

### When to use

When you want cleaner groundings that don't include dead-end proofs. Slightly more expensive than base BCGrounder due to the pruning pass.

---

## 3. BCProvsetGrounder — FC Provable Set

### Algorithm

Single-depth BC + FC provable set. Runs FC at initialization to compute the provable set `I_D`. During postprocessing, filters groundings: a grounding is kept only if all its body atoms are in `I_D`.

### Formal definition

```
Ground_Provset(Q, KB, D) -> G'
```

Uses `Ground_FC(KB, D_fc)` internally.

### Constructor

```python
BCProvsetGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,                      # BC depth
    max_states: int = None,
    max_total_groundings: int = 64,
    max_goals: int = None,
    compile_mode: str = "reduce-overhead",
    provable_set_method: str = "join",   # FC implementation
    fc_depth: int = 10,                  # FC iteration bound
    device: str = "cuda",
)
```

### Overrides

- `pre_ground`: runs FC to compute provable set
- `_postprocess`: checks body atoms against provable set

### Soundness

- **Sound**: Yes
- **Complete**: No — depth-1 rule matches only, filtered by FC provability
- **Bounds**: D (BC), D_fc (FC), K_max

### When to use

When you need provability guarantees on body atoms but don't need width control. Good for single-step grounding with strong filtering.

---

## 4. ParametrizedBCGrounder — Width + Depth Control

The most configurable BC grounder. Adds width parameter `W` controlling how many unproven body atoms are allowed per grounding.

### Algorithm

1. **Rule clustering**: group rules by head predicate, gather matching rules per query (`R_eff` per predicate)
2. **Fact-anchored enumeration (Direction A)**: for each matching rule, enumerate candidates from the first body atom with a free variable — look up facts matching `(pred, bound_arg, ?)` or `(pred, ?, bound_arg)` in the FactIndex
3. **Dual anchoring (Direction B)**: when a rule has multiple free-variable body atoms, enumerate from a second body atom as well (doubles coverage for rules like `r(X,Y) :- p(X,Z), q(Z,Y)` where Z is free)
4. **Width filtering**: count unproven body atoms per grounding. If count > W, discard. Provability checked via FC provable set (binary search on `provable_hashes`)
5. **Fixed-point pruning** (optional): iteratively remove groundings with unprovable body atoms
6. **Concatenate and flatten** to output shape `[B, tG, M, 3]`

### Formal definition

```
Ground_Param(Q, KB, D, W) -> G
```

### Constructor

```python
ParametrizedBCGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,                         # D — backward chaining depth
    width: int | None = 1,                  # W — max unproven body atoms (None = ∞)
    max_groundings_per_query: int = 32,     # Per-rule grounding budget
    max_total_groundings: int = 64,         # tG — total output budget
    prune_incomplete_proofs: bool = True,   # Enable fixed-point pruning
    provable_set_method: str = "join",      # FC implementation ("join", "spmm")
    fc_depth: int = 10,                     # FC iteration bound
    device: str = "cuda",
)
```

### Key methods

```python
def forward(
    self,
    queries: Tensor,            # [B, 3]
    query_mask: Tensor,         # [B]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Ground queries with width/depth control.

    Returns:
        body: [B, tG, M, 3]    — grounded body atoms
        mask: [B, tG]           — validity
        count: [B]              — groundings per query
        rule_idx: [B, tG]       — rule index per grounding
    """

def _check_provable(
    self,
    preds: Tensor,              # [N]
    subjs: Tensor,              # [N]
    objs: Tensor,               # [N]
) -> Tensor:                    # [N] bool
    """Check atom provability via binary search on provable_hashes."""
```

### Width semantics

| W | Behavior | Use case |
|---|----------|----------|
| `0` | Proven-only: all body atoms must be facts | Conservative grounding |
| `1` | Allow 1 unproven body atom (provability-checked via FC) | Default — good balance |
| `2` | Allow 2 unproven body atoms | Broader coverage |
| `None` | Full enumeration — delegates to FullBCGrounder | Exhaustive grounding |

### Internal features

- **Dual anchoring**: for rules with 2+ free-variable body atoms, enumerate candidates from both directions (A and B) to increase coverage
- **Rule clustering**: per-predicate rule index (`pred_rule_indices [P, R_eff]`) reduces the effective number of rules to consider per query from R to R_eff
- **Fact-anchoring**: enumerate candidate bindings from matching facts, not from all entities — dramatically reduces the search space vs FullBCGrounder

### Soundness

- **Sound**: Yes
- **Complete**: Deliberately incomplete — `Ground_Param(Q, KB, D, W) ⊆ Ground_BC(Q, KB, D)` for all finite W
- **Bounds**: D, W, K_max

### When to use

The default choice for most applications. W=1, D=2 is the practical sweet spot for KG completion.

---

## 5. FullBCGrounder — Full Entity Enumeration

### Algorithm

Full entity enumeration — enumerates ALL `E` entities for free variables, no fact-anchoring. Accepts all groundings regardless of provability. Only filter: query exclusion (body atom must not equal the query).

### Formal definition

```
Ground_Full(Q, KB) -> G
```

No depth, no width (single-step, all entities).

### Constructor

```python
FullBCGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    max_total_groundings: int = 64,     # tG — output budget
    device: str = "cuda",
)
```

### Key methods

```python
def forward(
    self,
    queries: Tensor,            # [B, 3]
    query_mask: Tensor,         # [B]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Full entity enumeration grounding.

    Output size: O(B * R_eff * E) — scales linearly with entity count.

    Returns:
        body: [B, tG, M, 3]
        mask: [B, tG]
        count: [B]
        rule_idx: [B, tG]
    """
```

### Soundness

- **Sound**: Yes
- **Complete**: Yes — enumerates all possible groundings
- **Bounds**: tG (output truncation only)

### When to use

Systems that need exact full grounding: LTN, SBR, MLNs. Automatically used by ParametrizedBCGrounder when `W >= M` or `W = None`. Warning: output explodes with entity count.

---

## 6. FCSemiNaiveGrounder — Semi-Naive Forward Chaining

### Algorithm

Semi-naive forward chaining. At each iteration, only considers rules that can fire using **newly derived** facts (delta) from the previous iteration. This avoids redundant derivation.

### Formal definition

```
Ground_FC(KB, D) -> I_D
```

`D` bounds the number of `T_P` iterations. May reach fixpoint earlier.

### Constructor

```python
FCSemiNaiveGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 10,                # D — max T_P iterations
    device: str = "cuda",
)
```

### Key methods

```python
def compute_provable_set(self) -> Tuple[Tensor, int]:
    """Iterate T_P up to depth times. Stops early at fixpoint.

    Returns:
        provable_hashes: [I_max] — sorted int64 hashes
        n_provable: int          — count of provable atoms
    """
```

### Soundness

- **Sound**: Yes
- **Complete**: Yes, if fixpoint reached within D iterations
- **Bounds**: D (iterations), I_max (capacity)

### When to use

Default FC implementation. General-purpose, works well for most rule sets. Used internally by ParametrizedBCGrounder and BCProvsetGrounder for provable set computation.

---

## 7. FCSPMMGrounder — Sparse Matrix Multiply FC

### Algorithm

Encodes rules as sparse matrices. Derives new facts via sparse matrix multiplication (SpMM). Same semantics as semi-naive — different implementation.

### Formal definition

```
Ground_FC(KB, D) -> I_D
```

Same as FCSemiNaiveGrounder.

### Constructor

Same as FCSemiNaiveGrounder.

```python
FCSPMMGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 10,
    device: str = "cuda",
)
```

### Soundness

- **Sound**: Yes
- **Complete**: Yes, if fixpoint reached within D
- **Bounds**: D, I_max

### When to use

Faster than semi-naive for dense rule sets (many rules, many derivations per iteration). Higher memory usage due to sparse matrix construction.

---

## 8. SamplerGrounder — Random Selection

Extends ParametrizedBCGrounder by oversampling then randomly selecting a subset.

### Algorithm

1. Call parent `forward()` with 4x the output budget (oversample)
2. Score: random for training (`rand(B, tG_in) * mask`), deterministic for eval (`mask.float()`)
3. Select top-k by score, gather corresponding outputs

### Constructor

```python
SamplerGrounder(
    # All ParametrizedBCGrounder args:
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    width: int | None = 1,
    max_groundings_per_query: int = 32,
    max_total_groundings: int = 64,
    prune_incomplete_proofs: bool = True,
    provable_set_method: str = "join",
    device: str = "cuda",
    # Additional:
    max_sample: int = 64,           # Output budget (parent gets 4x internally)
)
```

### Key behavior

- `effective_total_G = min(max_sample, parent.effective_total_G)`
- **Training**: random topk (exploration)
- **Eval**: deterministic valid-first (exploitation)
- Compile-safe: train/eval branching via separate CUDA graphs

### Soundness

- **Sound**: Yes — only selects from valid parent groundings
- **Complete**: No — random subset
- **Bounds**: D, W, max_sample

### When to use

Exploration during training. Avoids deterministic bias in grounding selection.

---

## 9. KGEGrounder — KGE-Scored Selection

Extends ParametrizedBCGrounder by scoring groundings with a KGE model and selecting the most plausible.

### Algorithm

1. Call parent `forward()` with 2x output budget (oversample)
2. Score each grounding's body atoms via `kge_model.score_atoms(preds, subjs, objs)`
3. Mask inactive atoms: `atom_scores[~active] = 1e9`
4. Conjunction: `scores = atom_scores.min(dim=-1)` — min across body atoms
5. Select top-k by score

### Constructor

```python
KGEGrounder(
    # All ParametrizedBCGrounder args:
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    width: int | None = 1,
    max_groundings_per_query: int = 32,
    max_total_groundings: int = 64,
    prune_incomplete_proofs: bool = True,
    provable_set_method: str = "join",
    device: str = "cuda",
    # Additional:
    kge_model: nn.Module,           # Must have .score_atoms(preds, subjs, objs) -> scores
    mode: str = "kge",              # "kge" only (use NeuralGrounder for attention)
    output_budget: int | None = None,  # Defaults to max_total_groundings
)
```

### KGE model interface

```python
class KGEModel(nn.Module):
    def score_atoms(
        self,
        preds: Tensor,          # [N]
        subjs: Tensor,          # [N]
        objs: Tensor,           # [N]
    ) -> Tensor:                # [N] — plausibility scores
        ...
```

### Soundness

- **Sound**: Yes — only selects from valid parent groundings
- **Complete**: No — top-k by KGE score
- **Bounds**: D, W, output_budget

### When to use

KGE-informed grounding selection. Prune implausible groundings based on learned entity/relation embeddings.

---

## 10. NeuralGrounder — Learned Attention Selection

Same as KGEGrounder but uses a learned attention MLP instead of KGE min-conjunction.

### Algorithm

1. Call parent `forward()` with 2x output budget
2. Embed body atoms: `emb = kge.embed_atoms(subjs, objs)` -> `[B, tG_in, M, E]`
3. Zero inactive: `emb[~active] = 0`
4. Flatten: `emb_flat = emb.reshape(B*tG_in, M*E)`
5. Score: `scores = attention_mlp(emb_flat)` -> `[B, tG_in]`
6. Select top-k

### Constructor

```python
NeuralGrounder(
    # All ParametrizedBCGrounder args:
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    width: int | None = 1,
    max_groundings_per_query: int = 32,
    max_total_groundings: int = 64,
    prune_incomplete_proofs: bool = True,
    provable_set_method: str = "join",
    device: str = "cuda",
    # Additional:
    kge_model: nn.Module,           # Must also have .embed_atoms(subjs, objs) -> [N, E]
    output_budget: int | None = None,
)
```

### Internal module

```python
class GroundingAttention(nn.Module):
    """MLP: M*E -> hidden -> 1"""

    def __init__(self, input_size: int) -> None:
        # input_size = M * embedding_dim

    def forward(self, body_emb_flat: Tensor) -> Tensor:
        """[N, M*E] -> [N, 1] attention scores"""
```

### Soundness

- **Sound**: Yes
- **Complete**: No — top-k by learned attention
- **Bounds**: D, W, output_budget

### When to use

Learn which groundings matter via end-to-end differentiable selection. More expressive than KGEGrounder's min-conjunction.

---

## 11. SoftGrounder — Soft Provability Scoring

Replaces hard provability pruning with **soft confidence scores**. Known atoms (facts or FC-provable) get score 1.0. Unknown atoms get a soft score via KGE or learned MLP.

### Algorithm

1. Call parent `forward()` with `prune_incomplete_proofs=False` (accept all groundings)
2. For each body atom in each grounding:
   - Check: `is_known = is_fact | is_provable` (via `_check_provable`)
   - If known: `score = 1.0`
   - If unknown (KGE mode): `score = sigmoid(kge.score_atoms(pred, subj, obj))`
   - If unknown (neural mode): `score = provability_mlp(kge.embed_atoms(subj, obj))`
3. Grounding confidence: `conf = score.prod(dim=-1)` — product across body atoms
4. Select top-k by confidence

### Constructor

```python
SoftGrounder(
    # All ParametrizedBCGrounder args:
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    width: int | None = 1,
    max_groundings_per_query: int = 32,
    max_total_groundings: int = 64,
    prune_incomplete_proofs: bool = False,   # NOTE: False by default (soft, not hard)
    provable_set_method: str = "join",
    device: str = "cuda",
    # Additional:
    kge_model: nn.Module,
    mode: str = "kge",              # "kge" (sigmoid) or "neural" (learned MLP)
    output_budget: int | None = None,
)
```

### Key methods

```python
def _check_provable(
    self,
    preds: Tensor,              # [N]
    subjs: Tensor,              # [N]
    objs: Tensor,               # [N]
) -> Tensor:                    # [N] bool
    """Binary search on provable_hashes to check provability."""
```

### Internal module (neural mode)

```python
class ProvabilityMLP(nn.Module):
    """MLP: E -> hidden -> 1 -> sigmoid"""

    def __init__(self, input_size: int) -> None:
        # input_size = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        """[N, E] -> [N] soft provability in [0, 1]"""
```

### Soundness

- **Sound**: Yes
- **Complete**: No — top-k by soft confidence
- **Bounds**: D, W, output_budget

### When to use

Bridge symbolic grounding with learned plausibility. No hard cutoff — unknown atoms get a soft score instead of being discarded. Best when you want gradients to flow through provability decisions.

---

## 12. LazyGrounder — Predicate Reachability Filtering

Filters rules by predicate reachability before grounding. BFS on the head→body predicate graph from query predicates. Only reachable rules are passed to the parent grounder.

### Algorithm

1. Build predicate dependency graph: for each rule, add edges from head predicate to body predicates
2. BFS from `query_predicates` to find all reachable predicates
3. Filter rules: keep only rules whose head predicate is reachable
4. Build inner ParametrizedBCGrounder with filtered rules
5. Delegate `forward()` to inner grounder

### Constructor

```python
LazyGrounder(
    # All ParametrizedBCGrounder args:
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    width: int | None = 1,
    max_groundings_per_query: int = 32,
    max_total_groundings: int = 64,
    prune_incomplete_proofs: bool = True,
    provable_set_method: str = "join",
    device: str = "cuda",
    # Additional:
    query_predicates: set[str] | None = None,  # Seed predicates for BFS (None = all)
)
```

### Key methods

```python
@staticmethod
def _compute_reachable_predicates(
    rules: List[Rule],
    query_predicates: set[str] | None = None,
) -> set[str]:
    """BFS on head->body predicate graph. Returns reachable predicate names."""
```

### Soundness

- **Sound**: Yes — only removes unreachable rules (cannot affect query results)
- **Complete**: Same as parent (ParametrizedBCGrounder with filtered rules)
- **Bounds**: D, W, reachable predicates

### When to use

Large KBs with many predicates where only a subset is relevant to the queries. Reduces rule set size and provable set computation cost.

---

## 13. Resolution: SLD (`resolution='sld'`)

Single-level SLD resolution. `K = K_f + K_r` — facts and rules are resolved **independently** against the selected goal, and children are concatenated.

This is the standard SLD resolution algorithm with full MGU-based unification (including var-var bindings).

### Algorithm

1. **Resolve facts**: look up facts matching the selected goal via FactIndex
2. **Resolve rules**: find rules whose head unifies with the goal, then assemble new goal states with `body + remaining`
3. Children from facts (`K_f`) and rules (`K_r`) are concatenated, producing `K = K_f + K_r` total children per state

### Usage

```python
grounder = BCGrounder(kb, resolution='sld', filter='prune', depth=2)
```

### Soundness

- **Sound**: Yes
- **Complete within D**: Yes, up to K_max truncation
- **Bounds**: D, K_max

### When to use

The default resolution for standard backward chaining. Use with `filter='prune'` for the most common setup.

---

## 14. Resolution: RTF (`resolution='rtf'`)

Two-level Rule-Then-Fact resolution. `K = K_f * K_r` — instead of resolving facts and rules independently (as SLD does with `K = K_f + K_r`), RTF first resolves against rule heads, then resolves the resulting body atoms against facts — producing a multiplicative combination.

### Algorithm

1. **Level 1 — Rule head unification**: resolve the selected goal against all matching rule heads. Produces `K_r` intermediate states, each containing the rule's body atoms (after substitution) concatenated with the remaining goals.
2. **Level 2 — Body-fact resolution**: for each of the `K_r` intermediate states, resolve the first body atom against facts via `targeted_lookup`, producing up to `K_f` children per intermediate state. Total: `K_r * K_f` children.

Fact resolution at the SELECT stage is empty — all work is done inside the rule resolution phase.

### Usage

```python
grounder = BCGrounder(kb, resolution='rtf', filter='prune', depth=2)
```

### Soundness

- **Sound**: Yes — every grounding is backed by valid rule head unification + fact matching
- **Complete within D**: Yes, up to `K_max` truncation
- **Bounds**: D, K = K_f * K_r, tG

### When to use

When rules have body atoms that benefit from immediate fact grounding within the same resolution step. The multiplicative `K = K_f * K_r` gives denser coverage per step at the cost of more children.

---

## 15. Factory

The factory creates grounders by parsing a dot-separated type string:

```python
def create_grounder(
    grounder_type: str,
    *,
    facts_idx: Tensor,
    rule_heads: Tensor,
    rule_bodies: Tensor,
    rule_lens: Tensor,
    constant_no: int,
    padding_idx: int,
    device: torch.device,
    predicate_no: Optional[int] = None,
    max_facts_per_query: int = 64,
    fact_index_type: str = "block_sparse",
    max_groundings: int = 32,
    max_total_groundings: int = 64,
    fc_method: str = "join",
    max_goals: int = 256,
    **kwargs,
) -> nn.Module: ...
```

### Naming convention

Format: `{resolution}[.{filter}].d{depth}[.w{width}]`

| Pattern | Config | Example |
|---------|--------|---------|
| `sld.d2` | resolution=sld, filter=none, depth=2 | `BCGrounder(kb, resolution='sld', depth=2)` |
| `sld.prune.d2` | resolution=sld, filter=prune, depth=2 | `BCGrounder(kb, resolution='sld', filter='prune', depth=2)` |
| `sld.provset.d3` | resolution=sld, filter=provset, depth=3 | `BCGrounder(kb, resolution='sld', filter='provset', depth=3)` |
| `rtf.d2` | resolution=rtf, depth=2 | `BCGrounder(kb, resolution='rtf', depth=2)` |
| `enum.prune.w1.d2` | resolution=enum, filter=prune, width=1, depth=2 | `BCGrounder(kb, resolution='enum', filter='prune', width=1, depth=2)` |
| `enum.full` | resolution=enum, all entities, depth=1 | `BCGrounder(kb, resolution='enum', width=None, depth=1)` |
| `lazy.enum.prune.w0.d1` | LazyGrounder wrapping enum | `LazyGrounder(kb, resolution='enum', ...)` |

### parse_grounder_type

```python
def parse_grounder_type(grounder_type: str) -> dict:
    """Parse grounder type string into config dict.

    'sld.prune.d2'         → {resolution:'sld', filter:'prune', depth:2, ...}
    'enum.prune.w1.d2'     → {resolution:'enum', filter:'prune', depth:2, width:1}
    'enum.full'            → {resolution:'enum', filter:'prune', depth:1, is_full:True}
    'lazy.enum.prune.w0.d1'→ {resolution:'enum', filter:'prune', depth:1, width:0, is_lazy:True}
    """
```
