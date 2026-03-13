# Grounders

Every grounder variant in NeSyGround, with algorithm, formal definition, full constructor signature, key methods, soundness properties, and usage guidance.

---

## Class Hierarchy

```
Grounder (nn.Module)                        # KB ownership (facts_idx, rule_index, etc.)
├── BCGrounder (abstract)                   # Pipeline skeleton: SELECT → RESOLVE → PACK
│   ├── PrologGrounder                      # Single-level SLD (K = K_f + K_r)
│   ├── RTFGrounder                         # Two-level Rule-Then-Fact (K = K_f * K_r)
│   ├── BCPruneGrounder                     # + fixed-point pruning
│   ├── BCProvsetGrounder                   # + FC provable set filtering
│   ├── ParametrizedBCGrounder              # + width control (W)
│   │   ├── SamplerGrounder                 # + random selection
│   │   ├── KGEGrounder                     # + KGE-scored selection
│   │   ├── NeuralGrounder                  # + learned attention selection
│   │   ├── SoftGrounder                    # + soft provability scoring
│   │   └── LazyGrounder                    # + predicate reachability filtering
│   └── FullBCGrounder                      # Full entity enumeration
├── FCSemiNaiveGrounder                     # Semi-naive forward chaining
│   └── FCSPMMGrounder                      # SpMM-based forward chaining
```

Two concrete resolution strategies implement BCGrounder's abstract `_resolve_facts` and `_resolve_rules`:

| Strategy | Class | K formula | How it works |
|----------|-------|-----------|--------------|
| Single-level | **PrologGrounder** | `K = K_f + K_r` | Facts and rules resolved independently, children concatenated |
| Two-level | **RTFGrounder** | `K = K_f * K_r` | Rules resolved first, then body atoms resolved against facts |

---

## 1. BCGrounder — Abstract Pipeline Base

The abstract base class for all backward chaining grounders. BCGrounder defines the 5-stage pipeline skeleton (SELECT, RESOLVE FACTS, RESOLVE RULES, PACK, POSTPROCESS) and implements the shared infrastructure: proof loop, state management, compilation, and the shared `_resolve_rule_heads()` method.

BCGrounder does **not** implement `_resolve_facts` or `_resolve_rules` — these are abstract and must be provided by a concrete subclass. The two concrete resolution strategies are:

- **PrologGrounder** (section 13): single-level `K = K_f + K_r`, facts and rules resolved independently
- **RTFGrounder** (section 14): two-level `K = K_f * K_r`, rules first then body atoms against facts

All other BC grounders (BCPrune, BCProvset, Parametrized, etc.) extend BCGrounder by adding filtering, scoring, or width control on top of the pipeline, and use one of these two concrete strategies for resolution.

### Pipeline

SLD resolution with MGU-based unification, up to depth `D`. Each step:
1. **SELECT** the first unresolved goal atom
2. **RESOLVE FACTS** (abstract) — match goal against facts
3. **RESOLVE RULES** (abstract) — match goal against rule heads, substitute body
4. **PACK** children into fixed-size state tensor, deduplicate, truncate
5. **POSTPROCESS** — prune ground facts, collect completed groundings

### Formal definition

```
Ground_BC(Q, KB, D) -> G
```

### Constructor

```python
BCGrounder(
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,                         # D — max resolution steps
    max_states: int = None,                 # S — states per query (auto-computed if None)
    max_total_groundings: int = 64,         # tG — output budget
    max_goals: int = None,                  # G — auto-computed as 1 + D*(M-1) if None
    compile_mode: str = "reduce-overhead",  # torch.compile mode
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
    """Multi-depth proof loop.

    Returns:
        body: [B, tG, M, 3]
        mask: [B, tG]
        count: [B]
        rule_idx: [B, tG]
    """

def _step_impl(self, proof_goals, state_valid, grounding_body, rule_idx, depth):
    """One SELECT -> RESOLVE -> PACK cycle."""

# --- Abstract methods (must be implemented by subclasses) ---

def _resolve_facts(self, queries, remaining, grounding_body, state_valid, active_mask):
    """Resolve selected goal against facts. Returns (goals, gbody, success)."""

def _resolve_rules(self, queries, remaining, grounding_body, state_valid, active_mask, next_var_indices):
    """Resolve selected goal against rules. Returns (goals, gbody, success, rule_idx)."""

# --- Shared infrastructure (used by both PrologGrounder and RTFGrounder) ---

def _resolve_rule_heads(self, queries, remaining, grounding_body, state_valid, active_mask, next_var_indices):
    """Level-1 rule head unification: lookup + standardize + unify + apply subs.
    Returns: (rule_body_subst, rule_remaining, rule_gbody, rule_success, sub_rule_idx, sub_lens, Bmax)
    """
```

### Not instantiated directly

BCGrounder is not used directly — instantiate PrologGrounder or RTFGrounder (or one of their descendants) instead.

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

## 13. PrologGrounder — Single-Level SLD Resolution

The concrete single-level resolution strategy for BCGrounder. `K = K_f + K_r` — facts and rules are resolved **independently** against the selected goal, and children are concatenated.

This is the standard SLD resolution algorithm with full MGU-based unification (including var-var bindings).

### Algorithm

1. **Resolve facts**: look up facts matching the selected goal via FactIndex. Two paths depending on the fact index type:
   - `ArgKeyFactIndex`: targeted lookup (`_resolve_facts_argkey`)
   - `InvertedFactIndex` / `BlockSparseFactIndex`: enumeration-based (`_resolve_facts_enumerate`)
2. **Resolve rules**: find rules whose head unifies with the goal via shared `_resolve_rule_heads()`, then assemble new goal states with `body + remaining`
3. Children from facts (`K_f`) and rules (`K_r`) are passed to PACK as separate sets, producing `K = K_f + K_r` total children per state

### Formal definition

```
Ground_BC(Q, KB, D) -> G
```

### Constructor

```python
PrologGrounder(
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

### Implements

- `_resolve_facts`: fact matching with full MGU (var-var bindings), dispatches to argkey or enumerate path
- `_resolve_rules`: rule head unification via `_resolve_rule_heads()` + goal assembly (body atoms prepended to remaining goals)

### Soundness

- **Sound**: Yes
- **Complete within D**: Yes, up to K_max truncation
- **Bounds**: D, K_max

### When to use

The default concrete BC grounder. Use when you need standard SLD resolution with additive `K = K_f + K_r` children. This is typically the base resolution strategy used by BCPruneGrounder, BCProvsetGrounder, and other pipeline-extending grounders.

---

## 14. RTFGrounder — Two-Level Rule-Then-Fact Resolution

The alternative concrete resolution strategy for BCGrounder. `K = K_f * K_r` — instead of resolving facts and rules independently (as PrologGrounder does with `K = K_f + K_r`), RTFGrounder first resolves against rule heads, then resolves the resulting body atoms against facts — producing a multiplicative combination.

### Algorithm

1. **Level 1 — Rule head unification**: resolve the selected goal against all matching rule heads via shared `_resolve_rule_heads()`. Produces `K_r` intermediate states, each containing the rule's body atoms (after substitution) concatenated with the remaining goals.
2. **Level 2 — Body-fact resolution**: for each of the `K_r` intermediate states, resolve the first body atom against facts via `targeted_lookup`, producing up to `K_f` children per intermediate state. Total: `K_r * K_f` children.

Fact resolution at the SELECT stage is empty — all work is done inside `_resolve_rules`.

### Formal definition

```
Ground_RTF(Q, KB, D) -> G
```

Same soundness as BCGrounder, different resolution strategy.

### Constructor

```python
RTFGrounder(
    # All BCGrounder args:
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    depth: int = 2,
    max_states: int = None,
    max_total_groundings: int = 64,
    max_goals: int = None,
    compile_mode: str = "reduce-overhead",
    device: str = "cuda",
    # Additional:
    body_order_agnostic: bool = False,  # Try all body atom positions against facts
    rtf_cascade: bool = False,          # Cascade: resolve multiple body atoms sequentially
)
```

### Options

**`body_order_agnostic`**: when `False` (default), only the first body atom is resolved against facts at level 2. When `True`, each body atom position is tried against facts independently, and results are concatenated. This produces `M * K_r * K_f` children but catches more groundings when the optimal resolution order is unknown.

**`rtf_cascade`**: when `True`, level-2 resolution cascades through all body atoms sequentially. After resolving the first body atom against facts, the second body atom of each surviving state is resolved against facts, and so on. This produces deeper resolution within a single step but is more expensive. Chunked execution (`cascade_step_chunked`) controls memory for large batches.

### Overrides

- `_compute_K_uncapped`: returns `K_f * K_r` (multiplicative, not additive)
- `_resolve_facts`: returns empty tensors (no standalone fact resolution)
- `_resolve_rules`: two-level resolution — rule heads then body-fact matching

### Key helper functions (module-level)

```python
def resolve_body_atom_with_facts(
    eng: RTFGrounder,
    atoms: Tensor,              # [B, K_r, 3] — body atoms to resolve
    remaining: Tensor,          # [B, K_r, G, 3] — remaining goals
    rule_success: Tensor,       # [B, K_r] — validity mask
    excluded_queries: Tensor | None,
) -> Tuple[Tensor, Tensor]:    # (derived_states [B, K_r*K_f, G, 3], valid [B, K_r*K_f])
    """Resolve a batch of atoms against facts using targeted_lookup."""

def resolve_rules_with_facts(
    eng: RTFGrounder,
    rule_states: Tensor,        # [B, K_r, M, 3] — intermediate states from level 1
    rule_success: Tensor,       # [B, K_r] — validity mask
    excluded_queries: Tensor | None,
) -> Tuple[Tensor, Tensor]:    # (resolved_states, resolved_valid)
    """Full two-level resolution. Dispatches to body_order_agnostic or cascade."""

def cascade_step_chunked(
    eng: RTFGrounder,
    cap_states: Tensor,         # [B, K_cap, G_cur, 3]
    cap_ok: Tensor,             # [B, K_cap]
    excluded_queries: Tensor | None,
    K_budget: int,
) -> Tuple[Tensor, Tensor]:
    """B-chunked cascade step with memory-aware execution."""
```

### Soundness

- **Sound**: Yes — every grounding is backed by valid rule head unification + fact matching
- **Complete within D**: Yes, up to `K_max` truncation (same as BCGrounder)
- **Bounds**: D, K = K_f * K_r, tG

### When to use

When rules have body atoms that benefit from immediate fact grounding within the same resolution step. The multiplicative `K = K_f * K_r` gives denser coverage per step at the cost of more children. Particularly useful with `rtf_cascade=True` for rules with multiple body atoms that can be resolved against facts in sequence. Also used by the RL layer (`RTFEngine`) which shares the module-level helper functions.

---

## 15. Factory

The factory creates grounders by parsing a type string:

```python
def create_grounder(
    grounder_type: str,
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    max_groundings: int,
    max_total_groundings: int,
    provable_set_method: str = "join",
    device: str = "cuda",
    kge_model: nn.Module | None = None,
) -> Grounder: ...
```

### Registry naming conventions

| Pattern | Grounder | Example |
|---------|----------|---------|
| `bc_{D}` | BCGrounder | `bc_2` |
| `bcprune_{D}` | BCPruneGrounder | `bcprune_2` |
| `bcprovset_{D}` | BCProvsetGrounder | `bcprovset_2` |
| `bcprolog_{D}` | PrologGrounder | `bcprolog_2` |
| `rtf_{D}` | RTFGrounder | `rtf_2` |
| `backward_{W}_{D}` | ParametrizedBCGrounder | `backward_1_2` |
| `full` | FullBCGrounder | `full` |
| `lazy_{W}_{D}` | LazyGrounder | `lazy_0_1` |
| `sampler_{W}_{D}` | SamplerGrounder | `sampler_1_2` |
| `kge_{W}_{D}` | KGEGrounder | `kge_1_2` |
| `neural_{W}_{D}` | NeuralGrounder | `neural_1_2` |
| `soft_{W}_{D}` | SoftGrounder | `soft_1_2` |

### parse_grounder_type

```python
def parse_grounder_type(grounder_type: str) -> Tuple[int, int]:
    """Parse grounder type string into (width, depth).

    "bc_2"          -> (1, 2)     # BC grounders: width defaults to 1
    "backward_1_2"  -> (1, 2)
    "lazy_0_1"      -> (0, 1)
    "full"          -> (None, 1)
    """
```
