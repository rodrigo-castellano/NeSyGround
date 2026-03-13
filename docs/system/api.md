# API Reference

Complete type and function specifications for NeSyGround. Every class, method, argument, type, default, and tensor shape.

---

## 1. Core Types

### KnowledgeBase

Holds the symbolic knowledge base: predicates, entity types, facts, and (optionally) rules.

```python
class KnowledgeBase:
    """Symbolic knowledge base with predicates, entities, and facts."""

    predicates: List[Predicate]
    name_to_predicate: Dict[str, Predicate]
    name_to_predicate_idx: Dict[str, int]
    domains: List[EntityType]
    name2domain: Dict[str, EntityType]
    facts: List[Tuple[str, str, str]]

    def __init__(
        self,
        domains: List[EntityType],
        predicates: List[Predicate],
        facts: Iterable[Union[Atom, str, Tuple]] = None,
        parse_format: str = "functional",
        entity_to_type_name: Dict[str, str] = None,
    ) -> None: ...

    @staticmethod
    def from_facts(
        facts: List[Tuple[str, str, str]],
        constants: List[str] = None,
        type_to_entities: Dict[str, List[str]] = None,
    ) -> "KnowledgeBase": ...

    def num_entities(self) -> int: ...
    def num_predicates(self) -> int: ...
```

### EntityType

```python
class EntityType:
    """A named set of entities (constants) sharing a domain."""

    name: str                       # Domain name (e.g., "person")
    constants: List[str]            # Entity instances (e.g., ["john", "mary"])
    has_features: bool = False      # Whether entities carry feature vectors
```

### Predicate

```python
class Predicate:
    """A named binary relation over entity types."""

    name: str                       # Predicate name (e.g., "father")
    domains: List[EntityType]       # [subject_type, object_type]
    arity: int                      # Always 2 in NeSyGround
    has_features: bool = False
```

### Atom

```python
class Atom:
    """A predicate applied to arguments (constants or variables)."""

    relation: str                   # Predicate name
    args: List[str]                 # [arg0, arg1] — constants or variables

    def __init__(
        self,
        relation: str = None,
        args: List[str] = None,
        s: str = None,
        parse_format: str = "functional",
    ) -> None: ...

    def ground(self, var_assignments: Dict[str, str]) -> bool: ...
    def to_tuple(self) -> Tuple[str, ...]: ...
```

### Rule

```python
class Rule:
    """A Horn clause: head :- body_1, body_2, ..., body_m."""

    name: str                       # Rule identifier
    weight: float = 1.0             # Rule weight (1.0 = hard rule)
    head: List[Tuple]               # Single head atom as [(pred, arg0, arg1)]
    body: List[Tuple]               # Body atoms as [(pred, arg0, arg1), ...]
    vars: List[str]                 # All variables (sorted)
    var_to_entity_type: OrderedDict[str, str]  # Variable -> domain mapping

    @property
    def head_atom(self) -> Tuple[str, str, str]:
        """The single head atom (always exactly one)."""

    @property
    def hard(self) -> bool:
        """True if weight == 1.0."""

    def __init__(
        self,
        name: str = None,
        weight: float = 1.0,
        body: List[str] = None,
        head: List[str] = None,
        body_atoms: List[Tuple] = None,
        head_atoms: List[Tuple] = None,
        var2domain: Dict[str, str] = None,
        s: str = None,
        parse_format: str = "functional",
    ) -> None: ...
```

### RuleGroundings

```python
class RuleGroundings:
    """Ground instantiations of a single rule."""

    name: str                                       # Rule name
    groundings: Iterable[GroundFormula]              # All groundings
    query2groundings: Dict[Query, GroundFormulas]    # Per-query groundings

    def __init__(
        self,
        name: str,
        groundings: Iterable[GroundFormula],
        query2groundings: Dict[Query, GroundFormulas] = {},
    ) -> None: ...
```

### Type aliases

```python
GroundAtom = Tuple[str, str, str]                       # (pred, arg0, arg1)
Query = GroundAtom
GroundHead = Tuple[GroundAtom]
GroundBody = Tuple[GroundAtom, ...]
GroundFormula = Tuple[GroundHead, GroundBody]            # (head, body)
GroundFormulas = Iterable[GroundFormula]
PerQueryGroundFormulas = Dict[Query, GroundFormulas]
```

---

## 2. Compiled Structures

### CompiledRule

Pre-processed rule with all metadata needed for tensor-based grounding.

```python
class CompiledRule:
    """A rule compiled into tensor-friendly metadata."""

    name: str                           # Rule name
    rule: Rule                          # Original rule object
    head_pred_idx: int                  # Predicate index of head
    head_var0: str                      # First head variable name
    head_var1: str                      # Second head variable name
    num_body: int                       # Number of body atoms
    body_pred_indices: List[int]        # Predicate indices (processing order)
    free_vars_list: List[str]           # Free variables (not in head)
    num_free: int                       # Count of free variables

    # Per-body-atom metadata (in processing order)
    body_patterns: List[Dict]           # Each dict has:
        # "pred_idx": int              — predicate index
        # "arg0_binding": int          — binding constant for arg0
        # "arg1_binding": int          — binding constant for arg1

    body_order: List[int]               # Original indices in processing order
    enum_meta: List[Dict]               # Each dict has:
        # "introduces_fv": int         — free var index introduced (-1 if none)
        # "enum_bound_src": int        — binding source for enumeration
        # "enum_direction": int        — 0 = enumerate objects, 1 = enumerate subjects
        # "enum_pred": int             — predicate to enumerate against

    def __init__(self, rule: Rule, pred_to_idx: Dict[str, int]) -> None: ...
```

**Binding constants:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `BINDING_HEAD_VAR0` | `0` | Argument is the first head variable |
| `BINDING_HEAD_VAR1` | `1` | Argument is the second head variable |
| `BINDING_FREE_VAR_OFFSET` | `2` | Free variable indices start here (2, 3, 4, ...) |
| `BINDING_NO_FREE_VAR` | `-1` | No free variable introduced by this body atom |

### Compiled KB Initialization

The `_bc_init` function compiles rules and registers all tensors as buffers on a grounder module:

```python
def _bc_init(
    module: nn.Module,              # Target grounder module
    fact_index: FactIndex,          # Fact index implementation
    rules: List[Rule],              # Symbolic rules
    kb: KnowledgeBase,              # Knowledge base
    depth: int,                     # Proof depth
    max_total_groundings: int,      # Output capacity
    max_goals: int,                 # Max goals per state
) -> None:
    """Compile rules and register tensor buffers on module.

    Registered buffers:
        head_preds: [R]                    — head predicate per rule
        body_preds: [R, M]                 — body predicates (padded)
        num_body_atoms: [R]                — actual body sizes
        has_free: [R]                      — whether rule has free variables
        enum_pred_a: [R]                   — enumeration predicate (direction A)
        enum_bound_binding_a: [R]          — bound argument source (direction A)
        enum_direction_a: [R]              — 0=objects, 1=subjects (direction A)
        check_arg_source_a: [R, M, 2]     — argument binding sources
        num_free_vars: [R]                 — count of free variables
        body_introduces_fv: [R, M]         — free var index per body atom
        body_enum_bound_src: [R, M]        — cascade bound source per body atom
        body_enum_direction: [R, M]        — cascade direction per body atom
        body_enum_pred: [R, M]             — cascade predicate per body atom
        pred_rule_indices: [P, R_eff]      — per-predicate rule mapping
        pred_rule_mask: [P, R_eff]         — rule validity per predicate

    Set attributes:
        fact_index, kb, depth, max_total_groundings, max_goals
        pred_to_idx: Dict[str, int]
        compiled_rules: List[CompiledRule]
        num_rules: int
        max_body_atoms: int (M)
        F_max: int
        R_eff: int
    """
```

---

## 3. FactIndex Protocol

The fact index provides efficient lookup and enumeration of facts. Three implementations are available, each optimized for different access patterns.

```python
@runtime_checkable
class FactIndex(Protocol):
    """Protocol for fact storage and retrieval."""

    facts_idx: Tensor               # [F, 3] all stored facts
    fact_hashes: Tensor             # [F] sorted int64 hashes
    pack_base: int                  # Base for pack_triples_64
```

### Methods

```python
def targeted_lookup(
    self,
    query_atoms: Tensor,            # [B, 3] — atoms to match against
    max_results: int,                # Maximum matches to return per query
) -> Tuple[Tensor, Tensor]:
    """Find facts matching each query atom.

    Returns:
        fact_indices: [B, K] — indices into facts_idx
        valid: [B, K] — boolean mask of valid matches
    """

def enumerate(
    self,
    preds: Tensor,                   # [N] — predicate indices
    bound_args: Tensor,              # [N] — bound argument values
    direction: Tensor,               # [N] — 0=enumerate objects, 1=enumerate subjects
) -> Tuple[Tensor, Tensor]:
    """Enumerate candidate bindings for free variables.

    Given a predicate and one bound argument, returns all entities
    that appear in the other argument position in matching facts.

    Returns:
        candidates: [N, K_f] — candidate entity indices
        valid: [N, K_f] — boolean mask of valid candidates
    """

def exists(
    self,
    preds: Tensor,                   # [N] — predicate indices
    subjs: Tensor,                   # [N] — subject indices
    objs: Tensor,                    # [N] — object indices
) -> Tensor:
    """Membership test. Returns [N] bool via binary search on fact_hashes."""
```

### Implementations

| Implementation | Access pattern | Lookup | Enumeration | Best for |
|---------------|---------------|--------|-------------|----------|
| `ArgKeyFactIndex` | BE-style targeted lookup | O(log F) | N/A | BCGrounder with targeted fact resolution |
| `InvertedFactIndex` | TS-style enumeration | N/A | O(K_f) | ParametrizedBCGrounder fact-anchored enumeration |
| `BlockSparseFactIndex` | Dense block reads | O(K) | O(1) amortized | Large KBs where memory allows dense blocks |

**ArgKeyFactIndex constructor:**

```python
def __init__(
    self,
    facts_idx: Tensor,              # [F, 3] facts
    constant_no: int,               # Number of entities (E)
    padding_idx: int,               # Padding value
    device: torch.device,
    pack_base: int = None,          # Defaults to constant_no + 2
) -> None: ...
```

**InvertedFactIndex constructor:**

```python
def __init__(
    self,
    facts_idx: Tensor,              # [F, 3] facts
    constant_no: int,
    padding_idx: int,
    device: torch.device,
    num_entities: int,              # E
    num_predicates: int,            # P
    max_facts_per_query: int = 64,  # K_f — output pad size
) -> None: ...
```

**BlockSparseFactIndex constructor:**

```python
def __init__(
    self,
    facts_idx: Tensor,              # [F, 3] facts
    constant_no: int,
    padding_idx: int,
    device: torch.device,
    num_entities: int,
    num_predicates: int,
    max_facts_per_query: int = 64,
    max_memory_mb: int = 256,       # Memory budget for dense blocks
) -> None: ...
```

---

## 4. Grounder Base

All grounders are `nn.Module` subclasses. The base interface:

```python
class Grounder(nn.Module):
    """Base class for all grounders. Owns the compiled KB."""

    # Attributes (set by _bc_init or constructor)
    fact_index: FactIndex
    depth: int
    max_total_groundings: int
    effective_total_G: int          # Actual output grounding capacity
    max_body_atoms: int             # M
    num_rules: int                  # R

    def __init__(
        self,
        fact_index: FactIndex,
        rules: List[Rule],
        kb: KnowledgeBase,
        depth: int,
        max_total_groundings: int,
        device: str = "cuda",
    ) -> None: ...

    def forward(
        self,
        queries: Tensor,            # [B, 3]
        query_mask: Tensor,         # [B] bool
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute groundings for a batch of queries.

        Returns:
            body: [B, tG, M, 3]    — grounded rule body atoms
            mask: [B, tG]           — validity mask
            count: [B]              — number of valid groundings per query
            rule_idx: [B, tG]       — which rule produced each grounding
        """
```

---

## 5. Backward Chaining Pipeline Methods

The BC pipeline consists of three stages: **SELECT**, **RESOLVE**, **PACK**. SELECT and PACK are implemented on BCGrounder (the abstract base). RESOLVE is split into two abstract methods (`_resolve_facts`, `_resolve_rules`) that must be implemented by a concrete subclass — PrologGrounder (single-level, `K = K_f + K_r`) or RTFGrounder (two-level, `K = K_f * K_r`). The shared `_resolve_rule_heads()` method on BCGrounder handles level-1 head unification for both strategies.

All shapes are fixed at construction time.

### SELECT

```python
def _select(
    self,
    proof_goals: Tensor,            # [B, S, G, 3] — current proof states
) -> Tuple[Tensor, Tensor, Tensor]:
    """Select the next goal to resolve from each state.

    Returns:
        query: [B, S, 3]           — selected goal atom
        remaining: [B, S, G, 3]    — remaining goals after selection
        active_mask: [B, S]        — which states have goals to resolve
    """
```

### RESOLVE

```python
def _resolve_facts(
    self,
    query: Tensor,                  # [B, S, 3] — selected goal
    remaining: Tensor,              # [B, S, G, 3] — remaining goals
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resolve goals against facts in the KB.

    Returns:
        children: [B, S, K_f, G, 3]   — child states from fact matches
        valid: [B, S, K_f]             — validity mask
        matched_facts: [B, S, K_f, 3]  — the facts that matched
    """

def _resolve_rules(
    self,
    query: Tensor,                  # [B, S, 3]
    remaining: Tensor,              # [B, S, G, 3]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve goals against rule heads in the KB.

    Returns:
        children: [B, S, K_r, G, 3]   — child states from rule matches
        valid: [B, S, K_r]             — validity mask
        rule_idx: [B, S, K_r]          — which rule matched
        body: [B, S, K_r, M, 3]       — instantiated rule bodies
    """

def _resolve_rule_heads(
    self,
    query: Tensor,                  # [B, S, 3]
    state_valid: Tensor,            # [B, S]
) -> Tuple[...]:                    # 7-tuple of unification results
    """Shared level-1 rule head unification.

    Finds all rules whose head predicate matches the selected goal
    and computes the MGU for head arguments.
    """
```

### PACK

```python
def _pack_step(
    self,
    fact_children: Tensor,          # [B, S, K_f, G, 3]
    rule_children: Tensor,          # [B, S, K_r, G, 3]
    fact_valid: Tensor,             # [B, S, K_f]
    rule_valid: Tensor,             # [B, S, K_r]
    prev_states: Tensor,            # [B, S, G, 3]
) -> Tuple[Tensor, Tensor]:
    """Compact children into fixed-size state tensor.

    Merges fact and rule children (S * K total), deduplicates,
    truncates to S slots, and detects terminal states.

    Returns:
        new_states: [B, S, G, 3]   — compacted proof states
        state_valid: [B, S]        — which states are active
    """
```

### Post-processing

```python
def _postprocess(
    self,
    states: Tensor,                 # [B, S, G, 3]
    state_valid: Tensor,            # [B, S]
    grounding_body: Tensor,         # [B, S, M, 3]
    rule_idx: Tensor,               # [B, S]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract completed groundings from terminal states.

    Returns:
        body: [B, tG, M, 3]       — grounded body atoms
        mask: [B, tG]              — validity
        count: [B]                 — count per query
        rule_idx: [B, tG]          — rule index per grounding
    """
```

### Full step

```python
def _step_impl(
    self,
    proof_goals: Tensor,            # [B, S, G, 3]
    state_valid: Tensor,            # [B, S]
    grounding_body: Tensor,         # [B, S, M, 3]
    rule_idx: Tensor,               # [B, S]
    depth: int,                     # Current depth
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """One complete BC step: SELECT -> RESOLVE -> PACK.

    Returns: updated (proof_goals, state_valid, grounding_body, rule_idx)
    with same shapes as inputs.
    """
```

---

## 6. Forward Chaining Pipeline Methods

FC computes the provable set iteratively. The pipeline has three stages: **MATCH**, **JOIN**, **MERGE**.

All shapes fixed. `I_max` (max provable atoms) determined at construction.

### Top-level

```python
def compute_provable_set(
    self,
    depth: int = None,              # Max iterations (defaults to constructor depth)
) -> Tuple[Tensor, int]:
    """Iterate T_P up to depth times. May reach fixpoint earlier.

    Returns:
        provable_hashes: [I_max]   — sorted int64 hashes of provable atoms
        n_provable: int            — number of valid entries
    """
```

### MATCH

```python
def match_rules(
    self,
    delta_hashes: Tensor,           # [delta_max] — newly derived atom hashes
    n_delta: int,                   # Number of valid delta entries
) -> Tuple[Tensor, int]:
    """Identify rules that can fire using delta facts (semi-naive).

    Returns:
        applicable: [A_max, ...]   — applicable rule instantiations
        n_applicable: int          — number of valid entries
    """
```

### JOIN

```python
def join(
    self,
    applicable: Tensor,             # [A_max, ...] — applicable rule instantiations
    n_applicable: int,
) -> Tuple[Tensor, int]:
    """Bind rule body atoms against fact set, compute head instantiations.

    Returns:
        new_heads: [H_max, 3]      — newly derived head atoms
        n_new: int                  — number of new atoms
    """
```

### MERGE

```python
def merge(
    self,
    current_hashes: Tensor,         # [I_max] — current provable set hashes
    n_current: int,
    new_atoms: Tensor,              # [H_max, 3] — new atoms to merge
    n_new: int,
) -> Tuple[Tensor, int, bool]:
    """Deduplicate and merge new atoms into provable set.

    Returns:
        updated_hashes: [I_max]    — updated sorted hashes
        n_updated: int             — new count
        is_fixpoint: bool          — True if no new atoms were added
    """
```

---

## 7. GroundingResult

The output of all grounders follows a uniform format:

```python
# Forward return tuple (not a named class — direct tensor tuple)
Tuple[
    body: Tensor,                   # [B, tG, M, 3] — grounded rule body atoms
    mask: Tensor,                   # [B, tG] — validity mask (bool)
    count: Tensor,                  # [B] — number of valid groundings per query
    rule_idx: Tensor,               # [B, tG] — which rule produced each grounding
]
```

**Dimension meanings:**

| Dim | Symbol | Description |
|-----|--------|-------------|
| 0 | `B` | Batch (one per query) |
| 1 | `tG` | Total grounding slots per query (`effective_total_G`) |
| 2 | `M` | Body atoms per grounding (`max_body_atoms`) |
| 3 | `3` | Triple components `(pred_idx, arg0_idx, arg1_idx)` |

The `mask` tensor determines which grounding slots contain valid data. Invalid slots are filled with `padding_idx`.

---

## 8. Factory

```python
def create_grounder(
    grounder_type: str,              # e.g., "bcprune_2", "lazy_0_1", "kge_1_2"
    fact_index: FactIndex,
    rules: List[Rule],
    kb: KnowledgeBase,
    max_groundings: int,             # Per-query budget
    max_total_groundings: int,       # Total output budget
    provable_set_method: str = "join",
    device: str = "cuda",
    kge_model: nn.Module = None,     # Required for kge/neural/soft grounders
) -> Grounder: ...

def parse_grounder_type(
    grounder_type: str,
) -> Tuple[int, int]:
    """Parse grounder type string into (width, depth).

    Examples:
        "bc_2"          -> (1, 2)
        "backward_1_2"  -> (1, 2)
        "lazy_0_1"      -> (0, 1)
        "full"          -> (None, 1)
    """
```

---

## 9. Module-Level Utilities

```python
def pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """[N, 3] -> [N] int64 hash keys.

    Formula: ((pred * base) + arg0) * base + arg1
    """

def fact_contains(
    atoms: Tensor,                   # [N, 3]
    fact_hashes: Tensor,             # [F] sorted
    pack_base: int,
) -> Tensor:                        # [N] bool
    """Membership test via searchsorted on sorted hashes."""
```
