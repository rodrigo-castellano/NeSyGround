# grounder

Unified backward-chaining Prolog grounder for knowledge graph reasoning.

Provides a fully compilable unification engine with fixed-shape tensors, masked operations, and `torch.compile` / CUDA graph compatibility.

## Usage

```python
from grounder import PrologGrounder

grounder = PrologGrounder(
    facts_idx=facts,          # [F, 3] (pred, arg0, arg1)
    rules_heads_idx=heads,    # [R, 3]
    rules_bodies_idx=bodies,  # [R, Bmax, 3]
    rule_lens=lens,           # [R]
    constant_no=C,
    padding_idx=P,
    device=dev,
    max_goals=G,
    depth=2,
)
result = grounder(queries, query_mask)
# result.collected_body:  [B, tG, M, 3]
# result.collected_mask:  [B, tG]
# result.collected_count: [B]
# result.collected_ridx:  [B, tG]
```

## Class Hierarchy

```
Grounder(nn.Module)          - base: owns KB state (facts, rules, indices)
  +- BCGrounder              - backward chaining: step() + multi-depth forward()
     +- PrologGrounder       - K = K_f + K_r, independent fact + rule resolution
     +- RTFGrounder          - K = K_f * K_r, two-level Rule-Then-Fact
```

## Package Structure

```
grounder/
+-- __init__.py           # Public exports
+-- grounder.py           # Class hierarchy
+-- primitives.py         # apply_substitutions, unify_one_to_one
+-- fact_index.py         # ArgKeyFactIndex, InvertedFactIndex, BlockSparseFactIndex
+-- rule_index.py         # RuleIndex (segment + table lookup)
+-- operations.py         # mgu_resolve_atom_facts, mgu_resolve_atom_rules
+-- packing.py            # pack_combined, compact_atoms, pack_fact_rule
+-- postprocessing.py     # prune_ground_facts, collect_groundings, dedup_groundings
+-- standardization.py    # standardize_vars_offset, standardize_vars_canonical
+-- types.py              # ForwardResult, StepResult, etc.
+-- tests/
    +-- test_primitives.py
    +-- test_fact_index.py
    +-- test_grounder.py
    +-- test_packing.py
```

## Testing

```bash
PYTHONPATH=/path/to/parent python -m pytest grounder/tests/ -v
```

## Tensor Conventions

- States: `[B, S, G, 3]` where B=batch, S=states, G=goals, 3=(pred, arg0, arg1)
- Constants: indices `0..constant_no`
- Variables: indices `>= constant_no + 1`
- Padding: `padding_idx` (must be outside constant/variable range)
