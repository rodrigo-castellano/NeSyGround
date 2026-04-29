# grounder

Unified backward-chaining Prolog grounder for knowledge graph reasoning.

Provides a fully compilable unification engine with fixed-shape tensors, masked operations, and `torch.compile` / CUDA graph compatibility.

## Usage

```python
from grounder import KB, BCGrounder

kb = KB(facts, heads, bodies, lens,
        constant_no=C, predicate_no=P, padding_idx=P, device=dev)
grounder = BCGrounder(kb, resolution='sld', filter='fp_batch', depth=2)
result = grounder(queries, query_mask)
# result.collected_body:  [B, tG, M, 3]
# result.collected_mask:  [B, tG]
# result.collected_count: [B]
# result.collected_ridx:  [B, tG]
```

## Class Hierarchy

```
Grounder(nn.Module)          - base: owns a KB reference
  +- BCGrounder              - backward chaining, configured via resolution/filter/depth
  +- LazyGrounder            - predicate-filtered wrapper around BCGrounder
```

Resolution is configured, not subclassed:
- `resolution='sld'`: K = K_f + K_r, independent fact + rule resolution
- `resolution='rtf'`: K = K_f * K_r, two-level Rule-Then-Fact
- `resolution='enum'`: full entity enumeration

### BC_{w,d} grounders need `filter='fp_batch'` and `all_anchors=True`

For the `enum` resolution, the BC_{w,d} family (e.g. `enum.fp_batch.w1.d2`) requires both:

1. **`filter='fp_batch'`** (the default for `enum`). With `filter='none'` the grounder admits rule applications whose unknown body atoms cannot be derived through the chain — these are pruned by keras-ns's `prune_incomplete_proofs=True` for `depth>1`. The `fp_batch` filter applies the equivalent Kleene fixed-point pruning over `_r2g_buffer` so `out.rule_groundings` matches keras-ns rule-by-rule.
2. **`all_anchors=True`** (forced by `BCGrounder.__init__` for `enum`). Anchoring only on the first body atom (e.g. `nb` in `nb(X,Y), loc(Y,Z) → loc(X,Z)`) misses bindings keras-ns finds when iterating each body position as anchor — anchoring on `loc(Y,Z)` admits Y values where `loc(Y,Z)` is fact and `nb(X,Y)` is the unknown. The dedup pass collapses the K_r anchor variants of the same logical rule application via `_variant_to_orig`, so consumers see a single entry per distinct rule application.

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
+-- postprocessing.py     # prune_ground_facts, collect_groundings
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
