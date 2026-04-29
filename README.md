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

### BC_{w,d,u} grounders: paper parametrization

The `enum` BC family follows the paper / keras-ns notation:

- `w` — `max_unknown_fact_count` at intermediate proof steps.
- `d` — `num_steps` (proof depth).
- `u` — `max_unknown_fact_count_last_step`. **Paper convention is `u=0`**: every leaf body atom must be a fact.

Use `grounder.make_bcwd(kb, w, d, u=0, ...)` for the parametrized constructor, or the type-string shorthands `bcWD` (paper, u=0) / `bcWDuU` (explicit u). Internal mapping: `u` → `BCGrounder.w_last_depth`.

```python
from grounder import KGDataset, make_bcwd
ds = KGDataset(...)
kb = ds.make_kb()
g = make_bcwd(kb, w=1, d=2)        # bc12 paper: u=0 → filter='fp_batch'
out = g(queries, query_mask)
# out.rule_groundings matches keras-ns rule-by-rule.
```

Default filter depends on `u`:

| `u` | default filter | matches keras-ns |
|-----|----------------|------------------|
| `u=0` (paper) | `fp_batch` | `prune_incomplete_proofs=True` (Kleene fixed-point pruning over the rule-application set) |
| `u>0` (rare)  | `none`     | `prune_incomplete_proofs=False` (admit unknown leaves) |

Under the paper convention, `bc01` and `bc11` produce **identical output** — at `d=1` the only step is the last step, and `u=0` caps unknown leaves regardless of `w`. They differ only for `d≥2` where `w` controls intermediate-step admissibility.

`all_anchors=True` is forced by `BCGrounder.__init__` for `enum`. Anchoring only on the first body atom (e.g. `nb` in `nb(X,Y), loc(Y,Z) → loc(X,Z)`) misses bindings keras-ns finds when iterating each body position as anchor. The dedup pass collapses the K_r anchor variants of the same logical rule application via `_variant_to_orig`.

**Paper rule sets.** Some datasets ship two rule files; the paper / IJCAI '25 numbers use the smaller curated set:

| dataset | paper rules | extended set |
|---|---|---|
| `family` | `rules_old.txt` (47 rules) | `rules.txt` (143 rules) |

Pass `KGDataset(..., rules_file='rules_old.txt')` to reproduce paper grounding counts on family. The 143-rule `rules.txt` is an automated expansion that blows up `K_r` and OOMs large query batches under `cartesian_product=True`.

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
