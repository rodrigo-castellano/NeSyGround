# Tensor Conventions

All tensor dimensions in NeSyGround are **fixed at construction time**. This is a hard requirement: CUDA graph capture needs static shapes, so no dimension changes during `forward()`.

---

## Dimension Table

| Symbol | Meaning | Determined by | Typical range |
|--------|---------|---------------|---------------|
| `B` | Batch size (number of queries) | Caller | 128 - 1024 |
| `S` | States per query | `max_states` constructor arg | 1 - 256 |
| `G` | Goals per state | `1 + D * (M - 1)` | 1 - 20 |
| `K` | Children per state (total) | `K_f + K_r` | dataset-specific |
| `K_f` | Max fact matches per goal | FactIndex capacity | dataset-specific |
| `K_r` | Max rule matches per goal | Number of matching rules | dataset-specific |
| `M` | Max body atoms per rule | Max across all rules in KB | 2 - 4 |
| `R` | Total number of rules | KB | dataset-specific |
| `R_eff` | Max rules per head predicate | Rule clustering | <= R |
| `F` | Total number of facts | KB | dataset-specific |
| `P` | Number of predicates | KB | dataset-specific |
| `E` | Number of entities | KB | dataset-specific |
| `D` | Depth | Constructor arg | 1 - 10 |
| `W` | Width (ParametrizedBC only) | Constructor arg | 0, 1, 2, None |
| `tG` | Total groundings per query | `effective_total_G` | 32 - 256 |
| `I_max` | Max provable atoms (FC) | Pre-allocated capacity | dataset-specific |

### Goals per state

The number of goal slots `G` grows with depth and rule body size:

```
G = 1 + D * (M - 1)
```

At depth 0, there is 1 goal (the query). Each resolution step replaces 1 goal with up to `M` body atoms, adding `M - 1` new goals. After `D` steps, the maximum is `1 + D * (M - 1)`.

---

## Validity Masks

Every variable-length dimension uses a boolean mask to distinguish real data from padding:

| Mask | Shape | Meaning |
|------|-------|---------|
| `state_valid` | `[B, S]` | Which states are active (not terminated or empty) |
| `goal_valid` | `[B, S, G]` | Which goals within each state are unresolved |
| `child_valid` | `[B, S, K]` | Which children from resolution are valid |
| `grounding_mask` | `[B, tG]` | Which output groundings are valid |
| `query_mask` | `[B]` | Which input queries are valid |

Masks are `bool` tensors. Invalid positions are filled with `padding_idx` in the corresponding data tensors.

---

## Padding

- **`padding_idx`**: distinguished integer used to fill inactive tensor slots. Set to `E` (one past the last entity index) so it never collides with real entities
- **`constant_no`**: alias for `E`, the number of real entities. Used in `pack_base` computation
- **`pack_base`**: set to `E + 2` — the base for atom hashing. The `+2` ensures padding and any sentinel values don't collide with valid hashes

Padding is applied at every stage:

1. FactIndex lookups pad results to `K_f` entries
2. Rule resolution pads to `K_r` entries
3. Pack stage pads states to `S` entries
4. Output pads groundings to `tG` entries

---

## Hashing

Atom deduplication and provable-set membership use `pack_triples_64`:

```python
def pack_triples_64(atoms: Tensor, base: int) -> Tensor:
    """[N, 3] -> [N] int64 keys via ((p * base) + a0) * base + a1"""
    return ((atoms[:, 0].long() * base) + atoms[:, 1].long()) * base + atoms[:, 2].long()
```

- `base = pack_base = E + 2`
- Output is `int64` to avoid overflow for large entity sets
- Hashes are **sorted** in provable set tensors to enable `O(log N)` binary search via `torch.searchsorted`

Membership test:

```python
def fact_contains(atoms: Tensor, fact_hashes: Tensor, pack_base: int) -> Tensor:
    """[N, 3] -> [N] bool via searchsorted on sorted hashes"""
    query_hashes = pack_triples_64(atoms, pack_base)
    idx = torch.searchsorted(fact_hashes, query_hashes)
    return (idx < len(fact_hashes)) & (fact_hashes[idx] == query_hashes)
```

---

## CUDA Graph Constraints

All grounder code must be compatible with `torch.compile(fullgraph=True, mode='reduce-overhead')`. This imposes strict constraints:

1. **No `.item()` calls** — cannot extract Python scalars from tensors during forward pass
2. **No dynamic control flow** — no `if tensor.sum() > 0:` or similar data-dependent branching
3. **No Python data-dependent branching** — tensor values must not influence Python control flow in `forward()`
4. **No dynamic shapes** — all output tensors have the same shape regardless of input content
5. **No in-place mutation of graph inputs** — use `torch.where()` or masked assignment instead

Train/eval mode branching (e.g., in SamplerGrounder) is handled by compiling separate CUDA graphs for each mode, not by runtime branching.
