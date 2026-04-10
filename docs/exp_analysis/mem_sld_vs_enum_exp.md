# Experimental Memory: SLD vs Enum Flat (countries_s3)

Single query (B=1), depth=3, width=1. CPU tensors.
Dataset: 983 facts, 3 rules, M=3, K_f=36, K_r=3.

## SLD d3

G=9, S=256, A=9, K=39

| Step | T_children | flat_goals MB | proof_goals MB | acc_body MB | grnd_body MB | valid_states | collected |
|------|-----------|--------------|---------------|------------|-------------|-------------|-----------|
| 0 |          3 | 0.0 MB | 0.1 MB | 0.0 MB | 0.0 MB |           3 | 0 |
| 1 |          0 | 0.2 MB | 0.1 MB | 0.1 MB | 0.0 MB |          10 | 0 |
| 2 |         15 | 0.2 MB | 0.1 MB | 0.1 MB | 0.0 MB |          45 | 3 |

**Total collected: 3**

## Enum Flat S=256

G=9, S=256, A=9, K=4096

| Step | T_children | flat_goals MB | proof_goals MB | acc_body MB | grnd_body MB | valid_states | collected |
|------|-----------|--------------|---------------|------------|-------------|-------------|-----------|
| 0 |        211 | 0.0 MB | 0.1 MB | 0.0 MB | 0.0 MB |         211 | 9 |
| 1 |     61,757 | 12.7 MB | 0.1 MB | 0.1 MB | 0.0 MB |         256 | 14 |
| 2 |      6,132 | 1.3 MB | 0.1 MB | 0.1 MB | 0.0 MB |         256 | 36 |

**Total collected: 36**

## Enum Flat S=4096

G=9, S=4096, A=9, K=4096

| Step | T_children | flat_goals MB | proof_goals MB | acc_body MB | grnd_body MB | valid_states | collected |
|------|-----------|--------------|---------------|------------|-------------|-------------|-----------|
| 0 |        211 | 0.0 MB | 0.8 MB | 0.0 MB | 0.3 MB |         211 | 9 |
| 1 |     61,757 | 12.7 MB | 0.8 MB | 0.8 MB | 0.3 MB |       4,096 | 48 |
| 2 |    170,022 | 35.0 MB | 0.8 MB | 0.8 MB | 0.3 MB |       4,096 | 70 |

**Total collected: 70**

## Enum Flat S=16384

G=9, S=16384, A=9, K=4096

| Step | T_children | flat_goals MB | proof_goals MB | acc_body MB | grnd_body MB | valid_states | collected |
|------|-----------|--------------|---------------|------------|-------------|-------------|-----------|
| 0 |        211 | 0.0 MB | 3.4 MB | 0.0 MB | 1.1 MB |         211 | 9 |
| 1 |     61,757 | 12.7 MB | 3.4 MB | 3.4 MB | 1.1 MB |      16,384 | 48 |
| 2 |    688,029 | 141.7 MB | 3.4 MB | 3.4 MB | 1.1 MB |      16,384 | 213 |

**Total collected: 213**

## Summary

| Grounder | proof_goals | flat_goals peak | Total state peak | Collected | Keras |
|----------|------------|----------------|-----------------|-----------|-------|
| SLD S=256 | 0.1 MB | 0.2 MB (dense) | ~0.5 MB | 3 | 3,349 |
| Enum Flat S=256 | 0.1 MB | 12.7 MB | ~13 MB | 36 | 3,349 |
| Enum Flat S=4096 | 0.8 MB | 35.0 MB | ~39 MB | 70 | 3,349 |
| Enum Flat S=16384 | 3.4 MB | 141.7 MB | ~153 MB | 213 | 3,349 |

### Memory bottleneck analysis

1. **State tensors** (`proof_goals`, `acc_body`, `grnd_body`) scale linearly with S_max.
   At S=16384: 3.4 + 3.4 + 1.1 = **7.9 MB**. Cheap.

2. **Flat resolve output** (`flat_goals [T, G, 3]`) is the peak intermediate.
   Step 2 at S=16384 produces 688K children → 142 MB. This scales as
   S_max × children_per_state × G × 3 × 8.

3. **Flat body_a** (inside resolve, freed after) also scales with N=B×S.
   At S=16384: N=16384 queries × K_r=3 × avg_k≈10 entries × 171 B ≈ 80 MB.

4. **No S×K quadratic**. Dense enum at S=4096 with K=64 would need
   [1, 4096, 64, 9, 3] = 54 MB for rule_goals alone. Flat K avoids this entirely.

### Why SLD finds only 3 groundings

SLD uses MGU resolution (fact + rule unification). With S=256 and K=39 (K_f=36 + K_r=3),
it only finds 3 terminal proofs at depth 3. The search space is heavily constrained by
the K_f cap from the arg_key fact index.

### Dynamic S + dedup results

With dynamic S (no fixed cap, `.item()` per step) and hash-based dedup
in pack_states_flat:

| Config | Keras | Fixed S=256 | Fixed S=4096 | Dynamic S + dedup |
|--------|-------|------------|-------------|-------------------|
| w0d1 | 42 | 42 | 42 | **42** (exact match) |
| w1d2 | 783 | 426 | 502 | **813** (+4%) |
| w1d3 | 3,349 | 599 | 1,037 | **2,677** (80%) |

Dynamic S dramatically closes the gap:
- w1d2: torch exceeds keras (+30) due to different dedup behavior
- w1d3: 80% of keras (was 18% at S=256, 31% at S=4096)

### Remaining w1d3 gap analysis (672 missing groundings)

Per-rule: torch finds more r0 (507 vs 83) but fewer r1 (358 vs 613)
and r2 (1812 vs 2653). The enumeration ORDER differs between systems:
- Keras: domain-wide enumeration, Python set dedup (insertion order)
- Torch: fact-anchored enumeration, hash-based dedup (keeps first by sort order)

Different dedup strategies keep different representatives when duplicates
exist, leading to different downstream exploration paths. The gap is NOT
from missing search coverage (dynamic S keeps all unique children) but
from which duplicate representatives are kept.

### Fixed S_max scaling (pre-dynamic-S measurements)

| S_max | Step 1: kept/total | Step 2 children | Collected | % of keras |
|-------|--------------------|----------------|-----------|------------|
| 256 | 256/61K (0.4%) | 6,132 | 36 | 1% |
| 4,096 | 4,096/61K (7%) | 170,022 | 70 | 2% |
| 16,384 | 16,384/61K (27%) | 688,029 | 213 | 6% |
| 65,536 | 43,679/61K (71%) | 2,641,056 | 1,195 | 36% |
