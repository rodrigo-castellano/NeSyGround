# SLD vs Enum — Memory Comparison

Tensor-by-tensor memory breakdown of the BCGrounder forward pipeline
on **fb15k237** (depth=2). Four configurations compared:

1. **SLD**: dense resolution, uncapped K_f
2. **Enum Dense**: dense body_a + dense resolve output [B,S,K,G,3]
3. **Enum Flat**: flat body_a (done) + dense resolve output
4. **Enum Flat+DynKS**: flat body_a + flat resolve output [T,G,3] + S_max

Shapes at **depth step 1** (peak memory). Values **uncapped**.

## Configuration

| Symbol | SLD | Enum Dense | Enum Flat | Flat+DynKS |
|--------|-----|-----------|-----------|------------|
| B | 192 | 192 | 192 | 192 |
| D | 2 | 2 | 2 | 2 |
| G | 4 | 4 | 4 | 4 |
| M | 2 | 2 | 2 | 2 |
| A | 4 | 4 | 4 | 4 |
| S | 256 | 256 | 256 | **4096** |
| K | K_f+K_r | 64 | 64 | **flat** |
| C | 64 | 64 | 64 | 64 |
| K_f | 3,612 | 0 | 0 | 0 |
| K_r | 30 | 30 | 30 | 30 |
| G_r | - | 3,612 | 3,612 | 3,612 |
| N (B*S) | 49,152 | 49,152 | 49,152 | 786,432 |

---

## Per-Phase Memory Summary

| Phase | SLD | Enum Dense | Enum Flat | Flat+DynKS | Shape |
|-------|-----|-----------|-----------|------------|-------|
| State tensors | 12 MB | 12 MB | 12 MB | 198 MB | [B, S, G+M+A+3, 3] |
| Resolve intermediate | 0 | 548.6 GB | 3.1 GB | 50.0 GB | body_a: dense [N,K_r,G_r,M,3] vs flat [T,M,3] |
| **Resolve output** | 38.4 GB | **699 MB** | **699 MB** | **936 MB** | dense [B,S,K,G,3] vs flat [T,G,3] |
| Pack output | 7 MB | 7 MB | 7 MB | 108 MB | [B, S_out, G+M, 3] |
| Collected | 1 MB | 1 MB | 1 MB | 1 MB | [B, C, A, 3] |

**Estimated totals** (intermediate + output + states, not accounting for memory reuse):

| | SLD | Enum Dense | Enum Flat | Flat+DynKS |
|--|-----|-----------|-----------|------------|
| Total | 38.4 GB | 549.3 GB | 3.8 GB | 51.3 GB |

---

## The S×K Bottleneck

The resolve output `rule_goals [B, S, K, G, 3]` has memory S×K×G:

| S | K | rule_goals | Notes |
|------|------|-----------|-------|
| 256 | 64 | 288 MB | current default |
| 1,024 | 64 | 1.2 GB |  |
| 4,096 | 64 | 4.6 GB |  |
| 256 | 4,096 | 18.4 GB |  |
| 4,096 | 4,096 | 294.9 GB | **impossible** on 24GB GPU |

With **flat K**, the resolve output is [T, G, 3] — no S×K product:

| S_max | State tensors | Flat resolve (est.) | Flat body_a (est.) | Total |
|-------|--------------|--------------------|--------------------|-------|
| 256 | 12 MB | 58 MB | 3.1 GB | 3.2 GB |
| 1,024 | 50 MB | 234 MB | 12.5 GB | 12.8 GB |
| 4,096 | 198 MB | 936 MB | 50.0 GB | 51.2 GB |
| 16,384 | 792 MB | 3.7 GB | 200.1 GB | 204.6 GB |
| 65,536 | 3.2 GB | 15.0 GB | 800.3 GB | 818.4 GB |

All costs **linear in S_max** (no quadratic S×K). But flat body_a
scales as N×K_r×avg_k = B×S_max×K_r×avg_k, which dominates at large S_max.

For fb15k237 (avg_k=13, K_r=30, B=192): flat body_a = S_max × 12.2 MB.
Practical S_max on 24GB GPU: **~1024** (total ~13 GB).

For smaller datasets (countries_s3: avg_k≈10, K_r=3, B=24):
flat body_a = S_max × 0.117 MB → S_max=65536 fits in ~1 GB.

---

## Compilation Impact

| Phase | SLD | Enum Dense | Enum Flat / Flat+DynKS |
|-------|-----|-----------|------------------------|
| KGE scoring | reduce-overhead | reduce-overhead | reduce-overhead |
| SELECT | reduce-overhead | reduce-overhead | mode=default, dynamic=True |
| RESOLVE | reduce-overhead | reduce-overhead | @torch.compiler.disable |
| PACK | reduce-overhead | reduce-overhead | mode=default, dynamic=True |
| POSTPROCESS | reduce-overhead | reduce-overhead | mode=default, dynamic=True |
| Reasoning | reduce-overhead | reduce-overhead | reduce-overhead |

SLD and dense enum are fully compiled with CUDA graphs (reduce-overhead).
Flat enum has one graph break at RESOLVE (data-dependent T from torch.nonzero).
Everything before and after is compiled with inductor (mode=default).
KGE and reasoning stay on reduce-overhead regardless of grounder type.

---

## Grounding Count Implications

countries_s3 comparison (keras-ns is uncapped reference):

| Config | Keras | Flat S=256 | Flat+DynKS S_max=4096 (est.) |
|--------|-------|-----------|-------------------------------|
| w0d1 | 42 | **42** | 42 (already matches) |
| w1d2 | 783 | 426 | ~750+ (16× more S slots) |
| w1d3 | 3349 | 599 | ~2500+ (16× more S slots) |

With S_max=4096, step 0 keeps up to 4096 children (vs 256 today).
The 16× increase in S slots directly translates to more branches
explored at each depth, closing the gap to keras.
