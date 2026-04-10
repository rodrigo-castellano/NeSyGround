# Dense vs Flat+Offsets (CSR) Benchmark Report

## Environment

- **GPU**: NVIDIA GeForce RTX 3090 (24 GB)
- **PyTorch**: 2.10.0+cu128
- **CUDA**: 12.8
- **Warmup iters**: 10
- **Measure iters**: 100 (median reported)
- **Date**: 2026-03-23

## Parameters tested

- B (batch): 1, 32, 128
- K_f (fact candidates): 18 (avg), 200 (p95), 3612 (max for fb15k237)
- S (state counts): 64, 256, 1024, 4096
- M (body atoms): 1, 2, 3
- G (goals): 7 (used in pipeline)
- fb15k237 scale: E=14541, P=237, F=310116 (capped to E=5000, F=200000 for memory feasibility in enumerate/pipeline benchmarks)

## Executive Summary

The benchmark reveals a **clear split** between operations:

| Operation | Memory winner | Speed winner (eager) | Speed winner (compiled) | Verdict |
|:----------|:-------------|:---------------------|:------------------------|:--------|
| Enumerate | Flat (13-2700x less) | Dense at small K_f; Flat at K_f>=200 | Dense (2-5x faster) | **Flat for memory; dense for compiled speed** |
| Conjunction | Dense (1.6x less) | Dense (2.8x faster) | Dense (4.2x faster) | **Dense wins** |
| Disjunction | Dense (1.6x less) | Dense (2.8x faster) | Dense (4.3x faster) | **Dense wins** |
| Pack/Compact | Dense (10-25x less) | Dense (1.5-5x faster) | Dense (1.5-6x faster) | **Dense wins decisively** |
| Scatter | Tied | Dense at small S; Flat at large S | Flat (1.2x faster) | **Marginal either way** |
| **Full Pipeline** | **Flat (78-2500x less)** | **Flat (1-182x faster)** | **Dense at tiny scale; Flat at realistic scale (3-18x)** | **Flat wins at production scale** |

---

## Detailed Results

### Benchmark 1: Enumerate (fact_index equivalent)

Dense: `[N, K_f_max]` padded output + validity mask.
Flat: `repeat_interleave(counts)` to build flat indices, return `[total_valid]` + offsets.

| Parameters | Dense mem (MB) | Flat mem (MB) | Mem ratio | Dense time (ms) | Flat time (ms) | Speed ratio | Dense compiled (ms) | Flat compiled (ms) | Compiled ratio |
|:-----------|---------------:|--------------:|----------:|----------------:|---------------:|------------:|--------------------:|-------------------:|---------------:|
| N=1024, K_f=18 | 0.47 | 0.04 | 13x | 0.14 | 0.26 | 0.54x | 0.058 | 0.287 | 0.20x |
| N=4096, K_f=18 | 1.89 | 0.14 | 14x | 0.15 | 0.26 | 0.56x | 0.064 | 0.317 | 0.20x |
| N=16384, K_f=18 | 7.56 | 0.55 | 14x | 0.14 | 0.53 | 0.27x | 0.067 | 0.346 | 0.19x |
| N=1024, K_f=200 | 5.10 | 0.04 | 141x | 0.14 | 0.26 | 0.54x | 0.071 | 0.336 | 0.21x |
| N=4096, K_f=200 | 20.38 | 0.14 | 148x | 0.15 | 0.26 | 0.58x | 0.066 | 0.325 | 0.20x |
| N=16384, K_f=200 | 84.50 | 0.55 | 155x | 0.36 | 0.28 | **1.27x** | 0.073 | 0.353 | 0.21x |
| N=1024, K_f=3612 | 91.75 | 0.04 | **2539x** | 0.40 | 0.27 | **1.45x** | 0.072 | 0.317 | 0.23x |
| N=4096, K_f=3612 | 366.93 | 0.14 | **2665x** | 1.46 | 0.27 | **5.49x** | 0.218 | 0.321 | 0.68x |
| N=16384, K_f=3612 | 1469.15 | 0.54 | **2699x** | 5.68 | 0.27 | **20.70x** | 0.974 | 0.338 | **2.88x** |

**Analysis**: Memory savings are massive (13x at K_f=18, up to 2700x at K_f=3612) because flat stores only actual valid entries while dense pads to K_f_max. For eager speed, dense is faster at small K_f because the dense path uses simple `arange+gather` which maps well to GPU parallelism, while flat's `repeat_interleave` has overhead from dynamic-shape operations. However at K_f=3612 (max), flat is 5-21x faster in eager because the dense path must allocate and fill enormous padded tensors.

**Critical finding on compiled speed**: `torch.compile` accelerates the dense path dramatically (2-5x over flat compiled) at all scales except K_f=3612/N=16384. This is because the dense path has static shapes which the compiler can optimize perfectly, while `repeat_interleave` in the flat path generates dynamic shapes that prevent compiler optimizations.

### Benchmark 2: Conjunction (min over body atoms)

Dense: `scores[B, tG, M].masked_fill(~mask, 1.0).min(dim=-1)`.
Flat: `scatter_reduce(scores_flat, seg_ids, 'amin')`.

| Parameters | Dense time (ms) | Flat time (ms) | Speed ratio | Dense compiled (ms) | Flat compiled (ms) | Compiled ratio |
|:-----------|----------------:|---------------:|------------:|--------------------:|-------------------:|---------------:|
| B=1, tG=64, M=1 | 0.043 | 0.118 | 0.37x | 0.051 | 0.236 | 0.22x |
| B=1, tG=1024, M=3 | 0.044 | 0.122 | 0.36x | 0.061 | 0.252 | 0.24x |
| B=32, tG=256, M=2 | 0.044 | 0.124 | 0.35x | 0.065 | 0.254 | 0.25x |
| B=32, tG=1024, M=3 | 0.044 | 0.130 | 0.34x | 0.064 | 0.264 | 0.24x |
| B=128, tG=256, M=3 | 0.043 | 0.129 | 0.33x | 0.062 | 0.273 | 0.23x |
| B=128, tG=1024, M=3 | 0.044 | 0.140 | 0.31x | 0.069 | 0.272 | 0.25x |

**Analysis**: Dense wins uniformly at **2.8-3.2x** in eager and **4-4.5x** in compiled mode. The flat `scatter_reduce` approach is fundamentally slower because: (1) it needs `repeat_interleave` to build segment IDs (dynamic shape), (2) `scatter_reduce` has more overhead than a simple `min(dim=-1)` on contiguous memory, and (3) M is very small (1-3), so the "waste" from padding is negligible. Memory is also ~1.6x *less* for dense because the flat path needs additional offset/segment-ID tensors.

**Verdict**: For conjunction, flat is strictly worse in all dimensions.

### Benchmark 3: Disjunction (max over groundings)

Dense: `scores[B, tG].masked_fill(~mask, -inf).max(dim=-1)`.
Flat: `scatter_reduce(scores_flat, seg_ids, 'amax')`.

Results mirror conjunction: dense is 2.6-3.2x faster in eager, 4-4.5x faster compiled. Memory is a wash or slightly favors dense.

**Verdict**: Same as conjunction. For simple reductions over the last dimension, dense + mask is the clear winner.

### Benchmark 4: Pack/Compact (select valid children)

Dense: `topk(S_out, dim=1)` on `[B, S*K, ...]` then `gather`.
Flat: `nonzero` on flat valid mask, `index_select`, rebuild offsets.

| Parameters | Dense mem (MB) | Flat mem (MB) | Mem ratio | Dense time (ms) | Flat time (ms) | Speed ratio | Compiled ratio |
|:-----------|---------------:|--------------:|----------:|----------------:|---------------:|------------:|---------------:|
| B=1, S=64, K=18 | 0.02 | 0.15 | 0.10x | 0.070 | 0.150 | 0.47x | 0.60x |
| B=1, S=4096, K=200 | 3.91 | 105.13 | 0.04x | 0.158 | 0.328 | 0.48x | 0.55x |
| B=32, S=256, K=200 | 7.84 | 210.27 | 0.04x | 0.179 | 0.590 | 0.30x | 0.31x |
| B=32, S=1024, K=200 | 32.25 | 839.96 | 0.04x | 0.463 | 2.206 | 0.21x | 0.18x |
| B=32, S=4096, K=200 | 126.00 | 3360.60 | 0.04x | 1.542 | 8.688 | 0.18x | 0.15x |
| B=128, S=1024, K=200 | 126.00 | 3360.00 | 0.04x | 1.548 | 8.684 | 0.18x | 0.15x |
| B=128, S=4096, K=18 | 126.00 | 1209.48 | 0.10x | 0.960 | 3.151 | 0.30x | 0.29x |

**Analysis**: Dense wins decisively in both memory AND speed. This is counter-intuitive for memory but makes sense: the flat pack uses `nonzero()` which returns a dynamic-shaped tensor of all valid positions, then does a full gather over a flat tensor. For the benchmark, ~70% of elements are valid, so `nonzero()` returns nearly all indices, consuming MORE memory than the compact topk approach. Dense `topk` benefits from highly optimized GPU sort kernels. The gap widens at large scales: at B=32/128, S=1024/4096, K=200, dense is **5-7x faster** in both eager and compiled modes.

**Note**: The flat Pack benchmark hit OOM at B=128, S=4096, K=200 due to the `nonzero` intermediate requiring ~11.5 GB.

**Verdict**: Pack/compact is the single worst operation for flat representation. A production CSR implementation would need a fundamentally different pack strategy (e.g., per-segment capped gather rather than global nonzero).

### Benchmark 5: Scatter (write children to parent positions)

Dense: `scatter_(1, indices, source)` on `[B, S_out, G, 3]`.
Flat: `flat_dst = batch_offset + indices; out[flat_dst] = source`.

| Parameters | Dense time (ms) | Flat time (ms) | Speed ratio | Compiled ratio |
|:-----------|----------------:|---------------:|------------:|---------------:|
| B=1, any S/K | 0.033-0.036 | 0.065-0.069 | 0.50-0.55x | 1.18-1.25x |
| B=32, S_out<=1024 | 0.034-0.035 | 0.065-0.068 | 0.51-0.53x | 1.22-1.28x |
| B=32, S_out=4096 | 0.069 | 0.066 | **1.03x** | 1.24-1.28x |
| B=128, S_out<=256 | 0.034-0.040 | 0.066-0.070 | 0.50-0.58x | 1.22-1.28x |
| B=128, S_out=1024 | 0.092 | 0.074-0.076 | **1.22-1.25x** | 1.23-1.24x |
| B=128, S_out=4096 | 0.254 | 0.150 | **1.69-1.70x** | **1.84x** |

**Analysis**: Mixed results. Dense scatter is faster at small scales (2x advantage at B=1) but flat catches up and surpasses at large B*S_out because the flat path avoids the 4D scatter_() which requires expanding indices to match the full [B, S_out, G, 3] shape. At B=128, S_out=4096, flat is 1.7x faster eager and 1.8x faster compiled. Memory is essentially identical.

**Verdict**: Marginal either way; depends on scale.

### Benchmark 6: Full Pipeline (enumerate -> fill_body -> exists -> filter -> pack)

This is the **critical benchmark** as it represents the actual `resolve_enum_step` flow.

Dense: Allocates `[B, S, R_eff*K_f, M, 3]` intermediate tensors for body atoms, runs vectorized exists + conjunction + topk.
Flat: Uses CSR gather for candidates, runs exists + conjunction on flat tensors, segment_reduce for aggregation.

| Parameters | Dense mem (MB) | Flat mem (MB) | Mem ratio | Dense time (ms) | Flat time (ms) | Speed ratio | Dense compiled (ms) | Flat compiled (ms) | Compiled ratio |
|:-----------|---------------:|--------------:|----------:|----------------:|---------------:|------------:|--------------------:|-------------------:|---------------:|
| B=1, S=64, K_f=18 | 0.65 | 0.01 | 78x | 0.41 | 0.47 | 0.88x | 0.14 | 0.40 | 0.33x |
| B=1, S=64, K_f=200 | 7.45 | 0.01 | 898x | 0.47 | 0.46 | 1.02x | 0.20 | 0.41 | 0.48x |
| B=1, S=256, K_f=18 | 3.45 | 0.01 | 235x | 0.47 | 0.47 | 1.00x | 0.21 | 0.46 | 0.46x |
| B=1, S=256, K_f=200 | 28.95 | 0.01 | 1976x | 0.48 | 0.47 | 1.02x | 0.22 | 0.46 | 0.47x |
| B=1, S=1024, K_f=18 | 11.26 | 0.05 | 228x | 0.48 | 0.47 | 1.01x | 0.21 | 0.47 | 0.44x |
| B=1, S=1024, K_f=200 | 116.53 | 0.05 | **2363x** | 0.91 | 0.47 | **1.94x** | 0.26 | 0.46 | 0.58x |
| B=32, S=64, K_f=18 | 21.69 | 0.09 | 234x | 0.43 | 0.47 | 0.91x | 0.15 | 0.47 | 0.33x |
| B=32, S=64, K_f=200 | 232.60 | 0.10 | **2406x** | 1.59 | 0.47 | **3.36x** | 0.41 | 0.48 | 0.85x |
| B=32, S=256, K_f=18 | 87.46 | 0.37 | 236x | 0.66 | 0.47 | **1.40x** | 0.22 | 0.49 | 0.44x |
| B=32, S=256, K_f=200 | 923.11 | 0.36 | **2544x** | 5.81 | 0.47 | **12.41x** | 1.47 | 0.48 | **3.07x** |
| B=32, S=1024, K_f=18 | 335.98 | 1.45 | 231x | 2.20 | 0.48 | **4.57x** | 0.57 | 0.60 | 0.95x |
| B=32, S=1024, K_f=200 | 3669.50 | 1.46 | **2515x** | 22.48 | 0.48 | **46.61x** | 5.45 | 0.60 | **9.02x** |
| B=128, S=64, K_f=18 | 87.46 | 0.37 | 236x | 0.65 | 0.48 | **1.36x** | 0.22 | 0.47 | 0.47x |
| B=128, S=64, K_f=200 | 923.11 | 0.37 | **2517x** | 5.82 | 0.47 | **12.44x** | 1.48 | 0.47 | **3.13x** |
| B=128, S=256, K_f=18 | 335.98 | 1.45 | 231x | 2.22 | 0.48 | **4.60x** | 0.59 | 0.60 | 0.99x |
| B=128, S=256, K_f=200 | 3669.50 | 1.45 | **2530x** | 22.47 | 0.48 | **46.59x** | 5.42 | 0.61 | **8.94x** |
| B=128, S=1024, K_f=18 | 1324.72 | 6.72 | 197x | 8.35 | 0.49 | **17.16x** | 2.05 | 1.20 | **1.71x** |
| B=128, S=1024, K_f=200 | **14679** | 6.73 | **2180x** | **89.22** | 0.49 | **182.29x** | 21.28 | 1.21 | **17.55x** |

**Analysis**: This is where the flat representation shows its true value.

**Memory**: At any realistic training scale (B>=32, S>=256, K_f>=200), dense requires 1-15 GB for intermediates while flat requires <7 MB. The ratio reaches **2500x** at production scales. The dense path allocates `[B, S, R_eff*K_f, M, 3]` body tensors which is `128 * 1024 * 5 * 200 * 2 * 3 * 8 bytes = ~15 GB` at the largest config, while flat only stores the actually-valid candidates (~5% of the padded volume).

**Eager speed**: Flat is faster at every production-relevant config. At B=128, S=1024, K_f=200 the dense path takes 89ms vs flat's 0.49ms -- a **182x** speedup. This is because the dense path spends the vast majority of time in memory allocation, memory traffic, and memset for the enormous intermediate tensors that are mostly padding. The flat path avoids all of this.

**Compiled speed**: Dense compiled is faster at small scales (B=1 or K_f=18) because torch.compile eliminates allocation overhead and the static shapes enable perfect kernel fusion. However at production scales (B>=32, S>=256, K_f=200), flat compiled is **3-18x faster** because even compiled code cannot avoid the memory bandwidth bottleneck of touching 1-15 GB of mostly-padding intermediate tensors.

---

## Overall Analysis

### Where dense wins

1. **Isolated small reductions** (conjunction, disjunction): Dense `min/max(dim=-1)` with mask is 3-4x faster than `scatter_reduce` because the last dimension is tiny (M=1-3) and the overhead of building segment IDs dominates.

2. **Pack/compact**: Dense `topk` is 2-7x faster than flat `nonzero + gather`, especially at large scales. This is the Achilles heel of a naive flat implementation.

3. **Compiled enumerate at small K_f**: When K_f <= 200, torch.compile optimizes the static-shaped dense path to 2-5x faster than the dynamic-shaped flat path.

### Where flat wins

1. **Full pipeline memory** (78-2500x less): This is the single most compelling finding. At production scales, the dense pipeline requires 1-15 GB of temporary memory for padded intermediates, while flat requires 0.01-7 MB. This is not a marginal improvement -- it fundamentally changes which experiments are feasible on a given GPU.

2. **Full pipeline speed at production scale** (3-182x faster eager, 1.7-18x faster compiled): The memory bandwidth savings of not touching padded regions dominate. Even though each individual operation may be slower, the total wall time is much less because the flat path avoids allocating/filling/reading through gigabytes of padding.

3. **Enumerate at K_f=3612** (5-21x faster eager, 2.9x faster compiled): When padding waste is extreme (max K_f), flat wins even for isolated enumerate.

### The compilation question

A key concern was whether flat+offsets is compatible with `torch.compile`. The answer is **partially**:

- **Bad for compile**: `repeat_interleave(counts)` generates dynamic shapes, forcing graph breaks. This hurts isolated operations like conjunction/disjunction where the operation itself is trivial.
- **Acceptable for compile at scale**: In the full pipeline, the compiled flat path is still 3-18x faster than compiled dense at production scales because memory bandwidth dominates. The compilation wins for dense (kernel fusion, static shape optimization) cannot overcome the fundamental overhead of touching 100-1000x more memory.
- **Opportunity**: A custom CUDA kernel for the flat reduce steps (or using `segment_csr` from `torch_scatter`) could eliminate the `repeat_interleave` overhead entirely, making flat even more compile-friendly.

---

## Failure Notes

- Pack benchmark OOM'd at B=128, S=4096, K=200 (flat path tried to allocate 11.5 GB for `nonzero` intermediates). This highlights that a naive flat pack implementation is *worse* than dense for memory when most elements are valid. A production implementation needs per-segment capped gather, not global nonzero.

---

## Recommendation

### Verdict: **CONDITIONAL GO**

The data supports migrating the pipeline's hot path to flat+offsets representation, but NOT as a blanket library rewrite.

### What to do

1. **YES -- Migrate the enumerate + fill_body intermediate tensors to flat+offsets.** This is the single highest-impact change: it eliminates the `[B, S, R_eff*K_f, M, 3]` intermediate tensor that dominates memory and time. Expected savings: 100-2500x memory, 2-50x wall time at production scales.

2. **NO -- Do NOT migrate conjunction/disjunction reductions to scatter_reduce.** Dense masked reductions are 3-4x faster for these operations. Keep them dense.

3. **Hybrid approach for pack/compact.** Do NOT use global `nonzero` for packing. Instead, use dense `topk` on per-query scores (which are already small), then gather from the flat storage. Or implement a per-segment capped gather.

4. **Investigate `torch_scatter.segment_csr`** for the flat reduce steps. This avoids `repeat_interleave` overhead and is more compile-friendly than `scatter_reduce`.

5. **Keep the final proof state tensors `[B, S, G, 3]` dense.** These are the inputs/outputs of each step and are small enough that padding waste is negligible. The flat representation should be used only for the large fan-out intermediates during resolution.

### Estimated effort

This is NOT a full library rewrite. The recommended changes are:
- Modify `resolve_enum` to output flat candidates + offsets instead of `[N, R_eff, G_use, ...]`
- Modify the body-filling logic to work on flat candidates
- Keep pack_states and the rest of the pipeline as-is

Estimated: 200-400 lines of code changes in `grounder/resolution/enum.py` and `grounder/bc/common.py`, not a from-scratch rewrite.
