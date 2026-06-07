# Grounder execution-strategy study — empirical

**Question.** For the grounder rewrite: what is the right way to handle dynamic
vs. static tensor shapes / compilation, and can the choice live behind one
module so the algorithm code is independent of it? This study measures every
realistic execution option on a small dataset (**family**) and a memory-stress
dataset (**fb15k237 bc13**), with proper numbers for speed, memory, where the
time/memory goes, and warmup/compile cost.

**Hardware/SW.** RTX 3090 (24 GB), torch 2.10.0+cu128. Grounder HEAD 4cf4705.

## Method (rigor)

- Each config runs in a **fresh process** (clean CUDA context + Dynamo/Inductor
  caches; fresh `TORCHINDUCTOR_CACHE_DIR` so "cold" is a genuine first compile).
- **Timing**: CUDA events, warm = median of N replays after the cold call;
  `cold` = first call (trace+compile+autotune+cudagraph-capture+run);
  `compile+capture` = cold − warm.
- **Memory**: `max_memory_allocated` and `max_memory_reserved`, peak reset per
  measurement. (Under reduce-overhead the CUDA-graph pool counts as *reserved*,
  so `reserved` is the honest footprint there, not `allocated`.)
- **Where time goes**: `torch.profiler` (CUDA self-time, grouped) + eager
  per-phase CUDA-event timing of SELECT/RESOLVE/PACK/POSTPROCESS.
- **Where memory goes**: profiler per-op `self_mem` + per-phase peak alloc +
  allocator snapshots.
- **OOM** captured structurally (which phase, how much) rather than crashing.

The harness drives all options through the single `Compiler.wrap` seam — which
already exists in the codebase. That it can express every option by patching one
function is itself the key feasibility result: **the "one module to choose
execution" design is achievable with low blast radius.**

## The option space

Three orthogonal `torch.compile` axes — `mode` {default, reduce-overhead,
max-autotune(-no-cudagraphs)} × `dynamic` {False, None(auto), True} ×
`fullgraph` {True, False} — plus non-compile strategies (manual CUDA graphs,
AOTInductor, compile-cache reuse, allocator config). Hard constraint:
**reduce-overhead (CUDA graphs) ⊥ dynamic=True**. Layout (compact/flat vs
padded/dense) is a *separate* axis the strategy must pick *with* the compile
policy, because data-dependent ops (`nonzero`, `.item()`) in the flat path are
eager-only under `fullgraph`.

## Family (w1d3, 64 queries, 143 rules, K_r=44, K_f=28)

| option | warm ms | cold ms | compile+capture | peak alloc MiB | peak resv MiB |
|---|--:|--:|--:|--:|--:|
| sld-reduce-overhead | **7.3** | 32636 | 32.6 s | 158 | 588 |
| sld-eager | 29.7 | 125 | — | 513 | 742 |
| eager-flat | 41.0 | 193 | — | 1571 | 2358 |
| dense-maxautotune | 106.3 | 64126 | 64.0 s | 1569 | 3060 |
| dense-reduce-overhead | 107.1 | 20124 | 20.0 s | 240* | 2282 |
| dense-inductor-static | 108.1 | 20128 | 20.0 s | 1569 | 3060 |
| dense-inductor-dynamic | 111.4 | 32913 | 32.8 s | 1569 | 2048 |
| eager-dense | 248.3 | 383 | — | 4290 | 7598 |

Phase split (eager): flat = resolve 49% / postproc 38% / pack 12%;
dense = resolve 75% / pack 20% / postproc 5%.
Hot ops: flat = `sort` + `index` (2.2 GiB transient); dense = `scatter_` +
`index` (20.3 GiB transient) + `eq`.

**Findings.**
1. **Compile pays for SLD, not enum.** SLD 29.7→7.3 ms with reduce-overhead
   (4×, launch-overhead-bound). Enum-dense 248→107 ms but still loses to
   eager-flat (41 ms). The right strategy is **resolution-dependent**.
2. **`dynamic=True` and `max-autotune` are dominated** here — slower to compile
   (33 s / 64 s) for no warm gain; dynamic's only edge is lower reserved mem.
3. **Dense's memory blow-up is partly an eager artifact** — compiling the same
   dense layout drops peak 4290→1569 MiB (inductor fuses the big transients).

## fb15k237 bc13 (w1d3, 16 queries, 399 rules, K_r=59, K_f=3612)

| option | warm ms | compile+capture | peak alloc MiB | peak resv MiB | status |
|---|--:|--:|--:|--:|--|
| eager-flat | **8.96** | — | **254** | 290 | ✓ |
| eager-dense | — | — | >27 GiB | — | ✗ OOM |
| dense-inductor-static | 52.1 | 20.4 s | 2694 | 3566 | ✓ |
| dense-inductor-dynamic | 52.3 | 31.4 s | 2694 | 3566 | ✓ |
| dense-reduce-overhead | 51.7 | 19.4 s | 123* | 3614 | ✓ |
| sld-eager | 137.2 | — | 7798 | 10656 | ✓ |
| sld-reduce-overhead | 58.0 | 31.5 s | 115* | 6240 | ✓ |

Phase split (eager): flat = postproc 68% / resolve 21%; per-phase peak: flat's
peak is resolve (254 MiB), SLD's peak is pack (7.8 GiB, a 10.7 GiB `cat`).
(These 16 test queries prove nothing at d3/w1 — the *enumeration* cost that
drives memory is unaffected by the result count.)

**Findings.**
1. **eager-dense OOMs at 16 queries (>27 GiB); compiled-dense fits in ~3.5 GiB.**
   `torch.compile` is an **enabler** for dense on big KBs, not just a speedup —
   inductor fuses away the ~27 GiB of transient candidate materialization.
2. **eager-flat dominates outright** — 9 ms / 254 MiB vs compiled-dense
   52 ms / 3.5 GiB (~6× faster, ~12× lighter), with ~0 warmup.

## Scaling with batch size (fb15k237, w1d3)

| config | nq | warm ms | peak alloc MiB | status |
|---|--:|--:|--:|--|
| flat single | 16 | 9.2 | 254 | ✓ |
| flat single | 24 | 10.1 | 347 | ✓ |
| flat single | 32 | 11.3 | 428 | ✓ |
| flat single | 48 | — | ~19712 | ✗ OOM |
| flat chunk=16 | 64/256/1024 | — | ~19707 | ✗ OOM (same chunk) |
| dense-RO | 16 | 51.6 | 123 | ✓ |
| dense-RO | 64 | 272 | 241 | ✓ |
| dense-RO | 256 | 880 | 616 | ✓ |
| dense-RO | 512 | 1688 | 1119 | ✓ |
| dense-RO | 1024 | 3327 | 2129 | ✓ |
| sld-RO | 16 | 58 | 115 | ✓ |
| sld-RO | 64/256/1024 | — | OOM | ✗ |

**This overturns "flat always wins."**
- **flat's memory cliff is per-QUERY, not per-batch.** Smooth 16→32 (~13 MiB/q),
  then OOM at 48 (~20 GiB). `chunk=16` OOMs at the *identical* ~19.7 GiB for all
  nq because it dies on the same chunk [32:48] — a single high-fan-out query
  (one query × 3612-fact predicate × d3) blows up the `nonzero` survivor set.
  **Chunking does not save flat; flat has no hard memory bound.**
- **G_r-capped compiled-dense is the robust scalable enum path**: linear, bounded
  (1024 queries → 2.1 GiB), and it produces real groundings. The `G_r` cap is a
  *hard* per-query bound; flat's `nonzero` materializes all survivors *before*
  capping, so it is unbounded.
- **The fix worth building**: apply the per-query/anchor cap in the flat path
  *before* the `nonzero` materialization (currently it `topk`-caps to K only
  *after*). That would give flat a hard bound = the best of both (compact AND
  safe), and is the concrete "L2" restructure.

## Where time / memory goes (synthesis)

- **flat**: time in resolve+postprocess; memory peak is the resolve-step
  `nonzero`/`index`/`sort` compaction. Memory tracks *actual survivors*, so it
  stays small even on fb15k.
- **dense (eager)**: time and memory both dominated by RESOLVE building the
  padded `[N,K_r,G_r,M,3]` candidate tensor; a single `index`/`scatter`
  transient is the OOM driver.
- **dense (compiled)**: inductor fuses the transients (memory) but the
  data-dependent `scatter_` stays a large irreducible cost (time).
- **SLD**: launch-overhead-bound; CUDA graphs give the biggest single win;
  memory peak is PACK.

## Design implications for the refactor

1. **The execution knob must be per-grounder, not library-global.** The data
   shows opposite optima: SLD wants reduce-overhead (CUDA graphs); enum wants
   eager-flat. A single global default would be wrong for one of them.
2. **Recommended defaults** (evidence-based; the optimum is FAN-OUT-dependent):
   - enum, low fan-out / small batch (e.g. family, K_f≈28) → **eager-flat**
     (41 ms, light; ~6× faster than anything compiled).
   - enum, high fan-out (e.g. fb15k, K_f≈3612) → **G_r-capped compiled-dense**
     (the only *bounded, non-OOM* enum path: 1024 q → 2.1 GiB, linear). flat
     OOMs here on single queries and chunking does not save it.
   - The selector should key on **fan-out (K_f) and per-query candidate
     estimate**, not on batch size — that's the signal that predicts the cliff.
   - sld → **reduce-overhead** when the same shape is replayed many times
     (4× warm), else eager (no 20–33 s warmup tax); chunk to bound memory.
   - eager-dense: never on large KBs (OOM); only meaningful as a small-KB
     debugging baseline.
3. **Drop `dynamic=True` and `max-autotune` from the default menu** — measured
   strictly worse (compile cost) with no warm benefit on these workloads.
4. **The real "flexible + compiled" future is the L2 restructure**, not
   partial-compile-flat: precompute actual per-call sizes outside the graph,
   `mark_dynamic`, fullgraph kernel with no in-graph `nonzero`/`.item()`.
5. **Low risk**: `Compiler.wrap` is already the single compile seam; the
   strategy module is mostly promoting that seam to also own layout + chunking +
   the (mode,dynamic,fullgraph) policy, and validating illegal combos.

## Caveats

- family uses the 143-rule `rules.txt` (not the 47-rule paper set); heavier,
  which only stresses the comparison more.
- fb15k 16-query slice yields 0 proofs at d3/w1 — enumeration cost (the memory
  driver) is unaffected, but absolute warm times for the collect path are a
  lower bound. A grounding-positive slice would raise postprocess cost slightly.
- Single-GPU, single-run medians; cross-cut sweeps (allocator
  `expandable_segments`, Mega-cache warm compile) not yet run.
