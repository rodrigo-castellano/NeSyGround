# Soundness and Completeness

Formal properties of each grounder in NeSyGround. Every grounder is **sound** — it never returns invalid groundings. Completeness varies by design.

---

## 1. Definitions

**Sound grounding**: every returned grounding corresponds to a valid proof in KB. Formally, if `g ∈ Ground(Q, KB, ...)`, then there exists a valid SLD derivation (or FC derivation) that produces `g`.

**Complete grounding (within bounds)**: every valid proof in KB (within the specified bounds) is returned. Formally, if there exists a valid derivation of `g` within bounds `D`, `W`, etc., then `g ∈ Ground(Q, KB, ...)`.

**Deliberate incompleteness**: some grounders intentionally sacrifice completeness for scalability. This is a feature, not a bug — complete grounding of a large KB is computationally infeasible.

---

## 2. Properties Table

| Grounder | Sound | Complete | Bounds | Notes |
|----------|-------|----------|--------|-------|
| PrologGrounder | Yes | Yes (up to K_max) | D, K=K_f+K_r | Single-level SLD, K additive |
| RTFGrounder | Yes | Yes (up to K_max) | D, K=K_f*K_r | Two-level rule-then-fact, K multiplicative |
| BCPruneGrounder | Yes | Yes minus pruned | D, K_max | Removes unsupported groundings |
| BCProvsetGrounder | Yes | No | D, D_fc, K_max | Depth-1 + FC provability filter |
| ParametrizedBC(D,W) | Yes | No (deliberately) | D, W, K_max | Width controls completeness |
| FullBCGrounder | Yes | Yes | tG | Enumerates all entities |
| FCSemiNaive | Yes | Yes if fixpoint reached | D (iterations) | Semi-naive T_P |
| FCSPMM | Yes | Yes if fixpoint reached | D (iterations) | SpMM-based T_P |
| SamplerGrounder | Yes | No | D, W, max_sample | Random subset of parent |
| KGEGrounder | Yes | No | D, W, output_budget | Top-k by KGE score |
| NeuralGrounder | Yes | No | D, W, output_budget | Top-k by learned attention |
| SoftGrounder | Yes | No | D, W, output_budget | Top-k by soft confidence |
| LazyGrounder | Yes | Same as parent | D, W, reachable preds | Removes unreachable rules only |

---

## 3. Width vs Completeness

The ParametrizedBCGrounder makes a deliberate trade-off between width and completeness.

### Theorem

For all finite W:

```
Ground_Param(Q, KB, D, W) ⊆ Ground_BC(Q, KB, D)
```

That is, every grounding returned by the parametrized grounder is also a valid BC grounding, but the parametrized grounder may miss some.

### Width spectrum

| W | Completeness | What's missed |
|---|-------------|---------------|
| `W = 0` | Maximally incomplete | All groundings with any unproven body atom |
| `W = 1` | Most groundings found | Groundings with 2+ unproven body atoms |
| `W = 2` | More complete | Groundings with 3+ unproven body atoms |
| `W = None` | Equivalent to FullBCGrounder | Nothing (full enumeration) |

### Practical sweet spot

`W = 1, D = 2` is the recommended default for KG completion tasks:

- `W = 1` captures groundings with at most one unproven body atom. For binary-predicate rules (typical in KG completion), this covers the vast majority of useful groundings.
- `D = 2` allows two levels of rule application, which handles composed rules like `grandparent(X,Y) :- parent(X,Z), parent(Z,Y)`.
- The provability check via FC ensures that the single allowed unproven atom is at least reachable by forward chaining.

---

## 4. Truncation Effects

Two sources of incompleteness affect all BC grounders:

### K_max truncation

When more than `K_max` children are produced by RESOLVE, excess children are dropped. This means valid groundings can be lost if a single resolution step produces too many matches.

**Mitigation**: increase `max_states` or `max_total_groundings`. Monitor `count` in the output — if it consistently hits the capacity limit, truncation is occurring.

### Depth truncation

All BC grounders miss proofs that require more than `D` resolution steps. A proof of depth `D+1` will not be found even if it exists.

**Mitigation**: increase `D`. Note that the state tensor grows as `G = 1 + D*(M-1)`, so deeper proofs require more memory.

### CUDA graph necessity

Both truncation types are **necessary** for CUDA graph compatibility. CUDA graphs require fixed tensor shapes, which means:

- The number of children per step must be bounded (K_max)
- The number of steps must be bounded (D)
- The output size must be bounded (tG)

Without these bounds, tensor shapes would be data-dependent and CUDA graph capture would fail.

---

## 5. FC Completeness

Forward chaining is complete **if and only if** the fixpoint is reached within `D` iterations.

### When fixpoint is guaranteed

- **Finite KB with no recursive rules**: fixpoint is always reached (in at most `|F|` iterations where `|F|` is bounded by `P * E^2`)
- **Acyclic rules**: fixpoint reached in depth equal to the longest rule chain

### When fixpoint may not be reached

- **Recursive rules with large entity sets**: the provable set can grow through recursive rules, requiring many iterations
- **Chains longer than D**: if the rule chain is longer than D, not all provable atoms will be discovered

In practice, `D = 10` is sufficient for most KG completion benchmarks.

---

## 6. Soundness of Scored Grounders

All scored grounders (Sampler, KGE, Neural, Soft) are sound because they only **select from** the parent ParametrizedBCGrounder's output. They never synthesize new groundings — they filter and rank existing ones.

The scoring function does not affect soundness:

- SamplerGrounder: random selection from valid groundings — still sound
- KGEGrounder: ranks by plausibility, keeps top-k — still sound
- NeuralGrounder: ranks by learned attention — still sound
- SoftGrounder: assigns soft scores, keeps top-k — still sound

What changes is the **selection bias**: which valid groundings are kept and which are dropped. This affects downstream accuracy but not logical correctness.
