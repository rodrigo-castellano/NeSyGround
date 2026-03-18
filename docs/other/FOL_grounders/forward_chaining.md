# Forward Chaining — Semi-Naive Evaluation

## Table of Contents

1. [General Algorithm](#1-general-algorithm)
2. [Semi-Naive Formula and Fixpoint Semantics](#2-semi-naive-formula-and-fixpoint-semantics)
3. [FCDynamic — Staged Ragged Join](#3-fcdynamic--staged-ragged-join)
4. [FCStatic — General Batched Kernel](#4-fcstatic--general-batched-kernel)
5. [Scalability](#5-scalability)
   - [5.1 Why FCDynamic Does Not Have the Memory Problem](#51-why-fcdynamic-does-not-have-the-memory-problem)
   - [5.2 Memory Parameters — K_max and N_max](#52-memory-parameters--k_max-and-n_max)
6. [Benchmarks](#6-benchmarks)

---

## 1. General Algorithm

Forward chaining derives all facts entailed by a rule set from a base fact set, repeating until no new facts can be derived (fixpoint).

**Notation:**
- `E`: number of entities
- `P`: number of predicates
- `F`: base facts  ⊆ P × E × E
- `R`: set of Horn rules, each of the form `h ← b_1, ..., b_m`
- `I_t`: the set of all known facts after step `t`
- `Δ_t`: the set of facts newly derived at step `t`

**Algorithm:**

1. **Initialise.** Set `I = F` (all known facts = base facts).

2. **Derive.** For each rule `r: h ← b_1, ..., b_m` and each substitution `θ` such that `b_i(θ) ∈ I` for all `i`, add `h(θ)` to the new-facts buffer `Δ`.

3. **Update.** Set `I_next = I ∪ Δ`.

4. **Check convergence.** If $I_{\text{next}} = I$, stop and return $I^* = I$. Otherwise set $I = I_{\text{next}}$ and go to step 2.

**Fixpoint guarantee.** The fact space $P \times E \times E$ is finite and $I$ grows monotonically at each step, so the algorithm always terminates at the unique least Herbrand model $I^*$.

**Immediate consequence operator:**

$$
T_{\mathcal{R}}(I) = \bigcup_{r \in \mathcal{R}} \bigl\{\, h(\theta) \mid \forall i: b_i(\theta) \in I \,\bigr\}
$$

$$
I_{t+1} = I_t \cup T_{\mathcal{R}}(I_t)
$$

---

## 2. Semi-Naive Formula and Fixpoint Semantics

Naively re-checking all substitutions at every step is wasteful — most were already valid in earlier steps. Semi-naive evaluation avoids redundant re-derivation by requiring at least one body atom to come from the *newly added* Δ.

**Formula:**

$$
I_0 = F, \qquad \Delta_0 = F
$$

$$
I_t = I_{t-1} \cup \Delta I_t
$$

$$
\Delta I_t = \bigcup_{r \in \mathcal{R}} \Delta T_r(I_{t-1},\, \Delta_{t-1})
$$

$$
\Delta T_r(I, \Delta) = \bigcup_{k=0}^{m-1} \bigl\{\, h(\theta) \;\big|\; b_k(\theta) \in \Delta,\; \forall j \neq k: b_j(\theta) \in I \,\bigr\}
$$

**Step-by-step explanation:**

- `k` is the **anchor position** — the one body atom that must come from the new delta Δ. This ensures the substitution wasn't already derivable in a previous step.
- All other body atoms (`j ≠ k`) are looked up in the accumulated `I` (base facts + all previously derived).
- There are `m` anchor choices per rule, generating up to `m` sub-derivations per rule per step.
- The union over all anchors ensures completeness: every new derivation has at least one body atom that is newly true.

**Why it is correct:**  any new derivation `h(θ)` at step `t` must have at least one body atom `b_k(θ)` that was not in `I_{t-1}` (otherwise it would have been derived earlier). That atom is in `Δ_{t-1}` = `I_{t-1} \ I_{t-2}`. So anchoring to Δ captures exactly the new derivations without repetition.

**Convergence** is guaranteed because each step either adds new facts (`|I|` strictly increases) or adds nothing (fixpoint). Maximum steps needed ≤ diameter of the rule dependency graph.

---

## 3. FCDynamic — Staged Ragged Join

`FCDynamic` is a CPU implementation. It processes rules and anchors sequentially via Python loops, joining only over *actually existing* tuples at each stage.

### Algorithm

**Preprocessing (once at init):**

1. For each rule `r`, compute a **greedy join order** over body atoms (`_compute_join_order_from`). The order guarantees every stage has at least one already-bound variable, avoiding cross-products.

> **Why join order matters — avoiding cross products**
>
> Evaluating a rule means finding all substitutions θ that satisfy every body atom simultaneously. The order in which body atoms are combined is the **join order**.
>
> A **cross product** occurs when two atoms are combined before they share any bound variable. If atom `a(X,Y)` has 100 matches and atom `b(W,V)` shares no variable with it, combining them naively produces 100 × |b| intermediate tuples — a combinatorial explosion — even if most are discarded by later stages.
>
> The **greedy join order** avoids this: at each step it picks the next atom that shares at least one variable with the already-bound set. This guarantees every stage is a **lookup** (PS or PO neighbor fetch for a bound key), not a cross product. The algorithm is greedy because it makes a locally optimal choice at each step — maximise shared variables — without backtracking. It doesn't find the globally optimal order, but it enforces the key invariant: **no stage ever starts with zero bound variables**.
>
> Atoms with only dangling variables (variables that appear in the body but not in the head and not as a lookup key in any subsequent EXPAND) are handled last as existence checks — does *any* matching fact exist? — rather than as neighbor fetches.

**Per step `t`, per rule `r`, per anchor `k`:**

> **What "anchor" means**
>
> For a rule with $m$ body atoms, the semi-naive formula (Section 2) requires that exactly one body atom comes from $\Delta$ (newly derived facts) while the rest are looked up in the accumulated $I$. The **anchor** $k$ is that designated atom index.
>
> For a 2-body rule there are 2 anchor choices run every step:
>
> | Anchor | Constraint |
> |--------|-----------|
> | $k=0$ | $b_0(\theta) \in \Delta,\; b_1(\theta) \in I$ |
> | $k=1$ | $b_0(\theta) \in I,\; b_1(\theta) \in \Delta$ |
>
> Their union covers all new derivations exactly once — any newly derivable $h(\theta)$ must have at least one body atom that just became true, so one of the two anchors will catch it.
>
> The join order is then computed *starting from the anchor atom* (it is the most constrained starting point: only $\Delta$ facts, not all of $I$), via `_compute_join_order_from(anchor_atom)`. Every (rule, anchor) pair gets its own join order.

1. **Stage 0 — Seed.** Collect all `(s, o)` pairs satisfying the anchor body atom `b_k` from `Δ_{t-1}`. These are the starting partial matches.
   - Implementation: CSR offset-array lookup into the delta index.
   - **Time:** $O(N_0)$ where $N_0 = |\Delta_{t-1}|$ at the anchor predicate.
   - **Space:** $O(N_0)$ to store the seed partial matches.

2. **Stages 1..m-1 — Extend.** For each subsequent body atom in join order, extend each partial match using PS (predicate-subject → objects) or PO (predicate-object → subjects) lookups into `I_{t-1}`.
   - **Case A** — both variables already bound: existence check (0 or 1 candidates, no new variable).
   - **Case B** — subject bound, object free: PS lookup, adds object variable.
   - **Case C** — object bound, subject free: PO lookup, adds subject variable.
   - After each stage, drop variables no longer needed by future stages or the head (frontier pruning).
   - **Time at stage** $j$ ($j = 1 \ldots m{-}1$): $O(N_0 \cdot K^{j-1} \cdot K) = O(N_0 \cdot K^j)$ — there are $N_0 K^{j-1}$ partial matches from the previous stage, each doing one $O(K)$ lookup. Total over all extend stages: $O(N_0 K) + O(N_0 K^2) + \cdots + O(N_0 K^{m-1}) = O(N_0 \cdot K^{m-1})$ (dominated by the last stage).
   - **Space at stage** $j$: $O(N_0 \cdot K^j)$ for the intermediate partial-match list. Peak at stage $m{-}1$: $O(N_0 \cdot K^{m-1})$.

3. **Output.** After all `m` stages, extract head variable pair `(h0, h1)` from each surviving partial match and emit hash $p_h \cdot E^2 + h_0 \cdot E + h_1$.
   - **Time:** $O(N_0 \cdot K^{m-1})$ — one hash per surviving partial match.
   - **Space:** $O(N_0 \cdot K^{m-1})$ for the output hash buffer (same size as the final partial-match list).

4. **Accumulate.** Collect all new hashes across all rules and anchors, deduplicate, filter against already-known facts, and add to `I`.
   - **Time:** $O(|\text{output}|) = O(N_0 \cdot K^{m-1})$ for deduplication and hash-set lookup.
   - **Space:** $O(|I|)$ for the accumulated fact set (grows monotonically; peak = $O(|I^*|)$).

### Complexity Summary

Steps 1–3 are dominated by stage $m{-}1$, giving $O(N_0 \cdot K^{m-1})$ per (rule, anchor, step). Summing over all rules, anchors, and steps:

$$\underbrace{R}_{\text{rules}} \times \underbrace{N_{\max}}_{\text{anchors}} \times \underbrace{S}_{\text{steps}} \times O\!\left(N_0 \cdot K^{N_{\max}-1}\right) = O\!\left(R \cdot N_{\max} \cdot S \cdot N_0 \cdot K^{N_{\max}-1}\right)$$

Since each atom enters $\Delta$ exactly once across all steps, $\sum_t N_{0,t} \leq |I^*|$, giving a tighter total:

$$\text{Total time} = O\!\left(R \cdot N_{\max} \cdot |I^*| \cdot K^{N_{\max}-1}\right)$$

For space, only one (rule, anchor) is processed at a time (sequential loop), so intermediate tensors are reused:

$$\text{Peak space} = O\!\left(|\Delta| \cdot K^{N_{\max}-1}\right) + O(|I^*|)$$

| | |
|---|---|
| **Time per (rule, anchor, step)** | $O(N_0 \cdot K^{N_{\max}-1})$ |
| **Total time** | $O\!\left(R \cdot N_{\max} \cdot \lvert I^*\rvert \cdot K^{N_{\max}-1}\right)$ |
| **Peak space (intermediates)** | $O\!\left(\lvert\Delta\rvert \cdot K^{N_{\max}-1}\right)$ — one (rule, anchor) at a time |
| **Persistent space** | $O(\lvert I^*\rvert)$ — accumulated fact set |

> **What $K$ (fanout) means and why it dominates complexity**
>
> **Fanout** is the number of neighbors a node has for a given predicate lookup. When extending a partial match at stage $k$, the kernel takes a bound entity $e$ and fetches all facts $\text{pred}(e, ?)$ — the count returned is the fanout for that $(pred, e)$ pair.
>
> $K$ is the **branching factor** per stage. At each stage every partial match can expand into up to $K$ new ones (one per neighbor). After $m-1$ stages the number of partial matches is at most:
>
> $$N_0 \;\xrightarrow{\times K}\; N_0 K \;\xrightarrow{\times K}\; N_0 K^2 \;\xrightarrow{\times K}\; \cdots \;\xrightarrow{\times K}\; N_0 K^{m-1}$$
>
> This is why $K$ appears as an exponent. In practice $K$ is the *average* fanout — paths that hit a dead end (fanout 0) are dropped immediately, keeping the actual intermediate size much smaller than the worst case. For kinship_family $K \approx 2$–$3$ (small family graphs); for wn18rr $K$ can reach 442 (dense hypernym chains), causing the atom explosion seen in the benchmarks.

Key property: memory scales with **active atoms only**, not with $E \times K^{N_{\max}}$ as in the static case.

> **Why this matters**
>
> "Active atoms" here means the same $K$ from above — the actual derived tuples that exist. FCDynamic only ever allocates memory for partial matches that correspond to real facts in the graph.
>
> The static case $E \times K^{N_{\max}}$ is fundamentally different: FCStatic pre-allocates a tensor slot for **every entity** $e \in \{0, \ldots, E{-}1\}$ and **every possible chain of length $N_{\max}$**, regardless of whether any facts exist there. The shape grows as:
>
> $$E \;\xrightarrow{\times K}\; E \cdot K \;\xrightarrow{\times K}\; E \cdot K^2 \;\xrightarrow{\times K}\; \cdots \;\xrightarrow{\times K}\; E \cdot K^{N_{\max}}$$
>
> because at each stage the kernel expands **every** entity slot by $K_{\max}$ (the neighbour fanout cap), whether or not that entity has any relevant neighbours. Compare this to FCDynamic, which starts from $N_0 \ll E$ real delta facts and only expands slots that actually matched.
>
> The full tensor shape is $[B,\, E,\, K_{\max}^{N_{\max}}]$ where $B = R \times N_{\max}$ — all (rule, anchor) pairs are batched together and allocated simultaneously. So the complete memory is $R \times N_{\max} \times E \times K_{\max}^{N_{\max}}$:
>
> | Dataset | $R$ | $N_{\max}$ | $E$ | $K_{\max}$ | Total slots ($B \times E \times K_{\max}^{N_{\max}}$) |
> |---------|-----|------------|-----|------------|-------------------------------------------------------|
> | kinship_family | 143 | 2 | 2,968 | 56 | $286 \times 2{,}968 \times 3{,}136 \approx 2.66\text{B}$ (~21 GB) |
> | wn18rr | 42 | 2 | 40,559 | 884 | $84 \times 40{,}559 \times 781{,}456 \approx 2.66\text{T}$ (>>TB) |
>
> For kinship_family FCDynamic derives only ~56k atoms — paying for a tiny fraction of the 2.66B static slots. For wn18rr the full static tensor would be in the terabyte range, making FCStatic completely infeasible regardless of available RAM. FCDynamic derives 33.5M atoms in 3 steps using only a few GB.

---

## 4. FCStatic — General Batched Kernel

`FCStatic` is a GPU implementation. It processes all $R \times N_{\max}$ (rule, anchor) pairs simultaneously in a single compiled kernel call per step, treating all entities in parallel.

### Op Type System

Each body atom in a rule is pre-classified into one of 8 op types during `__init__`. The kernel selects the correct operation for each stage via `torch.where` (no data-dependent branching).

| Code | Name | Bound var | New var | Effect |
|------|------|-----------|---------|--------|
| 0 | EXPAND_PS | cur = subject | new object → cur | K-gather neighbors; cur grows K× |
| 1 | EXPAND_PO | cur = object | new subject → cur | K-gather neighbors; cur grows K× |
| 2 | EXIST_ROOT_PS | root = subject | dangling object | identity expand; mask by root PS-existence |
| 3 | EXIST_ROOT_PO | root = object | dangling subject | identity expand; mask by root PO-existence |
| 4 | EXIST_CUR_PS | cur = subject | dangling object | identity expand; mask by cur PS-existence |
| 5 | EXIST_CUR_PO | cur = object | dangling subject | identity expand; mask by cur PO-existence |
| 6 | FILTER_BOTH | root + cur both bound | — | identity expand; mask by `cur ∈ neigh(root)` |
| 7 | PAD | identity | — | identity expand; mask = False |

**EXPAND** (ops 0–1): the kernel fetches K neighbors and replaces `cur` with them — the frontier genuinely moves.

**EXIST** (ops 2–5): one variable is dangling (appears in the body but not the head and not needed later). The kernel checks whether the bound variable has *any* neighbor under this predicate. `cur` stays the same (identity expand); invalid paths are masked out.

**FILTER_BOTH** (op 6): both `root` and `cur` are already bound. The kernel checks whether the edge `(root, cur)` exists under this predicate. This is a **set membership check** — not a neighbor fetch. This is the one operation that SpMM cannot express (no free dimension to sum over; it is an element-wise Hadamard, not a matrix multiply).

**PAD** (op 7): used to pad shorter rules up to N_max stages. Always produces a False mask.

### Rule Classification — `_classify_general_rule`

Called once at `__init__` for each rule. Tries every body atom as the starting atom and attempts to assign op types:

1. Pick a starting atom. Identify which head variable (`root_var`) is present in it; set `cur_var` = the other variable in that atom.
2. Assign op type 0 or 1 to the starting atom (EXPAND_PS or EXPAND_PO depending on which slot holds `root_var`).
3. For each subsequent atom in join order:
   - If both `root_var` and `cur_var` appear: **FILTER_BOTH** (op 6).
   - If `cur_var` appears and the new variable is needed later or is a head var: **EXPAND** (op 0 or 1), update `cur_var`.
   - If `cur_var` appears and the new variable is dangling: **EXIST_CUR** (op 4 or 5).
   - If `root_var` appears and the new variable is dangling: **EXIST_ROOT** (op 2 or 3).
4. Classification succeeds if after all stages `cur_var == other_head_var`.

Rules that require ≥3 simultaneously tracked variables fall through all starting atoms and return `None` — FCDynamic handles them.

### Shape Invariant

At every stage `k` the kernel tracks exactly two variables:

| Variable | Representation | Description |
|----------|---------------|-------------|
| `root` | E dimension (size E) | Fixed head variable — one slot per entity |
| `cur` | K^k trailing dimensions | Chain endpoint — active frontier |

Shape at stage $k$: $[B,\, E,\, K^k]$ where $B = R \times N_{\max}$.

Every op type maps $[B,\, E,\, K^k] \to [B,\, E,\, K^{k+1}]$:
- **EXPAND**: K-gather into neighbors of `cur` — K different new values.
- **EXIST / FILTER / PAD**: identity expand — K copies of the same `cur` value, validity controlled by mask.

This **uniform shape growth** is what makes the kernel CUDA-graph-compatible: `torch.compile(fullgraph=True, mode='reduce-overhead')` can capture a static computation graph with no dynamic shapes.

### Algorithm

**Preprocessing (once at init):**

1. Classify all rules with `_classify_general_rule`. Rules returning `None` are silently skipped (FCDynamic handles them).
2. Build neighbor buffers $[2,\, P{+}1,\, E,\, K_{\max}]$ for PS and PO lookups (one for accumulator, one for delta). The `+1` slot is the identity (entity → itself), used for PAD stages.
3. Precompute `_pred_seqs` $[R, N_{\max}]$, `_op_type_seqs` $[R, N_{\max}]$, `_filter_orient_seqs` $[R, N_{\max}]$, `_head_rev` $[R]$, `_anchor_ids` $[R \times N_{\max}]$.

**Per step `t`:**

1. **Update delta buffer.** Fill `delta_neigh / delta_mask` from `Δ_{t-1}` (new atoms since last step).
2. **Run kernel** (single batched call over all B = R × N_max combinations):
   - Stage k=0: seed `cur` from the anchor atom (using delta for the anchor, accumulator for non-anchors).
   - Stages $k=1\ldots N_{\max}{-}1$: apply op type for each stage via `torch.where`, building $[B, E, K^{k+1}]$.
   - Output: extract `(h0, h1)` pairs and compute hashes. Masked-out slots emit hash = -1.
3. **Deduplicate.** Filter output hashes against already-known atoms; add new ones to `I`.
4. **Update accumulator buffer.** Add new atoms to `accum_neigh / accum_mask`.

### Orient Encoding

```python
orient = op_k % 2          # EXPAND (0,1) and EXIST (2-5): 0=PS, 1=PO
orient = filter_orient_k   # FILTER_BOTH (6): stored per-rule in filter_orient_seqs
orient = 0                 # PAD (7): irrelevant; identity slot used
```

### head_reversed Flag

When `head_reversed=True`, the E-dimension represents `head_var1` (root = h1) rather than `head_var0`. The output hash is:

```python
h0   = where(rev, cur, x)    # head_var0
h1   = where(rev, x, cur)    # head_var1
hash = head_pred * E**2 + h0 * E + h1
```

### CUDA Graph Notes

- No `.item()` calls inside the kernel.
- No data-dependent branching — all 5 branch outcomes (EXPAND, EXIST_ROOT, EXIST_CUR, FILTER_BOTH, PAD) are computed simultaneously via `torch.where`; only the correct one is selected.
- `_fill_neigh_inplace` (buffer update) runs **outside** the compiled region — it may call `.item()`.
- The outer `for step in range(num_steps)` Python loop is the only dynamic control flow.

### Complexity Summary

| | |
|---|---|
| **Space (persistent)** | $O(P \cdot E \cdot K_{\max})$ for neighbor buffers |
| **Space (kernel peak)** | $O(R \cdot N_{\max} \cdot E \cdot K_{\max}^{N_{\max}})$ — ~12 branch tensors simultaneously |
| **Time per step** | $O(R \cdot N_{\max} \cdot E \cdot K_{\max}^{N_{\max}})$ — all entity slots processed regardless of sparsity |

---

## 5. Scalability

### 5.1 Why FCDynamic Does Not Have the Memory Problem

FCDynamic and FCStatic represent fundamentally different strategies for evaluating the same semi-naive formula.

**FCDynamic — sparse, sequential:**

```python
for rule in compiled_rules:          # R rules
    for anchor_idx in range(m):      # N_max anchors
        for s, o in delta[anchor]:   # only *active* Δ pairs
            join with accum[others]
            → emit new atom if match
```

At any moment FCDynamic holds only a ragged list of the tuples that actually exist. Memory scales with the number of active atoms, not with the full entity space.

**FCStatic — dense, parallel:**

```python
cur = allocate [B, E, K^1]    # ALL entities, ALL rules, ALL anchors — at once
cur = expand   [B, E, K^2]    # expand ALL, whether or not anything is there
→ emit from   [B, E, K^N_max]
```

Every entity slot is computed simultaneously. Most of the tensor is zeros or masked-out, but memory is still allocated for all of it.

**The sparse/dense gap:**

| Property | FCDynamic | FCStatic |
|---|---|---|
| Iterates over | active atoms only | all $E$ entities |
| Memory | $O(\text{active atoms})$ | $O(R \times E \times K^{N_{\max}})$ |
| Sparsity | exploited | ignored |
| Execution | sequential Python loops | single batched kernel |

For kinship_family: 56,389 atoms are derived from 2,968 entities × 12 predicates. Only ~1.8% of the dense $E \times K^2$ space is non-zero. FCDynamic pays for 1.8%; FCStatic pays for 100%.

FCStatic only wins when the graph is dense enough that GPU parallelism outweighs the wasted computation on empty slots — requiring small E, small K_max, or high derivation density.

**Relationship to SpMM:**

SpMM (Sparse Matrix–Matrix Multiplication) is an alternative dense-to-sparse strategy. A chain rule `r1(X,Y), r2(Y,Z) → r3(X,Z)` is exactly `A_r3 = A_r1 @ A_r2` where each relation is an adjacency matrix. SpMM exploits sparsity in the matrices and avoids padding to K_max.

SpMM + Hadamard covers all connected binary-head N-body rule types:
- **EXPAND / chain / fork rules** → SpMM with appropriate transposes: $A_{r_1}^\top A_{r_2}$, $A_{r_1} A_{r_2}^\top$, etc.
- **EXIST rules** (dangling variable) → SpMV with ones vector: $(A_r \mathbf{1}) > 0$.
- **FILTER_BOTH** (both variables bound, edge existence check) → Hadamard $A_{r_1} \odot A_{r_2}$. SpMM cannot handle this — there is no free dimension to sum over.

The reason FCStatic uses K-gather tensors rather than SpMM is CUDA-graph compatibility: SpSpMM (sparse × sparse) produces dynamic output sparsity, breaking `torch.compile(fullgraph=True)`. The K-gather approach preallocates a fixed-shape output at the cost of padding zeros.

### 5.2 Memory Parameters — K_max and N_max

#### K_max — neighbour fanout cap

K_max is the maximum number of objects the kernel fetches for any `(predicate, entity)` pair in a single stage.

$$
K_{\max} = \min\!\bigl(\text{max\_base\_degree} \times \text{safety\_factor},\; \text{hard\_cap}\bigr)
         = \min(28 \times 2,\; 128) = 56 \quad \text{(kinship\_family default)}
$$

`max_base_degree` is the highest out-degree of any `(pred, entity)` pair in the base facts. The $\times 2$ safety margin accounts for derived facts potentially having higher fanout.

**Why K_max dominates memory:** the kernel tensor shape is $[B, E, K^{N_{\max}}]$. $K$ appears as an *exponent*, so doubling $K$ quadruples memory for $N_{\max}=2$.

**K_max vs. num_steps:** `num_steps` (outer FC iterations) and memory are completely independent. Peak memory comes from inside a single kernel call — the $[B, E, K^{N_{\max}}]$ intermediate tensors — and that shape is the same whether you run 1 step or 100. Reducing `num_steps` derives fewer atoms but saves no memory.

#### N_max — max body atoms per rule

N_max = max rule body length across all compiled rules. For kinship_family all rules have ≤ 2 body atoms, so N_max = 2. This is **data-fixed** — it cannot be tuned without modifying the rule set.

$N_{\max}$ sets both the number of inner kernel stages and the exponent:

$$
[B,\, E,\, K^1] \;\to\; [B,\, E,\, K^2] \;\to\; \cdots \;\to\; [B,\, E,\, K^{N_{\max}}]
$$

For $N_{\max}=3$ memory would be $K\times$ worse than $N_{\max}=2$.

#### Peak memory estimate

`_estimate_static_bytes(R, N_max, E, P, K_max)` accounts for:

1. **Neighbour buffers:** $n_{\text{bufs}} \times [2,\, P{+}1,\, E,\, K_{\max}]$ (long + bool)
2. **Output tensor:** $[R \times N_{\max},\; E \times K^{N_{\max}}]$ (long)
3. **Kernel intermediates:** ~12 branch tensors of shape $[B, E, K^{N_{\max}}]$ live simultaneously at the deepest stage (`expand_n/m`, `exist_root_n/m`, `exist_cur_n/m`, `filter_n/m`, `pad_n/m`, `idx`). This term dominates for large $K_{\max}$.

At K_max=56 for kinship_family (K_base=28, safety_factor=2):
- Output tensor: ~21 GB
- Kernel intermediates: ~256 GB
- **Total estimate: ~277 GB**

The OOM pre-check in `_run_fc_benchmark` reads `/proc/meminfo` (CPU) or `torch.cuda.mem_get_info()` (CUDA) and skips FCStatic if the estimate exceeds 80% of available memory.

#### Tuning K_max to fit (kinship_family, 131 GB machine)

| K_max | Est. peak | Fits | Note |
|-------|-----------|------|------|
| 56 | 277 GB | ✗ | default (safety_factor=2) |
| 28 | 69 GB | ✓ | safety_factor=1; no truncation of base-fact lookups |
| 16 | 22 GB | ✓ | truncates ~43% of high-degree lookups |
| 8 | 5.7 GB | ✓ | aggressive truncation |

With K_max < max_base_degree (28), some `(pred, entity)` lookups are silently truncated and FCStatic derives fewer atoms than FCDynamic. K_max=28 is exact for base-fact lookups; derived atoms could in principle exceed 28 neighbours but rarely do in practice for kinship graphs.

---

## 6. Benchmarks

All runs use FCDynamic on CPU. FCStatic is OOM-skipped on the larger datasets due to K_max.

Run command:
```bash
conda run --no-capture-output -n gpu python -u \
  main/torch-ns/analysis/kg_complexity.py \
  -d <name> --benchmark-fc --fc-methods dynamic --num-steps <N>
```

---

> **Th. peak column** — formula: $|\Delta_t| \times K_{\max} \times 8$ bytes.
> Worst-case intermediate allocation if step $t$'s delta is used as anchor for the next derivation. This is a lower bound: actual memory is higher due to accumulated $|I_t|$, CSR index structures, and Python object overhead (~5–10× on large hash sets). OOM occurs when this estimate approaches available RAM.

> **Peak ΔRSS column** — empirical measurement: process RSS at end of step $t$ minus RSS at run start, read from `/proc/self/status`. Captures *persistent* growth (accumulated atom hash sets, CSR index arrays, Python object overhead) but not transient intermediates that are allocated and freed within a single (rule, anchor) pass. Th. peak >> Peak ΔRSS for dense steps because the partial-match lists are freed between rules.

---

### kinship_family

**Dataset:** 19,845 facts, E=2,968 entities, P=12 predicates, 143 rules (all 2-body), N_max=2, max_degree=2. For static setting (even if not reported results): K_max=56 (max_degree=28, safety_factor=2).

**Peak space derivation:**

**FCDynamic** — $O\!\left(|\Delta_t| \cdot K_{\max}^{N_{\max}-1}\right) + O(|I_t|)$

$N_{\max}-1 = 1$ extend stage: each of the $|\Delta_t|$ seed partial matches expands to at most $K_{\max}$ neighbors, stored as `int64` hashes (8 B each):

$$\text{Peak}_{\text{dyn}}(t) = |\Delta_t| \times K_{\max}^{1} \times 8\;\text{B}$$

Substituting $K_{\max}=56$ ($56\times8 = 448$ B/atom):

| Step | New atoms | New atoms × 56 × 8 B | Peak |
|------|-----------|----------------------|------|
| 0 | 32,287 | 14,464,576 | **14.5 MB** |
| 1 | 14,765 | 6,614,720 | **6.6 MB** |
| 2 | 6,055 | 2,712,640 | **2.7 MB** |
| 3 | 2,549 | 1,141,952 | **1.1 MB** |
| 4 | 733 | 328,384 | **328 KB** |
| 5 | 150 | 67,200 | **67 KB** |
| 6 | 27 | 12,096 | **12 KB** |

**FCStatic** — $O\!\left(R \cdot N_{\max} \cdot E \cdot K_{\max}^{N_{\max}}\right)$, $K_{\max} = \min(28\times2,\,128) = 56$

$B = R \times N_{\max} = 143\times2 = 286$; peak at the deepest kernel stage ($K_{\max}^2 = 3{,}136$):

| Term | Formula | Substitution | Size |
|------|---------|-------------|------|
| Neighbor buffers | $4(P{+}1)\,E\,K_{\max}\times9\;\text{B}$ | $4\times13\times2{,}968\times56\times9$ | **78 MB** |
| Output tensor | $B\,E\,K_{\max}^{2}\times8\;\text{B}$ | $286\times2{,}968\times3{,}136\times8$ | **21.3 GB** |
| Kernel intermediates | $12\,B\,E\,K_{\max}^{2}\times8\;\text{B}$ | $12\times21.3\;\text{GB}$ | **255.6 GB** |
| **Total** | | | **≈ 277 GB** |

FCStatic skipped: 277 GB > 80% × 131 GB = 105 GB.

**Rule classifier:** 143/143 rules classified by `_classify_general_rule`.

**Fully converges at step 6 (fixpoint). Total: 0.55s, 40 MB RSS.**

| Step | New atoms | Cumulative | Th. peak | Peak ΔRSS |
|------|-----------|------------|----------|-----------|
| 0 | +32,287 | 32,287 | 14.5 MB | 11 MB |
| 1 | +14,765 | 47,052 | 6.6 MB | 21 MB |
| 2 | +6,055 | 53,107 | 2.7 MB | 40 MB |
| 3 | +2,549 | 55,656 | 1.1 MB | 40 MB |
| 4 | +733 | 56,389 | 328 KB | 41 MB |
| 5 | +150 | 56,539 | 67 KB | 41 MB |
| 6 | +27 | 56,566 (fixpoint) | 12 KB | 41 MB |

The delta shrinks every step — classic semi-naive convergence. All theoretical peaks are negligible (<15 MB); the 40 MB actual RSS is Python interpreter and CSR index overhead. kinship_family is the only dataset here that fully converges (sparse family graph, bounded kinship depth).

---

### wn18rr

**Dataset:** 86,835 facts, E=40,559 entities, P=11 predicates, 42 rules, N_max=2, max_degree=442.

**Peak space derivation:**

**FCDynamic** — $O\!\left(|\Delta_t| \cdot K_{\max}^{N_{\max}-1}\right) + O(|I_t|)$

$K_{\max} = \min(442\times2,\,128) = 128$ (hard-cap limited). One extend stage ($128\times8 = 1{,}024$ B/atom):

$$\text{Peak}_{\text{dyn}}(t) = |\Delta_t| \times 128 \times 8\;\text{B}$$

| Step | New atoms | New atoms × 128 × 8 B | Peak |
|------|-----------|------------------------|------|
| 0 | 545,705 | 558,801,920 | **559 MB** |
| 1 | 2,131,849 | 2,183,013,376 | **2.18 GB** |
| 2 | 30,811,809 | 31,551,612,416 | **31.5 GB** |
| 3 (est.) | ~400,000,000 | ~409,600,000,000 | **~410 GB** (OOM) |

**FCStatic** — $O\!\left(R \cdot N_{\max} \cdot E \cdot K_{\max}^{N_{\max}}\right)$, $K_{\max} = 128$

$B = 42\times2 = 84$; $K_{\max}^2 = 16{,}384$:

| Term | Formula | Substitution | Size |
|------|---------|-------------|------|
| Neighbor buffers | $4(P{+}1)\,E\,K_{\max}\times9\;\text{B}$ | $4\times12\times40{,}559\times128\times9$ | **2.2 GB** |
| Output tensor | $B\,E\,K_{\max}^{2}\times8\;\text{B}$ | $84\times40{,}559\times16{,}384\times8$ | **446.6 GB** |
| Kernel intermediates | $12\,B\,E\,K_{\max}^{2}\times8\;\text{B}$ | $12\times446.6\;\text{GB}$ | **5.36 TB** |
| **Total** | | | **≈ 5.8 TB** |

FCStatic infeasible at any scale.

**Maximum feasible steps: 3.** Step 3 OOM-kills FCDynamic. Total: 20.15s, 274 MB RSS.

| Step | New atoms | Cumulative | Time | Th. peak | Peak ΔRSS |
|------|-----------|------------|------|----------|-----------|
| 0 | +545,705 | 545,705 | — | 559 MB | 14 MB |
| 1 | +2,131,849 | 2,677,554 | — | 2.18 GB | 284 MB |
| 2 | +30,811,809 | 33,489,363 | 20.15s total | 31.5 GB | 4.1 GB |
| 3 | OOM | — | — | est. ~410 GB (> 131 GB) | — |

The +30M jump at step 2 (~14× over step 1) reflects wn18rr's high-degree predicates (max degree 442). At step 3, the ~400M estimated new atoms would require ~410 GB just for intermediates — far beyond the 131 GB machine.

---

### FB15k237

**Dataset:** 272,115 facts, E=14,505 entities, P=237 predicates, 199 rules, N_max=2.

**Peak space derivation:**

**FCDynamic** — $O\!\left(|\Delta_t| \cdot K_{\max}^{N_{\max}-1}\right) + O(|I_t|)$

$K_{\max} = 128$ (hard-cap; confirmed by step 0: $659{,}485\times128\times8 = 675{,}312{,}640\;\text{B} = 675\;\text{MB}$). One extend stage:

$$\text{Peak}_{\text{dyn}}(t) = |\Delta_t| \times 128 \times 8\;\text{B}$$

| Step | New atoms | New atoms × 128 × 8 B | Peak |
|------|-----------|------------------------|------|
| 0 | 659,485 | 675,312,640 | **675 MB** |
| 1 | 8,107,610 | 8,302,192,640 | **8.3 GB** |
| 2 (est.) | ~98,000,000 | ~100,352,000,000 | **~100 GB** (OOM) |

**FCStatic** — $O\!\left(R \cdot N_{\max} \cdot E \cdot K_{\max}^{N_{\max}}\right)$, $K_{\max} = 128$

$B = 199\times2 = 398$; $K_{\max}^2 = 16{,}384$:

| Term | Formula | Substitution | Size |
|------|---------|-------------|------|
| Neighbor buffers | $4(P{+}1)\,E\,K_{\max}\times9\;\text{B}$ | $4\times238\times14{,}505\times128\times9$ | **15.9 GB** |
| Output tensor | $B\,E\,K_{\max}^{2}\times8\;\text{B}$ | $398\times14{,}505\times16{,}384\times8$ | **756.7 GB** |
| Kernel intermediates | $12\,B\,E\,K_{\max}^{2}\times8\;\text{B}$ | $12\times756.7\;\text{GB}$ | **9.08 TB** |
| **Total** | | | **≈ 9.9 TB** |

FCStatic infeasible at any scale.

**Maximum feasible steps: 2.** Step 2 OOM-kills FCDynamic. Total: 16.12s, 664 MB RSS.

| Step | New atoms | Cumulative | Time | Th. peak | Peak ΔRSS |
|------|-----------|------------|------|----------|-----------|
| 0 | +659,485 | 659,485 | — | 675 MB | 107 MB |
| 1 | +8,107,610 | 8,767,095 | 16.12s total | 8.3 GB | 3.5 GB |
| 2 | OOM | — | — | est. ~100 GB (> 131 GB with overhead) | — |

Step 0 is cheap (675 MB theoretical) due to FB15k237's many fine-grained predicates (237) keeping per-predicate degree low. Step 1 reaches 8.3 GB as 659k atoms fan out through 199 rules × 237 relations. At step 2, Python hash-set overhead on the ~100M+ estimated new atoms pushes total memory beyond the 131 GB limit.

---

### Summary

| Dataset | Facts | Rules | Max steps | Atoms at max | Time | Actual RSS | Th. peak (last step) |
|---------|-------|-------|-----------|--------------|------|------------|----------------------|
| kinship_family | 19,845 | 143 | converges (6) | 56,566 | 0.55s | 40 MB | 12 KB |
| wn18rr | 86,835 | 42 | 3 | 33,489,363 | 20.15s | 274 MB | 31.5 GB |
| FB15k237 | 272,115 | 199 | 2 | 8,767,095 | 16.12s | 664 MB | 8.3 GB |

wn18rr and FB15k237 grow without bound — the derived closure is exponentially larger than the base facts and hits RAM limits within 3 steps. kinship_family converges because the kinship relation graph has bounded depth and low branching.

The Th. peak at max step is the theoretical intermediate cost $|\Delta_{\text{last}}| \times K_{\max} \times 8$ for the final completed step. The *next* step's cost is what triggers OOM: wn18rr step 3 ≈ 410 GB, FB15k237 step 2 ≈ 100 GB (plus Python overhead). kinship's fixpoint delta is near zero so no OOM risk.

> **Why Th. peak >> Peak ΔRSS**
>
> Three effects compound to make the theoretical and empirical numbers diverge by 7–8× on dense datasets.
>
> **Effect 1 — the formula overcounts the anchor seed by ×P.**
> The formula `|Δ_t| × K_max × 8` treats the entire delta as the seed for one pass. In reality, each `(rule r, anchor k)` pass seeds only atoms in Δ_t whose predicate matches the anchor predicate — on average `|Δ_t| / P_active` atoms. For wn18rr step 2 with P=11 predicates this is 30.8M / 11 ≈ 2.8M atoms per pass, not 30.8M. Factor: ~11×.
>
> **Effect 2 — K_max is the worst-case fanout, not the average.**
> K_max is set from the single highest-fanout `(pred, entity)` pair in the graph (max_degree × safety_factor). The actual intermediate list size is `|anchor_seed| × actual_avg_fanout`, not `|anchor_seed| × K_max`. For wn18rr the mean out-degree per `(pred, entity)` pair in the base facts is `86,835 / (11 × 40,559) ≈ 0.19`; derived facts are denser but still far below 128. Most expand to 1–10 neighbors, not 128.
>
> **Effect 3 — intermediates are transient; Peak ΔRSS is sampled at step end.**
> The partial-match list for each `(rule, anchor)` pass is a temporary Python list that goes out of scope when the pass finishes. The RSS snapshot is taken after the full step completes — by then only persistent structures remain: the accumulated atom hash tensors and CSR index arrays, which scale with actual atom count `O(|I_t|)`. For wn18rr step 2 these account for roughly 268 MB (provable hashes) + 246 MB (delta hashes) + ~1 GB (CSR index values), with the remainder (~2.6 GB) coming from Python allocator fragmentation and object lifecycle overhead — totalling the observed 4.1 GB, far below the 31.5 GB Th. peak.
