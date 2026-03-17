# BCGrounder Tensor Pipeline — Architecture & Optimization Analysis

## Pipeline Overview

```
forward(queries, query_mask) -> GrounderOutput(state, evidence)
  |
  v
init_states ──> [states dict: 12 tensors]
  |
  v
for d in range(depth):
  |
  step(states, d)
  |   |
  |   +-- SELECT -----> queries [B,S,3], remaining [B,S,G,3], active [B,S]
  |   |                   |
  |   +-- RESOLVE ------> ResolvedChildren (9 tensors, M-scale)
  |   |                   |
  |   +-- FILTERS -------> ResolvedChildren (filtered rule_success)
  |   |                   |
  |   +-- HOOKS ---------> ResolvedChildren (hook-filtered)
  |   |                   |
  |   +-- PACK ----------> (states, SyncParams)       [M-scale]
  |   |                   |
  |   +-- POSTPROCESS
  |       |-- PRUNE_GOALS -> compact proof_goals       [G-scale]
  |       |-- SYNC -------> update accumulated_body    [G_body-scale]
  |       +-- COLLECT -----> append to collected_*     [G_body-scale]
  |
  v
filter_terminal(states) -> ProofEvidence
  |
  v
GrounderOutput(state=ProofState, evidence=ProofEvidence)
```

## Tensor Dimension Legend

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| B | Batch size | 32-192 |
| S | Proof states per batch | 50-550 |
| K_f | Fact children per state | 0-42 |
| K_r | Rule children per state | 4-30 |
| G | Max open goals | 4-100 |
| M | Max body atoms per rule | 1-3 |
| G_body | Accumulated body capacity (depth * M) | 2-45 |
| tG | Total collected groundings | 32-64 |

## Phase-by-Phase Data Flow

### States Dict (lives across all phases)

```
states = {
  "proof_goals":      [B, S, G, 3]       # open goals per branch
  "grounding_body":   [B, S, M_work, 3]   # current depth's body atoms (M-scale working buffer)
  "accumulated_body": [B, S, G_body, 3]   # ALL depths' body atoms (G_body-scale accumulator)
  "body_count":       [B, S]              # valid atoms in accumulated_body
  "top_ridx":         [B, S]              # first rule applied per branch
  "state_valid":      [B, S]              # alive branches
  "next_var_indices": [B]                 # variable counter
  "collected_body":   [B, tG, G_body, 3]  # output buffer (reused across depths)
  "collected_mask":   [B, tG]             # output validity
  "collected_ridx":   [B, tG]             # output rule indices
  "collected_bcount": [B, tG]             # output body counts
  "queries":          [B, 3]              # original queries (immutable)
}
```

### Phase 1: SELECT

```
proof_goals [B, S, G, 3]
    |
    +---> queries     = proof_goals[:, :, 0, :]   [B, S, 3]     (first goal)
    +---> remaining   = proof_goals.clone()         [B, S, G, 3]  (mask out slot 0)
    +---> active_mask = (goals[...,0,0] != pad)     [B, S]
```

**Alloc**: 1 clone of `[B, S, G, 3]`

### Phase 2: RESOLVE

```
queries [B,S,3] + remaining [B,S,G,3] + KB
    |
    +-- SLD: mgu_resolve_facts || mgu_resolve_rules
    |         |
    |         v
    |   ResolvedChildren:
    |     fact_goals   [B, S, K_f, G, 3]
    |     fact_gbody   [B, S, K_f, M, 3]
    |     fact_success [B, S, K_f]
    |     rule_goals   [B, S, K_r, G, 3]    <-- body + remaining assembled
    |     rule_gbody   [B, S, K_r, M, 3]
    |     rule_success [B, S, K_r]
    |     sub_rule_idx [B, S, K_r]
    |     fact_subs    [B, S, K_f, 2, 2]
    |     rule_subs    [B, S, K_r, 2, 2]
    |
    +-- ENUM: resolve_enum -> fill_body -> filter
              |
              v
        ResolvedChildren (K_f=0, rule_subs=padding)
```

**Alloc**: 9 output tensors + intermediates (body_subst, remaining assembly)

### Phase 3: PACK

```
ResolvedChildren [B, S, K_f/K_r, ...]
    |
    +-- Flatten facts:  [B, S*K_f, ...]   (inherit parent ridx, body_count)
    +-- Flatten rules:  [B, S*K_r, ...]   (capture body atoms, set ridx)
    |
    +-- Concatenate:    [B, N, ...]        where N = S*K_f + S*K_r
    |
    +-- Scatter-compact to S_out via cumsum + scatter_
    |
    v
  PackedStates:                            SyncParams:
    grounding_body [B, S_out, M, 3]          parent_map    [B, S_out]
    proof_goals    [B, S_out, G, 3]          winning_subs  [B, S_out, 2, 2]
    top_ridx       [B, S_out]                has_new_body  [B, S_out]
    state_valid    [B, S_out]                parent_bcount [B, S_out]
    body_count     [B, S_out]
```

**Alloc**: 8 output tensors (S_out+1 for overflow), 8 concat intermediates

### Phase 4a: PRUNE_GOALS

```
proof_goals [B, S, G, 3]
    |
    +-- prune_ground_facts: check atoms against fact_hashes
    |     mark ground facts for removal
    |
    +-- compact_atoms: left-align remaining atoms via argsort
    |
    v
  proof_goals [B, S, G, 3]   (same shape, gaps removed)
```

### Phase 4b: SYNC (G_body-scale, cold path)

```
SyncParams + accumulated_body [B, S_in, G_body, 3]
    |
    +-- Gather from parents:    acc = accumulated_body[parent_map]
    +-- Apply substitutions:    acc = apply_subs(acc, winning_subs)
    +-- Append new body atoms:  acc[body_count:body_count+M] = grounding_body
    +-- Update body_count:      body_count += new_lens
    |
    v
  accumulated_body [B, S_out, G_body, 3]   (updated in-place via scatter_)
  body_count       [B, S_out]
```

### Phase 4c: COLLECT

```
accumulated_body [B, S, G_body, 3] + proof_goals [B, S, G, 3]
    |
    +-- Detect terminal: all goals padding AND all body ground
    +-- Cat with collected_*: [B, tG+S, ...]
    +-- Dedup via prime-hash sort
    +-- Topk to select best tG
    |
    v
  collected_body  [B, tG, G_body, 3]  (updated)
  collected_mask  [B, tG]
  collected_ridx  [B, tG]
  collected_bcount [B, tG]
```

---

## Memory Profile Per Step

Estimates for B=32, S=50, K_r=8, G=10, M=3, G_body=6, tG=64.
Each `long` = 8 bytes, `bool` = 1 byte.

| Phase | Dominant Allocation | Shape | Size (MB) |
|-------|-------------------|-------|-----------|
| SELECT | remaining clone | [32,50,10,3] | 0.5 |
| RESOLVE (SLD) | rule_goals | [32,50,8,10,3] | 3.7 |
| RESOLVE (SLD) | fact_goals | [32,50,42,10,3] | 19.4 |
| RESOLVE (ENUM) | body_a (fill_body) | [1600,Re,G_use,3,3] | variable |
| PACK | all_goals (cat) | [32,650,10,3] | 5.0 |
| PACK | out_goals (scatter) | [32,51,10,3] | 0.5 |
| SYNC | acc (gather) | [32,50,6,3] | 0.3 |
| COLLECT | cb (cat) | [32,114,6,3] | 0.7 |
| COLLECT | atom_hashes (dedup) | [32,114,6] | 0.2 |
| **Total per step** | | | **~30-50 MB** |

The actual peak depends on which intermediates overlap in lifetime. The **RESOLVE** phase
dominates for SLD (K_f can be large), while **PACK** dominates for ENUM (large K_enum).

---

## Optimization Opportunities

### OPT-1: Collect Groundings — Avoid Full tG+S Materialization

**Current**: Concatenates `collected_*` ([B, tG]) with new groundings ([B, S]) into
`[B, tG+S]`, then dedup-sorts the full array, then topk back to tG.

```python
# Current: 4 cats + full dedup + topk + 4 gathers
cb = torch.cat([collected_body, body_new], dim=1)     # [B, tG+S, G_body, 3]
cm = torch.cat([collected_mask, valid_grounding], dim=1)
cr = torch.cat([collected_ridx, ridx_new], dim=1)
c_bc = torch.cat([collected_bcount, body_count], dim=1)
cm = _dedup_groundings(cb, cr, cm, G_body)             # sort [B, tG+S]
_, ki = cm.to(torch.int8).topk(n_k, ...)               # select tG from tG+S
out_body = cb.gather(1, ki...)                          # [B, tG, G_body, 3]
```

**Problem**: The concatenated `cb` has shape `[B, tG+S, G_body, 3]` — the largest
intermediate in the entire pipeline. For tG=64, S=50, G_body=6: this is 5x the output size.

**Proposed**: Write new groundings into reserved slots, dedup only the new entries.

```python
# Proposed: hash new groundings against existing, write non-dups to free slots
new_hashes = _hash_groundings(body_new, ridx_new)          # [B, S]
existing_hashes = _hash_groundings(collected_body, collected_ridx)  # [B, tG]
is_dup = _check_membership(new_hashes, existing_hashes)     # [B, S]
valid_new = valid_grounding & ~is_dup                       # [B, S]
# Scatter new entries into first free slots of collected_*
free_slots = (~collected_mask).long().cumsum(dim=1) - 1     # [B, tG]
# ... scatter valid_new into free_slots ...
```

**Savings**: Avoids `[B, tG+S, G_body, 3]` cat tensor. Dedup is O(S) instead of O(tG+S).
**Complexity**: Medium — need hash-set membership check instead of sort-dedup.
**Compile safety**: Fully static shapes, no dynamic indexing.

### OPT-2: Enum Dual Concat — Direct Assembly

**Current**: Pads direction A/B to same size, then concatenates:
```python
body_a = pad(body_a, ...)    # new alloc if sizes differ
body_b = pad(body_b, ...)    # new alloc if sizes differ
body_all = torch.cat([body_a, body_b], dim=2)  # another alloc
```

**Proposed**: Pre-allocate combined tensor, write A and B directly:
```python
G_total = G_use_a + G_use_b
body_all = torch.full((N, Re, G_total, M, 3), pad, device=dev)
body_all[:, :, :G_use_a] = body_a
body_all[:, :, G_use_a:G_use_a + G_use_b] = body_b[:, :, :G_use_b]
```

**Savings**: 1 allocation instead of 3 (2 pads + 1 cat).
**Complexity**: Low.

### OPT-3: SLD Rule Goals Assembly — Fuse with mgu_resolve_rules

**Current**: `mgu_resolve_rules` returns `(body_subst, remaining)`, then `resolve_sld`
allocates a new `rule_goals` and copies both in:
```python
rule_goals = torch.full((B, S, K_r, G, 3), pad, ...)   # new alloc
rule_goals[:, :, :, :Bmax, :] = rule_body_subst          # copy
rule_goals[:, :, :, Bmax:, :] = rule_remaining[:, :, :, :n_rem, :]  # copy
```

**Proposed**: Have `mgu_resolve_rules` return pre-assembled goals tensor:
```python
# Inside mgu_resolve_rules, assemble directly:
rule_goals = torch.full((B, S, K_r, G, 3), pad, ...)
rule_goals[..., :Bmax, :] = body_subst
rule_goals[..., Bmax:Bmax+n_rem, :] = remaining[..., :n_rem, :]
return rule_goals, rule_gbody, rule_success, sub_rule_idx, rule_subs
```

**Savings**: Eliminates 1 `[B, S, K_r, G, 3]` allocation + 2 copy ops per step.
**Complexity**: Low (move assembly into mgu_resolve_rules).

### OPT-4: Pack S_out+1 Overflow — Pre-allocate Persistent Buffers

**Current**: Every step allocates 7 tensors at `[B, S_out+1, ...]`, uses them for scatter,
then slices to `[B, S_out, ...]`.

**Proposed**: Pre-allocate once in `__init__` and reuse:
```python
# In __init__:
self._pack_buf_goals = torch.full((B_max, S+1, G, 3), pad, device=dev)
self._pack_buf_gbody = torch.full((B_max, S+1, M, 3), pad, device=dev)
# ...

# In pack_states: fill pre-allocated buffers, reset to pad before use
self._pack_buf_goals.fill_(pad)
self._pack_buf_goals.scatter_(1, ti, all_goals)
```

**Savings**: Eliminates 7 allocations per step (torch.full creates new memory each time).
**Complexity**: Medium — requires batch-size awareness and reset logic.
**Compile safety**: Pre-allocated buffers work with CUDA graphs (same memory address).
**Risk**: Requires B to be fixed at init time. Multi-batch-size support would need
multiple buffer sets or max-B pre-allocation.

### OPT-5: Fact Children Path — Skip When K_f=0

**Current**: Enum resolution returns K_f=0, but pack_states still allocates empty tensors
and concatenates them:
```python
f_gbody = torch.full((B, 0, M_work, 3), pad, ...)  # 0-sized but allocated
f_goals = torch.full((B, 0, G, 3), pad, ...)
# ... 6 more empty tensors
all_gbody = torch.cat([f_gbody, r_gbody], dim=1)    # cat with 0-sized = copy
```

**Proposed**: Check K_f early and skip the fact path entirely:
```python
if K_f == 0:
    all_gbody = r_gbody
    all_goals = r_goals
    # ... skip concatenation
else:
    # ... existing fact+rule cat logic
```

**Savings**: Eliminates ~8 empty tensor allocations + 8 no-op concatenations per step.
**Complexity**: Low.
**Note**: K_f=0 happens every step for enum resolution (the most common mode).

---

## Professionalization / Cleaning Opportunities

### CLEAN-1: Unused `Tuple` Import in Resolution Modules

After the `ResolvedChildren` refactor, `sld.py` and `rtf.py` no longer need `Tuple` in
their imports (it was removed from sld.py, but rtf.py still doesn't use it either).
Clean import: remove any remaining `Tuple` imports that aren't used.

### CLEAN-2: Type `Dict` Parameters in bc.py

Several methods use bare `Dict` for the states parameter:
```python
def step(self, states: Dict, d: int) -> Dict:
def _select(self, states: Dict) -> Tuple[...]:
def _postprocess(self, states: Dict, sync: SyncParams) -> Dict:
```

Could be tightened to `Dict[str, Tensor]` for documentation value. Not required
since the dict is internal, but improves IDE support.

### CLEAN-3: Hook Parameter Types in `__init__`

```python
def __init__(self, ..., fact_hook=None, rule_hook=None, ...):
```

Could be typed as:
```python
fact_hook: Optional[ResolutionFactHook] = None,
rule_hook: Optional[ResolutionRuleHook] = None,
```

### CLEAN-4: Factory Return Type

`factory.py:parse_grounder_type()` returns bare `dict`. Could be `Dict[str, Any]`.

### CLEAN-5: StepHook Docstring Shape

The StepHook protocol docstring says `body: [B, tG, M, 3]` but the actual tensor is
`[B, tG, G_body, 3]` where `G_body = depth * M`. The `M` in the docstring should
be `G_body` to match the accumulated body tensor shape.

### CLEAN-6: GroundingHook Docstring Shape

Same issue: `body: [B, tG, M, 3]` should be `body: [B, tG, G_body, 3]`.
