# Pipeline Tensors in BCGrounder

This document traces every tensor dimension through the BCGrounder pipeline:
initialization, resolution, packing, collection, and final output.

## Dimension glossary

| Symbol | Name | Definition | Typical value |
|--------|------|------------|---------------|
| B | batch | number of queries per batch | 64–256 |
| S | states | state budget (proof branches per query) | K^depth, capped |
| G | goals | max open atoms in a proof state (remaining goals) | `1 + depth * (M-1)` |
| G_body | body capacity | max accumulated body atoms across all depths | `depth * M` |
| M | body width | max atoms in the longest rule body | `max(rule_lens)` = `kb.M` |
| K | children | total children per state (K_f + K_r) | auto from KB |
| K_f | fact children | max fact unification matches per goal | auto |
| K_r | rule children | max rule head unification matches per goal | auto |
| tG | total groundings | max collected groundings per query | `effective_total_G` |
| pad | padding index | `kb.padding_idx` | 0 |

## Key design: M-working / G_body-accumulator split

The pipeline uses two separate body tensors to keep the hot path fast:

| Tensor | Shape | Role | In MGU resolution? | In pack clone+scatter? |
|--------|-------|------|-------------------|----------------------|
| `grounding_body` | `[B, S, M, 3]` | Current depth's rule body atoms (working buffer) | Yes (M-sized, fast) | Yes (M-sized clone, fast) |
| `accumulated_body` | `[B, S, G_body, 3]` | All depths' body atoms (output accumulator) | No (subs applied in SYNC) | No clone — gather-reindex only |

The expensive operations (MGU substitution, pack clone+scatter) only touch M-sized tensors.
The G_body accumulator is updated once per step via lightweight SYNC operations.

## Pipeline overview

```
init_states          →  proof_goals      [B, 1, G, 3]
                        grounding_body   [B, 1, M, 3]       (pad-filled, working)
                        accumulated_body [B, 1, G_body, 3]  (pad-filled, accumulator)
                        body_count       [B, 1]              (0)
                        top_ridx         [B, 1]              (-1 = no rule yet)
                        state_valid      [B, 1]
                        collected_body   [B, tG, G_body, 3]  (pad-filled)
                        collected_mask   [B, tG]              (False)
                        collected_ridx   [B, tG]

for d in range(depth):
    step():
        SELECT    →  queries     [B, S, 3]          first goal of each state
                     remaining   [B, S, G, 3]        goals after removing first

        RESOLVE   →  9-tuple:
                     fact_goals   [B, S, K_f, G, 3]      fact children proof states
                     fact_gbody   [B, S, K_f, M, 3]      parent's M-sized body (passthrough)
                     fact_success [B, S, K_f]             valid fact matches
                     fact_subs    [B, S, K_f, 2, 2]       MGU subs from fact unification
                     rule_goals   [B, S, K_r, G, 3]      rule children proof states
                     rule_gbody   [B, S, K_r, M, 3]      parent's M-sized body with subs
                     rule_success [B, S, K_r]             valid rule matches
                     sub_ridx     [B, S, K_r]             matched rule index
                     rule_subs    [B, S, K_r, 2, 2]       MGU subs from rule unification

        PACK      →  proof_goals     [B, S, G, 3]       (M-sized grounding_body)
                     grounding_body  [B, S, M, 3]        (current depth's body)
                     parent_map      [B, S]               (compaction index)
                     winning_subs    [B, S, 2, 2]         (subs for each winner)
                     has_new_body    [B, S]               (True for rule children)

        SYNC      →  accumulated_body [B, S, G_body, 3]  (reindex + subs + append M)
                     body_count       [B, S]              (updated)

        POSTPROCESS → collect terminal groundings from accumulated_body into collected_*

filter_terminal   →  GroundingResult(body, mask, count, rule_idx, body_count)
```

### Tensor meanings

| Tensor | Shape | Meaning |
|--------|-------|---------|
| **proof_goals** | `[B, S, G, 3]` | Open goals (atoms) still to prove in each state. Each goal is a triple `(predicate, subject, object)` encoded as 3 integer indices. Slot 0 is the next goal to resolve; padding fills unused slots. |
| **grounding_body** | `[B, S, M, 3]` | Working buffer: the current depth's rule body atoms. Captured from `rule_goals[:,:,:,:M,:]` when a rule is matched. M-sized for fast resolution and pack operations. |
| **accumulated_body** | `[B, S, G_body, 3]` | Accumulator: all body atoms from every rule application across all depths. Updated once per step in SYNC phase. Used for terminal grounding collection and final output. |
| **body_count** | `[B, S]` | Number of valid body atoms in each state's `accumulated_body`. Used as the write offset when appending new atoms in SYNC. |
| **top_ridx** | `[B, S]` | Index of the first rule applied in each proof branch. `-1` means no rule has been applied yet (initial state). |
| **state_valid** | `[B, S]` | Boolean mask: `True` for active proof states, `False` for dead-end or padding states. |
| **collected_body** | `[B, tG, G_body, 3]` | Buffer accumulating `accumulated_body` from terminal states (all goals resolved). |
| **collected_mask** | `[B, tG]` | Boolean mask: `True` for collected grounding slots that have been filled. |
| **collected_ridx** | `[B, tG]` | Rule index for each collected grounding. |
| **collected_bcount** | `[B, tG]` | Body count for each collected grounding. |
| **queries** | `[B, S, 3]` | The first non-padding goal extracted from each state for resolution. |
| **remaining** | `[B, S, G, 3]` | Goals left after removing the selected query. |
| **fact_goals** | `[B, S, K_f, G, 3]` | Child proof states from fact unification. |
| **fact_gbody** | `[B, S, K_f, M, 3]` | Parent's M-sized `grounding_body` passed through (facts don't introduce new body atoms). |
| **fact_success** | `[B, S, K_f]` | Which fact children produced a valid unification match. |
| **fact_subs** | `[B, S, K_f, 2, 2]` | MGU substitution pairs from fact unification. Propagated to SYNC for applying to `accumulated_body`. |
| **rule_goals** | `[B, S, K_r, G, 3]` | Child proof states from rule head unification. Slots `0..body_len-1` hold the new subgoals, rest hold remaining parent goals. |
| **rule_gbody** | `[B, S, K_r, M, 3]` | Parent's M-sized `grounding_body` with MGU substitutions applied. |
| **rule_success** | `[B, S, K_r]` | Which rule children produced a valid head unification match. |
| **sub_ridx** | `[B, S, K_r]` | Index of the matched rule for each rule child. |
| **rule_subs** | `[B, S, K_r, 2, 2]` | MGU substitution pairs from rule unification. Propagated to SYNC. |
| **parent_map** | `[B, S_out]` | Compaction index from pack_states: maps each output state to its parent state in S_in. Used by SYNC to gather-reindex `accumulated_body`. |
| **winning_subs** | `[B, S_out, 2, 2]` | The winning substitution pairs after compaction. Used by SYNC to apply MGU subs to `accumulated_body`. |
| **has_new_body** | `[B, S_out]` | True for output states that came from rule matches (have new body atoms to append). False for fact matches. |
| **GroundingResult.body** | `[B, tG, G_body, 3]` | Final accumulated body atoms from all rule applications, after soundness filtering. |
| **GroundingResult.mask** | `[B, tG]` | Final boolean mask: `True` for groundings that passed soundness checks. |
| **GroundingResult.count** | `[B]` | Number of valid groundings per query. |
| **GroundingResult.rule_idx** | `[B, tG]` | Rule index for each valid grounding. |
| **GroundingResult.body_count** | `[B, tG]` | Number of valid body atoms per grounding. Used by downstream consumers to build atom-level masks. |

## Phase 1: init_states

```python
proof_goals      = [B, 1, G, 3]       # query at slot 0, rest padding
grounding_body   = [B, 1, M, 3]       # all padding (working buffer, M-sized)
accumulated_body = [B, 1, G_body, 3]   # all padding (accumulator, G_body-sized)
body_count       = [B, 1]             # 0 (no body atoms accumulated yet)
top_ridx         = [B, 1]             # -1 (no rule applied yet)
state_valid      = [B, 1]             # True for valid queries
collected_body   = [B, tG, G_body, 3] # all padding
collected_mask   = [B, tG]            # all False
collected_ridx   = [B, tG]            # all 0
collected_bcount = [B, tG]            # all 0
```

## Phase 2: SELECT

Extracts the first non-padding goal from each state:

```
Input:  proof_goals  [B, S, G, 3]
Output: queries      [B, S, 3]       ← proof_goals[:, :, 0, :]
        remaining    [B, S, G, 3]    ← proof_goals[:, :, 1:, :] (shifted)
        active_mask  [B, S]          ← queries[:,:,0] != pad
```

## Phase 3: RESOLVE

Resolution produces children for each (batch, state) pair.
Returns a **9-tuple** (7 original tensors + fact_subs + rule_subs):

### SLD resolution

```
fact_goals    [B, S, K_f, G, 3]      # children from fact unification
fact_gbody    [B, S, K_f, M, 3]      # parent's M-sized grounding_body (passthrough)
fact_success  [B, S, K_f]            # which fact children are valid

rule_goals    [B, S, K_r, G, 3]      # children from rule head unification
rule_gbody    [B, S, K_r, M, 3]      # parent's M-sized grounding_body with MGU subs
rule_success  [B, S, K_r]            # which rule children are valid
sub_rule_idx  [B, S, K_r]            # which rule was matched

fact_subs     [B, S, K_f, 2, 2]      # MGU subs from fact unification
rule_subs     [B, S, K_r, 2, 2]      # MGU subs from rule head unification
```

### What mgu_resolve_rules returns for gbody

When `grounding_body` is passed (track=True):
- Substitutes variables in the parent's M-sized grounding_body using MGU bindings
- Returns `[B, S, K_r, M, 3]` with the updated body

When `grounding_body=None` (track=False):
- Returns `[B, S, K_r, 0, 3]` — empty

## Phase 4: PACK

`pack_states()` flattens K children into the S budget.
Works entirely with M-sized grounding_body (fast).

**Inputs:**
```
9-tuple from resolve   (fact/rule goals, gbody [M], success, subs, ridx)
top_ridx        [B, S]       parent's rule index
grounding_body  [B, S, M, 3] parent's M-sized working body
body_count      [B, S]       atoms already accumulated
```

**Grounding body handling (M-sized):**

For rule children: capture current rule's body atoms from `rule_goals[:,:,:,:M,:]`.
For fact children: set to padding (no new body atoms this depth).

**Additional outputs for SYNC:**
```
parent_map    [B, S_out]       which parent each output state came from
winning_subs  [B, S_out, 2, 2] MGU subs for each winning child
has_new_body  [B, S_out]       True for rule children, False for fact children
```

**Output:**
```
grounding_body  [B, S_out, M, 3]       (current depth's body atoms)
proof_goals     [B, S_out, G, 3]
top_ridx        [B, S_out]
state_valid     [B, S_out]
parent_map      [B, S_out]
winning_subs    [B, S_out, 2, 2]
has_new_body    [B, S_out]
```

## Phase 4b: SYNC (accumulated_body update)

After PACK reorders states, SYNC propagates and updates `accumulated_body`:

```python
# a. Gather accumulated_body from parents (reindex to new state order)
acc = accumulated_body.gather(1, parent_map)      # [B, S_out, G_body, 3]

# b. Apply winning substitutions (bind variables from this depth)
acc = apply_substitutions(acc, winning_subs)       # 2 torch.where ops

# c. Append new M body atoms at body_count offset
acc.scatter_(2, write_pos, new_atoms)              # where has_new_body

# d. Update body_count
body_count = parent_bcount + new_lens              # where has_new_body
```

All operations are static-shape and `torch.compile(fullgraph=True)` compatible.

## Phase 5: POSTPROCESS (collect_groundings)

After each step, terminal states (all goals resolved to ground facts) are
collected from `accumulated_body` into the `collected_*` buffers:

```
collected_body   [B, tG, G_body, 3]  ← accumulated_body of terminal states
collected_mask   [B, tG]             ← True for collected slots
collected_ridx   [B, tG]             ← rule index of terminal states
collected_bcount [B, tG]             ← body_count of terminal states
```

## Phase 6: filter_terminal → GroundingResult

After all depths, soundness filtering produces the final output:

```python
GroundingResult(
    body       = [B, tG, G_body, 3],  # accumulated body atoms from all rule applications
    mask       = [B, tG],              # which groundings are valid
    count      = [B],                  # number of valid groundings per query
    rule_idx   = [B, tG],              # which rule produced each grounding
    body_count = [B, tG],              # valid body atoms per grounding
)
```

## Shape relationships

```
G      = 1 + depth * (M - 1)  # max open goals (one removed, M-1 added per step)
G_body = depth * M             # max accumulated body atoms (M added per step, append-only)
M      = kb.M = max(rule_lens) # fixed — max body atoms in any single rule
S      = min(K^depth, hard_cap)# states grow exponentially, capped
tG     = effective_total_G     # max groundings to collect (configurable)
K      = K_f + K_r             # children per state per step
```

Note: G and G_body differ because goals have **replacement** semantics (select 1, add M → net +M-1)
while body has **append-only** semantics (add M per rule application → +M per step).

## Performance: why two body tensors?

The hot path (RESOLVE + PACK) runs at every depth and dominates GPU time.
By keeping these operations M-sized:

| Operation | Old (single G_body tensor) | New (M + G_body split) |
|-----------|---------------------------|----------------------|
| MGU substitution | `[N_r, G_body, 3]` | `[N_r, M, 3]` (fast) |
| Pack clone | `[B, n_r, G_body, 3]` | `[B, n_r, M, 3]` (fast) |
| Pack scatter | G_body-sized | M-sized (fast) |
| SYNC (gather+subs+append) | — | `[B, S_out, G_body, 3]` (once per step) |

For depth=4, M=2: M=2 vs G_body=8. The hot path is 4x smaller.
The SYNC adds a lightweight gather-reindex (reads S_out entries, not n_r)
plus 2 `torch.where` ops and a small scatter — all much cheaper than the
clone+scatter on the full n_r × G_body tensor.

## Memory impact

| Dataset | M | depth | G_body | Working (M) | Accumulator (G_body) |
|---------|---|-------|--------|-------------|---------------------|
| Family | 2 | 1 | 2 | 2 | 2 (same, no overhead) |
| Family | 2 | 2 | 4 | 2 | 4 |
| ablation | 2 | 4 | 8 | 2 | 8 |
| WN18RR | 3 | 3 | 9 | 3 | 9 |

For depth=1 (DpRL): G_body = M, so the accumulator adds zero overhead.
