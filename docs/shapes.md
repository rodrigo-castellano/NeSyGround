# Tensor Shapes in BCGrounder

This document traces every tensor dimension through the BCGrounder pipeline:
initialization, resolution, packing, collection, and final output.

## Dimension glossary

| Symbol | Name | Definition | Typical value |
|--------|------|------------|---------------|
| B | batch | number of queries per batch | 64–256 |
| S | states | state budget (proof branches per query) | K^depth, capped |
| G | goals | max atoms in a proof state (remaining goals) | `1 + depth * (M-1)` |
| M | body width | max atoms in the longest rule body | `max(rule_lens)` = `kb.M` |
| K | children | total children per state (K_f + K_r) | auto from KB |
| K_f | fact children | max fact unification matches per goal | auto |
| K_r | rule children | max rule head unification matches per goal | auto |
| tG | total groundings | max collected groundings per query | `effective_total_G` |
| pad | padding index | `kb.padding_idx` | 0 |

## Pipeline overview

```
init_states          →  proof_goals     [B, 1, G, 3]
                        grounding_body  [B, 1, M, 3]     (pad-filled)
                        top_ridx        [B, 1]            (-1 = no rule yet)
                        state_valid     [B, 1]
                        collected_body  [B, tG, M, 3]     (pad-filled)
                        collected_mask  [B, tG]            (False)
                        collected_ridx  [B, tG]

for d in range(depth):
    step():
        SELECT    →  queries     [B, S, 3]        first goal of each state
                     remaining   [B, S, G, 3]      goals after removing first
        RESOLVE   →  7-tuple: fact_goals, fact_gbody, fact_success,
                              rule_goals, rule_gbody, rule_success, sub_ridx
        PACK      →  proof_goals     [B, S, G, 3]
                     grounding_body  [B, S, M, 3]
                     top_ridx        [B, S]
                     state_valid     [B, S]
        POSTPROCESS → collect terminal groundings into collected_*

filter_terminal   →  GroundingResult(body, mask, count, rule_idx)
```

## Phase 1: init_states

```python
proof_goals     = [B, 1, G, 3]     # query at slot 0, rest padding
grounding_body  = [B, 1, M, 3]     # all padding (no body yet)
                                    # M=1 when track_grounding_body=False
top_ridx        = [B, 1]           # -1 (no rule applied yet)
state_valid     = [B, 1]           # True for valid queries
collected_body  = [B, tG, M, 3]   # all padding
collected_mask  = [B, tG]          # all False
collected_ridx  = [B, tG]          # all 0
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
Both SLD and RTF return the same 7-tuple format:

### SLD resolution

```
fact_goals    [B, S, K_f, G, 3]    # children from fact unification
fact_gbody    [B, S, K_f, M, 3]    # parent's grounding_body (passthrough)
fact_success  [B, S, K_f]          # which fact children are valid

rule_goals    [B, S, K_r, G, 3]    # children from rule head unification
                                    # slot 0..Bmax-1 = substituted rule body
                                    # slot Bmax..G   = remaining parent goals
rule_gbody    [B, S, K_r, M, 3]    # rule body atoms (from mgu_resolve_rules)
rule_success  [B, S, K_r]          # which rule children are valid
sub_rule_idx  [B, S, K_r]          # which rule was matched
```

### RTF resolution

Same output format, but:
- `fact_goals/gbody/success` are empty (K_f=0) — no standalone fact resolution
- `rule_goals` has K_rtf = K_r * K_f children (rules first, then facts on body[0])
- `rule_gbody` is propagated from the rule-level resolution (same M dimension)

### What mgu_resolve_rules returns for gbody

When `grounding_body` is passed (track=True):
- Substitutes variables in the parent's grounding_body using the MGU bindings
- Returns `[B, S, K_r, M, 3]` with the updated body

When `grounding_body=None` (track=False):
- Returns `[B, S, K_r, 0, 3]` — M=0, empty

## Phase 4: PACK

`pack_states()` flattens K children into the S budget.

**Inputs:**
```
7-tuple from resolve       (fact/rule goals, gbody, success, ridx)
top_ridx        [B, S]     parent's rule index
grounding_body  [B, S, M, 3]  parent's grounding body
```

**Critical grounding_body logic:**

```python
first = (top_ridx == -1)   # True at depth 0 (no rule applied yet)

r_gbody = torch.where(
    first,
    rule_goals[:, :, :, :M, :],   # ← FIRST resolution: capture rule body atoms
    rule_gbody,                    # ← LATER resolutions: keep parent's body
)
```

This means:
- At depth 0 (first rule application): `grounding_body` is set to the first M
  atoms of `rule_goals` (the matched rule's body atoms after substitution)
- At depth 1, 2, ...: `grounding_body` is **preserved unchanged** from the parent

**Output:**
```
grounding_body  [B, S_out, M, 3]
proof_goals     [B, S_out, G, 3]
top_ridx        [B, S_out]
state_valid     [B, S_out]
```

## Phase 5: POSTPROCESS (collect_groundings)

After each step, terminal states (all goals resolved to ground facts) are
collected into the `collected_*` buffers:

```
collected_body  [B, tG, M, 3]   ← grounding_body of terminal states
collected_mask  [B, tG]          ← True for collected slots
collected_ridx  [B, tG]          ← rule index of terminal states
```

## Phase 6: filter_terminal → GroundingResult

After all depths, soundness filtering produces the final output:

```python
GroundingResult(
    body     = [B, tG, M, 3],    # grounding body atoms
    mask     = [B, tG],           # which groundings are valid
    count    = [B],               # number of valid groundings per query
    rule_idx = [B, tG],           # which rule produced each grounding
)
```

## Shape relationships

```
G = 1 + depth * (M - 1)      # goals grow with depth (one goal replaced by M-1 new ones)
M = kb.M = max(rule_lens)    # fixed — max body atoms in any single rule
S = min(K^depth, hard_cap)   # states grow exponentially, capped
tG = effective_total_G       # max groundings to collect (configurable)
K = K_f + K_r                # children per state per step
```

---

## KNOWN LIMITATION: grounding_body does not accumulate across depths

### Current behavior

`grounding_body [B, S, M, 3]` captures only the **first rule's body atoms**.
Subsequent rule applications at deeper depths do NOT append their body atoms.

For a depth-2 proof: `Q ← R1(a, b) ← R2(c, d)`:
- `grounding_body` contains `[a, b, pad, ...]` (R1's body only)
- R2's body atoms `[c, d]` are **not recorded** in `grounding_body`

### keras-ns behavior (reference)

keras-ns accumulates all body atoms across depths into a single tuple:
- Each grounding stores `(head, (body_atom_1, body_atom_2, ..., body_atom_n))`
- At depth 2, body atoms from both R1 and R2 are included
- The tuple grows with depth

### Impact on reasoning modules

SBR/DCR/R2N compute fuzzy truth values over `GroundingResult.body` atoms.
With the current torch-ns behavior, the reasoner only scores the first rule's
body — missing evidence from deeper rule applications.

For depth-1 proofs (single rule application), this is correct.
For depth ≥ 2, the proof evidence is incomplete.

### Why the correct dimension is G, not M

The proof evidence is all body atoms that appeared as goals and got resolved
to ground facts — the "leaves" of the proof tree. Tracing a depth-2 proof:

```
depth 0:  goals = [Q]                          grounding_body should be []
          resolve Q with R1(h) :- b1, b2

depth 1:  goals = [b1, b2]                     grounding_body should be [b1, b2]
          resolve b1 with R2(h) :- b3, b4

depth 2:  goals = [b3, b4, b2]                 grounding_body should be [b1, b2, b3, b4]
          b3 ✓ fact, b4 ✓ fact, b2 ✓ fact → PROVED
```

The proof evidence `{b1, b2, b3, b4}` has up to `depth * M` atoms.
But `G = 1 + depth * (M - 1)` already bounds the maximum number of open
goals at any point. Since every body atom passes through `proof_goals` before
being resolved, **G is the natural upper bound for the grounding body**.

This means grounding_body should have shape `[B, S, G, 3]` (same as
proof_goals), not `[B, S, M, 3]`.

### What would full accumulation require

1. **Change grounding_body dimension from M to G**:
   `G = 1 + depth * (M - 1)` is already computed and static.
   Shape: `[B, S, G, 3]` — same as `proof_goals`.
   CUDA graph compatible (G is known at init time).

2. **Modify pack_states**: Instead of `torch.where(first, new, parent)`,
   append new body atoms after the parent's existing atoms:
   ```
   depth 0: grounding_body = [b1, b2, pad, pad, pad]
   depth 1: grounding_body = [b1, b2, b3, b4, pad]
   ```
   Track a `body_count` per state to know where to append.

3. **Resolution modules**: `mgu_resolve_rules` already applies substitutions
   to grounding_body (via concatenation with body + remaining). It would
   need to append the new rule's body atoms instead of just substituting
   the parent's body.

4. **Memory impact**: G vs M difference:
   - Family (M=2, depth=2): G=3 vs M=2 — 50% more
   - WN18RR (M=3, depth=3): G=7 vs M=3 — 133% more
   - Still modest: `B * S * G * 3 * 8` bytes

5. **GroundingResult.body changes**: `[B, tG, G, 3]` instead of `[B, tG, M, 3]`.
   Downstream consumers (reasoners, hooks) see more atoms per grounding.
