# grounder

Grounding techniques for neural-symbolic reasoning: given facts, rules and queries, the rule groundings a reasoner
scores. One class per technique over one data model (`KB`), on the GPU, with no caps: every lookup is enumerated in
full, a step's memory is bounded by streaming its work, and what does not fit runs out of memory.

| class | technique | output |
|---|---|---|
| `PBC` | parametrized backward chaining, BC_{w,d} (the IJCAI-25 grounders, keras-ns's prune) | `Groundings` |
| `SLD` | SLD resolution over proof states | `Proofs` |
| `Forward` | semi-naive forward chaining; `ground`: each derived atom's witness | `Closure`, `Groundings` |

```python
from grounder import KB, PBC, SLD, Forward, Guide

kb = KB.from_strings(facts, rules, entity2id, relation2id, device="cuda")   # facts [F, 3]; rules as parsed

g = PBC(kb, depth=2, width=1, prune="keras").ground(queries)        # queries [S, Q, 3]: S pools of Q queries
g = PBC.parse(kb, "enum.keras.w1.d2.flat")                          # torch-ns's type strings
proofs = SLD(kb, depth=3).prove(queries, mask)                      # [Q, 3] -> each query's proofs
closure = Forward(kb, depth=10).closure()                           # the derived atoms
witnessed = Forward(kb).ground(queries)                             # a derived query's first grounding
guided = PBC(kb, depth=3, width=1, guide=Guide(scorer, k=4))        # a model keeps k groundings per goal
```

`Groundings` holds each pool's atoms and its groundings (rule, head row, body rows), sorted and distinct; a pool is
what is grounded and pruned together, and a pool's output is what grounding it alone gives.

## BC_{w,d,u}

`w` the width (at most `w` body atoms of a grounding are not facts), `d` the depth (steps), `u` whether unknown leaves
may remain (the papers: no). `prune="keras"` is keras-ns's `ApproximateBackwardChainingGrounder` as the IJCAI-25 code
runs it (width `w` at every step, then its proof walk); `prune="fp_batch"` makes the last step width 0 and keeps what
is proved within `d` rounds. `CLAUDE.md` has the parity record against keras-ns.

## torch-ns

torch-ns builds its grounder through `grounder.api.rule_grounder.create_grounder(type, ...)` (PBC types, or `closure`
for `Forward.ground`) and reads each pool as a `RuleGroundings`.

## Tests

```bash
python -m pytest tests/ -q      # unit tests and a brute-force oracle (GPU)
python tests/gate.py            # < 90 s: grounding counts and hashes, forward chaining's closures, a speed ratchet
```

`docs/design.md` is the design reference.
