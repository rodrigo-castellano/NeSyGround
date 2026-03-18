# Example: Proving `locatedInCR(france, europe)` — Prolog Strategy

**Goal**: prove that France is in Europe — a fact **not stored** in the knowledge graph.

**Known facts**: `neighborOf(france, italy)`, `locatedInCR(italy, europe)`, ...
**Rules**: `R1: neighborOf(X, Y), locatedInCR(Y, Z) → locatedInCR(X, Z)`

At each step the engine resolves atom[0] against all facts + rules, producing derived states. The agent picks one.

---

## Step 1

```
  CURRENT STATE                        DERIVED STATES (agent chooses one)
 ┌────────────────────────────────┐
 │ locatedInCR(france, europe)    │──→  [0] R1 → { neighborOf(france, Y), locatedInCR(Y, europe) }   ← agent picks ★
 └────────────────────────────────┘     [1] R2 → { neighborOf(france, Y), neighborOf(Y, K), locCR(K, europe) }
                                        [2] end-proof (give up)
```

Unification: goal matches R1 head `locatedInCR(X, Z)` with `{X→france, Z→europe}`, producing the rule body as new open goals.

## Step 2

```
  CURRENT STATE                                 DERIVED STATES (agent chooses one)
 ┌──────────────────────────────────────────┐
 │ neighborOf(france, Y), locatedInCR(Y, eu)│──→  [0] Y→andorra → { locatedInCR(andorra, europe) }    not in KB
 └──────────────────────────────────────────┘     [1] Y→belgium → { locatedInCR(belgium, europe) }    not in KB
        ↑                                         [2] Y→germany → { locatedInCR(germany, europe) }    not in KB
   resolve atom[0] against facts                  [3] Y→italy   → { ∅ }  PROOF DONE ✓                ← agent picks ★
                                                  [4] Y→spain   → { locatedInCR(spain, europe) }      not in KB
                                                  ...
```

Unification: atom[0] `neighborOf(france, Y)` matches 8 facts. Each binding propagates to atom[1]:
- Y→italy makes atom[1] = `locatedInCR(italy, europe)` → **in KB → auto-pruned** → no goals left → done!

## Result: reward +1

```
  locatedInCR(france, europe)
    ├── neighborOf(france, italy)    ✓ fact
    └── locatedInCR(italy, europe)   ✓ fact
```

**The unification engine computes the options; the RL agent learns to pick the right ones.**
