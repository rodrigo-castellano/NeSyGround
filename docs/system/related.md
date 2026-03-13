# Related Systems

How NeSyGround compares to other neuro-symbolic reasoning systems. NeSyGround is a **grounder**, not a model — it produces ground rule instantiations, not predictions.

---

## Comparison Table

| System | Grounding method | Compiled? | GPU? | Batched? | RL step()? | Grounder separate from model? |
|--------|-----------------|-----------|------|----------|-----------|-------------------------------|
| ProbLog | Symbolic SLD resolution | No | No | No | No | No (integrated) |
| DeepProbLog | Differentiable SLD | No | Partial | No | No | No (integrated) |
| NTP | Attention-based unification | No | Yes | Yes | No | No (end-to-end) |
| CTP | Conditional theorem proving | No | Yes | Yes | No | No (end-to-end) |
| pLogicNet | EM over Markov Logic | No | No | No | No | Partially |
| DRUM | End-to-end differentiable rules | N/A | Yes | Yes | No | No (no explicit grounding) |
| NeuralLP | Differentiable rule learning | N/A | Yes | Yes | No | No (no explicit grounding) |
| LTN | Full ground truth tensor | N/A | Yes | Yes | No | No (pre-computed) |
| **NeSyGround** | Compiled tensor BC/FC | **Yes** | **Yes** (CUDA graphs) | **Yes** | **Yes** | **Yes** |

---

## Key Differentiators

### 1. Static-shape CUDA graph operations

NeSyGround is the only system where all tensor shapes are determined at construction time. This enables `torch.compile(fullgraph=True, mode='reduce-overhead')` with CUDA graph capture, eliminating kernel launch overhead and enabling maximum GPU utilization.

Other systems either use dynamic shapes (ProbLog, DeepProbLog, NTP) or don't compile to CUDA graphs (CTP, pLogicNet).

### 2. BC + FC composition

NeSyGround cleanly composes backward chaining with forward chaining. The FC provable set pre-computation feeds into BC pruning/filtering, combining query-directed search (BC) with data-driven completeness checking (FC).

Systems like ProbLog and DeepProbLog use only BC. Systems like DRUM and NeuralLP use only FC-like mechanisms. None compose both in a modular way.

### 3. step() API for RL

The `step()` method exposes a single resolution step as a first-class operation. RL agents can call `step()` once per decision, making the grounder an environment rather than a black box.

No other system provides this interface.

### 4. Grounder separated from scoring

NeSyGround produces structural groundings (which rules apply to which entities). Scoring — how to combine body atom truth values into a head atom truth value — is entirely downstream.

This separation means the same grounder can feed into:
- Product t-norm scoring (SBR)
- R2N attention-based scoring
- DCR differentiable counting
- MLN weight learning
- Any future scoring mechanism

Most other systems bundle grounding and scoring together, making it impossible to swap one without the other.

---

## System-by-System Comparison

### ProbLog / DeepProbLog

ProbLog performs symbolic SLD resolution on CPU with Prolog-style backtracking. DeepProbLog extends this with neural predicates whose truth values come from neural networks.

**vs NeSyGround**: ProbLog's grounding is symbolic and sequential — no batching, no GPU. DeepProbLog adds neural predicates but still uses Prolog's SLD engine. NeSyGround tensorizes the same SLD algorithm for GPU execution with batched queries.

### Neural Theorem Provers (NTP)

NTP replaces symbolic unification with differentiable attention over embeddings. Every unification step computes soft similarity between query and KB atoms.

**vs NeSyGround**: NTP makes unification differentiable at the cost of making it approximate. NeSyGround keeps unification exact (MGU-based) and leaves differentiability to the downstream scorer. NTP cannot guarantee soundness; NeSyGround can.

### Conditional Theorem Provers (CTP)

CTP extends NTP with conditional computation — only expands proof branches that pass a learned gate.

**vs NeSyGround**: CTP's gating is analogous to NeSyGround's scored grounders (KGEGrounder, NeuralGrounder), but CTP integrates gating into the proof search itself. NeSyGround separates search (grounder) from selection (scored wrapper), making each independently configurable.

### pLogicNet

pLogicNet uses Markov Logic Networks with EM: E-step grounds rules, M-step learns weights.

**vs NeSyGround**: pLogicNet's grounding is rule-level enumeration similar to ParametrizedBCGrounder. But it's implemented on CPU without batching. NeSyGround's ParametrizedBCGrounder provides the same grounding semantics on GPU with batched execution.

### DRUM / NeuralLP

DRUM and NeuralLP learn rule weights end-to-end without explicit rule grounding. They implicitly enumerate all length-D predicate chains via matrix multiplication.

**vs NeSyGround**: These systems don't perform explicit grounding — they learn soft rules via attention over predicate sequences. NeSyGround takes explicit rules as input and produces explicit groundings. The approaches are complementary: DRUM/NeuralLP discover rules; NeSyGround grounds given rules efficiently.

### Logic Tensor Networks (LTN)

LTN pre-computes the full grounding tensor: for each rule, enumerate all entity combinations for all variables. This produces an exact but potentially enormous tensor.

**vs NeSyGround**: LTN's approach corresponds to NeSyGround's FullBCGrounder, which enumerates all `E` entities for each free variable. NeSyGround's ParametrizedBCGrounder provides scalable alternatives (fact-anchoring, width bounding, provability pruning) that produce much smaller grounding sets while maintaining soundness.

---

## Positioning

NeSyGround occupies a specific niche:

```
                    Exact grounding ◄────────────► Approximate grounding
                         │                                  │
                    NeSyGround                          NTP, CTP
                    ProbLog                             DRUM
                    pLogicNet                           NeuralLP
                         │                                  │
                    Explicit rules ◄─────────────► Learned rules
```

NeSyGround is for systems that have **explicit rules** and need **exact, sound groundings** at GPU speed. If you need to learn rules from data, use DRUM/NeuralLP. If you need approximate differentiable unification, use NTP/CTP. If you have rules and need to ground them efficiently on GPU, use NeSyGround.
