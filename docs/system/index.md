# NeSyGround

**A compiled, CUDA-graph-safe FOL grounding library for neuro-symbolic reasoning.**

NeSyGround turns symbolic first-order logic grounding into batched tensor operations on GPU. It takes a knowledge base of facts and Horn clause rules, and produces ground rule instantiations as fixed-shape tensors — ready for consumption by any differentiable reasoning system.

## What NeSyGround is

- A **grounding library**: it computes which rule instantiations are relevant for a set of queries or a knowledge base
- **Compiled for GPU**: all operations run as static-shape CUDA graph nodes via `torch.compile(fullgraph=True)`
- **Paradigm-agnostic**: supports both backward chaining (query-driven) and forward chaining (data-driven) grounding
- **Modular**: grounders are `nn.Module` subclasses that compose, extend, and interoperate cleanly

## What NeSyGround is NOT

- **Not a model**: it produces groundings, not predictions. Scoring and learning happen downstream
- **Not a reasoner**: it does not compute truth values, probabilities, or proofs
- **Not a training framework**: it has no loss functions, optimizers, or training loops

## Expressiveness boundary

NeSyGround operates over **Horn clauses with binary predicates** (triples). This means:

- All atoms are of the form `p(X, Y)` where `p` is a predicate and `X`, `Y` are constants or variables
- Rules have exactly one head atom and one or more body atoms
- No negation, no function symbols, no n-ary predicates

This is a deliberate design choice — it is sufficient for knowledge graph completion and enables full tensorization with fixed shapes.

## Consumers

NeSyGround is designed to feed into:

- **Differentiable reasoners**: SBR, R2N, DCR, and similar systems that score groundings with t-norms
- **RL agents**: via the `step()` API for single-step grounding expansion
- **Proof extractors**: systems that need to enumerate valid proof paths

## Documentation

| File | Description |
|------|-------------|
| [foundations.md](foundations.md) | FOL definitions, grounding signatures, tensor mapping |
| [architecture.md](architecture.md) | System diagram, BC and FC pipelines, composition patterns |
| [api.md](api.md) | Core types, interfaces, full function specs with args/types/shapes |
| [grounders.md](grounders.md) | Every grounder variant: algorithm, constructor, methods, trade-offs |
| [tensors.md](tensors.md) | Dimension conventions, masks, padding, hashing, CUDA constraints |
| [soundness.md](soundness.md) | Formal soundness and completeness properties per grounder |
| [walkthrough.md](walkthrough.md) | Full tensor trace example with a family/kinship knowledge base |
| [extension.md](extension.md) | Step-by-step guide to adding a new grounder |
| [related.md](related.md) | Comparison with other NeSy grounding systems |
