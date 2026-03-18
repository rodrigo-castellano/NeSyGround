# Related Work: Forward/Backward Chaining and Optimized Variants

This document surveys the external literature relevant to the tensorized grounding algorithms in torch-ns. It is a companion to `Recursive_query_processing.md` (Bancilhon & Ramakrishnan 1986), which covers the foundational taxonomy of evaluation strategies.

---

## 1. Foundations

### 1.1 Datalog and Horn Clause Evaluation

The standard reference for bottom-up evaluation is **Bancilhon & Ramakrishnan (1986)** (`Recursive_query_processing.md`). They establish the three performance axes — duplication, potentially-relevant-facts, and arity of intermediate relations — and rank naive, semi-naive, QSQ, Magic Sets, Counting, and Henschen-Naqvi against four benchmark queries. Key takeaway: for restricted linear rules, Counting and Henschen-Naqvi are orders of magnitude better than semi-naive; for non-linear rules, QSQR (recursive memoized top-down) is uniquely good.

**Brass & Stephan (2015)** extend this with a push-based perspective: rather than pulling new tuples into a fixpoint loop, facts are *pushed* to all rules that consume them, enabling event-driven rather than batch evaluation. This maps naturally to sparse or streamed fact sets.

> S. Brass and H.-J. Stephan. *Bottom-Up Evaluation of Datalog: Preliminary Report.* WLP 2015.
> [[PDF]](https://users.informatik.uni-halle.de/~brass/push/publ/wlp15.pdf)

A standard pedagogical treatment of naive and semi-naive evaluation (including worked examples with delta relations) is available in:

> UW-Madison CS838 Lecture Notes: *Datalog: Evaluation.*
> [[PDF]](https://pages.cs.wisc.edu/~paris/cs838-s16/lecture-notes/lecture8.pdf)

---

## 2. Optimized Bottom-Up Evaluation

### 2.1 Semi-Naive as the Standard

Semi-naive evaluation avoids recomputing tuples derived in previous iterations by maintaining a *delta* relation `Δp` of newly derived facts per round. The main loop body becomes:

```
new_p := φ(Δp₁, p₂, ...) ∪ φ(p₁, Δp₂, ...) ∪ ...   -- at-least-one-delta term
Δp  := new_p \ p
p   := p ∪ Δp
```

For linear rules this simplifies to replacing `p` with `Δp` in the body. For non-linear rules each body position must be taken as delta in turn (as detailed in Bancilhon & Ramakrishnan §3.2.2). This is exactly the *incremental SpMM* strategy described in `FOL_grounders/forward_chaining.md`.

The higher-order generalization of semi-naive to the **Datafun** language (first-class relations, set comprehensions) is treated in:

> M. Arntzenius and N. Krishnaswami. *Seminaïve Evaluation for a Higher-Order Functional Language.* POPL 2020.
> [[PDF]](https://www.cl.cam.ac.uk/~nk480/seminaive-datafun.pdf)

This paper shows the semi-naive transformation can be derived systematically as a *derivative* of the fixpoint operator — the same idea underlying the delta-relation rewrite.

### 2.2 Compilation to Imperative Code

The dominant practical approach since Soufflé (2016) is to compile Datalog to C++ implementing a specialised semi-naive loop with optimised data structures (B-trees, tries). Recent work extends this:

- **Ascent (2022)**: compile Datalog to Rust via macros, embedding semi-naive directly in the host language.
- **Making Formulog Fast (2024)**: argues that unconventional evaluation orders — interleaving bottom-up derivation with functional computation — outperform compiled semi-naive for SMT-enriched Datalog.

> T. Gilray et al. *Making Formulog Fast: An Argument for Unconventional Datalog Evaluation.* OOPSLA 2024.
> [[arXiv]](https://arxiv.org/html/2408.14017v1)

### 2.3 Distributed and Streaming Evaluation

**Nexus (2022)** evaluates Datalog with semi-naive using incremental and asynchronous iteration over distributed partitions:

> Y. Gao et al. *Fast Datalog Evaluation for Batch and Stream Graph Processing.* WWW Journal 2022.
> [[Springer]](https://link.springer.com/article/10.1007/s11280-021-00960-w)

Key insight: asynchronous iteration (workers do not wait for a global barrier) reduces wall-clock time when fact propagation is uneven across partitions — relevant to knowledge graphs where some entities have high fan-out.

### 2.4 GPU-Based Evaluation

The most directly relevant modern work for torch-ns:

> T. Sahoo et al. *Modern Datalog on the GPU.* arXiv 2023.
> [[arXiv]](https://arxiv.org/html/2311.02206v3)

This paper implements Datalog fixpoint evaluation on GPU using:
- **Relation storage**: sorted arrays on device memory, supporting set-difference and union via merge operations
- **Join evaluation**: hash join on GPU with radix partitioning
- **Semi-naive loop**: delta arrays updated in-place per iteration

They achieve 10–100× speedup over CPU Soufflé on graph-structured Datalog. The tradeoff versus torch-ns: their approach materialises full intermediate relations (exact), while torch-ns operates per-batch with bounded groundings (approximate but on-the-fly). Their memory model is `O(|R|)` per relation; torch-ns is `O(batch × max_groundings)`.

The large-scale semi-naive variant on Hadoop (MapReduce-style) is:

> A. Shkapsky et al. *Optimizing Large-Scale Semi-Naïve Datalog Evaluation in Hadoop.* RR 2012.
> [[Springer]](https://link.springer.com/chapter/10.1007/978-3-642-32925-8_17)

---

## 3. Top-Down Optimizations: Magic Sets and SLD

### 3.1 Magic Sets

Magic Sets (Bancilhon et al. 1986) rewrite a Datalog program so that bottom-up evaluation only derives tuples reachable from the query constants. The transformation introduces *magic* predicates that propagate binding patterns downward through rule bodies, simulating Prolog's top-down binding propagation within a bottom-up engine.

As shown in `Recursive_query_processing.md` §3.3.3, Magic Sets is equivalent to QSQR in the linear case but degenerates to semi-naive for non-linear rules (it cannot determine the relevant subgraph). Magic Sets is therefore optimal for the 2-body linear Horn rules that constitute the majority of KGC rules.

### 3.2 Magic Sets vs. SLD-Resolution

The key algorithmic question for torch-ns: when should grounding use bottom-up (Magic Sets / semi-naive) vs. top-down (SLD / backward chaining)?

> A. Bry. *Magic Sets vs. SLD-Resolution.* 1996.
> [[ResearchGate]](https://www.researchgate.net/publication/2616842_Magic_Sets_vs_SLD-resolution)

Main result: SLD-resolution is significantly more efficient than Magic Sets on **tail-recursive programs** (rules of the form `p(X,Y) :- base(X,Z), p(Z,Y)`). Magic Sets must still compute the full magic set before beginning the bottom-up loop; SLD directly follows the chain. For KGC with path-following rules, this favours backward chaining (torch-ns `FullGraphGrounder`) over materialised provable sets.

### 3.3 SLDMagic: Combining Both

> W. Lüttringhaus-Kobs. *SLDMagic — An Improved Magic Set Technique.* 1996.
> [[ResearchGate]](https://www.researchgate.net/publication/2827190_SLDMagic_---_An_Improved_Magic_Set_Technique)

SLDMagic derives a new transformation based on partial evaluation of a bottom-up meta-interpreter for SLD. It gains tail-recursion optimisation (matching SLD on linear rules) while retaining bottom-up parallelism. In the torch-ns context, this corresponds to the hybrid strategy of using the precomputed provable set (bottom-up) as a filter for per-query backward chaining — which is exactly the `FullGraphGrounder` design in `grounding_systems.md`.

### 3.4 Tabling (SLG Resolution)

Tabling extends SLD with memoisation of sub-goal answers. When a sub-goal is encountered for the second time, tabling suspends the call and resumes with previously computed answers instead of re-deriving them. This solves the two main failures of pure Prolog: infinite loops on cyclic data and redundant recomputation.

> SWI-Prolog Tabling Documentation (SLG Resolution).
> [[Docs]](https://www.swi-prolog.org/pldoc/man?section=tabling)

Tabling corresponds to QSQR in the Bancilhon taxonomy. In the tensor setting, torch-ns implements an implicit form of tabling: the provable set `P` (computed once per training step) memoises all provable atoms, and the backward chaining grounder looks up `P` rather than recursing.

---

## 4. Logical Characterisation: Unifying FC and BC

> K. Chaudhuri and F. Pfenning. *A Logical Characterization of Forward and Backward Chaining in the Inverse Method.* IJCAR 2006.
> [[PDF]](https://www.cs.cmu.edu/~fp/papers/ijcar06.pdf)

This paper defines a **focusing bias** for atoms in sequent calculus. Atoms with a *positive bias* (produced) give rise to forward chaining (hyperresolution); atoms with a *negative bias* (consumed) give rise to backward chaining (SLD resolution). On the Horn fragment, the two coincide for ground atoms. The key insight: the choice of bias determines efficiency, not correctness — a pure forward chainer and a pure backward chainer compute the same answers, they just traverse different subsets of the proof space. This is the theoretical foundation for why torch-ns can use either `forward_chaining.md` (provable set precomputation) or `backward_chaining.md` (per-query BC) interchangeably.

---

## 5. Forward vs. Backward in Ontology-Based QA

> ForBackBench: *A Benchmark for Chasing vs. Query-Rewriting.* VLDB 2022.
> [[ACM DL]](https://dl.acm.org/doi/abs/10.14778/3529337.3529338)

In ontology-based data access (OBDA), the **chase** (forward chaining) materialises a complete Skolem instance of the data, while **query rewriting** (backward chaining) rewrites the query over the source schema. This paper benchmarks both approaches across existential rules (TGDs) and finds:
- Forward (chase) wins when queries are complex and data is small
- Backward (rewriting) wins when data is large and queries are simple

The same tradeoff appears in KGC: torch-ns forward-chains a provable set once per epoch (cheap when rules are few), then uses it to answer all queries (cheap when queries are many). For settings with very many rules but few queries, full per-query backward chaining (keras-ns style) would be preferable.

---

## 6. Neural Guidance of Backward Chaining

### 6.1 Learning Subgoal Ordering

> J. Durbin and J. Heflin. *Learning a More Efficient Backward-Chaining Reasoner.* ACS 2022.
> [[PDF]](https://par.nsf.gov/servlets/purl/10415017)

Trains a feed-forward network to predict which rule to apply and which subgoal to pursue first at each BC step. Embeddings for logical atoms are derived from unification structure. The paper reports large reductions in nodes expanded. Directly relevant to torch-ns if the grounder is extended to support rule scoring / priority queues.

A follow-up evaluation of multiple meta-reasoning strategies:

> J. Durbin and J. Heflin. *An Evaluation of Strategies to Train More Efficient Backward-Chaining Reasoners.* KCAP 2023.
> [[PDF]](https://www.cse.lehigh.edu/~heflin/pubs/KCap23_Meta_Reasoning.pdf)

### 6.2 LLM-Driven Backward Chaining

> H. Kazemi et al. *LAMBADA: Backward Chaining for Automated Reasoning in Language Models.* ACL 2023.
> [[PDF]](https://aclanthology.org/2023.acl-long.361.pdf)

Uses an LLM as the rule-matcher at each BC step. The BC structure prunes the search space and forces the model to commit to intermediate conclusions, improving multi-hop NLI accuracy. Unlike torch-ns, this is a symbolic-first approach with learned rule matching rather than end-to-end differentiable grounding.

> X. Ye et al. *SymBa: Symbolic Backward Chaining for Structured Natural Language Reasoning.* arXiv 2024.
> [[arXiv]](https://arxiv.org/abs/2402.12806)

Extends LAMBADA with structured proof trees and a verifier, improving faithfulness of the proof trace.

---

## 7. Summary Table

| Paper | Year | Method | Relevance to torch-ns |
|-------|------|--------|----------------------|
| Bancilhon & Ramakrishnan | 1986 | Taxonomy: naive/SN/QSQ/Magic/Counting | Foundational; `Recursive_query_processing.md` |
| Brass & Stephan | 2015 | Push-based bottom-up | Event-driven alternative to batch fixpoint |
| Arntzenius & Krishnaswami | 2020 | Semi-naive as fixpoint derivative | Formal basis of delta-relation rewrite |
| Modern Datalog on the GPU | 2023 | GPU semi-naive with exact materialisation | Complementary to torch-ns: exact but static |
| Gilray et al. (Formulog) | 2024 | Unconventional eval orders | SMT-enriched; out of scope for Horn KGC |
| Bry | 1996 | Magic Sets vs. SLD on tail-recursive rules | Justifies BC for path rules |
| Lüttringhaus-Kobs (SLDMagic) | 1996 | Partial eval of SLD meta-interpreter | Theoretical basis of provable-set + BC hybrid |
| SWI-Prolog Tabling | — | SLG resolution / memoised BC | torch-ns provable set = implicit tabling |
| Chaudhuri & Pfenning | 2006 | Focusing bias unifies FC and BC | Correctness equivalence of FC and BC |
| ForBackBench VLDB | 2022 | Chase vs. query-rewriting benchmark | FC vs. BC tradeoff in OBDA |
| Durbin & Heflin | 2022/2023 | Learned BC rule ordering | Neural guidance of grounder search |
| Kazemi et al. (LAMBADA) | 2023 | LLM-driven BC | Symbolic-first alternative to end-to-end |
| Ye et al. (SymBa) | 2024 | Structured BC with verifier | NLR application; not KGC |
