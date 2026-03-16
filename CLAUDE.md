# CLAUDE.md — NeSyGround (grounder)

This file provides guidance to Claude Code when working with the grounder library.

## Project Overview

NeSyGround is a compiled CUDA-graph-safe FOL grounding library for neuro-symbolic reasoning. It provides backward-chaining Prolog unification with fixed-shape tensors and `torch.compile` compatibility.

## Environment

```bash
conda activate gpu
```

## Testing — Verification After Every Change

**Every change to the grounder lib requires ALL THREE levels of testing.**

**Run all tests sequentially (never in parallel)** — speed regression tests compare against baselines, so concurrent GPU load would cause false failures.

### 1. Grounder own tests (run first)

From the grounder directory (`torch-ns/grounder/`):

```bash
cd /home/castellanoontiv/probfol-llm-swarm/main/submodules/torch-ns/grounder
PYTHONUNBUFFERED=1 python -m pytest tests/ -v
```

### 2. torch-ns precommit tests (probfol-llm-swarm)

From the torch-ns root:

```bash
cd /home/castellanoontiv/probfol-llm-swarm/main/submodules/torch-ns
PYTHONUNBUFFERED=1 python -u tests/run.py precommit
```

This runs 3 regression tests: speed, groundings, MRR.

### 3. DpRL-KGR parity tests

**Before running**: sync the grounder lib to the DpRL repo (see Syncing below).

```bash
cd /home/castellanoontiv/DpRL-KGR-swarm/main
PYTHONUNBUFFERED=1 python -m pytest kge_experiments/tests/parity/test_run.py kge_experiments/tests/parity/test_speed_profile.py -v
```

## Syncing the Grounder Between Repos

The grounder lib is a submodule in two repos:
- **probfol-llm-swarm**: `main/submodules/torch-ns/grounder/` (origin: grounders_KG.git)
- **DpRL-KGR-swarm**: `main/grounder/` (origin: NeSyGround.git)

After making changes in one, **copy the changed files** to the other before running its tests:

```bash
# Example: sync from probfol-llm-swarm → DpRL-KGR-swarm
rsync -av --exclude='__pycache__' --exclude='.git' \
  /home/castellanoontiv/probfol-llm-swarm/main/submodules/torch-ns/grounder/ \
  /home/castellanoontiv/DpRL-KGR-swarm/main/grounder/
```

## Package Structure

```
grounder/
├── __init__.py           # Public exports
├── base.py               # Grounder base class (nn.Module)
├── bc/                   # Backward chaining core
├── fc/                   # Forward chaining
├── resolution/           # Unification / resolution
├── filters/              # Grounding filters
├── nesy/                 # Neuro-symbolic scoring interface
├── data_loader.py        # Dataset loading
├── fact_index.py         # Fact indexing strategies
├── rule_index.py         # Rule indexing
├── factory.py            # Grounder factory
├── primitives.py         # Core tensor primitives
├── types.py              # Type definitions
├── tests/                # Unit tests
│   ├── test_grounder.py
│   ├── test_primitives.py
│   ├── test_fact_index.py
│   ├── test_packing.py
│   ├── test_filters.py
│   ├── test_rtf.py
│   └── test_datasets.py
└── docs/                 # Documentation
```

## Coding Standards

- **Static tensor shapes**: All tensors must have fixed shapes for CUDA graph capture
- **No `.item()` in forward**: Breaks `torch.compile(fullgraph=True)`
- **No dynamic control flow**: No Python data-dependent branching in compiled paths
- **Compile per step, not per depth**: `torch.compile` must wrap the single-step function (one depth iteration), not the full multi-depth loop. The outer depth loop stays in plain Python. This avoids CUDA graph explosion at large depths — compiling the full unrolled loop creates a separate graph per depth, causing recompilation and OOM. One compiled step reused D times is both faster and memory-safe.
- **Tensor shape annotations**: Document shapes as `[B, S, G, 3]` with comments
- **Type hints**: All function signatures must include type hints
