# CLAUDE.md

This file defines repository-wide guidance for the `grounder` repository.

## Scope

- Applies to this `grounder` repository.
- If this repository is checked out inside another project, also follow that parent repository's local instructions.
- Keep the same section order in future nested `CLAUDE.md` files so local docs stay predictable.

## Project Overview

NeSyGround is a compiled, fixed-shape grounding library for neuro-symbolic reasoning. It provides backward-chaining resolution, filtering, KB indexing, and optional neural/KGE hooks in a form that stays compatible with `torch.compile` and CUDA-graph-friendly execution.

## Architecture

Current package ownership:

- `grounder/data/`: dataset loading, KB construction, fact/rule indexing
- `grounder/bc/`: backward-chaining execution
- `grounder/fc/`: forward-chaining execution
- `grounder/resolution/`: unification primitives, SLD, RTF, enumeration, standardization
- `grounder/filters/`: search and soundness filters plus hooks
- `grounder/nesy/`: neural/KGE scoring helpers and hooks
- `grounder/factory.py`: grounder construction entry point
- `grounder/types.py`, `grounder/utils.py`: shared types and utilities
- `grounder/analysis/`: comparison, gold-standard, and depth-generation scripts
- `grounder/tests/`: unit and regression tests
- `grounder/docs/`: package documentation

## Running Experiments

This repository is primarily a library, not a training entry point.

Use it in three ways:

- run focused grounder tests from this directory
- run grounder analysis scripts such as:

```bash
cd /path/to/grounder
python -m grounder.analysis.compare_groundings --help
```

Do not add standalone training scripts to `grounder/` unless they are genuinely grounder-specific analysis tools.

## Logging Experiments

- Standalone runtime outputs live under the repo root `output/`.
- `output/runs/<experiment_name>/<run_name>/` is the canonical run bundle for analysis scripts.
- Each run stores `manifest.json`, `config.json`, `stdout.log`, `events.jsonl`, `metrics.json`, and optional `artifacts/`.
- `config.json` and `metrics.json` are analysis-script-defined; the shared logger only fixes the bundle layout.
- `report.md` is optional and is only written when an agent or human explicitly requests it.
- `output/registry/<experiment_name>/<run_name>/` is a manually promoted copy of the same run bundle.
- `output/legacy/` is reserved for migrated historical artifacts only.
- Keep analysis outputs out of importable library modules and out of curated docs directories by default.

## Testing

Local grounder tests:

```bash
cd /path/to/grounder
PYTHONUNBUFFERED=1 python -m pytest tests/ -v
PYTHONUNBUFFERED=1 python -m pytest tests/test_groundings.py -v
```

Rules:

- Run timing-sensitive suites sequentially.
- Run `tests/test_groundings.py` when grounding counts, resolution, filters, or dataset loading may have changed.
- If this repository is mirrored into another checkout, sync the changed files there and rerun the relevant integration tests in that mirror as needed.

## Documentation

- Update `grounder/README.md` for public API or usage changes.
- Update the closest relevant doc in `grounder/docs/` when tensor flow, filters, indexing, or grounding semantics change.
- If a new analysis script or output convention is introduced, document where its artifacts belong.
- Keep mirrored copies of the grounder docs conceptually aligned when the code is meant to stay shared across repositories.

## Adding or Changing Code

- Each module should own one clear responsibility.
- Before creating a new file, extend the module that already owns the behavior.
- Create a new file only when no current module owns that functionality, or when extending the current one would mix unrelated responsibilities.
- Do not create parallel implementations of the same resolution, filter, or indexing logic unless there is a clear algorithmic distinction.

Modification discipline:

- Prefer modifying one existing owner module over spreading one feature across many files.
- Do not split one resolution, filter, or indexing responsibility across multiple scripts without a strong architectural reason.
- Do not create files named `*_new`, `*_v2`, `*_copy`, `tmp_*`, or similar variants.
- If similar logic already exists in multiple places, consolidate it instead of adding another copy.
- Shared logic should live in one reusable module; callers should import it rather than duplicate it.
- New files require a clear reason: missing responsibility, clean extraction of a coherent unit, or reuse by multiple callers.
- If a new file is created by extraction, remove the superseded duplicated logic from the old location.

Placement rules:

- data loading, KB wiring, indexing: `grounder/data/`
- resolution, substitutions, standardization, search expansion: `grounder/resolution/`
- backward/forward execution loops: `grounder/bc/`, `grounder/fc/`
- filter logic and hooks: `grounder/filters/`
- neural or KGE-assisted scoring: `grounder/nesy/`
- one-off comparison and reporting scripts: `grounder/analysis/`

## Naming Convention

Standard symbols for tensor dimensions and layout parameters. Use these
consistently in code, comments, and documentation.

| Symbol | Description | Formula / Source |
|--------|------------|-----------------|
| `D` | depth (proof steps) | user param |
| `W` | width (unknown tolerance) | user param |
| `B` | batch size | user param |
| `N` | flattened queries | B * S |
| `G` | goals per state | M + (M-1)*D |
| `M` | body atoms per rule | from KB |
| `A` | accumulated body capacity | D * M |
| `S` | states per step | 256 default |
| `C` | collected groundings budget | user param |
| `K` | children per state | SLD: K_f+K_r, RTF: K_f*K_r, Enum: min(K_r*G_r, K_max) |
| `K_f` | fact children (SLD/RTF) | from fact index |
| `K_r` | rules per predicate | from rule index |
| `G_r` | groundings per rule (enum) | user param |
| `K_v` | candidates per free var (enum) | min(K_f, G_r) |
| `V` | free vars per rule (enum) | from rules |
| `K_max` | children cap | 550 default |
| `pad` | padding index | from KB |

Public API aliases (for backward compatibility with experiments/model.py):
- `effective_total_G` = `C`
- `max_body_capacity` = `A`

## Coding Standards

- Keep tensors statically shaped wherever code is intended for compiled execution.
- Avoid `.item()` and Python data-dependent branching inside compiled forward/step paths.
- Compile a single step, not an entire multi-depth loop.
- Add type hints to function signatures.
- Document important tensor shapes with comments using the standard symbols above (e.g. `[B, S, G, 3]`).
- Prefer vectorized tensor code over Python loops in hot paths.
- Keep comments concise and focused on non-obvious behavior.

## Technical Rules

- Never revert or restore files without explicit user permission.
- Fix bugs forward; do not hide them with clamps or silent fallbacks.
- If a path is meant to run in `torch.compile` / CUDA-graph-friendly mode, solve the root issue there instead of silently switching to a slower dynamic path.
- Keep timing-sensitive tests and benchmarks sequential.
- Use a git worktree when you need to compare with an older commit.
- Keep mirrored grounder copies synchronized when the intent is shared behavior across repos.
- Do not leave scratch artifacts inside package directories.
- Prefer the smallest coherent change that keeps one owner per responsibility; avoid scattering one feature across multiple modules.
- torch-kge-kernels is a sibling repo at `~/repos/torch-kge-kernels-swarm/main/`, installed as pip-editable. Edit it there, commit there, push there. The SHA pin in this repo's `pyproject.toml` must be bumped whenever the editable HEAD moves — the pre-commit hook (`scripts/check_editable_pins.py`, wired via `.pre-commit-config.yaml`) refuses commits when the pin and the editable HEAD disagree or when the editable HEAD is unpushed. Setup once with `conda activate gpu && pre-commit install`. Bypass only with `SKIP=check-editable-pins git commit ...` for genuinely unrelated commits during an in-flight cascade.

## Verification Checklist

- any code change: `python -m pytest tests/ -v` (skip the `test_keras_*` files unless GPU + tensorflow are available)
- grounding semantics or counts changed: `python -m pytest tests/test_groundings.py -v`
- mirrored change intended: sync the other grounder copy or checkout and rerun its relevant tests
- before any commit: the `check-editable-pins` pre-commit hook runs automatically (if installed) and blocks the commit if the `torch-kge-kernels` SHA pin in `pyproject.toml` has drifted from the editable install or points at an unpushed HEAD. To run it manually: `python scripts/check_editable_pins.py`.
