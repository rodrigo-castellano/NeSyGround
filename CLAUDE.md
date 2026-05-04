# CLAUDE.md

This file defines repository-wide guidance for the `grounder` repository.

## Scope

- Applies to this `grounder` repository.
- If this repository is checked out inside another project, also follow that parent repository's local instructions.
- Keep the same section order in future nested `CLAUDE.md` files so local docs stay predictable.

## Project Overview

NeSyGround is a compiled, fixed-shape grounding library for neuro-symbolic reasoning. It provides backward-chaining resolution, filtering, KB indexing, and optional neural/KGE hooks in a form that stays compatible with `torch.compile` and CUDA-graph-friendly execution.

## BC_{w,d,u} grounders: paper parametrization

The `enum` BC family is parametrized by **`(w, d, u)`** matching the paper / keras-ns notation:

- `w` — `max_unknown_fact_count` at intermediate proof steps.
- `d` — `num_steps` (proof depth).
- `u` — `max_unknown_fact_count_last_step` (last-depth cap). The paper convention is **`u=0`**: every leaf body atom must be a fact. The IJCAI '25 experiments use this everywhere.

Use `grounder.factory.make_bcwd(kb, w, d, u=0, ...)` (or the type-string shorthand `bcWD` / `bcWDuU`) to build a fully-configured BC grounder. Internal mapping: `u` → `BCGrounder.w_last_depth`.

### Default filter depends on `u`

When `filter` is omitted, both `BCGrounder.__init__` and `make_bcwd` derive it from `u`:

| `u` | default filter | matches keras-ns |
|-----|----------------|------------------|
| `u=0` (paper) | `fp_batch` | `prune_incomplete_proofs=True` (Kleene fixed-point pruning over the rule-application set) |
| `u>0` (rare)  | `none`     | `prune_incomplete_proofs=False` (admit unknown leaves; downstream scorer weights them) |

With `u=0` and the implied `fp_batch`, `out.rule_groundings` matches keras rule-by-rule. Note that under the paper convention `bc{0}{1}` and `bc{1}{1}` produce **identical output** — at depth 1 the only step *is* the last step, and `u=0` caps unknown leaves to 0 regardless of `w`.

### Other forced defaults

- `all_anchors=True` is forced for `enum` in `BCGrounder.__init__` even if the caller passes `False`. Anchoring only on the first body atom misses bindings keras finds when iterating each body position as anchor — for `nb(X,Y), loc(Y,Z) → loc(X,Z)`, anchoring on `loc` admits Y values where `nb(X,Y)` is unknown but `loc(Y,Z)` is fact (and vice versa). The dedup pipeline uses `_variant_to_orig` so the K_r anchor variants of the same logical rule application collapse to a single entry.
- `flat_intermediate=True` is the `make_bcwd` default — zero grounding loss when V≥2; falls through to the dense path for V<2.

### Paper rule sets

Some datasets ship two rule files. The paper / IJCAI '25 numbers use the smaller, hand-curated set:

| dataset | paper rules | extended set | notes |
|---|---|---|---|
| `family` | `rules_old.txt` (47 rules) | `rules.txt` (143 rules) | Use `KGDataset(..., rules_file='rules_old.txt')` to reproduce paper grounding counts. The 143-rule set is an automated expansion that blows up `K_r` and OOMs on large query batches with `cartesian_product=True`. |

### Known parity quirk: keras 1-body shortcut

For 1-body rules (e.g. symmetry `also_see(y,x) → also_see(x,y)`), keras-ns's `approximate_backward_chaining_grounding_one_rule` takes a shortcut that bypasses the `max_unknown_fact_count` cap entirely (line ~70: `if len(rule.body) == 1: new_ground_atoms.add(...); continue`). It then relies on `PruneIncompleteProofs` to drop apps whose body atoms aren't proved by other rules.

Torch's enum applies the width filter uniformly to 1-body rules, so when `u=0` and the body atom isn't a fact, torch drops the app. Keras's prune may resurrect these via mutual chains (other rules independently deriving the head, then the symmetric body atom is "proved" via that chain).

This causes an over-count in keras vs torch on datasets with 1-body symmetry rules and reciprocal test pairs (`also_see(A,B)` and `also_see(B,A)` both queried). The gap **amplifies through depths** because `PruneIncompleteProofs` admits more atoms via the shortcut at each iteration. On wn18rr 50 queries: bc01 +2, bc12 +100 (entirely localised to r2 = `hypernym ← der_form(x,z), der_form(z,y)` whose body atoms lean on the der_form symmetry), bc13 +60. Per the paper convention `u=0`, torch is the strict / correct behaviour; keras is the lenient one.

## V≥1 flat resolution path

`_resolve_enum_step_flat` is enabled for `V >= 1` (was V≥2 before 2026-04-30). Two coupled fixes were needed to make V=1 datasets correct on this path:

1. `_PatternVariant` propagates `_orig_body_patterns` / `_orig_body_pred_indices` from the base `RulePattern`. Without this, anchor variants carry *anchor-permuted* body order in `arg_source_dep` / `body_preds_dep`, so logically-equivalent apps don't share a key under the terminal `(orig_rule_idx, head, sorted_body)` dedup.
2. `_enumerate_cartesian_flat` applies `active_mask` to all rule slots (not just the slot-0 carve-out for `~has_free` rules). Padded K_r positions (rule_idx clamped to 0 because the predicate has fewer than K_r matching variants) used to leak candidates that got recorded as spurious apps under wrong heads.

Effect on the parity sweep (50 queries, GPU, default config):

| Dataset | bc01 | bc12 | bc13 | bc12 speedup | bc13 speedup |
|---|---|---|---|---|---|
| ablation_d2 | ✓ 0 | ✓ 118 | ✓ 377 | 0.30× | 0.33× |
| ablation_d3 | ✓ 0 | ✓ 0 | ✓ 252 | 0.14× | 0.27× |
| countries_s2 | ✓ 67 | ✓ 119 | ✓ 165 | 0.28× | 0.36× |
| countries_s3 | ✓ 42 | ✓ 783 | ✓ 3349 | **10.79×** | **2.73×** |
| family rules_old | ✓ 195 | ✓ 610 | ✓ 787 | 1.16× | **2.24×** |
| wn18rr | -2 (1-body quirk) | -100 (1-body quirk amplified) | -60 (1-body quirk amplified) | 0.60× | 0.66× |

16/18 cells full parity. The 3 wn18rr cells trace entirely to the 1-body keras shortcut amplification documented above. Small-workload cells (ablation, countries_s2) are slower than keras because the flat path runs eager (the compiled step-fast-path is gated off when `flat_intermediate=True`); for medium+ workloads (countries_s3, family bc13) torch beats keras-ns substantially.

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

Layout:

```
tests/
├── unit/                            # Pure-Python / CPU pytest suite (~3.5s)
│   ├── test_primitives.py             unify, apply_substitutions
│   ├── test_packing.py                pack_states, compact_atoms
│   ├── test_fact_index.py             ArgKey / Inverted / BlockSparse
│   ├── test_filters.py                fp_batch, fp_global
│   ├── test_grounder.py               BCGrounder + SLD end-to-end (toy KBs)
│   ├── test_rtf.py                    BCGrounder + RTF smoke
│   ├── test_datasets.py               real-KB integration (family/wn18rr/fb15k237)
│   └── test_grounding_baseline.py     per-query exact-count regression vs JSON baseline
├── _runners.py                      # Shared grounder construction (build_torch_grounder, build_keras_grounder, DEFAULT_COMPILE_MODES)
├── profile_speed.py                 # Generic timing helpers (time_runner, time_grounder)
├── test_groundings.py               # Cross-system count sweep — 4 tables (considered/ground_rules/ground_proofs/timing) over (datasets × grounders)
├── test_speed.py                    # Cross-system speed sweep — wall-clock per cell with per-grounder compile_mode
├── precommit.py                     # Runs test_groundings + test_speed on wn18rr × {keras-BC:w1d3, SLD:d4, enum-flat:w1d3, enum-dense:w1d3, FC:fp_global}
└── baselines/
    ├── comparison.json                count baseline (test_groundings.py)
    ├── speed.json                     timing baseline (test_speed.py)
    └── groundings.json                per-query baseline (unit/test_grounding_baseline.py)
```

Quick runs:

```bash
cd /path/to/grounder
# Unit tests only (CPU, fast):
PYTHONUNBUFFERED=1 python -m pytest tests/unit/ -v
# Per-query regression:
PYTHONUNBUFFERED=1 python -m pytest tests/unit/test_grounding_baseline.py -v
# Combined precommit (counts + speed on wn18rr × 5 grounders):
python tests/precommit.py
# Full count sweep (all datasets × all configs):
python tests/test_groundings.py
# Full speed sweep (all datasets × all configs, with per-grounder compile_mode):
python tests/test_speed.py
```

Rules:

- Run timing-sensitive suites sequentially.
- `precommit.py` is the gate before commits — it surfaces both correctness and wall-clock regressions on the smallest grid that exercises every grounder family.
- Run `tests/unit/test_grounding_baseline.py` when grounding counts, resolution, filters, or dataset loading may have changed.
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

- any code change: `python -m pytest tests/unit/ -v` (the unit suite is GPU-light; integration vs keras lives in `tests/test_groundings.py` / `tests/test_speed.py`).
- grounding semantics or counts changed: `python -m pytest tests/unit/test_grounding_baseline.py -v`
- before commit: `python tests/precommit.py` (counts + speed on the precommit grid).
- mirrored change intended: sync the other grounder copy or checkout and rerun its relevant tests
- before any commit: the `check-editable-pins` pre-commit hook runs automatically (if installed) and blocks the commit if the `torch-kge-kernels` SHA pin in `pyproject.toml` has drifted from the editable install or points at an unpushed HEAD. To run it manually: `python scripts/check_editable_pins.py`.
