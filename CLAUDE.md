# CLAUDE.md

This file defines repository-wide guidance for the `grounder` repository.

## Scope

- Applies to this `grounder` repository.
- If this repository is checked out inside another project, also follow that parent repository's local instructions.
- Keep the same section order in future nested `CLAUDE.md` files so local docs stay predictable.

## Project Overview

grounder is a library of grounding techniques for neural-symbolic reasoning: given facts, rules and queries, it
produces the rule groundings a reasoner scores. One class per technique, one data model (`kb.py`), one output type
(`types.py`): `PBC` (parametrized backward chaining, BC_{w,d}), `SLD` (SLD resolution) and `Forward` (forward
chaining). No caps: every lookup is enumerated in full, and a call whose output does not fit runs out of memory (the
explosion is the point the RL papers make). `docs/design.md` is the design reference.

## BC_{w,d,u}: the papers' parametrization

- `w` — the width: at most `w` body atoms of a kept grounding are not facts.
- `d` — the depth: grounding steps (each grounds the previous step's unknown body atoms).
- `u` — whether unknown leaves may remain after the last step. The papers use **u = false**: what is not proved is
  pruned. `PBC.parse` types: `enum.keras.wW.dD.flat` (keras-ns's prune, the papers' protocol) and
  `enum.fp_batch.wW.dD.flat` (fp_batch); BC_{0,1,0} = BC_{1,1,0}.

### keras (the papers' protocol)

The IJCAI-25 code (`PhD-papers/ijcai25_grounding_methods/repro/src_paper`, `BuildGrounder` with `backward_W_D`) builds
keras-ns's `ApproximateBackwardChainingGrounder` with width `W` at **every** step, the last one included
(`max_unknown_fact_count_last_step=W`), `prune_incomplete_proofs=True` and no per-rule cap. `enum.keras.wW.dD` does
the same: width `W` at every step, then keras-ns's proof walk (`D - 1` rounds; `.r<N>` sets them: the later keras-ns
of XAI-25 / NeSy-25 walks `D`), its one-body rules (added untested, the walk decides), its cycle rule and its goals.
(keras-ns-swarm's own `grounder_factory.py` sets the last step's width to 0: not what the paper ran.)

Parity: every grounding equal to keras-ns's on whole query sets — WN18RR test and 20k training queries at BC01 and
BC12, FB15k-237 at BC01, Countries S2/S3 at BC21/22/23 (loss, gradients and the Adam step too), Family and WN18RR
batches at BC01 — and `tests/counts.py` holds those semantics. keras-ns walks a (rule, step) block's goals in a
Python set's order, so its own output varies with the hash seed (a few groundings in thousands); here in atom order,
one of the orders it can take. Not compared with keras-ns directly: depth 3 on Family / WN18RR / FB15k-237, YAGO3-10.

### fp_batch

The last step's width is 0 (every leaf a fact), then the groundings whose body is proved within `d` rounds of
propagation from the facts. Differs from keras where an unknown atom is proved through another chain.

### Paper rule sets

Family: the paper uses the 47 hand-curated rules — `rules.txt` in data-swarm (`rules_old.txt` is the same set;
`rules_new.txt`, 143 rules, is an automated expansion).

## Architecture

- `kb.py`: parsing, rule compilation (anchor variants), `Facts`, `Rules`, `KB`
- `ops.py`, `types.py`: keys, decoding, grouping; `Groundings`, `Proofs`, `Closure`
- `pbc/`: `PBC` — `tables.py` (per-variant tables), `kernels.py` (Triton: fact hash set, the fused last stage),
  `engine.py` (the steps, streamed; the prunes; the canonical output), `guide.py` (`Guide`), `sizing.py` (worst case)
- `sld/`: `SLD` — `resolve.py` (unify, substitute, lookups), `state.py` (pack, compact, rename, trail, harvest)
- `forward/`: `Forward` — `spmm/` (semi-naive sparse matmul), `join/` (staged join), `router.py`
- `api/rule_grounder.py`: the adapter torch-ns calls (`create_grounder`, `RuleGrounder`)
- `docs/`: `design.md`
- `tests/`: unit and regression tests
- Still present, to be deleted once consumers move: the old general engine (`backward/`, `resolution/`,
  `execution/`, `filters/`, `data/`, `base/`, `core.py`, `vocab/`)

## Running Experiments

This repository is primarily a library, not a training entry point.

Use it from its consumers (torch-ns: `PBC` through `api/rule_grounder.py`; probfol-llm: `SLD`, `Forward`) and run
its tests from this directory. Do not add training scripts here.

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

```
tests/
├── test_pbc.py              PBC on Countries S3 / Family: Full, the guide, chunked steps change nothing
├── test_pbc_oracle.py       PBC (fp_batch) against a brute-force reference on random KBs
├── test_keras_filter.py     the keras prune (keras-ns's BC_{w,d}): its cycle rule, one-body rules, walk rounds
├── test_rule_grounder.py    the torch-ns adapter (create_grounder, RuleGrounder)
├── test_sld.py              SLD's derive / prove on toy KBs
├── counts.py                optional: grounding counts on real KGs against baselines/grounding_counts.json
├── probfol_record.py        optional: probfol-llm's SLD / FC calls against their recorded outputs
├── fc_fingerprint.py        optional: forward chaining's closures against baselines
└── guided_ab.py             older guided harness (imports the old engine)
```

```bash
python -m pytest tests/ -q          # the suite (GPU: PBC's kernels are Triton)
python tests/counts.py              # optional, < 90 s: Family / WN18RR / Countries S3 test queries (and a Family
                                    # train slice) x fp_batch and keras grounders at depths 1-3, in batches of 256:
                                    # each step's goals, the kept firings, atoms, firings per rule and an output
                                    # hash exactly
python tests/counts.py --update     # re-record after an intended change of what is grounded
```

Run `tests/counts.py` whenever grounding semantics, the prunes, PBC's engine or dataset loading may have changed; an
optimisation must leave every checked field unchanged (it may shrink the raw groundings a step writes, which the
script reports but does not check).

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

- parsing, rule compilation, fact and rule indexes: `kb.py`
- shared tensor operations and output types: `ops.py`, `types.py`
- a technique's steps, prunes and kernels: its package (`pbc/`, `sld/`, `forward/`)
- models that guide grounding: outside this repo, through `pbc.Guide`'s `Scorer`

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

- PBC's kernels (Triton) take their sizes at run time; size buffers from actual counts, never from a worst case.
- Keep host syncs (`.item()`, `.tolist()`) to one per chunk or kernel launch; none per row.
- Add type hints to function signatures.
- Document important tensor shapes with comments using the standard symbols above (e.g. `[B, S, G, 3]`).
- Prefer vectorized tensor code over Python loops in hot paths.
- Keep comments concise and focused on non-obvious behavior.

## Technical Rules

- Never revert or restore files without explicit user permission.
- Fix bugs forward; do not hide them with clamps or silent fallbacks.
- Never truncate (no fact, rule, step or children caps in PBC): bound a step's memory by streaming its work in chunks, and keep the output identical to the unchunked one.
- Keep timing-sensitive tests and benchmarks sequential.
- Use a git worktree when you need to compare with an older commit.
- Keep mirrored grounder copies synchronized when the intent is shared behavior across repos.
- Do not leave scratch artifacts inside package directories.
- Prefer the smallest coherent change that keeps one owner per responsibility; avoid scattering one feature across multiple modules.
- torch-kge-kernels is a sibling repo at `~/repos/torch-kge-kernels-swarm/main/`, installed as pip-editable. Edit it there, commit there, push there. The SHA pin in this repo's `pyproject.toml` must be bumped whenever the editable HEAD moves — the pre-commit hook (`scripts/check_editable_pins.py`, wired via `.pre-commit-config.yaml`) refuses commits when the pin and the editable HEAD disagree or when the editable HEAD is unpushed. Setup once with `conda activate gpu && pre-commit install`. Bypass only with `SKIP=check-editable-pins git commit ...` for genuinely unrelated commits during an in-flight cascade.

## Verification Checklist

- any code change: `python -m pytest tests/ -q`.
- grounding semantics, the prunes or PBC's engine changed: `python tests/counts.py` (exact counts and hashes).
- before commit: both, then torch-ns's gate (`python tests/gate.py` there: a Family BC12 run end to end).
- mirrored change intended: sync the other grounder copy or checkout and rerun its relevant tests
- before any commit: the `check-editable-pins` pre-commit hook runs automatically (if installed) and blocks the commit if the `torch-kge-kernels` SHA pin in `pyproject.toml` has drifted from the editable install or points at an unpushed HEAD. To run it manually: `python scripts/check_editable_pins.py`.
