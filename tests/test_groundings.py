"""Comprehensive grounding-count comparison across every grounder.

Builds a 2-D table where ROWS are (grounder × config) and COLUMNS are
datasets. Each cell records both grounding count and wall-clock time
for that grounder on that dataset's test split. Result is written to
``tests/baselines/comparison.json`` and printed in two tables (counts,
timings).

Default grounders / configs:
  * keras-ns BC         : w0d1, w1d2, w1d3        (paper baseline)
  * SLD                 : d1, d2, d3, d4 (w=1)    (torch SLD reference)
  * enum-flat           : w0d1, w1d2, w1d3        (torch dynamic path)
  * enum-dense          : w0d1, w1d2, w1d3        (torch static-shape path)
  * FC fp_global        : closure                 (forward-chaining provable set)

Default datasets: ablation_d2, ablation_d3, countries_s2, countries_s3,
family, wn18rr — all test split only.

Run as precommit (requires keras-ns + tensorflow on CPU). Use args to
narrow scope:

    # Quick smoke run on one dataset, one grounder, fewer queries
    python tests/test_groundings.py \\
        --datasets ablation_d2 --rows enum-flat:w1d2 --max-queries 25

    # Full sweep, write baseline
    python tests/test_groundings.py --output tests/baselines/comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force TensorFlow off the GPU before any keras-ns import — keras-ns
# eagerly imports TF and TF defaults to grabbing all visible memory.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

# keras-ns is sibling-style (not pip-installed because its top-level
# ``ns_lib`` collides with torch-ns). Resolve via ``KERAS_NS_ROOT``.
TESTS_DIR = Path(__file__).resolve().parent
KERAS_NS_ROOT = Path(os.environ.get(
    "KERAS_NS_ROOT",
    str(Path.home() / "repos" / "keras-ns-swarm" / "main"),
))
if str(KERAS_NS_ROOT) not in sys.path:
    sys.path.insert(0, str(KERAS_NS_ROOT))

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

from grounder.data.loader import KGDataset
from grounder.bc.bc import BCGrounder    # used by run_fc directly (fp_global)
from grounder.groundings import (
    atom_hash, count_proof_trees, evidence_unique_app_count,
)

# Reuse keras-ns and torch grounder builders from the shared
# ``_runners`` module so the count and speed sweeps stay aligned.
sys.path.insert(0, str(TESTS_DIR))
from _runners import (  # noqa: E402
    DEFAULT_COMPILE_MODES, build_keras_grounder, build_torch_grounder,
)

# ── Default grid ──────────────────────────────────────────────────────

DEFAULT_DATASETS: List[str] = [
    "ablation_d2", "ablation_d3",
    "countries_s2", "countries_s3",
    "family", "wn18rr",
    # fb15k237 is included for the FC row only — it's the largest
    # benchmark we run forward-chaining on, used to size-test the SpMM
    # closure path. The other grounders (BC variants) would either OOM
    # or not produce meaningful numbers at this scale within a sweep.
    "fb15k237",
]


# Datasets that participate in the BC sweep. Datasets outside this set
# only run the ``FC`` row.
_BC_SWEEP_DATASETS = {
    "ablation_d2", "ablation_d3",
    "countries_s2", "countries_s3",
    "family", "wn18rr",
}


def _rows_for_dataset(dataset: str, rows):
    """Filter the row list to those that make sense for a dataset.

    ``fb15k237`` is too big for the BC grounders in this sweep — we
    only run the FC row to record the closure benchmark.
    """
    if dataset not in _BC_SWEEP_DATASETS:
        return [r for r in rows if r[0] == "FC"]
    return rows


# Per-dataset default for ``--max-queries`` when the user passes 0
# (= "use full test split"). fb15k237's test split is huge (20k
# queries); 50 is what other datasets use anyway, and FC's per-query
# work scales linearly so we want a bounded number.
_DEFAULT_MAX_QUERIES = {
    "fb15k237": 50,
}

# Each row = (grounder kind, config label, kwargs).
# ``kind`` selects the runner; ``cfg`` is the config label printed in
# the row header; ``kwargs`` is whatever the runner needs.
DEFAULT_ROWS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("keras-BC",   "w0d1", dict(w=0, d=1)),
    ("keras-BC",   "w1d2", dict(w=1, d=2)),
    ("keras-BC",   "w1d3", dict(w=1, d=3)),
    # SLD is parameterised by depth only — width is an enum-only knob.
    ("SLD",        "d1",   dict(depth=1)),
    ("SLD",        "d2",   dict(depth=2)),
    ("SLD",        "d3",   dict(depth=3)),
    ("SLD",        "d4",   dict(depth=4)),
    ("enum-flat",  "w0d1", dict(w=0, d=1, flat=True)),
    ("enum-flat",  "w1d2", dict(w=1, d=2, flat=True)),
    ("enum-flat",  "w1d3", dict(w=1, d=3, flat=True)),
    ("enum-dense", "w0d1", dict(w=0, d=1, flat=False)),
    ("enum-dense", "w1d2", dict(w=1, d=2, flat=False)),
    ("enum-dense", "w1d3", dict(w=1, d=3, flat=False)),
    ("FC",         "fp_global", dict()),
]


# ── Per-dataset defaults ──────────────────────────────────────────────

# Test-split sizes (queries) and per-dataset rules file. Family ships
# two rule sets; the IJCAI '25 paper uses the smaller hand-curated
# ``rules_old.txt``. Other datasets use ``rules.txt``.
def _rules_file_for(label: str) -> str:
    return "rules_old.txt" if "family" in label else "rules.txt"


# Per-dataset batch size (chunking) for the dense / enum paths so the
# big test sets don't OOM.
def _batch_size_for(label: str) -> int:
    if "wn18rr" in label:    return 10
    if "family" in label:    return 100
    if "countries_s3" in label: return 50
    return 0    # full batch


# ── Runners ───────────────────────────────────────────────────────────

def _sync(device) -> None:
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()
    elif hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize()


# ``atom_hash`` and ``count_proof_trees`` (the proof-tree fixpoint
# counter) and ``evidence_unique_app_count`` (the keras-comparable
# rule-grounding metric) live in :mod:`grounder.groundings` so they
# can be reused by callers outside this test harness. We import them
# at the top of the file.


def run_keras(ds: KGDataset, ds_path: Path, kb, queries: torch.Tensor,
              cfg: Dict[str, Any]) -> Tuple[int, Optional[int], Optional[int], float]:
    """Run the keras-ns ApproximateBackwardChainingGrounder.

    Returns ``(considered_rules, ground_rules, ground_proofs, ms)``.
    Keras-ns only exposes its ``rule2groundings`` accumulator total —
    its considered-rules metric. It does **not** distinguish "rule
    apps appearing in a completed proof tree" from "rule apps
    considered", and it does **not** expose a per-query proof-tree
    counter (the closed-form recursion overflows on recursive rule
    sets like ablation's ``locatedInCR``). We therefore report
    ``ground_rules`` and ``ground_proofs`` as ``None`` for keras, so
    the table renders ``—`` rather than misleadingly echoing the
    considered count.
    """
    test_tuples = [
        (ds.idx2pred[queries[i, 0].item()],
         ds.idx2entity[queries[i, 1].item()],
         ds.idx2entity[queries[i, 2].item()])
        for i in range(queries.shape[0])
    ]
    fact_tuples = list(ds._facts_raw)
    kg, _rules = build_keras_grounder(ds, ds_path, cfg["w"], cfg["d"])
    t0 = time.perf_counter()
    kg.ground(fact_tuples, test_tuples)
    elapsed = (time.perf_counter() - t0) * 1000.0
    total = sum(len(v) for v in kg.rule2groundings.values())
    return total, None, None, elapsed


def run_torch_bc(kb, queries: torch.Tensor, qmask: torch.Tensor,
                 cfg: Dict[str, Any], bs: int) -> Tuple[int, int, int, float]:
    """Run a BCGrounder (SLD / enum-flat / enum-dense).

    Returns ``(ground_rules, ground_proofs, ms)``:

    * ``ground_rules`` — unique ``(rule, head, body)`` triples from
      ``out.rule_groundings.A_in`` where every atom is fully ground
      (constants only). Variable-bearing entries are filtered out, so
      this column is directly comparable to keras's metric.
    * ``ground_proofs`` — per-query proof-tree count summed across
      queries (``evidence.count.sum()``). For enum this equals the
      number of complete derivation trees the grounder produced;
      for SLD it counts terminating SLD branches. 0 if the grounder
      doesn't populate ``evidence``.
    """
    is_sld = "depth" in cfg
    kind = "SLD" if is_sld else (
        "enum-flat" if cfg.get("flat", True) else "enum-dense")
    # Compile mode comes from the same ``DEFAULT_COMPILE_MODES`` table
    # ``test_speed.py`` uses. BCGrounder routes ``sld`` / ``rtf``
    # through the outer-compile + chunked-replay path; ``enum`` uses
    # per-step inner compile.
    #
    # SLD additionally drops ``collect_rule_groundings`` because that
    # accumulator's per-step body is appended to a Python list that
    # holds CUDA-graph-private-pool memory across chunks — which gets
    # overwritten by each subsequent compiled chunk call. Dropping it
    # gives us a None ``considered_rules`` for SLD; the metric is
    # meaningfully different for SLD anyway (raw branch count, not
    # unique apps), so we surface ``—`` for it. ``ground_rules`` and
    # ``ground_proofs`` come from evidence which IS cloned by the
    # outer-compile path.
    grounder_kwargs: Dict[str, Any] = {}
    if is_sld:
        grounder_kwargs["collect_rule_groundings"] = False
    g = build_torch_grounder(
        kind, kb, cfg,
        compile_mode=DEFAULT_COMPILE_MODES.get(kind),
        **grounder_kwargs)
    # Warmup (first call may allocate buffers); timed call is the second.
    fwd_kwargs = {"batch_size": bs} if bs > 0 else {}
    with torch.no_grad():
        _ = g(queries, qmask, **fwd_kwargs)
    _sync(queries.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = g(queries, qmask, **fwd_kwargs)
    _sync(queries.device)
    elapsed = (time.perf_counter() - t0) * 1000.0
    # Three metrics reported per cell:
    #
    # ``considered_rules`` — every rule application the grounder
    #   *considered* during search. For enum this is the in-pipeline
    #   ``rule_groundings`` accumulator (post-filter, pre-proof);
    #   matches keras's ``rule2groundings`` semantics. For SLD this
    #   includes pre-substitution branches (every rule fire across
    #   every SLD step), so it can be much larger than the in-proof
    #   count below.
    #
    # ``ground_rules`` — unique ``(rule, head, sorted_body)`` tuples
    #   that appear in some completed proof tree (= keras-comparable
    #   "rule applications used in proofs"). Derived from
    #   ``out.evidence`` (post-substitution).
    #
    # ``ground_proofs`` — ``evidence.count.sum`` — the number of
    #   distinct proof trees rooted at the queries.
    pad = kb.padding_idx
    if out.rule_groundings is not None:
        considered_rules = sum(
            out.rule_groundings.A_in[r].shape[0]
            for r in out.rule_groundings.A_in)
    elif is_sld:
        # SLD's outer-compile path drops the rule_groundings
        # accumulator (see ``grounder_kwargs`` above). Surface ``—``
        # in the considered-rules table rather than a misleading 0.
        considered_rules = None
    else:
        considered_rules = 0
    ground_rules = evidence_unique_app_count(out.evidence, pad)
    ground_proofs = 0
    if out.evidence is not None and out.evidence.count is not None:
        ground_proofs = int(out.evidence.count.sum().item())
    return considered_rules, ground_rules, ground_proofs, elapsed


def run_fc(kb, queries: torch.Tensor, qmask: torch.Tensor,
           _cfg) -> Tuple[int, int, float]:
    """Pure forward chaining (fp_global) — closure to rule apps.

    Pure FC has no depth or width. Implementation:

      1. Run ``run_forward_chaining`` to fixpoint over (facts, rules).
         Returns the set of all provable atoms (the closure).
      2. Count distinct rule groundings using only closure atoms:
         for each rule, enumerate every ``(head, body)`` grounding
         whose body atoms are all in the closure. Implemented as a
         single BC enum step with ``filter='fp_global'`` (which uses
         the precomputed closure as its provable-atom oracle) and
         ``depth=1, width=None``.

    Returns ``(ground_rules, ground_proofs, ms)``. By construction
    ``ground_rules`` is the upper bound on every other grounder's
    rule-application count for the same (facts, rules, queries) —
    every BC-style grounder at any depth/width admits a subset of
    these apps.
    """
    # For very large KBs (e.g. fb15k237), the BC enum step over the
    # closure can't fit on a 24 GB GPU even at batch=1 — its dense
    # tensors are O(B*S*K_r*G_r*M*3) and K_r alone is in the hundreds.
    # In that regime, skip BCGrounder construction (which would also
    # try to build a full-depth closure with fc_depth=10) and go
    # straight to the closure-only fallback at d=3. The threshold here
    # is conservative: family / wn18rr are well under 100k facts.
    if kb.num_facts > 200_000:
        return _fc_closure_only(kb, queries)

    bs = _batch_size_for_kb(kb)
    fwd_kwargs = {"batch_size": bs} if bs > 0 else {}

    def _try_depth(d: int) -> Tuple[int, float]:
        # ``flat_intermediate=True`` (dynamic shapes) avoids the
        # ``[B*S, K_r, G_r, M, 3]`` blowup of the dense path —
        # essential at depth 10 with width=None.
        g = BCGrounder(
            kb, resolution="enum", width=None, depth=d,
            filter="fp_batch",
            max_groundings_per_query=4096,
            max_total_groundings=4096,
            max_states=256, fc_method="spmm", prune_facts=True,
            flat_intermediate=True, bump_s_to_k=False,
            init_state_shape="minimal",
            collect_rule_groundings=True,
        )
        with torch.no_grad():
            _ = g(queries, qmask, **fwd_kwargs)   # warmup
        _sync(queries.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = g(queries, qmask, **fwd_kwargs)
        _sync(queries.device)
        ms = (time.perf_counter() - t0) * 1000.0
        if out.rule_groundings is None:
            return 0, ms
        n = sum(out.rule_groundings.A_in[r].shape[0]
                for r in out.rule_groundings.A_in)
        return n, ms

    # Aggressively free GPU memory before FC — by the time FC runs,
    # the previous 13 grounders have left fragmented allocations.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Adaptive (depth, batch_size) schedule: try the most permissive
    # config that fits in memory. d=3 + width=None dominates every
    # standard BC config (which all use d ≤ 3 and width ≤ 1). For
    # high-fan-out KBs we shrink the chunk progressively.
    K_f = kb.K_f if hasattr(kb, "K_f") else 0
    # FC has no depth or width — those belong to BC_{w,d}. The
    # implementation here uses a single BC enum step with
    # ``filter='fp_global'``: that filter precomputes the FC closure
    # (all atoms derivable from facts + rules) and uses it as the
    # provable-atom oracle for the BC step. The result is exactly
    # the set of distinct rule groundings ``(rule, head, body)``
    # whose body atoms are derivable in the closure — i.e., what
    # pure FC would produce as its rule-application output.
    def _attempt(target_kb, target_q, target_qm, chunk_bs: int) -> Tuple[int, int, int, float]:
        g = BCGrounder(
            target_kb, resolution="enum", width=None, depth=1,
            filter="fp_global",
            max_groundings_per_query=4096,
            max_total_groundings=4096,
            max_states=256, fc_method="spmm", prune_facts=True,
            flat_intermediate=True, bump_s_to_k=False,
            init_state_shape="minimal",
            collect_rule_groundings=True,
        )
        kw = {"batch_size": chunk_bs} if chunk_bs > 0 else {}
        with torch.no_grad():
            _ = g(target_q, target_qm, **kw)
        _sync(target_q.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = g(target_q, target_qm, **kw)
        _sync(target_q.device)
        ms = (time.perf_counter() - t0) * 1000.0
        if out.rule_groundings is not None:
            considered = sum(out.rule_groundings.A_in[r].shape[0]
                             for r in out.rule_groundings.A_in)
        else:
            considered = 0
        n_rules = evidence_unique_app_count(out.evidence, target_kb.padding_idx)
        n_proofs = 0
        if out.evidence is not None and out.evidence.count is not None:
            n_proofs = int(out.evidence.count.sum().item())
        return considered, n_rules, n_proofs, ms

    bs_options = [bs or 100, 50, 20, 10, 5, 2, 1]
    # First pass: GPU.
    for chunk_bs in bs_options:
        try:
            considered, n_rules, n_proofs, ms = _attempt(
                kb, queries, qmask, chunk_bs)
            print(f"    [FC bs={chunk_bs} → considered={considered}, "
                  f"in-proof={n_rules}, proofs={n_proofs}]")
            return considered, n_rules, n_proofs, ms
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e)[:80]
            print(f"    [FC bs={chunk_bs} OOM: {msg!r}]")
            gc.collect()
            torch.cuda.empty_cache()

    # GPU exhausted. fp_global precomputes the FC closure at grounder
    # construction; on high-fan-out KBs (wn18rr) that join can blow
    # past 20 GiB regardless of batch_size. Fall back to CPU — the
    # closure compute still terminates (8M sorted_hashes for wn18rr)
    # and the BC step over it is small.
    print("    [FC: GPU exhausted; retrying on CPU]")
    try:
        kb_cpu = kb.to("cpu") if hasattr(kb, "to") else kb
        q_cpu = queries.to("cpu")
        m_cpu = qmask.to("cpu")
        for chunk_bs in (50, 10, 1):
            try:
                considered, n_rules, n_proofs, ms = _attempt(
                    kb_cpu, q_cpu, m_cpu, chunk_bs)
                print(f"    [FC CPU bs={chunk_bs} → considered={considered}, "
                      f"in-proof={n_rules}, proofs={n_proofs}]")
                return considered, n_rules, n_proofs, ms
            except (RuntimeError, MemoryError) as e:
                print(f"    [FC CPU bs={chunk_bs} failed: {str(e)[:80]}]")
    except Exception as e:
        print(f"    [FC CPU fallback failed at setup: {str(e)[:80]}]")

    # Closure-only fallback when both the GPU and CPU BC-enum paths
    # failed at OOM. Records the SpMM closure size as the FC metric.
    print("    [FC: BC-enum step infeasible; recording closure-only metric]")
    return _fc_closure_only(kb, queries)


def _fc_closure_only(kb, queries) -> Tuple[int, int, Optional[int], float]:
    """SpMM forward-chaining closure size as the FC metric, depth-capped.

    Used when the post-closure BC enum step is infeasible (memory or
    fan-out blows up on the dense path). The returned counts are
    semantically a different metric from the post-closure BC numbers
    in other cells:

      * ``considered_rules`` = ``ground_rules`` = SpMM closure size
        (number of provable atoms at depth 3). Every closure atom is
        derivable; closure-only mode doesn't enumerate per-rule-app
        groundings so we don't distinguish "considered" from "in-proof".
      * ``ground_proofs`` = ``None`` (renders as ``—``). We don't
        enumerate proof trees in this path.

    Cap at depth 3 — for fb15k237 the closure is still growing at
    d=3 (113M new atoms in step 2) so ``fixpoint`` would blow past
    available memory.
    """
    import gc
    try:
        from grounder.fc.fc import run_forward_chaining
        from grounder.data.rule_index import compile_rules
        rules = compile_rules(
            kb.rule_index.rules_heads_sorted,
            kb.rule_index.rules_bodies_sorted,
            kb.rule_index.rule_lens_sorted,
            kb.fact_index._num_entities)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = str(queries.device)
        t0 = time.perf_counter()
        # Cap at depth=3 — for fb15k237 the closure is still growing
        # at d=3 (113M new atoms in step 2) so ``fixpoint`` would blow
        # past available memory.
        _, n_atoms = run_forward_chaining(
            compiled_rules=rules,
            facts_idx=kb.fact_index.facts_idx,
            num_entities=kb.fact_index._num_entities,
            num_predicates=kb.fact_index._num_predicates,
            depth=3, device=device)
        _sync(queries.device)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"    [FC closure-only: {n_atoms} provable atoms, {ms:.0f}ms]")
        return n_atoms, n_atoms, None, ms
    except Exception as e:
        print(f"    [FC closure-only failed: {type(e).__name__}: {str(e)[:80]}]")
        return 0, 0, None, 0.0


def _batch_size_for_kb(kb) -> int:
    """Chunk size for FC's BC pass — keep per-step memory bounded.

    The body tensor at d=10 grows as ``B*S*K_r*G_r*M*3``; chunking
    keeps that under ~1 GiB on typical 24 GiB GPUs.
    """
    K_f = kb.K_f if hasattr(kb, "K_f") else 0
    if K_f >= 400:        return 10        # wn18rr-class
    if K_f >= 100:        return 50        # countries_s3-class
    return 0                                # ablation, family — fits


# ── Sweep driver ──────────────────────────────────────────────────────

def run_cell(kind: str, ds: KGDataset, ds_path: Path, kb,
             queries, qmask, cfg, bs: int) -> Tuple[int, int, int, float]:
    """Dispatch a single (grounder, dataset) cell to its runner.

    Returns ``(considered_rules, ground_rules, ground_proofs, ms)``.
    See runner docstrings for the precise definitions of each metric.
    """
    if kind == "keras-BC":
        if not HAS_KERAS:
            return 0, 0, 0, 0.0
        return run_keras(ds, ds_path, kb, queries, cfg)
    if kind in ("SLD", "enum-flat", "enum-dense"):
        return run_torch_bc(kb, queries, qmask, cfg, bs)
    if kind == "FC":
        return run_fc(kb, queries, qmask, cfg)
    raise ValueError(f"unknown grounder kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="Comma-separated dataset names (resolved "
                             "under --data-root).")
    parser.add_argument("--data-root",
                        default=str(Path.home() / "repos" / "data-swarm"
                                    / "main"))
    parser.add_argument("--rows", default=None,
                        help="Comma-separated 'kind:config' filters "
                             "(e.g. 'enum-flat:w1d2,SLD:d2'). Default: "
                             "every default row.")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Truncate test split to first N queries "
                             "(0 = use all).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None,
                        help="Write the JSON baseline to this path. "
                             "Default: tests/baselines/comparison.json. "
                             "Pass empty string to skip.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge results into the existing JSON "
                             "rather than overwriting. Used when "
                             "running grounder-by-grounder so each "
                             "subprocess only contributes its own row.")
    args = parser.parse_args()

    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    rows_to_run: List[Tuple[str, str, Dict[str, Any]]] = DEFAULT_ROWS
    if args.rows:
        wanted = {r.strip() for r in args.rows.split(",") if r.strip()}
        rows_to_run = [r for r in DEFAULT_ROWS
                       if f"{r[0]}:{r[1]}" in wanted]

    # Cell results keyed by (dataset, kind, config) → {count, ms}.
    cells: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ds_name in dataset_names:
        # Force a fresh CUDA allocator state for each dataset. Without
        # this, the previous dataset's KB / grounder buffers stay in
        # the cache and fragment the GPU memory just enough to push
        # FC's d=10 / d=3 BC pass over the budget on family / wn18rr.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        ds_path = Path(args.data_root).expanduser() / ds_name
        rules_file = _rules_file_for(ds_name)
        ds = KGDataset(str(ds_path), device=args.device,
                       rules_file=rules_file)
        kb = ds.make_kb(max_facts_per_query=4096,
                        fact_index_type="block_sparse")
        queries = ds.get_queries("test")
        max_q = args.max_queries
        if max_q == 0:
            max_q = _DEFAULT_MAX_QUERIES.get(ds_name, 0)
        if max_q > 0:
            queries = queries[:max_q]
        qmask = torch.ones(queries.shape[0], dtype=torch.bool,
                           device=queries.device)
        bs = _batch_size_for(ds_name)
        n_q = queries.shape[0]
        print(f"\n### {ds_name}  rules={rules_file}  "
              f"facts={kb.num_facts}  queries={n_q}  batch_size={bs or 'full'}")
        cells[ds_name] = {"_n_queries": n_q}

        # Run FC LAST per dataset. The fp_global filter side-effect
        # is to MUTATE the kb's fact_index by augmenting it with the
        # full FC closure (see ``_build_fp_global_set`` in bc/bc.py).
        # Any subsequent grounder running on that mutated kb sees
        # closure-derived atoms as if they were base facts, which both
        # changes its grounding count and (if the closure spilled to
        # CPU) leaves the kb's tensors on mixed devices. Order
        # FC-last so every other grounder runs on the un-mutated kb.
        # ``torch.cuda.empty_cache()`` between cells keeps the GPU
        # tidy enough for FC's closure build.
        ds_rows = _rows_for_dataset(ds_name, rows_to_run)
        ordered = sorted(ds_rows, key=lambda r: 1 if r[0] == "FC" else 0)
        for kind, cfg_label, cfg in ordered:
            torch.cuda.empty_cache()
            try:
                considered, rules, proofs, ms = run_cell(
                    kind, ds, ds_path, kb, queries, qmask, cfg, bs)
                ok = True
                err = None
            except Exception as e:
                considered, rules, proofs, ms = 0, 0, 0, 0.0
                ok = False
                err = f"{type(e).__name__}: {str(e)[:80]}"
            tag = f"{kind}:{cfg_label}"
            cells[ds_name][tag] = {
                "considered_rules": considered,
                "ground_rules": rules,
                "ground_proofs": proofs,
                "ms": round(ms, 2),
                "ok": ok,
                "error": err,
                # Backwards-compat: keep ``count`` as alias of
                # ``ground_rules`` so older readers don't break.
                "count": rules,
            }
            if ok:
                def _s(v):
                    return f"{v:>6d}" if isinstance(v, int) else "    —"
                status = (f"considered={_s(considered)}  "
                          f"in-proof={_s(rules)}  proofs={_s(proofs)}  "
                          f"{ms:>9.1f}ms")
            else:
                status = f"FAIL ({err})"
            print(f"  {tag:<22s} {status}")

    # ── Print 2-D tables ──
    row_keys = [f"{k}:{c}" for (k, c, _) in rows_to_run]
    col_keys = dataset_names
    width = max(22, max(len(r) for r in row_keys) + 2)
    cw = 12  # per-cell column width

    def _print_table(title: str, value_fn) -> None:
        print(f"\n{'='*(width + cw*len(col_keys) + 4)}")
        print(title)
        print(f"{'='*(width + cw*len(col_keys) + 4)}")
        header = f"{'Grounder:Config':<{width}}  " + "".join(
            f"{ds[:cw-1]:>{cw}}" for ds in col_keys)
        print(header)
        print("-" * len(header))
        for rk in row_keys:
            row = f"{rk:<{width}}  "
            for ds in col_keys:
                cell = cells.get(ds, {}).get(rk, {})
                row += f"{value_fn(cell):>{cw}}"
            print(row)

    def _fmt_count(c, key, fallback_key=None):
        if not c:
            return "—"
        if not c.get("ok"):
            return "FAIL"
        v = c.get(key)
        if v is None and fallback_key is not None:
            v = c.get(fallback_key)
        if v is None:
            return "—"
        return str(v)

    def _fmt_ms(c):
        if not c:
            return "—"
        if not c.get("ok"):
            return "FAIL"
        v = c.get("ms")
        return "—" if v is None else f"{v:.0f}"

    _print_table(
        "CONSIDERED RULES (every rule app the grounder considered — "
        "keras's rule2groundings semantics; "
        "for SLD includes pre-substitution branches)",
        # No ``count`` fallback for considered_rules — when a grounder
        # legitimately doesn't expose this metric (SLD compiled-mode,
        # keras-BC), the cell stores ``None`` and we want ``—``, not
        # the ground_rules count.
        lambda c: _fmt_count(c, "considered_rules"))
    _print_table(
        "GROUND RULES (unique (rule,head,body) appearing in some "
        "completed proof tree — evidence-derived)",
        lambda c: _fmt_count(c, "ground_rules", "count"))
    _print_table(
        "GROUND PROOFS (per-query proof-tree count summed)",
        lambda c: _fmt_count(c, "ground_proofs"))
    _print_table(
        "TIMING (ms)",
        _fmt_ms)

    # ── Write JSON baseline ──
    out_path = args.output
    if out_path is None:
        out_path = str(TESTS_DIR / "baselines" / "comparison.json")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if args.merge and Path(out_path).exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        merged_cells = existing.get("cells", {}) if args.merge else {}
        for ds_name, row_dict in cells.items():
            target = merged_cells.setdefault(ds_name, {})
            for k, v in row_dict.items():
                target[k] = v
        # Union the row list (existing + new), preserving DEFAULT_ROWS
        # ordering when present.
        all_row_tags = list({tag for ds_d in merged_cells.values()
                             for tag in ds_d.keys() if tag != "_n_queries"})
        canonical = [f"{k}:{c}" for (k, c, _) in DEFAULT_ROWS]
        ordered_rows = [r for r in canonical if r in all_row_tags] + [
            r for r in all_row_tags if r not in canonical]
        baseline = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datasets": list(merged_cells.keys()),
            "rows": ordered_rows,
            "max_queries": args.max_queries,
            "device": args.device,
            "cells": merged_cells,
        }
        with open(out_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"\n[json] wrote {out_path}")

    # Surface cell failures via process exit code (precommit / CI
    # driver expects this).
    n_fail = sum(
        1 for ds_d in cells.values()
        for tag, c in ds_d.items()
        if tag != "_n_queries" and isinstance(c, dict)
        and c.get("ok") is False
    )
    if n_fail > 0:
        print(f"\n[exit 1] {n_fail} cell(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
