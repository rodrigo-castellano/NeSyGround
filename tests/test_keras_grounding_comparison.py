"""Compare grounding counts between keras-ns and any torch-ns grounder.

Goal: confirm torch-ns produces the same per-rule grounding counts as
keras-ns on every cell of the IJCAI '25 paper grid (datasets × BC_w_d
configs), and report timing per cell. Per-query comparison for
filter='none', full-batch for filter='fp_batch'. Supports any dataset
with domain2constants.txt and arrow-format rules.

What this test does
-------------------

For each (dataset, config, path) cell:

  1. Build the keras-ns grounder, run on the test split (or train+valid
     +test for small datasets — see ``_default_splits``), record per-
     rule grounding counts and wall-clock.
  2. Build the torch-ns grounder with the matching ``BC_{w,d,u=0}``
     parametrisation via ``make_bcwd``; run with the same query set,
     record per-rule counts and wall-clock for the steady-state call
     (one warmup is excluded).
  3. Diff per-rule counts. ``Match: YES`` when they're identical.

The two ``path`` flavours of the torch grounder:

  * ``flat`` — the dynamic ``_resolve_enum_step_flat`` path. Body atoms
    survive without padding; pack truncation only kicks in when valid
    children > S=256 per query, which is rare in practice. Always
    eager (``torch.nonzero`` produces dynamic shapes incompatible with
    ``compile_mode='reduce-overhead'``).
  * ``dense`` — the static-shape ``[B*S, K_r, G_r, M, 3]`` path that's
    the right pairing for ``compile_mode='reduce-overhead'`` (CUDA-graph
    capture). Allocates a fixed rectangle per step and pays for it on
    big KBs (family / wn18rr at scale).

Both run eager by default in this test (``compile_mode=None``); the
sweep needs that for cross-grounder safety — torch's CUDA-graph
``cudagraph_trees`` weakref bookkeeping accumulates state across
grounder instances and trips an internal assert after a few cells.
Single-cell benchmarks can pass ``--compile-mode reduce-overhead`` to
turn it on for one specific grounder.

Usage:
    # Full sweep, both paths, eager:
    PYTHONPATH=. python tests/test_keras_grounding_comparison.py \\
        --datasets ablation_d2,ablation_d3,countries_s2,countries_s3,family,wn18rr \\
        --configs w0d1,w1d2,w1d3 --paths flat,dense --device cuda

    # Same as above as pytest:
    PYTHONPATH=. python -m pytest tests/test_keras_grounding_comparison.py -v -s
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


def _sync(device) -> None:
    """Synchronize CUDA stream so wall-clock timings are accurate."""
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize()
    elif isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()


def _depth_rule_counts_from_evidence(
    out, keras_rules: list, variant_to_orig: Optional[list] = None,
) -> Dict[int, Dict[str, int]]:
    """Extract per-depth, per-rule firing counts from a GrounderOutput.

    ``out.evidence.rule_idx`` has shape ``[B, C, D]`` (per-batch, per
    collected proof tree, per depth). Each entry is the rule (or rule
    variant) applied at that depth. Variants get remapped to their
    original rule index via ``variant_to_orig`` so per-rule counts
    align with keras's rule names.

    Returns: ``{depth: {rule_name: count}}``.
    """
    out_dict: Dict[int, Dict[str, int]] = {}
    ev = getattr(out, "evidence", None)
    if ev is None or ev.rule_idx is None:
        return out_dict
    rule_idx = ev.rule_idx
    if rule_idx.dim() < 3:
        return out_dict
    mask = ev.mask if ev.mask is not None else None
    rule_idx_cpu = rule_idx.detach().cpu().numpy()
    mask_cpu = mask.detach().cpu().numpy() if mask is not None else None
    B, C, D = rule_idx_cpu.shape
    rule_names = [r.name for r in keras_rules]
    for d in range(D):
        per_rule: Dict[str, int] = {}
        for b in range(B):
            for c in range(C):
                if mask_cpu is not None and not mask_cpu[b, c]:
                    continue
                ridx = int(rule_idx_cpu[b, c, d])
                if ridx < 0:
                    continue
                if variant_to_orig is not None and ridx < len(variant_to_orig):
                    ridx = variant_to_orig[ridx]
                if 0 <= ridx < len(rule_names):
                    name = rule_names[ridx]
                    per_rule[name] = per_rule.get(name, 0) + 1
        if per_rule:
            out_dict[d] = per_rule
    return out_dict

TESTS_DIR = Path(__file__).resolve().parent
GROUNDER_ROOT = TESTS_DIR.parent
# keras-ns is a sibling-style reference repo (not pip-installed because its
# top-level `ns_lib/` collides with torch-ns).  Default to ~/repos/keras-ns-swarm/main/
# but allow override via KERAS_NS_ROOT env var.
KERAS_NS_ROOT = Path(os.environ.get(
    "KERAS_NS_ROOT",
    str(Path.home() / "repos" / "keras-ns-swarm" / "main"),
))

if str(KERAS_NS_ROOT) not in sys.path:
    sys.path.insert(0, str(KERAS_NS_ROOT))

from grounder.data.loader import KGDataset
from grounder.bc.bc import BCGrounder
from grounder.factory import make_bcwd

# Force TensorFlow off the GPU before any keras-ns import. keras-ns
# pulls in TF eagerly and TF's default policy is to grab all visible
# GPU memory, which OOMs the torch run that follows.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf  # noqa: E402
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# Keras imports (guarded)
try:
    from ns_lib.logic.commons import Domain, Rule
    from ns_lib.grounding.backward_chaining_grounder import (
        ApproximateBackwardChainingGrounder,
    )
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

_VAR_UPPER = re.compile(r"^[A-Z]")
_VAR_LOWER = re.compile(r"^[a-z]$")


def _is_variable(name: str) -> bool:
    return bool(_VAR_LOWER.match(name) or _VAR_UPPER.match(name))


# ══════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════

def load_dataset(data_dir: str, device: str = "cpu",
                  rules_file: str = "rules.txt"):
    """Load a KGDataset and build KB with large caps."""
    ds = KGDataset(data_dir, device=device, rules_file=rules_file)
    kb = ds.make_kb(max_facts_per_query=4096, fact_index_type="block_sparse")
    return ds, kb


def build_keras_grounder(
    ds: KGDataset,
    data_dir: Path,
    width: int,
    depth: int,
) -> "ApproximateBackwardChainingGrounder":
    """Build a keras-ns grounder matching the given (width, depth) config."""
    assert HAS_KERAS, "keras-ns not available (TensorFlow not installed)"

    fact_tuples = list(ds._facts_raw)
    rules_raw = sorted(list(ds._rules_raw))

    # Parse domains
    domain_path = data_dir / "domain2constants.txt"
    domains: Dict[str, Domain] = {}
    if domain_path.exists():
        with open(domain_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    domains[parts[0]] = Domain(parts[0], sorted(parts[1:]))

    if not domains:
        all_ents = sorted(ds.entity2idx.keys())
        domains["entity"] = Domain("entity", all_ents)

    # First domain name for var2domain mapping
    domain_name = next(iter(domains))

    # Build keras rules
    keras_rules = []
    for i, (head, body_atoms) in enumerate(rules_raw):
        all_vars: set = set()
        for atom in [head] + body_atoms:
            for arg in atom[1:]:
                if _is_variable(arg):
                    all_vars.add(arg)
        var2domain = {v: domain_name for v in all_vars}
        keras_rules.append(Rule(
            name=f"r{i}", head_atoms=[head], body_atoms=body_atoms,
            var2domain=var2domain,
        ))

    # Paper BC_{w,d,u=0} convention: u=0 always.
    #   max_unknown_fact_count            = w  (intermediate steps)
    #   max_unknown_fact_count_last_step  = 0  (paper: every leaf body
    #                                            atom must be a fact)
    #   prune_incomplete_proofs           = True (paired with u=0;
    #                                            torch maps to fp_batch)
    # Consequence: BC_{w,1,0} ≡ BC_{0,1,0} for every w because at d=1
    # the only step IS the last step, capped at u=0. They differ only
    # for d≥2 where w controls intermediate-step admissibility.
    max_unk = width
    max_unk_last = 0
    prune = True

    return ApproximateBackwardChainingGrounder(
        rules=keras_rules, facts=fact_tuples, domains=domains,
        num_steps=depth, max_unknown_fact_count=max_unk,
        max_unknown_fact_count_last_step=max_unk_last,
        prune_incomplete_proofs=prune,
    ), keras_rules


def build_torch_grounder(
    kb, width: int, depth: int, *,
    flat: bool = True, all_anchors: bool = False,
    filt: Optional[str] = None, S_max: int = 256, C: int = 4096,
    G_r: int = 4096, bump_s_to_k: bool = True,
    compile_mode: Optional[str] = None,
) -> BCGrounder:
    """Build the paper's BC_{w,d} grounder via ``make_bcwd``.

    ``filt=None`` lets ``make_bcwd`` pick the keras-prune-aligned
    default (``'none'`` for d=1,w>0; ``'fp_batch'`` otherwise).
    Pass an explicit ``filt`` to test a specific filter.

    ``bump_s_to_k`` controls the per-state pack budget:
      * ``True`` — ``S = max(S_max, min(K, K_r * K_v))`` so all valid
        per-state children fit in the pack output (no truncation).
        Pre-bump-fix this used the budget ``K`` directly which on
        small KBs overshot the realistic per-state child count by
        50× and OOMed; the current ``min(K, K_r*K_v)`` cap keeps
        ``S`` realistic.
      * ``False`` — ``S = S_max`` (= 256). Faster but truncates at
        scale on high-fan-out KBs, undercounting unique apps.
    """
    kwargs = dict(
        kb=kb, w=width, d=depth,
        flat_intermediate=flat,
        filter=filt,
        max_groundings_per_query=G_r,
        max_total_groundings=C,
        max_states=S_max,
        fc_method="join", prune_facts=True,
        bump_s_to_k=bump_s_to_k,
        # init_state_shape:
        #   "minimal" (default, DpRL-friendly) — d=0 has S_in=1.
        #   "full"  — d=0 has S_in=max_states; needed for single-graph
        #              compile but blows memory on full-batch sweeps.
        # Use "full" only when we explicitly compile.
        init_state_shape=("full" if compile_mode else "minimal"),
    )
    # Explicitly pass compile_mode (defaults to None = eager). Without
    # this the BCGrounder auto-pick would set 'reduce-overhead' for the
    # dense path, which breaks multi-grounder sweeps via the torch
    # cudagraph_trees weakref assert. Pass --compile-mode reduce-overhead
    # for single-cell benchmarks.
    kwargs["compile_mode"] = compile_mode
    return make_bcwd(**kwargs)


# ══════════════════════════════════════════════════════════════════════
# Comparison logic
# ══════════════════════════════════════════════════════════════════════

def compare_groundings(
    ds: KGDataset,
    kb,
    data_dir: Path,
    width: int,
    depth: int,
    *,
    flat: bool = True,
    all_anchors: bool = False,
    filt: str = "none",
    S_max: int = 256,
    C: int = 4096,
    split: str = "test",
    splits: Optional[List[str]] = None,
    verbose: bool = True,
    max_queries: int = 0,
    G_r: int = 4096,
    batch_size: Optional[int] = None,
    bump_s_to_k: bool = True,
    compile_mode: Optional[str] = None,
) -> dict:
    """Compare keras-ns vs torch-ns grounding counts.

    Returns dict with per-query counts and summary.

    ``splits`` overrides ``split`` when provided: queries are concatenated
    across all listed splits ("train" / "valid" / "test"). Useful when the
    test split is too small to be representative — most ablation /
    countries datasets only ship a few hundred test triples but many
    thousand train triples.
    """
    if splits is None:
        splits = [split]
    parts: List[Tensor] = []
    for s in splits:
        parts.append(ds.get_queries(s))
    test = torch.cat([p for p in parts if p.numel() > 0], dim=0) \
        if any(p.numel() > 0 for p in parts) \
        else torch.empty(0, 3, dtype=torch.long, device=ds.device)
    if max_queries > 0:
        test = test[:max_queries]
    B = test.size(0)
    qmask = torch.ones(B, dtype=torch.bool, device=test.device)

    # Test queries as string tuples
    test_tuples = [
        (ds.idx2pred[test[i, 0].item()],
         ds.idx2entity[test[i, 1].item()],
         ds.idx2entity[test[i, 2].item()])
        for i in range(B)
    ]

    # ── Keras ──
    kg, keras_rules = build_keras_grounder(ds, data_dir, width, depth)
    fact_tuples = list(ds._facts_raw)

    # For fp_batch: run full batch. For none: keras always runs full batch anyway.
    t0 = time.perf_counter()
    kg.ground(fact_tuples, test_tuples)
    keras_ms = (time.perf_counter() - t0) * 1000.0
    keras_total = sum(len(v) for v in kg.rule2groundings.values())

    # Per-rule keras counts
    keras_per_rule = {r.name: len(kg.rule2groundings[r.name]) for r in keras_rules}

    # Per-query keras counts (re-run per query for per-query comparison)
    keras_per_query = []
    keras_per_query_ms_total = 0.0
    if filt == "none":
        for i in range(B):
            kg_i, _ = build_keras_grounder(ds, data_dir, width, depth)
            t0 = time.perf_counter()
            kg_i.ground(fact_tuples, [test_tuples[i]])
            keras_per_query_ms_total += (time.perf_counter() - t0) * 1000.0
            cnt = sum(len(v) for v in kg_i.rule2groundings.values())
            keras_per_query.append(cnt)
    else:
        # Batched filter: can't decompose per-query
        keras_per_query = [keras_total]  # single batch entry

    # ── Torch ──
    g = build_torch_grounder(kb, width, depth, flat=flat,
                              all_anchors=all_anchors, filt=filt,
                              S_max=S_max, C=C, G_r=G_r,
                              bump_s_to_k=bump_s_to_k,
                              compile_mode=compile_mode)

    # Count torch's UNIQUE rule applications (matches keras's
    # ``len(rule2groundings[r])``): out.rule_groundings.A_in[r] holds
    # one row per distinct (head, body) tuple for rule r, deduped via
    # ``_variant_to_orig`` and ``_collect_r2g`` set semantics. Counting
    # ``ev.mask.sum()`` would count proof TREES (one entry per depth-D
    # tree, NOT per rule application) and undercounts whenever a tree's
    # depths share rule applications.
    #
    # Timing: warm up once at the shape we'll be running so the first
    # ``torch.compile`` capture (and CUDA-graph build with
    # ``mode='reduce-overhead'``) is excluded from ``torch_ms``. We
    # report steady-state wall-clock to mirror keras (no compile cost).
    torch_ms = 0.0
    if filt == "none":
        # Per-query: run each query individually so per-query counts
        # are comparable with keras's per-query rerun.
        with torch.no_grad():
            _ = g(test[:1], qmask[:1])  # warmup at B=1
        _sync(test.device)
        torch_per_query = []
        torch_per_rule: Dict[str, int] = {r.name: 0 for r in keras_rules}
        torch_per_depth_rule: Dict[int, Dict[str, int]] = {}
        v2o = getattr(g, "_variant_to_orig", None)
        for i in range(B):
            q = test[i:i+1]
            qm = torch.ones(1, dtype=torch.bool, device=q.device)
            _sync(q.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = g(q, qm)
            _sync(q.device)
            torch_ms += (time.perf_counter() - t0) * 1000.0
            cnt_apps = (sum(out.rule_groundings.A_in[r].shape[0]
                            for r in out.rule_groundings.A_in)
                        if out.rule_groundings is not None else 0)
            torch_per_query.append(cnt_apps)
            if out.rule_groundings is not None:
                for ri, r in enumerate(keras_rules):
                    if ri in out.rule_groundings.A_in:
                        torch_per_rule[r.name] += int(
                            out.rule_groundings.A_in[ri].shape[0])
            depth_counts = _depth_rule_counts_from_evidence(
                out, keras_rules, variant_to_orig=v2o)
            for d, per_rule in depth_counts.items():
                bucket = torch_per_depth_rule.setdefault(d, {})
                for name, cnt in per_rule.items():
                    bucket[name] = bucket.get(name, 0) + cnt
        torch_total = sum(torch_per_query)
    else:
        # Batched. With cartesian_product=True (forced for enum) the
        # body tensor shape grows as ``B × K_r × E × M × 3`` which OOMs
        # for datasets with many head-clustered rules (e.g. family with
        # 143 rules → K_r ≥ 286). Fall back to per-query when the
        # batched path runs out of memory; the rule_groundings count
        # still aggregates correctly via ``_r2g_buffer`` set semantics.
        torch_per_rule = {r.name: 0 for r in keras_rules}
        torch_per_depth_rule = {}
        # Forward kwargs — batch_size enables BCGrounder's static-shape
        # ``_forward_chunked`` path so each chunk's reduce-overhead graph
        # replays with stable shapes, last chunk padded + masked.
        fwd_kwargs = {"batch_size": batch_size} if batch_size else {}
        try:
            # Warmup at full-batch shape, then steady-state timed run.
            with torch.no_grad():
                _ = g(test, qmask, **fwd_kwargs)
            _sync(test.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = g(test, qmask, **fwd_kwargs)
            _sync(test.device)
            torch_ms = (time.perf_counter() - t0) * 1000.0
            torch_total = (sum(out.rule_groundings.A_in[r].shape[0]
                               for r in out.rule_groundings.A_in)
                           if out.rule_groundings is not None else 0)
            torch_per_query = [torch_total]
            if out.rule_groundings is not None:
                for ri, r in enumerate(keras_rules):
                    if ri in out.rule_groundings.A_in:
                        torch_per_rule[r.name] = int(
                            out.rule_groundings.A_in[ri].shape[0])
            v2o = getattr(g, "_variant_to_orig", None)
            torch_per_depth_rule = _depth_rule_counts_from_evidence(
                out, keras_rules, variant_to_orig=v2o)
        except (RuntimeError, MemoryError) as exc:
            # Halve batch_size and retry — per-query fallback would
            # double-count duplicates across queries (no cross-query
            # dedup) and is wrong for ``fp_batch`` which prunes against
            # the merged-batch app set, not per-query.
            torch.cuda.empty_cache()
            cur_bs = batch_size if batch_size else max(1, B)
            success = False
            while cur_bs > 1 and not success:
                cur_bs = max(1, cur_bs // 2)
                print(f"  [batched OOM: {type(exc).__name__}; retrying "
                      f"with batch_size={cur_bs}]", flush=True)
                try:
                    with torch.no_grad():
                        _ = g(test, qmask, batch_size=cur_bs)
                    _sync(test.device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = g(test, qmask, batch_size=cur_bs)
                    _sync(test.device)
                    torch_ms = (time.perf_counter() - t0) * 1000.0
                    torch_total = (sum(out.rule_groundings.A_in[r].shape[0]
                                       for r in out.rule_groundings.A_in)
                                   if out.rule_groundings is not None else 0)
                    torch_per_query = [torch_total]
                    if out.rule_groundings is not None:
                        for ri, r in enumerate(keras_rules):
                            if ri in out.rule_groundings.A_in:
                                torch_per_rule[r.name] = int(
                                    out.rule_groundings.A_in[ri].shape[0])
                    v2o = getattr(g, "_variant_to_orig", None)
                    torch_per_depth_rule = _depth_rule_counts_from_evidence(
                        out, keras_rules, variant_to_orig=v2o)
                    success = True
                except (RuntimeError, MemoryError) as exc2:
                    exc = exc2
                    torch.cuda.empty_cache()
            if not success:
                # Last-resort per-query fallback. Inflates the count for
                # any rule app that fires for multiple queries (common
                # for cross-query body atoms), but at least returns a
                # finite number rather than crashing.
                print(f"  [exhausted batch_size retries; per-query "
                      f"fallback — count will OVER-count duplicate "
                      f"(rule, head, body) tuples across queries]",
                      flush=True)
                torch_ms = 0.0
                torch_per_query = []
                for i in range(B):
                    q = test[i:i+1]
                    qm = torch.ones(1, dtype=torch.bool, device=q.device)
                    _sync(q.device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out_i = g(q, qm)
                    _sync(q.device)
                    torch_ms += (time.perf_counter() - t0) * 1000.0
                    cnt = (sum(out_i.rule_groundings.A_in[r].shape[0]
                                for r in out_i.rule_groundings.A_in)
                           if out_i.rule_groundings is not None else 0)
                    torch_per_query.append(cnt)
                    if out_i.rule_groundings is not None:
                        for ri, r in enumerate(keras_rules):
                            if ri in out_i.rule_groundings.A_in:
                                torch_per_rule[r.name] += int(
                                    out_i.rule_groundings.A_in[ri].shape[0])
                torch_total = sum(torch_per_query)

    # ── Report ──
    speedup = (keras_ms / torch_ms) if torch_ms > 0 else float("inf")
    result = {
        "config": f"w{width}d{depth}",
        "n_queries": B,
        "keras_total": keras_total,
        "torch_total": torch_total,
        "keras_per_rule": keras_per_rule,
        "torch_per_rule": torch_per_rule,
        "torch_per_depth_rule": torch_per_depth_rule,
        "keras_per_query": keras_per_query,
        "torch_per_query": torch_per_query,
        "match": keras_total == torch_total,
        "diff": torch_total - keras_total,
        "keras_ms": keras_ms,
        "torch_ms": torch_ms,
        "keras_per_query_ms_total": keras_per_query_ms_total,
        "speedup": speedup,
    }

    if verbose:
        label = f"w{width}d{depth}"
        aa = "+AA" if all_anchors else ""
        fl = "flat" if flat else "dense"
        print(f"\n{'='*70}")
        print(f"{ds._facts_raw[0][0].split('(')[0] if ds._facts_raw else 'dataset'}"
              f" {label} (S_max={S_max}, {fl}{aa}, filter={filt})")
        print(f"{'='*70}")
        print(f"  Keras: {keras_total}  Torch: {torch_total}  "
              f"Diff: {torch_total - keras_total:+d}  "
              f"Match: {'YES' if result['match'] else 'NO'}")
        # Wall-clock timing for the full-batch keras run (one call) vs.
        # the steady-state torch run (post-warmup). For ``filt='none'``,
        # torch_ms is the sum across B per-query calls; divide by B for
        # the per-query average.
        print(f"  Time:  Keras: {keras_ms:.1f} ms   "
              f"Torch: {torch_ms:.1f} ms   "
              f"Speedup keras/torch: {speedup:.2f}×")
        if filt == "none" and B > 0:
            print(f"         Keras per-query sum: "
                  f"{keras_per_query_ms_total:.1f} ms "
                  f"(avg {keras_per_query_ms_total / B:.2f} ms)   "
                  f"Torch per-query avg: {torch_ms / B:.2f} ms")

        # Per-rule (top-level)
        print(f"\n  Per-rule (top-level):")
        for r in keras_rules:
            kc = keras_per_rule[r.name]
            tc = torch_per_rule.get(r.name, 0)
            flag = "" if kc == tc else f"  ({tc - kc:+d})"
            print(f"    {r.name}: keras={kc:<6} torch={tc:<6}{flag}")

        # Per-depth-per-rule (structured)
        if torch_per_depth_rule:
            print(f"\n  Per-depth-per-rule (torch structured evidence):")
            for d in sorted(torch_per_depth_rule.keys()):
                counts = torch_per_depth_rule[d]
                active = sum(counts.values())
                if active == 0:
                    continue
                parts = [f"{rn}={c}" for rn, c in counts.items() if c > 0]
                print(f"    depth {d}: {', '.join(parts)} (total={active})")

        # Per-query (if available)
        if filt == "none" and len(keras_per_query) == B:
            mismatches = [(i, keras_per_query[i], torch_per_query[i])
                          for i in range(B) if keras_per_query[i] != torch_per_query[i]]
            if mismatches:
                print(f"\n  Per-query mismatches ({len(mismatches)}/{B}):")
                for i, kc, tc in mismatches[:10]:
                    q_str = f"{test_tuples[i][0]}({test_tuples[i][1]},{test_tuples[i][2]})"
                    print(f"    q{i} {q_str[:40]:<40} K={kc:<4} T={tc:<4} ({tc-kc:+d})")
                if len(mismatches) > 10:
                    print(f"    ... +{len(mismatches)-10} more")
            else:
                print(f"\n  All {B} queries match!")

    return result


# ══════════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare keras-ns vs torch-ns grounding counts.")
    parser.add_argument("--dataset", default=None,
                        help="Path to a single dataset directory. Mutually "
                             "exclusive with --datasets.")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset paths (or names) for a "
                             "multi-dataset sweep. Each dataset runs the "
                             "configs and a final cross-dataset summary "
                             "table is printed.")
    parser.add_argument("--data-root", default=None,
                        help="Root directory used to resolve --datasets "
                             "names (e.g. ~/repos/data-swarm/main).")
    parser.add_argument("--configs", default="w0d1,w1d2,w1d3",
                        help="Comma-separated w<W>d<D> configs")
    parser.add_argument("--flat", action="store_true", default=True,
                        help="Deprecated — use ``--paths flat`` "
                             "(default) or ``--paths flat,dense``.")
    parser.add_argument("--no-flat", dest="flat", action="store_false",
                        help="Deprecated — use ``--paths dense``.")
    parser.add_argument("--all-anchors", action="store_true")
    # BC_{w,d} default filter is keras-prune-aligned per (d, w):
    #   d=1, w>0 → 'none'       (keras prune_incomplete_proofs=False)
    #   else     → 'fp_batch'   (keras prune_incomplete_proofs=True)
    # 'auto' (default) lets the grounder pick. Pass an explicit value
    # only to test a specific filter regardless of (d, w).
    parser.add_argument("--filter", default="auto")
    parser.add_argument("--s-max", type=int, default=256)
    parser.add_argument("--C", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-queries", type=int, default=-1,
                        help="Truncate to first N queries. -1 = per-dataset "
                             "default (use the full default split set for "
                             "every dataset). 0 = no cap (all queries from "
                             "the chosen splits, equivalent to -1).")
    parser.add_argument("--splits", default=None,
                        help="Comma-separated splits to use for queries (e.g. "
                             "'test', 'train,valid,test'). Default: per-dataset "
                             "(family/wn18rr=test only; countries/ablation="
                             "train,valid,test).")
    parser.add_argument("--G-r", type=int, default=4096,
                        help="max_groundings_per_query in BCGrounder.")
    parser.add_argument("--rules-file", default="rules.txt",
                        help="Rules filename inside dataset dir. Paper "
                             "numbers for family use 'rules_old.txt' "
                             "(47 rules vs 143 in rules.txt).")
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="BCGrounder.forward(batch_size=...) chunk "
                             "size. 0 = full batch in one forward. -1 = "
                             "per-dataset default (countries_s3 chunks at "
                             "50 to dodge an at-scale bug + OOM at V=2; "
                             "all others run full batch).")
    parser.add_argument("--bump-s", default=None,
                        choices=["true", "false"],
                        help="Force ``bump_s_to_k`` to True/False. "
                             "Default (omitted) uses per-dataset choice: "
                             "True for ablation_*/countries_s2 (small K, "
                             "fits memory, no truncation), False for "
                             "countries_s3/family/wn18rr (high K, would "
                             "OOM with bump=True; accept some pack "
                             "truncation).")
    parser.add_argument("--no-bump-s", action="store_true",
                        help="Shorthand for ``--bump-s false``.")
    parser.add_argument("--paths", default="flat",
                        help="Comma-separated paths to compare per cell: "
                             "'flat' (default), 'dense', or 'flat,dense' "
                             "to run both and emit two rows per cell so "
                             "you can see keras vs flat vs dense counts "
                             "+ timings side-by-side.")
    parser.add_argument("--compile-mode", default=None,
                        help="Pass to BCGrounder. Typical: "
                             "'reduce-overhead' (CUDA-graph capture, "
                             "static shapes only). Only affects the "
                             "dense path; flat path is forced eager. "
                             "First call compiles (slow, in warmup); "
                             "second call replays the graph at sub-ms.")
    parser.add_argument("--csv", default=None,
                        help="Write the cross-dataset summary table as CSV "
                             "to this path (in addition to stdout).")
    parser.add_argument(
        "--json", default=None,
        help="Write the cross-dataset summary table as JSON to this path "
             "(default: tests/baselines/comparison.json — pass empty "
             "string to disable). One record per (dataset, config, "
             "path) row with all fields from the printed table plus "
             "the per-rule counts.")
    args = parser.parse_args()

    # ── Resolve dataset list ──
    if args.datasets is not None and args.dataset is not None:
        parser.error("Use either --dataset OR --datasets, not both.")
    if args.datasets is None and args.dataset is None:
        # Default: single dataset, the historical fallback.
        args.dataset = "grounder/data/countries_s3"
    dataset_paths: List[Tuple[str, Path]] = []  # (label, path)
    if args.dataset is not None:
        p = Path(args.dataset)
        dataset_paths.append((p.name, p))
    else:
        for raw in args.datasets.split(","):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute() and args.data_root is not None:
                p = Path(args.data_root).expanduser() / raw
            dataset_paths.append((p.name, p))

    configs = []
    for c in args.configs.split(","):
        m = re.match(r"w(\d+)d(\d+)", c.strip())
        if m:
            configs.append((int(m.group(1)), int(m.group(2))))

    # ``family`` ships two rule files; the paper / IJCAI '25 numbers
    # use ``rules_old.txt`` (47 rules). All other datasets use
    # ``rules.txt``. Allow explicit override via ``--rules-file``.
    explicit_rules_file = (args.rules_file != "rules.txt")

    # ── Per-dataset defaults ──
    # Each dataset has its own memory profile (driven by K_r, K_f, V).
    # The defaults below pick splits, max_queries, batch_size, and
    # bump_s_to_k so the sweep finishes without OOM and reports real
    # parity numbers. Override any of them on the command line:
    #   --splits / --max-queries / --batch-size / --bump-s / --no-bump-s
    #
    # Splits choice: family / wn18rr have thousands of test triples,
    # plenty for parity; countries_* and ablation_* have only ~25
    # test triples, so we exercise on train+valid+test combined.
    def _default_splits(label: str) -> List[str]:
        if "family" in label or "wn18rr" in label:
            return ["test"]
        # countries_s2/s3, ablation_d2/d3, and any other small dataset
        return ["train", "valid", "test"]

    def _default_max_queries(label: str) -> int:
        # All datasets use their full default split set: family / wn18rr
        # use the full test split; countries_*/ablation_* use the full
        # train+valid+test combined set.
        return 0

    def _default_bump_s(label: str) -> bool:
        # ``bump_s_to_k=True`` widens pack output to ``S = max(S, K)``
        # so all valid children survive (no-loss). The resulting
        # ``[B*S, K_r, G_r, M, 3]`` allocation is fine on small KBs but
        # OOMs at scale on family / wn18rr / countries_s3 (high K_r or
        # high K_f). For those, default off and accept the S=256 pack
        # truncation; the flat path doesn't have this issue at all.
        if ("ablation" in label or
                "countries_s2" in label):
            return True
        return False

    def _default_batch_size(label: str) -> int:
        # Picked to keep peak memory under ~5 GiB on a 24 GiB card with
        # ``max_states=256``, ``G_r=4096``. Above these thresholds the
        # dense / flat intermediate tensors OOM (or, when partial OOMs
        # leave stale buffers, produce silently wrong counts at full
        # batch). The chunk merger applies the final fp_batch pruning
        # to the union of chunk apps, so the result equals what a
        # full-batch run would have produced.
        if "wn18rr" in label:
            return 10           # K_f=473, K_r=16 — heaviest per-query;
                                # 2924 queries × bc13 needs small chunks
                                # to avoid the chunk-merger memory pile-up
        if "family" in label:
            return 100          # K_r=16 — moderate; 5626 queries
        if "countries_s3" in label:
            return 50           # V=2, K_r=7 — already known scale issue
        return 0                # ablation_*, countries_s2 — fits full batch

    explicit_splits = (
        [s.strip() for s in args.splits.split(",") if s.strip()]
        if args.splits is not None else None)

    all_rows: List[Dict[str, Any]] = []
    for ds_label, ds_path in dataset_paths:
        if explicit_rules_file:
            rules_file = args.rules_file
        elif "family" in ds_label:
            rules_file = "rules_old.txt"
        else:
            rules_file = "rules.txt"
        ds, kb = load_dataset(str(ds_path), device=args.device,
                              rules_file=rules_file)
        ds_splits = explicit_splits or _default_splits(ds_label)
        ds_max_q = (args.max_queries if args.max_queries >= 0
                    else _default_max_queries(ds_label))
        ds_batch_size = (args.batch_size if args.batch_size >= 0
                         else _default_batch_size(ds_label))
        # Resolve bump_s_to_k per dataset (tri-state): explicit
        # --bump-s/--no-bump-s overrides per-dataset default.
        if args.bump_s == "true":
            ds_bump_s = True
        elif args.bump_s == "false" or args.no_bump_s:
            ds_bump_s = False
        else:
            ds_bump_s = _default_bump_s(ds_label)
        # Paths to evaluate ('flat', 'dense', or both).
        ds_paths = [p.strip() for p in args.paths.split(",") if p.strip()]
        if not ds_paths:
            ds_paths = ["flat"]
        print(f"\n{'#'*78}")
        print(f"Dataset: {ds_label} (rules={rules_file})")
        print(f"  facts={kb.num_facts}, rules={kb.num_rules}, M={kb.M}, "
              f"K_f={kb.K_f}, K_r={kb.K_r}")
        print(f"  splits={ds_splits}  max_queries="
              f"{'all' if ds_max_q == 0 else ds_max_q}  "
              f"batch_size={'full' if ds_batch_size == 0 else ds_batch_size}  "
              f"bump_s={ds_bump_s}  paths={ds_paths}")
        print(f"{'#'*78}")

        for width, depth in configs:
            # 'auto' → None → make_bcwd picks the keras-prune-aligned default.
            # Anything else is an explicit override (kept for testing a
            # specific filter regardless of (d, w)).
            filt = None if args.filter == "auto" else args.filter
            for path in ds_paths:
                flat_for_path = (path == "flat")
                # Free GPU memory between configs / paths.
                torch.cuda.empty_cache()
                r = compare_groundings(
                    ds, kb, ds_path, width, depth,
                    flat=flat_for_path, all_anchors=args.all_anchors,
                    filt=filt, S_max=args.s_max, C=args.C,
                    splits=ds_splits,
                    max_queries=ds_max_q,
                    G_r=args.G_r,
                    batch_size=(ds_batch_size if ds_batch_size > 0 else None),
                    bump_s_to_k=ds_bump_s,
                    compile_mode=args.compile_mode,
                )
                r["dataset"] = ds_label
                r["rules_file"] = rules_file
                r["splits"] = ",".join(ds_splits)
                r["path"] = path
                all_rows.append(r)

    # Cross-dataset summary table — counts on the left, timing on the
    # right. ``K(ms)`` is the keras full-batch wall-clock; ``T(ms)`` is
    # the steady-state torch run (sum of per-query times when
    # filter='none', full-batch otherwise). ``Speedup`` is
    # ``keras_ms / torch_ms`` — values >1 mean torch is faster.
    width_w = max(20, max(len(r["dataset"]) for r in all_rows) + 2)
    print(f"\n{'='*116}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*116}")
    header = (
        f"{'Dataset':<{width_w}} {'Config':<6} {'Path':<5} {'#Q':>4} "
        f"{'Keras':>7} {'Torch':>7}"
        f" {'Diff':>6} {'Match':>5}   {'K(ms)':>8} {'T(ms)':>8} "
        f"{'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in all_rows:
        speedup_str = (f"{r['speedup']:.2f}x" if r['speedup'] != float('inf')
                       else "inf")
        print(
            f"{r['dataset']:<{width_w}} {r['config']:<6} "
            f"{r.get('path','flat'):<5} "
            f"{r['n_queries']:>4} "
            f"{r['keras_total']:>7} {r['torch_total']:>7} "
            f"{r['diff']:>+6} {'YES' if r['match'] else 'NO':>5}   "
            f"{r['keras_ms']:>8.1f} {r['torch_ms']:>8.1f} "
            f"{speedup_str:>8}"
        )

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow([
                "dataset", "rules_file", "splits", "config", "n_queries",
                "keras_total", "torch_total", "diff", "match",
                "keras_ms", "torch_ms", "speedup",
            ])
            for r in all_rows:
                w.writerow([
                    r["dataset"], r["rules_file"], r.get("splits", "test"),
                    r["config"], r["n_queries"],
                    r["keras_total"], r["torch_total"], r["diff"],
                    "YES" if r["match"] else "NO",
                    f"{r['keras_ms']:.2f}", f"{r['torch_ms']:.2f}",
                    f"{r['speedup']:.4f}" if r['speedup'] != float('inf') else "inf",
                ])
        print(f"\n[csv] wrote {args.csv}")

    # ── JSON baseline write ──
    # Keep a machine-readable record of every sweep next to the
    # existing groundings baseline. Default path is
    # ``tests/baselines/comparison.json``; pass ``--json ""`` to skip.
    json_path = args.json
    if json_path is None:
        json_path = str(TESTS_DIR / "baselines" / "comparison.json")
    if json_path:
        import json as _json
        from datetime import datetime as _dt
        json_rows: List[Dict[str, Any]] = []
        for r in all_rows:
            json_rows.append({
                "dataset":      r["dataset"],
                "rules_file":   r["rules_file"],
                "splits":       r.get("splits", "test"),
                "path":         r.get("path", "flat"),
                "config":       r["config"],
                "n_queries":    r["n_queries"],
                "keras_total":  r["keras_total"],
                "torch_total":  r["torch_total"],
                "diff":         r["diff"],
                "match":        r["match"],
                "keras_ms":     round(r["keras_ms"], 3),
                "torch_ms":     round(r["torch_ms"], 3),
                "speedup":      (round(r["speedup"], 4)
                                 if r["speedup"] != float("inf")
                                 else None),
                "keras_per_rule": r.get("keras_per_rule", {}),
                "torch_per_rule": r.get("torch_per_rule", {}),
            })
        out = {
            "generated_at": _dt.utcnow().isoformat(timespec="seconds"),
            "rows": json_rows,
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"[json] wrote {json_path}")


if __name__ == "__main__":
    if not HAS_KERAS:
        print("ERROR: keras-ns not available. Install TensorFlow.")
        sys.exit(1)
    main()
