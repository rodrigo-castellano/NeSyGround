"""Compare grounding counts between keras-ns and any torch-ns grounder.

Per-query comparison for filter='none', full-batch for filter='fp_batch'.
Supports any dataset with domain2constants.txt and arrow-format rules.

Usage:
    cd torch-ns
    PYTHONPATH=. python -m pytest grounder/tests/test_keras_grounding_comparison.py -v -s
    PYTHONPATH=. python grounder/tests/test_keras_grounding_comparison.py  # standalone
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch


def _sync(device) -> None:
    """Synchronize CUDA stream so wall-clock timings are accurate."""
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize()
    elif isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()

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
) -> BCGrounder:
    """Build the paper's BC_{w,d} grounder via ``make_bcwd``.

    ``filt=None`` lets ``make_bcwd`` pick the keras-prune-aligned
    default (``'none'`` for d=1,w>0; ``'fp_batch'`` otherwise).
    Pass an explicit ``filt`` to test a specific filter.

    ``bump_s_to_k=True`` (default) keeps the no-loss ``S = max(S_max, K)``
    bump at depth>1, width>0 — slow on high-fan-out KBs.
    ``False`` keeps ``S = S_max`` (fast, accepts truncation).
    """
    return make_bcwd(
        kb, w=width, d=depth,
        flat_intermediate=flat,
        filter=filt,
        max_groundings_per_query=G_r,
        max_total_groundings=C,
        max_states=S_max,
        fc_method="join", prune_facts=True,
        bump_s_to_k=bump_s_to_k,
    )


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
    verbose: bool = True,
    max_queries: int = 0,
    G_r: int = 4096,
    batch_size: Optional[int] = None,
    bump_s_to_k: bool = True,
) -> dict:
    """Compare keras-ns vs torch-ns grounding counts.

    Returns dict with per-query counts and summary.
    """
    test = ds.get_queries(split)
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
                              bump_s_to_k=bump_s_to_k)

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
        except (RuntimeError, MemoryError) as exc:
            # Per-query fallback (slower but bounded memory).
            print(f"  [batched OOM: {type(exc).__name__}; falling back "
                  f"to per-query]", flush=True)
            with torch.no_grad():
                _ = g(test[:1], qmask[:1])  # warmup at B=1
            _sync(test.device)
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
    parser.add_argument("--dataset", default="grounder/data/countries_s3")
    parser.add_argument("--configs", default="w0d1,w1d2,w1d3",
                        help="Comma-separated w<W>d<D> configs")
    parser.add_argument("--flat", action="store_true", default=True)
    parser.add_argument("--no-flat", dest="flat", action="store_false")
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
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Truncate test split to first N queries (0 = all).")
    parser.add_argument("--G-r", type=int, default=4096,
                        help="max_groundings_per_query in BCGrounder.")
    parser.add_argument("--rules-file", default="rules.txt",
                        help="Rules filename inside dataset dir. Paper "
                             "numbers for family use 'rules_old.txt' "
                             "(47 rules vs 143 in rules.txt).")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="BCGrounder.forward(batch_size=...) chunk "
                             "size. 0 = full batch in one forward.")
    parser.add_argument("--no-bump-s", action="store_true",
                        help="Disable the ``S = max(S_max, K)`` bump for "
                             "depth>1, width>0. Default keeps the bump "
                             "for no-loss parity. Disable for speed on "
                             "high-fan-out KBs (wn18rr, family at d=3).")
    args = parser.parse_args()

    data_dir = Path(args.dataset)
    ds, kb = load_dataset(str(data_dir), device=args.device,
                          rules_file=args.rules_file)

    print(f"Dataset: {data_dir.name}")
    print(f"  facts={kb.num_facts}, rules={kb.num_rules}, M={kb.M}, "
          f"K_f={kb.K_f}, K_r={kb.K_r}")

    configs = []
    for c in args.configs.split(","):
        m = re.match(r"w(\d+)d(\d+)", c.strip())
        if m:
            configs.append((int(m.group(1)), int(m.group(2))))

    results = []
    for width, depth in configs:
        # 'auto' → None → make_bcwd picks the keras-prune-aligned default.
        # Anything else is an explicit override (kept for testing a
        # specific filter regardless of (d, w)).
        filt = None if args.filter == "auto" else args.filter
        r = compare_groundings(
            ds, kb, data_dir, width, depth,
            flat=args.flat, all_anchors=args.all_anchors,
            filt=filt, S_max=args.s_max, C=args.C,
            max_queries=args.max_queries,
            G_r=args.G_r,
            batch_size=(args.batch_size if args.batch_size > 0 else None),
            bump_s_to_k=not args.no_bump_s,
        )
        results.append(r)

    # Summary table — counts on the left, timing on the right.
    # ``K(ms)`` is the keras full-batch wall-clock; ``T(ms)`` is the
    # steady-state torch run (sum of per-query times when filter='none',
    # full-batch otherwise). Speedup is ``keras / torch`` — values >1
    # mean torch is faster.
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    print(f"{'Config':<8} {'Keras':>8} {'Torch':>8} {'Diff':>6} {'Match':>5}"
          f"   {'K(ms)':>8} {'T(ms)':>8} {'Speedup':>8}")
    print("-" * 78)
    for r in results:
        speedup_str = (f"{r['speedup']:.2f}x" if r['speedup'] != float('inf')
                       else "inf")
        print(f"{r['config']:<8} {r['keras_total']:>8} {r['torch_total']:>8} "
              f"{r['diff']:>+6} {'YES' if r['match'] else 'NO':>5}   "
              f"{r['keras_ms']:>8.1f} {r['torch_ms']:>8.1f} "
              f"{speedup_str:>8}")


if __name__ == "__main__":
    if not HAS_KERAS:
        print("ERROR: keras-ns not available. Install TensorFlow.")
        sys.exit(1)
    main()
