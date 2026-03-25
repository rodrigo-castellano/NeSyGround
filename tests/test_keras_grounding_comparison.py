"""Compare grounding counts between keras-ns and any torch-ns grounder.

Per-query comparison for filter='none', full-batch for filter='fp_batch'.
Supports any dataset with domain2constants.txt and arrow-format rules.

Usage:
    cd torch-ns
    PYTHONPATH=. python -m pytest grounder/tests/test_keras_grounding_comparison.py -v -s
    PYTHONPATH=. python grounder/tests/test_keras_grounding_comparison.py  # standalone
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

TESTS_DIR = Path(__file__).resolve().parent
GROUNDER_ROOT = TESTS_DIR.parent
TORCH_NS_ROOT = GROUNDER_ROOT.parent
KERAS_NS_ROOT = TORCH_NS_ROOT.parent / "keras-ns"

if str(KERAS_NS_ROOT) not in sys.path:
    sys.path.insert(0, str(KERAS_NS_ROOT))

from grounder.data.loader import KGDataset
from grounder.bc.bc import BCGrounder

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

def load_dataset(data_dir: str, device: str = "cpu"):
    """Load a KGDataset and build KB with large caps."""
    ds = KGDataset(data_dir, device=device)
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

    # Map (width, depth) to keras params
    # width=W at step i<D-1, width=0 at last step
    if depth == 1:
        max_unk = width
        max_unk_last = width
        prune = (width == 0)
    else:
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
    filt: str = "none", S_max: int = 256, C: int = 4096,
) -> BCGrounder:
    """Build a torch-ns BCGrounder for enum resolution."""
    return BCGrounder(
        kb, resolution="enum", filter=filt,
        depth=depth, width=width,
        max_groundings_per_query=4096,
        max_total_groundings=C,
        max_states=S_max,
        fc_method="join", prune_facts=True,
        flat_intermediate=flat, all_anchors=all_anchors,
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
) -> dict:
    """Compare keras-ns vs torch-ns grounding counts.

    Returns dict with per-query counts and summary.
    """
    test = ds.get_queries(split)
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
    kg.ground(fact_tuples, test_tuples)
    keras_total = sum(len(v) for v in kg.rule2groundings.values())

    # Per-rule keras counts
    keras_per_rule = {r.name: len(kg.rule2groundings[r.name]) for r in keras_rules}

    # Per-query keras counts (re-run per query for per-query comparison)
    keras_per_query = []
    if filt == "none":
        for i in range(B):
            kg_i, _ = build_keras_grounder(ds, data_dir, width, depth)
            kg_i.ground(fact_tuples, [test_tuples[i]])
            cnt = sum(len(v) for v in kg_i.rule2groundings.values())
            keras_per_query.append(cnt)
    else:
        # Batched filter: can't decompose per-query
        keras_per_query = [keras_total]  # single batch entry

    # ── Torch ──
    g = build_torch_grounder(kb, width, depth, flat=flat,
                              all_anchors=all_anchors, filt=filt,
                              S_max=S_max, C=C)

    if filt == "none":
        # Per-query: run each query individually
        torch_per_query = []
        torch_per_rule = {r.name: 0 for r in keras_rules}
        # Per-depth-per-rule counts (structured evidence)
        torch_per_depth_rule: Dict[int, Dict[str, int]] = {}
        for i in range(B):
            q = test[i:i+1]
            qm = torch.ones(1, dtype=torch.bool, device=q.device)
            with torch.no_grad():
                out = g(q, qm)
            ev = out.evidence
            cnt = int(ev.mask.sum().item())
            torch_per_query.append(cnt)
            # Top-level rule (depth 0)
            ridx_top = ev.rule_idx_top  # [1, C]
            for ri, r in enumerate(keras_rules):
                torch_per_rule[r.name] += int(
                    ((ridx_top == ri) & ev.mask).sum().item())
            # Per-depth rule counts (if structured)
            if ev.D > 0:
                for d in range(ev.D):
                    if d not in torch_per_depth_rule:
                        torch_per_depth_rule[d] = {r.name: 0 for r in keras_rules}
                    ridx_d = ev.rule_idx[:, :, d]  # [1, C]
                    for ri, r in enumerate(keras_rules):
                        torch_per_depth_rule[d][r.name] += int(
                            ((ridx_d == ri) & ev.mask).sum().item())
        torch_total = sum(torch_per_query)
    else:
        # Batched
        with torch.no_grad():
            out = g(test, qmask)
        ev = out.evidence
        torch_total = int(ev.mask.sum().item())
        torch_per_query = [torch_total]
        torch_per_rule = {}
        torch_per_depth_rule = {}
        ridx_top = ev.rule_idx_top
        for ri, r in enumerate(keras_rules):
            torch_per_rule[r.name] = int(
                ((ridx_top == ri) & ev.mask).sum().item())
        if ev.D > 0:
            for d in range(ev.D):
                torch_per_depth_rule[d] = {}
                ridx_d = ev.rule_idx[:, :, d]
                for ri, r in enumerate(keras_rules):
                    torch_per_depth_rule[d][r.name] = int(
                        ((ridx_d == ri) & ev.mask).sum().item())

    # ── Report ──
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
    parser.add_argument("--filter", default="none")
    parser.add_argument("--s-max", type=int, default=256)
    parser.add_argument("--C", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.dataset)
    ds, kb = load_dataset(str(data_dir), device=args.device)

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
        filt = "fp_batch" if width == 0 and depth == 1 else args.filter
        r = compare_groundings(
            ds, kb, data_dir, width, depth,
            flat=args.flat, all_anchors=args.all_anchors,
            filt=filt, S_max=args.s_max, C=args.C,
        )
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<10} {'Keras':>8} {'Torch':>8} {'Diff':>8} {'Match':>6}")
    print("-" * 42)
    for r in results:
        print(f"{r['config']:<10} {r['keras_total']:>8} {r['torch_total']:>8} "
              f"{r['diff']:>+8} {'YES' if r['match'] else 'NO':>6}")


if __name__ == "__main__":
    if not HAS_KERAS:
        print("ERROR: keras-ns not available. Install TensorFlow.")
        sys.exit(1)
    main()
