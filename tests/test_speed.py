"""Speed sweep across grounders × datasets — the timing-only
counterpart to ``test_groundings.py`` (which measures counts).

Per cell: 1 warmup + 1 timed call. The compile mode for each grounder
follows :data:`_runners.DEFAULT_COMPILE_MODES`:

    keras-BC      eager (n/a — keras-ns doesn't use torch.compile)
    SLD           torch.compile mode='reduce-overhead'
    enum-flat     eager (flat path uses dynamic shapes; not
                  reduce-overhead-compatible)
    enum-dense    torch.compile mode='reduce-overhead'
    FC fp_global  eager (closure compute + fp_global filter)

Output goes to ``tests/baselines/speed.json``. The default sweep mirrors
``test_groundings.py``'s grid except SLD uses ``d=4`` and enum/keras use
``w=1, d=3`` per the deepest paper config:

    keras-BC:w1d3, SLD:d4, enum-flat:w1d3, enum-dense:w1d3, FC:fp_global

Documented FAILs:
  * family ``SLD:d4`` — depth-4 SLD on family OOMs (~1 GB CUDA temp);
    see ``baselines/comparison.json`` for the same FAIL on the count
    sweep. Override with ``--rows`` if a different SLD depth is wanted.

Usage:

    python tests/test_speed.py                     # full sweep
    python tests/test_speed.py --datasets wn18rr   # one dataset
    python tests/test_speed.py --datasets wn18rr \\
        --rows enum-flat:w1d3,FC:fp_global         # one dataset, two cells
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force TF off the GPU before any keras-ns import (mirrors
# test_groundings.py — same TF dependency landmine).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

TESTS_DIR = Path(__file__).resolve().parent
KERAS_NS_ROOT = Path(os.environ.get(
    "KERAS_NS_ROOT",
    str(Path.home() / "repos" / "keras-ns-swarm" / "main"),
))
if str(KERAS_NS_ROOT) not in sys.path:
    sys.path.insert(0, str(KERAS_NS_ROOT))

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from grounder.data.loader import KGDataset
from grounder.bc.bc import BCGrounder

from _runners import (
    DEFAULT_COMPILE_MODES, HAS_KERAS,
    build_keras_grounder, build_torch_grounder,
)
from profile_speed import time_grounder, time_runner


# ── Default sweep grid ────────────────────────────────────────────────


DEFAULT_DATASETS: List[str] = [
    "ablation_d2", "ablation_d3",
    "countries_s2", "countries_s3",
    "family", "wn18rr",
]

# (kind, config_label, cfg). One row per kind; this is the deepest
# paper config per family. Override per-cell via ``--rows``.
DEFAULT_ROWS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("keras-BC",    "w1d3", dict(w=1, d=3)),
    ("SLD",         "d4",   dict(depth=4)),
    ("enum-flat",   "w1d3", dict(w=1, d=3, flat=True)),
    ("enum-dense",  "w1d3", dict(w=1, d=3, flat=False)),
    ("FC",          "fp_global", dict()),
]


def _rules_file_for(label: str) -> str:
    return "rules_old.txt" if "family" in label else "rules.txt"


def _batch_size_for(label: str) -> int:
    if "wn18rr" in label:
        return 10
    if "family" in label:
        return 100
    if "countries_s3" in label:
        return 50
    return 0


# ── Cell runners ──────────────────────────────────────────────────────


def _sync(device) -> None:
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()
    elif hasattr(device, "type") and device.type == "cuda":
        torch.cuda.synchronize()


def _reset_gpu_state() -> None:
    """Aggressive between-cell cleanup so reduce-overhead-compiled
    grounders can coexist in one process.

    ``torch.cuda.empty_cache()`` alone doesn't cut it: dynamo's compile
    cache holds strong refs to captured CUDA graphs, so the underlying
    memory blocks remain reserved across grounder invocations. Calling
    :func:`torch._dynamo.reset` clears those caches, after which Python
    GC + ``empty_cache`` actually returns the memory to the driver.
    """
    gc.collect()
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# Order grounders within a dataset to minimise inter-cell GPU pressure.
# Smaller integer = run earlier. reduce-overhead cells go first so they
# capture CUDA graphs against a clean GPU; eager torch cells follow;
# keras-BC (CPU-only) and FC (closure rebuild during construction) run
# last where the GPU's accumulated state can be most fragmented without
# affecting them.
_KIND_ORDER = {
    "enum-dense":  0,   # reduce-overhead — wants pristine GPU
    "SLD":         1,   # reduce-overhead — wants pristine GPU
    "enum-flat":   2,   # eager torch
    "FC":          3,   # eager + builds closure during __init__
    "keras-BC":    4,   # CPU-only — order indifferent
}


def _time_torch(kb, queries, qmask, kind, cfg, bs, *,
                n_warmup, n_timed) -> Dict[str, Any]:
    """Time SLD / enum-flat / enum-dense via the shared builder.

    Compile mode comes from :data:`DEFAULT_COMPILE_MODES`. BCGrounder
    picks the right inner / outer compile path itself (see bc.py
    ``_uses_outer_compile``); both paths use the same chunked forward
    machinery, replaying the captured CUDA graph against padded
    chunks of size ``batch_size``.

    Collection (``evidence`` / ``rule_groundings``) is disabled —
    the speed sweep only times the forward.
    """
    compile_mode = DEFAULT_COMPILE_MODES.get(kind)
    g = build_torch_grounder(
        kind, kb, cfg, compile_mode=compile_mode,
        collect_evidence=False, collect_rule_groundings=False)
    fwd_kwargs = {"batch_size": bs} if bs > 0 else {}
    res = time_grounder(g, queries, qmask, fwd_kwargs,
                        n_warmup=n_warmup, n_timed=n_timed)
    res["compile_mode_used"] = compile_mode
    return res


def _time_keras(ds, ds_path, queries, cfg, *,
                n_warmup, n_timed) -> Dict[str, Any]:
    """Time keras-ns. keras-ns invokes ``kg.ground(facts, test_tuples)``
    rather than a torch forward, so we wrap in ``time_runner``."""
    test_tuples = [
        (ds.idx2pred[queries[i, 0].item()],
         ds.idx2entity[queries[i, 1].item()],
         ds.idx2entity[queries[i, 2].item()])
        for i in range(queries.shape[0])
    ]
    fact_tuples = list(ds._facts_raw)
    kg, _rules = build_keras_grounder(ds, ds_path, cfg["w"], cfg["d"])

    def _call():
        kg.ground(fact_tuples, test_tuples)

    # keras runs on CPU; no CUDA sync needed.
    return time_runner(_call, n_warmup=n_warmup, n_timed=n_timed)


def _time_fc(kb, queries, qmask, *, n_warmup, n_timed) -> Dict[str, Any]:
    """Time the FC fp_global path. Mirrors ``test_groundings.run_fc``'s
    setup but only measures the timed forward call (closure built
    during BCGrounder construction is not counted).

    ``fc_depth=3`` caps the SpMM closure depth: at ``fc_depth=10``
    (the BCGrounder default) wn18rr's closure compute exceeds 24 GB
    on a single GPU because ``cusparseSpGEMM_workEstimation`` blows
    up at deep iterations. ``test_groundings`` handles this via a
    closure-only fallback after BC enumeration OOMs; the speed sweep
    avoids it upfront. d=3 is the closure size other grounders see
    in practice.
    """
    bs = _batch_size_for_kb(kb)
    g = BCGrounder(
        kb, resolution="enum", width=None, depth=1,
        filter="fp_global",
        max_groundings_per_query=4096,
        max_total_groundings=4096,
        max_states=256,
        fc_method="spmm", fc_depth=3,
        prune_facts=True,
        flat_intermediate=True, bump_s_to_k=False,
        init_state_shape="minimal",
        collect_evidence=False,
        collect_rule_groundings=False,
        # FC: eager per DEFAULT_COMPILE_MODES.
    )
    fwd_kwargs = {"batch_size": bs} if bs > 0 else {}
    return time_grounder(
        g, queries, qmask, fwd_kwargs,
        n_warmup=n_warmup, n_timed=n_timed)


def _batch_size_for_kb(kb) -> int:
    """Chunk size for FC's BC pass — keep per-step memory bounded."""
    K_f = kb.K_f if hasattr(kb, "K_f") else 0
    if K_f >= 400:
        return 10
    if K_f >= 100:
        return 50
    return 0


def run_cell(kind: str, ds: KGDataset, ds_path: Path, kb,
             queries: torch.Tensor, qmask: torch.Tensor,
             cfg: Dict[str, Any], bs: int, *,
             n_warmup: int = 1,
             n_timed: int = 1) -> Dict[str, Any]:
    """Dispatch a (grounder, dataset) cell to its timer.

    Returns the dict from ``time_runner`` / ``time_grounder``, or a
    dict with ``ok=False`` and ``error`` on failure.
    """
    try:
        if kind == "keras-BC":
            if not HAS_KERAS:
                return {"ok": False, "error": "keras-ns unavailable"}
            res = _time_keras(ds, ds_path, queries, cfg,
                              n_warmup=n_warmup, n_timed=n_timed)
        elif kind in ("SLD", "enum-flat", "enum-dense"):
            res = _time_torch(kb, queries, qmask, kind, cfg, bs,
                              n_warmup=n_warmup, n_timed=n_timed)
        elif kind == "FC":
            res = _time_fc(kb, queries, qmask,
                           n_warmup=n_warmup, n_timed=n_timed)
        else:
            return {"ok": False, "error": f"unknown kind {kind!r}"}
        res["ok"] = True
        res["error"] = None
        return res
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


# ── Driver ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="Comma-separated dataset names.")
    parser.add_argument("--data-root",
                        default=str(Path.home() / "repos" / "data-swarm"
                                    / "main"))
    parser.add_argument("--rows", default=None,
                        help="Comma-separated 'kind:config' filters.")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Truncate test split to first N queries.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-timed", type=int, default=1)
    parser.add_argument("--output", default=None,
                        help="Write the JSON baseline to this path. "
                             "Default: tests/baselines/speed.json. "
                             "Pass empty string to skip.")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    rows_to_run: List[Tuple[str, str, Dict[str, Any]]] = DEFAULT_ROWS
    if args.rows:
        wanted = {r.strip() for r in args.rows.split(",") if r.strip()}
        rows_to_run = [r for r in DEFAULT_ROWS
                       if f"{r[0]}:{r[1]}" in wanted]

    cells: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ds_name in dataset_names:
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
        if args.max_queries > 0:
            queries = queries[:args.max_queries]
        qmask = torch.ones(queries.shape[0], dtype=torch.bool,
                           device=queries.device)
        bs = _batch_size_for(ds_name)
        n_q = queries.shape[0]
        print(f"\n### {ds_name}  rules={rules_file}  "
              f"facts={kb.num_facts}  queries={n_q}  "
              f"batch_size={bs or 'full'}")
        cells[ds_name] = {"_n_queries": n_q}

        # Order grounders by ``_KIND_ORDER``: reduce-overhead first so
        # they capture CUDA graphs against a clean GPU; eager torch
        # grounders second; FC (closure rebuild) and keras-BC (CPU)
        # last where allocator fragmentation matters less.
        ordered = sorted(rows_to_run,
                         key=lambda r: _KIND_ORDER.get(r[0], 99))
        for kind, cfg_label, cfg in ordered:
            _reset_gpu_state()
            res = run_cell(
                kind, ds, ds_path, kb, queries, qmask, cfg, bs,
                n_warmup=args.n_warmup, n_timed=args.n_timed)
            tag = f"{kind}:{cfg_label}"
            cells[ds_name][tag] = res
            if res.get("ok"):
                ms = res["ms_median"]
                if args.n_timed > 1:
                    spread = f"  (min={res['ms_min']:.1f}, " \
                             f"max={res['ms_max']:.1f})"
                else:
                    spread = ""
                print(f"  {tag:<22s} {ms:>9.1f}ms{spread}")
            else:
                print(f"  {tag:<22s} FAIL ({res.get('error')})")

    # ── Print the timing table ──
    row_keys = [f"{k}:{c}" for (k, c, _) in rows_to_run]
    col_keys = dataset_names
    width = max(22, max((len(r) for r in row_keys), default=0) + 2)
    cw = 12

    print(f"\n{'='*(width + cw*len(col_keys) + 4)}")
    print("TIMING (ms_median; 1 warmup + N timed; "
          "compile_mode per kind, see _runners.DEFAULT_COMPILE_MODES)")
    print(f"{'='*(width + cw*len(col_keys) + 4)}")
    header = f"{'Grounder:Config':<{width}}  " + "".join(
        f"{ds[:cw-1]:>{cw}}" for ds in col_keys)
    print(header)
    print("-" * len(header))
    for rk in row_keys:
        row = f"{rk:<{width}}  "
        for ds in col_keys:
            c = cells.get(ds, {}).get(rk, {})
            if not c:
                v = "—"
            elif c.get("ok"):
                v = f"{c['ms_median']:.0f}"
            else:
                v = "FAIL"
            row += f"{v:>{cw}}"
        print(row)

    # ── Write baseline JSON ──
    out_path = args.output
    if out_path is None:
        out_path = str(TESTS_DIR / "baselines" / "speed.json")
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
        all_row_tags = list({tag for ds_d in merged_cells.values()
                             for tag in ds_d.keys() if tag != "_n_queries"})
        canonical = [f"{k}:{c}" for (k, c, _) in DEFAULT_ROWS]
        ordered_rows = [r for r in canonical if r in all_row_tags] + [
            r for r in all_row_tags if r not in canonical]
        baseline = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datasets": list(merged_cells.keys()),
            "rows": ordered_rows,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "max_queries": args.max_queries,
            "device": args.device,
            "compile_modes": {k: v for k, v in DEFAULT_COMPILE_MODES.items()},
            "cells": merged_cells,
        }
        with open(out_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"\n[json] wrote {out_path}")

    # Surface cell failures via process exit code so the precommit
    # driver (or any caller) can detect them. Cells that didn't run
    # (filtered out) don't count.
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
