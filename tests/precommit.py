"""Combined precommit suite: counts + speeds, narrowed to the smallest
failure-detecting grid.

Runs both ``test_groundings.py`` (count metrics, correctness) and
``test_speed.py`` (wall-clock per cell) on a single dataset / row set
that's wide enough to exercise every grounder family but small enough
to finish in a couple of minutes:

    dataset:   wn18rr
    rows:      keras-BC:w1d3, SLD:d4, enum-flat:w1d3,
               enum-dense:w1d3, FC:fp_global

If any cell fails or any number drifts more than the documented
tolerance, the precommit summary makes it visible. The full sweeps
(all 6+1 datasets, every config) live in ``test_groundings.py`` and
``test_speed.py`` themselves; run those manually for a release-grade
baseline refresh.

Usage:

    python tests/precommit.py
    python tests/precommit.py --skip-counts        # speed only
    python tests/precommit.py --skip-speed         # counts only
    python tests/precommit.py --datasets wn18rr,family   # override
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
GROUNDER_ROOT = TESTS_DIR.parent

# Single dataset + ROW SET narrow enough to run in ~2 min total.
PRECOMMIT_DATASETS = "wn18rr"
PRECOMMIT_ROWS = (
    "keras-BC:w1d3,SLD:d4,enum-flat:w1d3,enum-dense:w1d3,FC:fp_global"
)


def _run(label: str, cmd: list, dry_run: bool) -> int:
    print(f"\n{'='*70}")
    print(f"# {label}")
    print(f"# {' '.join(cmd)}")
    print(f"{'='*70}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(GROUNDER_ROOT))
    if proc.returncode != 0:
        print(f"\n!!! {label} FAILED (exit {proc.returncode}) !!!")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=PRECOMMIT_DATASETS,
                        help=f"Default: {PRECOMMIT_DATASETS}")
    parser.add_argument("--rows", default=PRECOMMIT_ROWS,
                        help=f"Default: {PRECOMMIT_ROWS}")
    parser.add_argument("--skip-counts", action="store_true",
                        help="Skip the count-comparison sweep.")
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip the speed sweep.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Truncate test split to first N queries "
                             "(0 = full split).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--no-merge", action="store_true",
                        help="Overwrite the baseline JSONs instead of "
                             "merging into the existing ones (default "
                             "is --merge so precommit doesn't wipe out "
                             "cells for datasets/rows it doesn't run).")
    args = parser.parse_args()

    common = [
        "--datasets", args.datasets,
        "--rows", args.rows,
        "--device", args.device,
    ]
    if args.max_queries > 0:
        common += ["--max-queries", str(args.max_queries)]
    if not args.no_merge:
        common += ["--merge"]

    fail = 0

    if not args.skip_counts:
        rc = _run(
            "COUNTS — test_groundings.py (correctness; counts table)",
            [sys.executable, "-u",
             str(TESTS_DIR / "test_groundings.py")] + common,
            args.dry_run,
        )
        if rc != 0:
            fail += 1

    if not args.skip_speed:
        # Speed sweep is run **one cell per subprocess** so each grounder
        # sees a clean GPU. Necessary because reduce-overhead-compiled
        # cells (SLD, enum-dense) capture multi-GB CUDA graphs that
        # don't share well with the next cell's allocations even after
        # ``torch._dynamo.reset()``. Subprocess startup is ~1.5s, which
        # is a small premium for guaranteed isolation. Always pass
        # ``--merge`` so each subprocess only contributes its row to
        # ``baselines/speed.json`` without wiping the rest.
        cells = [r.strip() for r in args.rows.split(",") if r.strip()]
        speed_common = [
            "--datasets", args.datasets,
            "--device", args.device,
            "--merge",
        ]
        if args.max_queries > 0:
            speed_common += ["--max-queries", str(args.max_queries)]
        for cell in cells:
            rc = _run(
                f"SPEED — test_speed.py [{cell}]",
                [sys.executable, "-u",
                 str(TESTS_DIR / "test_speed.py"),
                 "--rows", cell] + speed_common,
                args.dry_run,
            )
            if rc != 0:
                fail += 1

    total = 0
    if not args.skip_counts:
        total += 1
    if not args.skip_speed:
        total += len([r.strip() for r in args.rows.split(",") if r.strip()])
    print(f"\n{'='*70}")
    if fail:
        print(f"PRECOMMIT FAILED ({fail} / {total} cells)")
        sys.exit(1)
    print(f"PRECOMMIT OK ({total} cells)")


if __name__ == "__main__":
    main()
