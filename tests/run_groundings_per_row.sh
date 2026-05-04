#!/bin/bash
# Run tests/test_groundings.py one (grounder, config) row per
# subprocess so each row sees a clean GPU. Each subprocess merges
# its result into tests/baselines/comparison.json.
#
# Total wall-clock ≈ 12–15 min on a single 24 GiB GPU.
#
# Usage:
#     bash tests/run_comparison_per_row.sh                 # full grid
#     bash tests/run_comparison_per_row.sh --max-queries 25  # quick
set -euo pipefail
cd "$(dirname "$0")/.."

ROWS=(
    "keras-BC:w0d1"
    "keras-BC:w1d2"
    "keras-BC:w1d3"
    "SLD:d1"
    "SLD:d2"
    "SLD:d3"
    "SLD:d4"
    "enum-flat:w0d1"
    "enum-flat:w1d2"
    "enum-flat:w1d3"
    "enum-dense:w0d1"
    "enum-dense:w1d2"
    "enum-dense:w1d3"
    "FC:fp_global"
)

# Reset the baseline at the start; subsequent rows merge.
BASELINE=tests/baselines/comparison.json
echo "{}" > "$BASELINE"

start=$(date +%s)
for row in "${ROWS[@]}"; do
    row_start=$(date +%s)
    echo "──────────────────────────────────────────────────────────────"
    echo "ROW: $row"
    echo "──────────────────────────────────────────────────────────────"
    conda run --no-capture-output -n gpu \
        python tests/test_groundings.py \
            --rows "$row" --merge "$@" \
        || echo "  [row $row failed]"
    echo "  ($(($(date +%s) - row_start))s)"
done

total=$(($(date +%s) - start))
echo
echo "Total wall-clock: ${total}s ($((total / 60))m $((total % 60))s)"
echo "Baseline written: $BASELINE"
