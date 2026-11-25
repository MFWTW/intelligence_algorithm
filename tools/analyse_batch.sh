#!/usr/bin/env bash
# Post-process a batch of round directories: structured analysis, per-object
# split, dwell episodes, dynamic-obstacle clearance, and the figures.
#
#   tools/analyse_batch.sh runs/2026-11-25_1113_rule1 runs/..._rule2 runs/..._rule3
set -uo pipefail
ROOT=/home/xu/intelligence_algorithm-main
DIRS=("$@")
[ ${#DIRS[@]} -gt 0 ] || { echo "usage: analyse_batch.sh <dir> [dir ...]"; exit 1; }

for d in "${DIRS[@]}"; do
    echo "== $d"
    python3 "$ROOT/tools/round_analysis.py" "$d" > "$d/round_analysis.txt" 2>&1
    echo "   round_analysis rc=$?"
    python3 "$ROOT/tools/per_object_timings.py" --dir "$d" > "$d/timings.txt" 2>&1
    python3 "$ROOT/tools/per_object_timings.py" --dir "$d" --json > "$d/objects.json" 2>&1
    echo "   objects: $(python3 -c "import json,sys;print(len(json.load(open('$d/objects.json'))))" 2>/dev/null)"
done

mkdir -p "$ROOT/analysis"
echo "== dwell episodes"
python3 "$ROOT/tools/dwell_report.py" "${DIRS[@]/#/--dir }" --csv "$ROOT/analysis/dwells.csv" 2>&1 | tail -40
echo "== dynamic obstacle clearance"
python3 "$ROOT/tools/mover_proximity.py" "${DIRS[@]/#/--dir }" --csv "$ROOT/analysis/proximity.csv" 2>&1 | tail -40
echo "== figures"
python3 "$ROOT/tools/plot_round_analysis.py" "${DIRS[@]/#/--dir }" --out "$ROOT/analysis" 2>&1 | tail -10
echo "== full report"
python3 "$ROOT/tools/build_report.py" "${DIRS[@]}" \
    --out "$ROOT/runs/REPORT_3ROUNDS_$(date +%Y-%m-%d).md" 2>&1 | tail -5
