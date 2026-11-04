#!/bin/bash
# Harness experiments on top of the reproduced decision layer (run scripts/reproduce_best.sh first).
#   - H6 signature vote (teammate's specification: >= 2 public neighbours, 100% agreement, model confidence <= 0.9)
#     with the official 데이터셋_라벨 (LB-tested: 0.93) and with 재판정_라벨 (8 rows -> 중복형 count above 35)
#   - validation harness: train-side simulation of the twin rules + conformal prediction sets (koaltq2.harness_val)
# usage: bash scripts/run_harness_experiments.sh [outputs_dir]       (PYTHON=... to choose the interpreter)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTD=${1:-$ROOT/outputs}
PY=${PYTHON:-python3}
test -f "$OUTD/harness_in_pred.npy" || { echo "run scripts/reproduce_best.sh first (needs $OUTD/harness_in_*.npy)"; exit 1; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -R "$ROOT/model/runs_frozen/." "$WORK/"
export KOALTQ_BUNDLE="$ROOT/data/koaltq_bundle" KOALTQ_RUNS="$WORK" PYTHONPATH="$ROOT/src"
cd "$ROOT"
"$PY" -m koaltq2.run harness "$OUTD" --s6 --gate 0.9 --label dataset --tag _H6_dataset
"$PY" -m koaltq2.run harness "$OUTD" --s6 --gate 0.9 --label rejudge --tag _H6_rejudge
"$PY" -m koaltq2.run harness "$OUTD" --s6 --gate 1.0 --label rejudge --tag _H6_rejudge_nogate
cp "$WORK"/submissions/HARNESS_*.csv "$OUTD/"
"$PY" -m koaltq2.run hval "$OUTD" --reps 30
"$PY" - "$ROOT/submission/final_submission_0.9357.csv" "$OUTD" <<'PY'
import collections, csv, glob, sys
rd = lambda p: {r["record_id"]: r["label"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
best = rd(sys.argv[1])
for p in sorted(glob.glob(sys.argv[2] + "/HARNESS_*.csv")):
    x = rd(p)
    ch = collections.Counter(f"{best[k]}->{x[k]}" for k in best if best[k] != x[k])
    print(f"{p.rsplit('/', 1)[-1]}: {sum(ch.values())} rows differ from the final submission {dict(ch)}")
PY
