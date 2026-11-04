#!/bin/bash
# Re-creates the final submission (LB 0.9357) from the saved model outputs in model/runs_frozen/ and checks it
# byte-for-byte (sha256) against submission/final_submission_0.9357.csv.  CPU only, about 3-8 minutes.
#
# Decision layer (src/koaltq2/x8_blend.py + src/koaltq2/harness.py):
#   1) log-linear blend: 0.6 x (0.9 log P[x17 stacker] + 0.1 log P[x15 judge LR]) + 0.4 x mean log P[27B, 9B LoRA]
#   2) class quotas on the rows not fixed by rules (derived test class distribution)
#   3) harness H0-H5 (public 109 rows, guideline rules, template twins, sole-link constraint, structure twins,
#      audit-agreement veto) -> outputs/harness_log.jsonl records every rule decision with its evidence
# usage: bash scripts/reproduce_best.sh [out_csv]        (PYTHON=... to choose the interpreter)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-$ROOT/outputs/reproduced_submission.csv}
PY=${PYTHON:-python3}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -R "$ROOT/model/runs_frozen/." "$WORK/"
# the stacker's TEST probabilities come from x17fx: the same LightGBM recipe trained after the rot1/rot6 judge runs
# existed (378 features vs 364) on train + 109 public rows + 133 rule-fixed test rows (R2 116, R4' 17); its out-of-fold
# train probabilities (used to choose the blend rule and weight) stay those of x17 (README 4절, 5-1)
cp "$WORK/x17_stack_lgbm_b109fx/test.npy" "$WORK/x17_stack_lgbm_b109/test.npy"
rm -rf "$WORK/x17_stack_lgbm_b109fx" "$WORK/x6_qwen38_27b_fx_lora_full" "$WORK/x6_qwen38_27b_fx3_lora_full"
cd "$ROOT"
KOALTQ_HARNESS=1 KOALTQ_BUNDLE="$ROOT/data/koaltq_bundle" KOALTQ_RUNS="$WORK" PYTHONPATH="$ROOT/src" \
  "$PY" -m koaltq2.run x8 --objective lbq --wlora 0.4
mkdir -p "$(dirname "$OUT")"
cp "$WORK/submissions/S3_ensemble_quota_l0_lbq.csv" "$OUT"
cp "$WORK/x8_blend_lbq/harness_log.jsonl" "$WORK/x8_blend_lbq/harness_summary.json" "$(dirname "$OUT")/"
cp "$WORK/x8_blend_lbq/harness_in_pred.npy" "$WORK/x8_blend_lbq/harness_in_fixed.npy" "$WORK/x8_blend_lbq/harness_in_logits.npy" \
   "$WORK/x8_blend_lbq/test_logits_final.npy" "$WORK/x8_blend_lbq/honest_preds.npz" "$WORK/x8_blend_lbq/report.json" "$(dirname "$OUT")/"
EXPECT=$(cut -d' ' -f1 "$ROOT/submission/final_submission_0.9357.csv.sha256")
GOT=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$OUT")
echo "reproduced: $OUT"
echo "sha256 $GOT"
if [ "$GOT" = "$EXPECT" ]; then echo "OK - identical to submission/final_submission_0.9357.csv"; else echo "MISMATCH (expected $EXPECT)"; exit 1; fi
