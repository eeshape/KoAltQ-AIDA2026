#!/bin/bash
# 데모 영상용 실행 스크립트: 저장된 모델 출력 → 최종 결정 단계(결합 → 할당량 → 하네스) → 제출 CSV 생성 → 결과 확인
#   1) scripts/reproduce_best.sh  : model/runs_frozen/의 모델 출력으로 결정 단계를 실행해 CSV를 만들고 sha256을 대조
#   2) 생성된 CSV의 앞부분과 라벨별 개수 출력
#   3) scripts/make_result_viewer.py : 이미지·alt·판정 라벨·하네스 근거를 보는 HTML 뷰어 생성
#   4) CSV와 뷰어를 기본 앱으로 열기 (DEMO_NO_OPEN=1이면 열지 않음)
#      DEMO_TOUR=1: 뷰어를 자동 시연 모드로 만들고, CSV를 몇 초 보여 준 뒤 뷰어를 연다
#                   (macOS에 Chrome이 있으면 주소창·북마크 없는 앱 창으로 열고, DEMO_WINDOW=가로,세로 로 창 크기 지정)
# 학습·GPU·모델 다운로드는 필요 없음 (CPU 약 3~8분). usage: [PYTHON=...] bash demo/run_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PYTHON:-python3}
OUT="$ROOT/outputs/reproduced_submission.csv"
cd "$ROOT"

echo "=================================================================="
echo " KoAltQ 데모: 저장된 모델 출력 → 결정 단계 → 제출 CSV"
echo "=================================================================="
echo "[1/3] 결정 단계 실행 (CPU, 학습 없음)  $(date '+%H:%M:%S')"
T0=$(date +%s)
PYTHON="$PY" bash scripts/reproduce_best.sh "$OUT"
echo "      소요 $(( $(date +%s) - T0 ))초"

echo
echo "[2/3] 생성된 제출 파일: ${OUT#$ROOT/}"
ls -l "$OUT"
echo "--- 앞 10행 ---"
head -n 11 "$OUT"
echo "--- 라벨별 개수 (600행) ---"
"$PY" - "$OUT" <<'EOF'
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig", newline="")))
c = Counter(r["label"] for r in rows)
for k in ["적절", "무의미형", "파일명형", "중복형", "장식오용형", "불충분형", "무관형"]:
    print(f"  {k}{' ' * (10 - 2 * len(k))} {c.get(k, 0):4d}  " + "#" * round(c.get(k, 0) / 10))
print(f"  합계{' ' * 6} {len(rows):4d}")
EOF

echo
echo "[3/3] 결과 뷰어 생성"
TOUR_ARG=(); [ "${DEMO_TOUR:-0}" = "1" ] && TOUR_ARG=(--tour)
"$PY" scripts/make_result_viewer.py "$OUT" "${TOUR_ARG[@]+"${TOUR_ARG[@]}"}" | tail -n 2
VIEW="$ROOT/outputs/result_viewer.html"

if [ "${DEMO_NO_OPEN:-0}" != "1" ]; then
  if command -v open >/dev/null 2>&1; then  # macOS: CSV는 Numbers, 없으면 텍스트 편집기 (Excel은 BOM 없는 UTF-8 한글이 깨질 수 있음)
    if [ -d /Applications/Numbers.app ]; then open -a Numbers "$OUT" || true; else open -e "$OUT" || true; fi
    if [ "${DEMO_TOUR:-0}" = "1" ]; then
      sleep 6
      if [ -d "/Applications/Google Chrome.app" ]; then
        open -na "Google Chrome" --args --app="file://$VIEW" --window-position=0,0 --window-size="${DEMO_WINDOW:-1400,900}"
      else open "$VIEW"; fi
    else open "$VIEW"; fi
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$VIEW" >/dev/null 2>&1 || true
  else echo "브라우저에서 여세요: $VIEW"; fi
fi
echo "완료  $(date '+%H:%M:%S')"
