# demo/ — 데모영상

| 파일 | 내용 |
|---|---|
| demo_reproduce_best.mp4 | 최종 제출 파일 재현 과정 (24.6초, 1600×900, H.264) |
| demo_log.txt | 영상의 원본: 실제 실행 출력 한 줄마다 경과 시간을 기록한 로그 |

## 촬영 방법
- Colab CPU 런타임의 **빈 폴더에 제출물을 풀고** 다음 세 명령을 실제로 실행했습니다.
  1. `bash scripts/reproduce_best.sh`
  2. `head -5 outputs/reproduced_submission.csv`
  3. `cat outputs/harness_summary.json`
- 출력이 나온 시각을 줄 단위로 기록해 터미널 화면으로 다시 그렸습니다(`demo_log.txt`).
- 실제 소요 시간은 467초입니다. 영상에서는 2초가 넘는 대기 구간을 2초로 압축했습니다.
- 화면 녹화가 아니라 실제 실행 출력의 재생입니다. 출력 내용은 편집하지 않았습니다(경고 줄만 생략).

## 영상에서 확인할 수 있는 것
1. 결정 단계 실행: 결합 규칙 선택(blend:x17_stack_lgbm_b109, 판정기 LR 0.1), LoRA 비중 0.4
2. 하네스 모듈별 적용 수: H0 109, H1 8+116, H2 17, H3 1, H4 6(+H5 거부 3)
3. sha256 `87c3de68…` → "OK - identical to submission/final_submission_0.9357.csv"
4. 재현된 제출 파일의 앞부분과 `outputs/harness_summary.json`
