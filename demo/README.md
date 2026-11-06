# demo/ — 데모영상

| 파일 | 내용 |
|---|---|
| demo_screen_recording.mp4 | **추론(결정) 과정 화면 녹화**: `bash demo/run_demo.sh` 실행 → 제출 CSV 생성·sha256 대조 → 결과 뷰어 → CSV 열기 (macOS, 2026-11-06, 3분 57초, 2560×1352) |
| run_demo.sh | 위 영상에서 실행한 데모 스크립트 |
| demo_reproduce_best.mp4 | 이전 데모: Colab에서 실행한 재현 출력을 터미널 화면으로 다시 그린 영상 (24.6초, 1600×900, H.264) |
| demo_log.txt | 이전 데모의 원본: 실제 실행 출력 한 줄마다 경과 시간을 기록한 로그 |

## 데모에서 실행하는 것과 하지 않는 것
- **실행함**: `model/runs_frozen/`에 저장된 학습된 모델들의 출력(OOF·test 확률)으로 최종 결정 단계(결합 → 할당량 → 하네스)를 CPU에서 실행해 평가 데이터 600행의 제출 CSV를 만듭니다.
- **하지 않음**: 모델 학습과 GPU VLM 추론은 영상에서 다시 하지 않습니다. 최종 LoRA 가중치는 보존되지 못했기 때문입니다(README 1절). 처음부터 학습·추론하는 절차는 README 5-3·5-4절과 노트북 02에 있습니다.

## demo_screen_recording.mp4 (화면 녹화)
1. `bash demo/run_demo.sh`
   - [1/3] 결정 단계 실행: 결합 규칙 선택(blend:x17_stack_lgbm_b109, 판정기 LR 0.1), LoRA 비중 0.4, 할당량, 하네스
   - sha256 `87c3de68…` → "OK - identical to submission/final_submission_0.9357.csv"
   - [2/3] 생성된 `outputs/reproduced_submission.csv`의 앞 10행과 라벨별 개수(600행)
   - [3/3] 결과 뷰어 `outputs/result_viewer.html` 생성
2. 결과 뷰어에서 각 행의 이미지·alt·판정 라벨을 보고, 라벨 필터와 '하네스가 정한 행만' 보기로 규칙이 정한 행의 근거를 확인합니다.
3. 생성된 CSV를 열어 `record_id,label` 600행을 확인합니다.
- 화면을 그대로 녹화했고, 실행 출력과 시간은 편집하지 않았습니다(결정 단계의 실제 소요 시간 그대로).
- 영상 가공은 다음 세 가지뿐입니다.
  - 오디오 제거
  - 2분 48초 이후 브라우저 도구 막대·북마크 영역을 흐리게 처리 (개인 북마크 보호)
  - 해상도를 2560×1352로 축소

같은 데모를 직접 실행하려면 저장소 루트에서 `bash demo/run_demo.sh`를 실행합니다(CPU, 약 3~8분, README 0절의 환경 필요).

## demo_reproduce_best.mp4 (이전 데모)
- Colab CPU 런타임의 **빈 폴더에 제출물을 풀고** 다음 세 명령을 실제로 실행했습니다.
  1. `bash scripts/reproduce_best.sh`
  2. `head -5 outputs/reproduced_submission.csv`
  3. `cat outputs/harness_summary.json`
- 출력이 나온 시각을 줄 단위로 기록해 터미널 화면으로 다시 그렸습니다(`demo_log.txt`).
- 실제 소요 시간은 467초입니다. 영상에서는 2초가 넘는 대기 구간을 2초로 압축했습니다.
- 화면 녹화가 아니라 실제 실행 출력의 재생입니다. 출력 내용은 편집하지 않았습니다(경고 줄만 생략).
