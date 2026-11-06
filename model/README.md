# model/ — 학습된 모델의 출력

> **이 폴더에는 학습된 가중치 파일이 없습니다.**
> - 최종 제출에 쓰인 LoRA 분류기(Qwen3.8-27B, Qwen3.5-9B 전체 학습본)의 어댑터 가중치는 Colab 세션을 종료하기 전에 백업 대상에서 빠져 보존되지 못했습니다.
> - 대신 학습된 모델들의 출력(OOF·test 확률)을 `runs_frozen/`에 모두 보존했고, 이 출력만으로 최종 제출 파일을 바이트 단위로 다시 만듭니다(`scripts/reproduce_best.sh`).
> - 가중치는 학습 레시피(아래 '학습된 가중치', `notebooks/02_full_pipeline_colab.ipynb`)로 다시 학습할 수 있습니다. GPU 비결정성 때문에 확률은 조금 달라질 수 있습니다.

## runs_frozen/ (최종 결정 단계의 입력, 정확 재현용)

- 각 하위 폴더는 학습 단계 하나의 출력이며, 이름은 `src/koaltq2`의 단계 이름과 같습니다.
- 공통 파일
  - `oof.npy`: 학습 2,400행의 fold 밖 7범주 확률
  - `oof_mask.npy`
  - `test.npy`: 평가 600행 확률
  - `metrics.json`: OOF 지표, 혼동행렬, 특징 수 등
- `scripts/reproduce_best.sh`가 이 폴더의 복사본만으로 최종 제출 파일을 바이트 단위로 다시 만듭니다.
- `logs/`: 원래 Colab 학습 실행 로그(단계별 시작 인자·소요 시간·지표). `docs/pipeline_provenance.md`의 근거입니다.
- 제출본 정리(2026-11-06): 최종 파일에 영향이 없는 실험 출력(x11 교차 인코더, x13 희소 클래스 실험)과 그 로그는 뺐습니다. 뺀 뒤에도 재현 파일의 sha256이 같음을 확인했습니다.

| 폴더 | 모델 | 최종 결정에서 |
|---|---|---|
| x1_* | 규칙·구조 특징 + TF-IDF → GBM / LR | 후보 비교만 |
| x2_ocr, x2_hgb_ocr | PaddleOCR PP-OCRv5 korean 글자 특징 → HGB | x2_ocr의 글자 수를 하네스가 사용 |
| x3_* | 고정 VLM 임베딩(Qwen3.5-9B, InternVL3.5-14B) → LR / MLP 헤드 | 후보 비교만. 임베딩 원본 `feats.npy`(약 0.45GB)는 미포함, `meta.json`만 있음 |
| x12_* | VLM 예/아니오 속성 질의 (Qwen3.5-9B, Qwen3.8-27B, Qwen3-VL-8B/32B) | 사진 여부 투표를 하네스가 사용 |
| x14_* | 가이드라인 판정기 7범주 확률: Qwen3.8-27B 10변형(`x14_qwen38_27b_long*`) + Qwen3.5-9B·Qwen3-VL-8B/32B | 27B long 10변형 평균을 하네스 H5b가 사용 |
| x15_judge_lr | 판정기·속성 질의 출력(124개) → 로지스틱 회귀 | **결합 0.1** |
| x16_* | 예/아니오 판정기 (불충분 여부, 적절 여부) | 스태커 특징으로만 |
| x7*_stack_* | 특징 + 판정기 → 스태커 (HGB / LightGBM) | 후보 비교만 |
| x17_stack_lgbm_b109 | LightGBM 스태커(특징 364개) + 공개 109건 학습 | **결합 0.9** (OOF로 규칙·비중 선택) |
| x17_stack_lgbm_b109fx | 같은 레시피, 판정기 15개 시점(특징 378개) + 규칙 확정 test 133행 학습 | **test 확률을 x17 대신 사용** (`test.npy`만 있음) |
| x6_qwen38_27b_lora_fold / _full | Qwen3.8-27B LoRA 분류기 (fold 0 검증 / 전체 학습 test 확률) | **결합 0.4의 절반** (test는 _full) |
| x6_qwen35_9b_lora_fold / _full | Qwen3.5-9B LoRA 분류기 | **결합 0.4의 절반** (test는 _full) |

## 학습된 가중치

- 기반 모델(Qwen, InternVL, PaddleOCR)은 공개 가중치를 내려받아 씁니다. 출처·라이선스·리비전은 README 6절에 있습니다.
- LoRA 분류기 레시피(`src/koaltq2/x6_lora.py`, 각 run의 `metrics.json`의 `meta`)
  - 공통: r=16, lr 1e-4, gradient accumulation 16, seed 20261102, 이미지 + 증거 카드 입력, 희소 클래스 반복 샘플링 + 로짓 보정
  - epoch: 9B 2.0, 27B 1.5
  - 전체 학습본: 학습 2,400 + 공개 109건
  - A100 소요 시간: 9B fold 85분 / 전체 73분, 27B fold 85분 / 전체 약 105분
- 최종 제출에 쓰인 LoRA 전체 학습본(27B·9B)의 어댑터 가중치는 Colab 세션 종료 전 백업 대상에서 빠져 보존되지 못했습니다.
  - 그 모델들의 확률(`x6_*_lora_full/test.npy`)만 남아 있습니다.
  - 다시 학습하려면 `notebooks/02_full_pipeline_colab.ipynb`의 x6 단계를 실행합니다. GPU 비결정성 때문에 확률이 조금 달라질 수 있습니다.
- LightGBM 스태커는 고정 시드의 단일 모델입니다.
  - 원래 runs 폴더 전체(x3 임베딩 원본 포함)가 있으면 다시 학습해도 test 확률이 비트 단위로 같습니다(Colab에서 확인).
  - 이 폴더만으로 다시 학습하면 특징 구성이 달라 argmax 97.5%만 일치합니다.
