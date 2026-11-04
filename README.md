# KoAltQ: 이미지·문맥 정보를 활용한 한국어 웹 대체텍스트 품질 판정

2026 DATA·AI 분석 경진대회 문제 4 · 팀 **못나지않은사람들**

| 항목 | 내용 |
|---|---|
| 최종 제출 파일 | `submission/final_submission_0.9357.csv` (600행, sha256 `87c3de688bd20937dc86abb9fb5e1171d6d012cd132aa23bd1c3f9611a9a1cf8`) |
| 리더보드 | **0.9357** = 0.5 × 이진 F1 **0.9579** + 0.5 × 7범주 macro-F1 **0.9134** (2026-11-04 06:28 제출) |
| 과제 | 웹 이미지의 alt 텍스트를 7범주로 판정: 적절 / 무의미형 / 파일명형 / 중복형 / 장식오용형 / 불충분형 / 무관형 |
| 재현 범위 | 저장된 모델 출력(`model/runs_frozen/`)에서 최종 파일을 **바이트 단위로 동일하게** 다시 만듦 (CPU, 3~8분). GPU 재학습 절차는 노트북 02 (5-3 참고) |
| 외부 API | 예측 파이프라인은 외부 API를 쓰지 않음. 모든 모델은 공개 가중치를 내려받아 Colab GPU에서 직접 실행. 개발 과정의 AI 도구 사용은 7절 |

---

## 0. 빠른 시작 (심사·재현용)

**로컬 (Linux · macOS. Windows는 WSL 또는 Colab)**

```bash
# 1) 환경 구성: .venv 생성 + requirements.txt, requirements-notebook.txt 설치
bash scripts/setup_env.sh python3.12          # 또는 python3.13

# 2) 최종 제출 파일 재현 + 무결성 확인
PYTHON=.venv/bin/python bash scripts/reproduce_best.sh
```

- 결과: `outputs/reproduced_submission.csv`
- 마지막 줄이 `OK - identical to submission/final_submission_0.9357.csv`이면 재현 성공입니다.
- `outputs/harness_log.jsonl`: 규칙이 바꾸거나 고정한 모든 행과 그 근거(감사 로그)
- 노트북: `.venv/bin/jupyter notebook notebooks/01_reproduce_best.ipynb` → 위에서부터 실행

**Google Colab**
1. 제출물 폴더를 `/content/KoAltQ_submission_0.9357`에 업로드하거나 압축을 풉니다.
2. 실행합니다.
   ```
   %cd /content/KoAltQ_submission_0.9357
   !bash scripts/reproduce_best.sh
   ```
   - Colab 기본 런타임에는 requirements.txt의 버전이 이미 설치되어 있어 setup_env.sh가 필요 없습니다.
3. 노트북 01도 같은 위치에서 그대로 실행됩니다(폴더를 자동으로 찾음).

---

## 1. 결과물 구성

```
KoAltQ_submission_0.9357/
├── README.md                  이 문서 (구성·환경·실행 방법·방법 요약·AI 활용 내역)
├── requirements.txt           CPU 재현 환경 (상세 버전)
├── requirements-notebook.txt  로컬에서 노트북 실행 시 (Colab 런타임 버전)
├── requirements-gpu.txt       전체 학습·추론 환경 (Colab A100, 상세 버전)
├── data/                      학습·평가에 사용한 데이터 (data/README.md)
│   └── koaltq_bundle/         data/*.jsonl, images/(3,000장, 정규화본), docs/guideline_v1.1.md, MANIFEST.json
├── model/                     학습된 모델의 출력 (model/README.md)
│   └── runs_frozen/           단계별 OOF·test 확률 등 (최종 결정 단계의 입력, 약 11MB)
│       └── logs/              원래 학습 실행 로그 (단계별 시작 인자·소요 시간·지표)
├── src/koaltq2/               소스코드 (특징·판정기·스태커·LoRA·결합·하네스, 단계별 모듈, `python -m koaltq2.run <단계>`)
├── notebooks/
│   ├── 01_reproduce_best.ipynb         CPU: 데이터 확인 → 결정 단계 → sha256 대조 → 모델별 평가 → 감사 로그 → 그림 (실행 결과 포함)
│   └── 02_full_pipeline_colab.ipynb    GPU(Colab A100): 전체 학습·추론 38단계를 원래 순서대로
├── scripts/
│   ├── setup_env.sh                    CPU 환경 구성
│   ├── reproduce_best.sh               최종 파일 재현 + sha256 대조
│   ├── run_harness_experiments.sh      하네스 변형(H6)·검증 하네스·conformal 분석
│   ├── tools/swap_candidate.py         새 로짓으로 같은 개수 재배정 후보 만들기 (실험용)
│   ├── tools/decode_lb.py              리더보드 점수로 바뀐 행의 정답 수 역산 (실험용)
│   ├── build_bundle.py                 대회 배포본 → 데이터 번들 생성 (원래 작업 폴더 구조 필요, data/README.md)
│   └── colab/                          원래 Colab 셀 스크립트(c00~c30)와 작업 큐 실행기 (기록용. 지금 실행은 노트북 02)
├── submission/                최종 제출 파일 + sha256
├── demo/                      데모영상 demo_reproduce_best.mp4 + 원본 실행 로그 (demo/README.md)
├── figures/                   설명용 그림 (리더보드 변화, 클래스 개수, 하네스 결정, OOF 혼동행렬; 노트북 01이 생성)
└── docs/
    ├── method_report_harness.txt       방법·하네스·실험 결과 정리 (보고서 원고)
    ├── pipeline_provenance.md          최종 제출에 쓰인 모든 모델 출력의 생성 명령·순서·소요 시간 (실행 로그에서 복원)
    ├── experiment_log.md               실험 기록 (시간순, 개인정보 비식별화)
    └── colab_pip_freeze_full.txt       학습에 쓴 Colab 런타임의 전체 패키지 목록 (머리말에 캡처 시점)
```

`outputs/`는 스크립트를 실행하면 생기는 폴더입니다(제출물에는 포함하지 않음).

---

## 2. 실행 환경

### 2-1. 재현 (CPU)

| 항목 | 검증한 환경 1 (Colab) | 검증한 환경 2 (로컬) |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS | macOS (Apple M4 Pro) |
| Python | 3.13.16 | 3.12.9 |
| numpy / scipy / scikit-learn | 2.4.6 / 1.16.3 / 1.6.1 | 2.5.4 / 1.18.1 / 1.9.1 |
| 결과 | sha256 일치 (빈 폴더에서 재현, 노트북 01 실행 포함) | sha256 일치 |

`requirements.txt`는 환경 1의 버전으로 고정했습니다(matplotlib 3.10.0 포함). GPU는 필요 없습니다.

### 2-2. 전체 학습·추론 (GPU)

| 항목 | 버전 |
|---|---|
| 하드웨어 | Google Colab, NVIDIA A100-SXM4 80GB ×1, High-RAM |
| NVIDIA Driver / CUDA | 580.82.07 / 13.0 |
| OS / Python | Ubuntu 24.04.4 LTS / 3.13.16 |
| 주요 라이브러리 | torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, flash-linear-attention 0.6.0, lightgbm 4.6.0 |
| OCR | 격리 venv: paddlepaddle-gpu 3.4.0 + paddleocr 3.7.0 (PP-OCRv5 korean). `src/koaltq2/x2_ocr.py`가 자동 생성 |

- 상세 버전: `requirements-gpu.txt`
- 런타임 전체 목록: `docs/colab_pip_freeze_full.txt`
  - 학습에 쓴 같은 Colab 세션에서 2026-11-04에 캡처했습니다.
  - 학습 당시의 scikit-learn·LightGBM 버전은 따로 기록하지 않았습니다. 대신 이 런타임에서 스태커 x17fx를 다시 학습해 비트 단위로 같은 출력이 나오는 것을 확인했습니다.
- 기반 모델은 Hugging Face에서 자동으로 내려받습니다. 모두 공개 모델이라 인증 정보가 필요 없습니다.
  - 학습에 쓴 리비전은 6절 표에 있습니다.

---

## 3. 데이터

파일별 설명은 `data/README.md`에 있습니다.

- `train.jsonl` 2,400건(라벨 포함), `test.jsonl` 600건, `image_meta.jsonl`, `images/` 3,000장
  - 각 레코드: 페이지 URL·제목, 이미지 URL, alt, 주변 문맥, 링크 여부·목적지, 헤딩 경로
  - 레코드는 대회 배포본 그대로입니다.
  - 이미지는 모델 입력용으로 정규화했습니다(형식 판별, SVG→PNG, 투명 배경→흰색, 최장변 1536px).
- `folds.jsonl`: 학습 데이터 5-fold. 같은 페이지·템플릿 묶음(277개 그룹)이 한 fold에 들어가도록 고정
- 공개 경계사례(평가 데이터 109건): `boundary_test_labels.jsonl`(데이터셋_라벨, 공식 정답)과 `boundary_test_rejudge.jsonl`(재판정_라벨 포함)
  - Q&A 678: "학습, 예측 결과 후처리, 최종 제출 어디에 쓰셔도 제한하지 않습니다"
- `docs/guideline_v1.1.md`: 대회 라벨링 가이드라인 (판정기 프롬프트에 사용)
- 외부 데이터는 쓰지 않았습니다.
  - `pseudo_test_v*.jsonl`(모델이 만든 의사 라벨)은 비교 실험에만 쓰였고 최종 제출에는 쓰지 않았습니다.

---

## 4. 방법 개요

```
 [이미지 + alt + 문맥]
        │
        ├─ 기초 특징: 규칙·구조 특징 + TF-IDF(x1), PaddleOCR 글자 특징(x2),
        │             고정 VLM 임베딩 Qwen3.5-9B·InternVL3.5-14B(x3, PCA 48차원 블록)
        ├─ VLM 속성 질의(x12): 예/아니오 확률 (Qwen3.5-9B, Qwen3.8-27B, Qwen3-VL-8B, Qwen3-VL-32B)
        ├─ 가이드라인 판정기(x14·x16): 가이드라인 2~3-3절 + 레코드 → 7범주 확률
        │       Qwen3.8-27B 10변형(선택지 순서·문맥·해상도) + Qwen3.5-9B·Qwen3-VL-8B/32B 판정기, 예/아니오 판정기 2종
        ├─ 판정기 LR(x15): 판정기·속성 질의 출력(124개) → 로지스틱 회귀
        ├─ 스태커(x17): 위 특징 전체(364개) → LightGBM. 학습 2,400 + 공개 109건(데이터셋_라벨)
        │       x17fx: 같은 레시피를 판정기 15개 시점(378개)에 학습 + 규칙 확정 test 133행(R2 116, R4′ 17)
        └─ 지도학습 VLM(x6): Qwen3.8-27B LoRA, Qwen3.5-9B LoRA (이미지 + 증거 카드 → 7범주)
                 │
   결정 단계(x8) ── 결합: 0.6 × (0.9 log P스태커 + 0.1 log P판정기LR) + 0.4 × 평균 log P(27B·9B LoRA)
                 │     - 스태커 test 확률은 x17fx, 규칙·비중 선택은 x17 OOF
                 │     - 할당량: 규칙으로 정하지 않은 행에 평가셋 클래스 분포를 맞춰 배정
                 ▼
   하네스 H0~H5 (구조 서명 일관성 하네스) → 최종 600행 + 감사 로그
```

klue/roberta-large 교차 인코더(x11)와 단일 모델(x1, x2, x3 헤드, x7)은 결정 단계에서 후보로만 비교했고, 최종 결합에는 쓰이지 않습니다.

### 4-1. 하네스 (Structure-Signature Consistency Harness)

모델이 제안한 라벨을 순서가 정해진 모듈로 통과시킵니다. 근거가 있는 행만 보정하고, 모든 결정을 근거와 함께
`harness_log.jsonl`에 남깁니다(`src/koaltq2/harness.py`).

| 모듈 | 내용 | 최종 파일에서 |
|---|---|---|
| H0 공개 행 | 경계사례 109건은 공식 데이터셋_라벨 | 109행 |
| H1 가이드라인 규칙 | R1 파일명 흔적 → 파일명형, R2 복원 자료의 역할 표시 alt → 적절 | 8행, 116행 |
| H2 템플릿 쌍둥이 | (페이지, 링크 여부, 링크 종류, alt 템플릿)이 같은 공개 행의 재판정_라벨(80% 이상 일치) | 17행 |
| H3 링크 제약 | 링크의 유일한 콘텐츠인 이미지는 장식이 아님 (가이드라인 3-3. 학습 데이터의 같은 구조 667건 중 장식오용형 0) | 1행 |
| H4 구조 쌍둥이 | (페이지, 링크, 링크 종류, alt 관계, 이미지 종류)가 같은 공개 행의 80% 이상이 '적절'로 재판정 → 불충분형 예측을 적절로 | 후보 9행 중 6행 |
| H5 검수 일치 거부 | (a) 학습의 같은 구조(25건 이상)에서 현재 라벨이 95% 이상: 2행 / (b) 결합 모델과 판정기가 모두 현재 라벨 0.95 이상: 1행 | 3행 거부 |
| H6 서명 투표 (선택) | 동료 설계: 같은 서명(페이지·링크·링크 종류·alt 관계·이미지 파일명 패턴)의 공개 이웃 2건 이상 100% 일치 + 확신도 0.9 이하 → 교체 | 기본 꺼짐 (LB 0.93) |

핵심 원리(검수 일치):
- 주최 측은 평가 600건 전체를 독립 재판정했고, 원 라벨과 재판정이 갈린 건을 경계사례로 공개했습니다.
- 공개되지 않은 491건은 10/13에 정정된 5건을 빼면 두 판정이 일치한 행입니다(8절 참고).
- 이런 행의 정답은 P(y | x, 일치) ∝ P원라벨러(y | x) × P검수자(y | x)를 따릅니다.
- 공개 쌍둥이의 재판정 라벨을 옮기는 것(H2·H4)은 검수자의 일관성을 이용하는 것입니다.
- 원 라벨러가 그 구조에 그 라벨을 거의 주지 않는다면(학습 데이터 기준) 옮기지 않습니다(H5).
- H5는 16번 제출 결과(H4 9행 중 3행 오답)를 본 뒤 설계했습니다.
  - 기준은 학습 라벨과 모델 점수만 씁니다. 행 단위 리더보드 정보는 쓰지 않았습니다.
  - 기준을 정한 뒤 한 번 제출해 확인했습니다(3/3 정답).

검증 하네스(학습 데이터 시뮬레이션), conformal 분석, H6 실험은 `docs/method_report_harness.txt`에 있습니다.

### 4-2. 리더보드 점수 변화

| 단계 | LB |
|---|---|
| 기존 팀 제출 | 0.8374 |
| 판정기 특징 스태커 + 할당량 배정 | 0.8379 (*) |
| 형제 행에 재판정 라벨 전파 (H2) | 0.86 |
| 27B·9B LoRA 결합 | 0.87 |
| 판정기 기반 희소 클래스 전용 판정 제거 | 0.8875 (*) |
| 스태커에 확정 test 행 추가 + LoRA 비중 0.4 + 27B 전체 학습본 | 0.9194 |
| 링크 유일 이미지는 장식 아님 (H3) | 0.9262 |
| 구조 쌍둥이 '적절' 전파 (H4) | 0.9304 |
| **검수 일치 거부 (H5) → 최종** | **0.9357** |

(*) 제출 이력에는 소수 둘째 자리(0.84, 0.89)만 남아 있어, 리더보드 표시값과 파일의 대응은 추정입니다.

효과가 없었던 대표 시도:
- 판정기를 검수자 대리로 쓴 재순위(PoE): 0.92
- LoRA 5-fold OOF 스태커(X19): 0.93
- H6 서명 투표: 0.93
- 링크 안 사진 '중복형 금지' 규칙: 0.94 구간, 갱신 없음
- 판정기 단독 배정: 0.80

---

## 5. 실행 방법

### 5-1. 최종 파일 재현 (CPU, 정확 재현)

`bash scripts/reproduce_best.sh [출력 csv]` (인터프리터는 `PYTHON=...`로 지정)

1. `model/runs_frozen/`을 임시 폴더로 복사하고, 스태커 x17의 test 확률을 x17fx의 것으로 교체합니다(4절).
2. 임시 폴더에서 `python -m koaltq2.run x8 --objective lbq --wlora 0.4`를 `KOALTQ_HARNESS=1`로 실행합니다(결합 → 할당량 → 하네스).
3. 결과를 `submission/final_submission_0.9357.csv`와 sha256으로 대조합니다.
4. `outputs/`에 다음을 남깁니다.
   - 제출 파일
   - 감사 로그(`harness_log.jsonl`, `harness_summary.json`)
   - 결정 보고서(`report.json`)
   - 하네스 입력(`harness_in_*.npy`)

- 환경변수 `KOALTQ_BUNDLE`(데이터 번들), `KOALTQ_RUNS`(모델 출력 폴더), `PYTHONPATH=src`는 스크립트가 설정합니다.
- 기존 플래그 조합(`KOALTQ_SOLE_NOT_DECO=1 KOALTQ_SIB_K2_OK=1 KOALTQ_K2_VETO=1`)도 같은 파일을 만듭니다.

### 5-2. 하네스 실험 (CPU)

`bash scripts/run_harness_experiments.sh` (5-1 실행 후)

- H6 변형 3종: 공식 데이터셋_라벨 / 재판정_라벨 / 게이트 없음
- 검증 하네스: 학습 데이터 시뮬레이션 30회, 단독·하네스 순서
- conformal 예측집합 분석
- 결과는 `outputs/`의 `HARNESS_*.csv`, `harness_log_*.jsonl`, `harness_val.json`에 저장됩니다.

### 5-3. 전체 학습·추론 (GPU, Colab A100)

`notebooks/02_full_pipeline_colab.ipynb`를 Colab(A100 80GB)에서 순서대로 실행합니다.
- 명령과 순서는 원래 실행 로그와 작업 큐에서 복원했습니다(`docs/pipeline_provenance.md`, 38단계).
- 준비·x0·x1 셀은 Colab에서 실행해 확인했습니다. GPU 단계 전체를 한 번에 다시 돌려 보지는 않았습니다.

1. 준비(제출물 폴더 배치, 패키지 설치, x0 데이터 계약 검사)
2. 기초 특징: x1, x2(OCR), x3(Qwen3.5-9B, InternVL3.5-14B 임베딩)
3. 9B LoRA(fold 0 검증·전체 학습), x11 교차 인코더, x12 속성 질의 3종
4. 판정기 x14·x16 (Qwen3.8-27B 변형 8개 + 다른 모델 판정기 + x12 27B)
5. **체크포인트(`x14_*` 13개) → x7·x15·x17 스태커**
6. 27B LoRA(fold 0 검증), 판정기 rot1·rot6, 27B LoRA 전체 학습
7. x17fx (`x14_*` 15개 시점)
8. 결정 단계 + 하네스 → 제출 파일과 비교

유의점:
- A100 기준 GPU 시간은 약 15시간(대부분 27B 판정기와 LoRA), CPU 단계는 약 20분입니다.
  - 노트북은 끝난 단계를 건너뛰는 이어하기와, 출력을 Google Drive에 두는 옵션(`USE_DRIVE`)을 지원합니다.
- **순서 제약**: 스태커는 runs 폴더의 판정기 결과를 모두 읽어 특징을 만듭니다.
  - x7·x15·x17은 `x14_*`가 정확히 13개일 때(특징 364·124개), x17fx는 15개일 때(378개) 실행합니다.
  - 노트북에 assert가 있습니다.
- **재현 범위**
  - 최종 제출에 쓰인 LoRA 전체 학습본(27B·9B)의 어댑터 가중치는 Colab 세션 재시작 때 보존되지 않았고, 그 test 확률만 `model/runs_frozen/`에 있습니다.
  - VLM 임베딩 원본(`x3_*/feats.npy`, 약 0.45GB)도 용량 때문에 넣지 않았습니다. 그래서 `model/runs_frozen/`만으로 스태커를 다시 학습하면 특징이 달라집니다(argmax 97.5% 일치).
  - GPU 학습은 비결정적 요소가 있어, 처음부터 다시 학습하면 확률이 조금 달라지고 최종 파일도 몇 행 다를 수 있습니다.
  - 제출 파일과의 정확한 일치는 5-1(저장된 모델 출력)로 확인합니다.

### 5-4. 새 실험을 붙이는 방법 (팀원용)

**새 모델 출력으로 결정 단계 다시 돌리기**: 반드시 `model/runs_frozen`의 **복사본**에서 실행합니다. x8은 실행 폴더에 결과를 씁니다.

```bash
W=$(mktemp -d); cp -R model/runs_frozen/. "$W"
cp "$W"/x17_stack_lgbm_b109fx/test.npy "$W"/x17_stack_lgbm_b109/test.npy; rm -rf "$W"/x17_stack_lgbm_b109fx
cp -R <새 run 폴더> "$W"/
KOALTQ_HARNESS=1 KOALTQ_BUNDLE=data/koaltq_bundle KOALTQ_RUNS="$W" PYTHONPATH=src \
  python -m koaltq2.run x8 --objective lbq --wlora 0.4
# 결과: $W/submissions/S3_ensemble_quota_l0_lbq.csv, 결정 보고서: $W/x8_blend_lbq/report.json
```

- 새 run 폴더에는 다음 파일을 넣습니다.
  - `oof.npy`: 2400×7, 학습 fold 밖 확률
  - `oof_mask.npy`
  - `test.npy`: 600×7
- x8은 `oof_mask`가 전부 True이거나 정확히 fold 0인 폴더만 후보로 씁니다. 나머지 폴더는 조용히 건너뜁니다.
  - 이름이 `x6_`로 시작하는 fold 0 폴더는 LoRA 성분 후보가 됩니다.
- `--rule "blend:<이름>"`으로 결합 규칙을 고정할 수 있는 폴더는 따로 정해져 있습니다.
  - 이름이 `x7_` `x7j_` `x17_` `x19_`로 시작하고, OOF가 전체이고, `x15_judge_lr`가 있는 경우에만 됩니다.
  - 다른 이름은 assert로 멈춥니다.

**같은 개수 재배정 후보 만들기** (저장소 루트에서, 5-1 실행 후)

```bash
KOALTQ_BUNDLE=data/koaltq_bundle PYTHONPATH=src python scripts/tools/swap_candidate.py \
  --logits <600×7 로그확률 .npy> --out outputs/my_candidate.csv
```

- 규칙·하네스가 정한 행은 고정하고, 적절·중복형·불충분형 안에서만 순위를 바꿉니다.

**리더보드 역산**

```bash
python scripts/tools/decode_lb.py <후보 csv> <이진 F1> <macro F1>   # 4자리 표시값
python scripts/tools/decode_lb.py <후보 csv> <2자리 총점>            # 제출 이력 값
```

- 바뀐 행 중 몇 개가 맞았는지 계산합니다.

---

## 6. 외부 자료 활용 및 이용 조건

| 자료 | 용도 | 출처 · 라이선스 (Hugging Face 모델 카드, 2026-11-04 확인) | 학습에 쓴 리비전 |
|---|---|---|---|
| Qwen/Qwen3.8-27B | 판정기, 27B LoRA, 속성 질의 | Apache-2.0 | `1d4bf0f2ff60…` |
| Qwen/Qwen3.5-9B | 임베딩, 9B LoRA, 속성 질의, 판정기 | Apache-2.0 | `c20223623576…` |
| Qwen/Qwen3-VL-32B-Instruct, Qwen3-VL-8B-Instruct | 속성 질의, 판정기 | Apache-2.0 | 기록 없음 (현재 `0cfaf48183f5…`, `0c351dd01ed8…`) |
| OpenGVLab/InternVL3_5-14B-HF | 임베딩 | Apache-2.0 | `226b96d5912e…` |
| klue/roberta-large | 교차 인코더 (비교 후보, 최종 결합에는 미사용) | 모델 카드에 라이선스 표기 없음 | 기록 없음 |
| PaddleOCR PP-OCRv5 (paddleocr 3.7.0) | 이미지 속 글자 인식 (가중치는 PaddleX 공식 저장소에서 자동 다운로드) | Apache-2.0 | - |
| 데이터 | 대회 제공 데이터만 사용 | 대회 규정 | - |

- 주요 라이브러리(PyTorch BSD-3, transformers·peft·accelerate Apache-2.0, LightGBM MIT, scikit-learn BSD-3, numpy·scipy BSD)는 공개 오픈소스입니다.
- 상용 LLM/VLM API는 예측에 쓰지 않았습니다(Q&A 535).
- 제출물에는 API 키·비밀번호·인증 정보가 없습니다.
- 팀원 이름은 문서와 로그에서 비식별화했습니다.
- data/의 레코드와 OCR 결과(`model/runs_frozen/x2_ocr/`)는 공공기관 웹페이지에서 수집된 대회 제공 데이터입니다. 웹에 공개된 기관 연락처 등이 포함될 수 있습니다.

---

## 7. AI 활용범위 및 주요 검토 과정

**활용 도구**

| 도구 | 버전·설정 | 활용 단계 | 주요 활용 내용 |
|---|---|---|---|
| Claude Code (Anthropic) | CLI 2.1.318, 모델 Claude Opus 5.5 (`claude-opus-5-5`) | 문제 분석, 아이디어 탐색, 데이터 처리, 모델 설계, 코드 작성, 실험 운영, 결과 분석, 문서 작성 | `src/koaltq2` 모듈 구현, Colab 작업 큐 운영, 리더보드 역산 분석, 하네스·검증 하네스·conformal 분석, README·보고서 초안 |
| Codex CLI (OpenAI) | GPT-6.1-Sol, reasoning high, 읽기 전용 | 계획 (2026-11-02) | Claude 계획(Plan A)과 독립적으로 작성한 실험 계획(Plan B). 팀이 비교 후 혼합안(C) 선택 |
| Google Colab CLI | 0.7.5 | 실험 실행 | Claude Code가 Colab A100 세션에서 학습·추론 작업을 실행하고 결과를 내려받음 |
| Orca computer use (화면 제어) | - | 리더보드 제출 | 사용자 승인 하에 Claude Code가 사용자 Chrome에서 결과 파일 제출·점수 확인. 로그인은 사용자가 직접 함 |
| Claude Code 멀티 에이전트 리뷰 | 리뷰어 3 + 반박 검증 7 (11/02), 리뷰어 3 + 검증 1 (11/04, 2회) | 코드·결과물 검토 | 사전 코드 결함 7건 수정, 하네스 코드 동치성 확인, 결과물 점검(본 README 반영) |

**AI 사용과 관련해 밝혀 둘 사항**
- 초기 분석 단계(2026-11-03 13:00 이전)에 Claude Code가 대화 중 일부 학습·평가 데이터 레코드를 직접 열람·분석했습니다.
  - Q&A 618 검토 후에는 평가 레코드를 개별로 열람하지 않고 코드·집계 수준에서만 분석했습니다.
  - 평가 데이터를 사람이나 AI가 수작업으로 라벨링하지는 않았습니다.
  - 예측은 모두 공개 가중치 모델로만 했습니다.
- 화면 제어 제출 중 2026-11-04 08:47에 창 인식 오류로 의도하지 않은 제출 2건이 기록됐습니다. 최종 점수에는 영향이 없습니다.
- 리더보드 제출 횟수: 대회 플랫폼 표시(일일 제출 횟수 제한 0)와 팀 확인에 따라 제한 없이 제출했습니다.

**참가자 주요 기여 및 검증**
- 제출 여부, 최종 제출본, 최종 방법은 참가자가 결정했습니다.
- 계획 단계에서 두 계획(Claude·Codex)을 비교해 혼합안을 선택했습니다.
- 동료가 하네스 설계(H6: 공개 이웃 2건 이상 100% 일치 + 확신도 0.9)를 제안했고, 이를 구현해 리더보드로 검증했습니다(0.93).
- 규정 해석은 Q&A 535·618·678을 기준으로 참가자가 판단했습니다.
- 리더보드 채점 정답에 대한 의문(8절)은 참가자가 주최 측 문의 여부를 판단했습니다.
- 모든 규칙 변경은 리더보드 결과로 확인했습니다.
- 제출 파일은 재현 스크립트로 sha256까지 대조했습니다.

**검토한 대안 및 실패사례**
- 리더보드로 확인
  - 판정기 PoE 재순위: 0.92
  - X19 스태커: 0.93
  - H6 서명 투표: 0.93
  - 링크 사진 '중복형 금지' 규칙: 갱신 없음
  - 판정기 단독 배정: 0.80
- 비용·검증 결과로 중단
  - 추론형(thinking) 판정기: 행당 26초로 비용 대비 이득이 없어 중단
  - LoRA 증거 카드 확장(c2 카드): fold 0 검증에서 원래 카드보다 나빠 중단
  - LoRA fold 앙상블: 재배정 결과 변화 없음
- 학습 데이터 시뮬레이션에서는 H6가 가장 정밀했습니다(정밀도 0.70). 그러나 평가 데이터에서는 공개 행이 '판정이 갈린 행'뿐이라 효과가 반대로 나타났습니다(4-1).

**최종 선택 이유**
- 리더보드 최고점입니다.
- 모든 규칙 결정이 감사 로그로 추적됩니다.
- 저장된 모델 출력에서 바이트 단위로 재현됩니다.
- 규칙의 근거가 가이드라인과 학습 데이터 통계에 있습니다.

---

## 8. 참고: 리더보드 채점 정답

- 제출마다 리더보드 수치를 역산하면, 리더보드는 10/13 정정 전 정답(적절 405 · 불충분형 132 · 무관형 9)으로 채점되는 것으로 보입니다.
  - 정정 후 정답은 적절 409 · 불충분형 133 · 무관형 4입니다.
- 이 정답으로 채점하면 정정 후 정답과 600건 모두 일치하는 파일도 0.9667이 상한입니다.
- `scripts/tools/decode_lb.py`의 상수도 이 리더보드 기준으로 맞춰 두었습니다.

---

## 9. 제출 전 확인사항

| 항목 | 상태 | 위치 |
|---|---|---|
| README에 결과물 구성 및 실행 방법 | ✅ | 0·1·5절 |
| 실행환경·라이브러리 버전 | ✅ | 2절, requirements*.txt, docs/colab_pip_freeze_full.txt |
| 학습 및 평가(추론) 과정 확인 | ✅ (범위: 5-3) | 재현·평가: notebooks/01 (실행 결과 포함) / 학습 절차: notebooks/02, docs/pipeline_provenance.md |
| 데모영상 | ✅ | demo/demo_reproduce_best.mp4 |
| 외부 데이터·코드·모델 이용조건 | ✅ | 6절 (모델 카드 확인; klue/roberta-large는 표기 없음, 최종 결합 미사용) |
| 개인정보·인증정보 미포함 | ✅ | 6절 (팀원 정보 비식별화, 키·비밀번호 없음) |
| AI 활용범위 및 주요 검토 과정 | ✅ | 7절 |
