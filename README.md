# KoAltQ: 이미지·문맥 정보를 활용한 한국어 웹 대체텍스트 품질 판정

2026 DATA·AI 분석 경진대회 문제 4 · 팀 **못나지않은사람들**

> 웹 이미지와 alt 텍스트, 주변 문맥, OCR 정보를 활용해 대체텍스트 품질을 7개 범주로 분류합니다.
> 저장된 모델 출력으로 최종 제출 파일을 CPU에서 바이트 단위로 재현하는 방법과, Google Colab·Linux GPU 서버에서 처음부터 학습·추론하는 절차를 제공합니다.

| 항목 | 내용 |
|---|---|
| 최종 제출 파일 | `submission/final_submission_0.9357.csv` (600행, sha256 `87c3de688bd20937dc86abb9fb5e1171d6d012cd132aa23bd1c3f9611a9a1cf8`) |
| 리더보드 | **0.9357** = 0.5 × 이진 F1 **0.9579** + 0.5 × 7범주 macro-F1 **0.9134** (2026-11-04 06:28 제출) |
| 과제 | 웹 이미지의 alt 텍스트를 7범주로 판정: 적절 / 무의미형 / 파일명형 / 중복형 / 장식오용형 / 불충분형 / 무관형 |
| 재현 범위 | 저장된 모델 출력(`model/runs_frozen/`)에서 최종 파일을 **바이트 단위로 동일하게** 다시 만듦 (CPU, 3~8분). GPU 재학습 절차는 5-3·5-4절 |
| 학습된 가중치 | **제출물에 없음.** 최종 LoRA 어댑터는 보존되지 못했고, 학습된 모델들의 출력(확률)만 보존 (1절, `model/README.md`) |
| 외부 API | 예측 파이프라인은 외부 API를 쓰지 않음. 모든 모델은 공개 가중치를 내려받아 GPU에서 직접 실행. 개발 과정의 AI 도구 사용은 7절 |

**핵심 요약**
- **성능**: 리더보드 0.9357. 기존 팀 제출(0.8374)보다 0.098 높습니다. 단계별 효과는 4-2절에 있습니다.
- **방법**
  - 지도학습 모델: LightGBM 스태커와 Qwen LoRA 분류기
  - 라벨링 가이드라인을 그대로 읽힌 VLM 판정기의 확률을 스태커 특징으로 사용
  - 하네스: 주최 측의 '전수 재판정' 구조에서 끌어낸 **검수 일치 원리**(4-1절)로, 근거가 있는 행만 보정
- **검증 가능성**
  - 규칙이 정하거나 바꾼 260행은 모두 근거와 함께 감사 로그로 남습니다.
  - 저장된 모델 출력으로 CPU에서 약 3분 만에 최종 파일을 바이트 단위로 재현합니다. 제출 압축본을 새 폴더에 풀어서도 확인했습니다.
- **누수 통제**
  - 같은 페이지·템플릿 묶음(277개)이 한 fold에 들어가도록 5-fold를 고정했습니다.
  - 결합 규칙은 할당량 배정까지 흉내 낸 fold별 정직 시뮬레이션으로 골랐습니다(4-3절).
  - 행 단위 리더보드 정보는 쓰지 않았습니다.
- **한계**
  - 최종 LoRA 가중치는 보존되지 못해 출력만 제출합니다(1절).
  - GPU 재학습은 비트 단위 재현 대상이 아닙니다(5-4절).

---

## 0. 빠른 시작 (심사·재현용)

**로컬 (Linux · macOS. Windows는 WSL 또는 Colab)**

```bash
# 1) 환경 구성: .venv 생성 + requirements.txt, requirements-notebook.txt 설치
bash scripts/setup_env.sh python3.12          # 또는 python3.13

# 2) 최종 제출 파일 재현 + 무결성 확인
PYTHON=.venv/bin/python bash scripts/reproduce_best.sh

# 3) (선택) 데모: 재현 → CSV 확인 → 결과 뷰어 열기
PYTHON=.venv/bin/python bash demo/run_demo.sh
```

- 결과: `outputs/reproduced_submission.csv`
- 마지막 줄이 `OK - identical to submission/final_submission_0.9357.csv`이면 재현 성공입니다.
- `outputs/harness_log.jsonl`: 규칙이 바꾸거나 고정한 모든 행과 그 근거(감사 로그)
- `outputs/result_viewer.html`: 600행의 이미지·alt·판정 라벨·하네스 근거를 보는 결과 뷰어 (데모 스크립트가 생성)
- 노트북: `.venv/bin/jupyter notebook notebooks/01_reproduce_best.ipynb` → 위에서부터 실행
- 이 경로는 저장된 `model/runs_frozen/`을 쓰므로 GPU, 모델 다운로드, 재학습이 필요 없습니다.

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
├── requirements-gpu.txt       전체 학습·추론 환경 (Colab A100 기준, 상세 버전)
├── data/                      학습·평가에 사용한 데이터 (data/README.md)
│   └── koaltq_bundle/         data/*.jsonl, images/(3,000장, 정규화본), docs/guideline_v1.1.md, MANIFEST.json
├── model/                     학습된 모델의 출력 (model/README.md). 가중치 파일은 없음
│   └── runs_frozen/           단계별 OOF·test 확률 (최종 결정 단계의 입력)
│       └── logs/              원래 학습 실행 로그 (단계별 시작 인자·소요 시간·지표)
├── src/koaltq2/               소스코드 (단계별 모듈, `python -m koaltq2.run <단계>`)
├── notebooks/
│   ├── 01_reproduce_best.ipynb         CPU: 데이터 확인 → 결정 단계 → sha256 대조 → 모델별 평가 → 감사 로그 → 그림 (실행 결과 포함)
│   └── 02_full_pipeline_colab.ipynb    GPU(Colab A100): 전체 학습·추론을 원래 순서대로
├── scripts/
│   ├── setup_env.sh                    CPU 환경 구성
│   ├── reproduce_best.sh               최종 파일 재현 + sha256 대조
│   ├── make_result_viewer.py           제출 CSV → 이미지·alt·판정 라벨 결과 뷰어(HTML)
│   ├── build_bundle.py                 대회 배포본 → 데이터 번들 생성 (원래 작업 폴더 구조 필요, data/README.md)
│   └── colab/                          원래 Colab 셀 스크립트와 작업 큐 실행기 (기록용. 지금 실행은 노트북 02)
├── submission/                최종 제출 파일 + sha256
├── demo/                      데모영상, 데모 실행 스크립트 run_demo.sh (demo/README.md)
├── figures/                   설명용 그림 (리더보드 변화, 클래스 개수, 하네스 결정, OOF 혼동행렬; 노트북 01이 생성)
└── docs/
    ├── method_report_harness.txt       방법·하네스·실험 결과 정리 (보고서 원고)
    ├── pipeline_provenance.md          최종 제출에 쓰인 모든 모델 출력의 생성 명령·순서·소요 시간 (실행 로그에서 복원)
    ├── experiment_log.md               실험 기록 (시간순, 개인정보 비식별화)
    └── colab_pip_freeze_full.txt       학습에 쓴 Colab 런타임의 전체 패키지 목록 (머리말에 캡처 시점)
```

- `outputs/`는 스크립트를 실행하면 생기는 폴더입니다(제출물에는 포함하지 않음).
- **model/ (학습된 모델 파일)**
  - 최종 제출에 쓰인 LoRA 분류기(Qwen3.8-27B, Qwen3.5-9B 전체 학습본)의 어댑터 가중치는 Colab 세션을 종료하기 전에 백업 대상에서 빠져 보존되지 못했습니다.
  - 대신 학습된 모델들의 출력(OOF·test 확률)을 모두 보존했고, 이 출력으로 최종 제출 파일을 바이트 단위로 다시 만듭니다(5-1절).
  - 가중치는 같은 레시피로 다시 학습할 수 있습니다(5-3·5-4절). GPU 비결정성 때문에 확률은 조금 달라질 수 있습니다.
- **제출본 정리**: 최종 파일에 쓰이지 않은 실험 모듈(교차 인코더 x11, 희소 클래스 실험 x13, 추론형 판정기 x18, LoRA 스태커 x19 등)과 리더보드 분석 도구는 제출본에서 뺐습니다. 관련 실험 출력도 함께 뺐으며, 뺀 뒤에도 재현 파일의 sha256이 같음을 확인했습니다. 실험 결과와 경과는 `docs/`에 기록으로 남겼습니다.

**소스코드 (`src/koaltq2/`)**

| 모듈 | 역할 |
|---|---|
| `common.py`, `rules.py`, `prompting.py`, `submit.py` | 데이터 로딩·평가 지표, 가이드라인 규칙과 구조 특징, VLM 입력(증거 카드), 제출 파일 쓰기 |
| `x0_setup.py` | 데이터 계약 검사 (파일 수, 건수, fold 해시) |
| `x1_gbm.py` | 규칙·구조 특징 + TF-IDF → GBM / LR, 교차 검증 공통 함수 |
| `x2_ocr.py`, `ocr_worker.py` | PaddleOCR PP-OCRv5 korean (격리 venv에서 실행) → OCR 특징 |
| `x3_frozen.py` | 고정 VLM 임베딩 (Qwen3.5-9B, InternVL3.5-14B) |
| `x6_lora.py` | LoRA 분류기 (Qwen3.8-27B, Qwen3.5-9B) |
| `x12_attr.py` | VLM 예/아니오 속성 질의 |
| `x14_judge.py` | 가이드라인 판정기 (7범주 확률; `--ask`로 예/아니오 판정기 x16) |
| `x15_judge_lr.py` | 판정기 출력 → 로지스틱 회귀 |
| `x7_stack.py`, `x17_boundary.py` | 스태커 특징 조립, 최종 LightGBM 스태커 x17·x17fx |
| `x8_blend.py`, `lbest.py` | 결정 단계: 결합 → 할당량 → 하네스 (`lbest`는 리더보드 산식을 학습 데이터로 모의 계산해 결합 규칙을 고르는 목적함수) |
| `harness.py` | 구조 서명 일관성 하네스 H0~H6 (4-1절) |
| `run.py` | 단계 실행기 |

---

## 2. 실행 환경

### 2-1. 재현 (CPU)

| 항목 | 검증한 환경 1 (Colab) | 검증한 환경 2 (로컬) |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS | macOS (Apple M4 Pro) |
| Python | 3.13.16 | 3.12.9 |
| numpy / scipy / scikit-learn | 2.4.6 / 1.16.3 / 1.6.1 | 2.5.4 / 1.18.1 / 1.9.1 |
| 결과 | sha256 일치 (빈 폴더에서 재현, 노트북 01 실행 포함) | sha256 일치 (제출본 정리 후 2026-11-06 재확인) |

`requirements.txt`는 환경 1의 버전으로 고정했습니다(matplotlib 3.10.0 포함). GPU는 필요 없습니다.

### 2-2. 전체 학습·추론 (GPU, Google Colab)

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

### 2-3. 전체 학습·추론 (GPU, Linux SSH 서버)

팀원이 2026-11-05에 연구실 서버에서 다음 구성을 직접 확인했습니다.

| 항목 | 확인값 |
|---|---|
| OS / Python | Ubuntu 24.04.4 LTS / Python 3.12.3 |
| GPU | NVIDIA RTX PRO 5000 Blackwell 48GB × 3 |
| NVIDIA Driver | 595.71.05 |
| `nvidia-smi` CUDA Version | 13.2 |
| PyTorch / CUDA 빌드 | torch 2.11.0+cu130 / `torch.version.cuda` 13.0 |
| 주요 패키지 | torchvision 0.26.0+cu130, transformers 5.22.1, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, triton 3.6.0, flash-linear-attention 0.6.0 |

- `nvidia-smi`의 `CUDA Version 13.2`는 드라이버가 지원하는 최고 버전입니다. PyTorch는 CUDA 13.0 휠(`cu130`)로 빌드되었고 이 드라이버에서 실행됩니다.
- 5-3절의 환경 변수와 의존성을 설정하면 일반 Linux GPU 서버에서도 같은 단계를 실행할 수 있습니다. 27B·32B 모델은 80GB급 GPU가 필요합니다.

---

## 3. 데이터

파일별 설명은 `data/README.md`에 있습니다.

- `train.jsonl` 2,400건(라벨 포함), `test.jsonl` 600건, `image_meta.jsonl`, `images/` 3,000장
  - 각 레코드: 페이지 URL·제목, 이미지 URL, alt, 주변 문맥, 링크 여부·목적지, 헤딩 경로
  - 레코드는 대회 배포본 그대로입니다.
  - 이미지는 모델 입력용으로 정규화했습니다(형식 판별, SVG→PNG, 투명 배경→흰색, 최장변 1536px).
- `folds.jsonl`: 학습 데이터 5-fold. 같은 페이지·템플릿 묶음(277개 그룹)이 한 fold에 들어가도록 고정해 교차 검증 누수를 줄였습니다.
- 공개 경계사례(평가 데이터 109건): `boundary_test_labels.jsonl`(데이터셋_라벨, 공식 정답)과 `boundary_test_rejudge.jsonl`(재판정_라벨 포함)
  - Q&A 678: "학습, 예측 결과 후처리, 최종 제출 어디에 쓰셔도 제한하지 않습니다"
  - 코드에 명시된 단계(스태커 학습, LoRA 전체 학습, 하네스)에서만 쓰며, 5-fold OOF 평가와는 구분합니다.
- `docs/guideline_v1.1.md`: 대회 라벨링 가이드라인 (판정기 프롬프트에 사용)
- 외부 데이터는 쓰지 않았습니다.
  - `pseudo_test_v*.jsonl`(모델이 만든 의사 라벨)은 비교 실험에만 쓰였고 최종 제출에는 쓰지 않았습니다.

7개 라벨(가이드라인 2절, `src/koaltq2/common.py`의 순서)

| 라벨 | 정의 (가이드라인 v1.1) |
|---|---|
| 적절 | 이미지의 역할에 맞는 정보를 전달 |
| 무의미형 | 내용 없는 상투어·기호 나열 ("이미지", "사진", "*" 등) |
| 파일명형 | 파일명·경로·확장자 그대로 |
| 중복형 | 인접 텍스트(캡션·제목·링크 문구)를 그대로 반복해 정보가 0 |
| 장식오용형 | 순수 장식 이미지에 내용성 alt를 부여 (`alt=""`가 정답인 경우) |
| 불충분형 | 방향은 맞으나 핵심 정보 누락 |
| 무관형 | 이미지 내용과 불일치 |

여러 유형에 해당하면 `파일명형 > 무관형 > 무의미형 > 장식오용형 > 중복형 > 불충분형` 순으로 하나만 붙입니다(가이드라인 3-2절). 세부 경계 규칙은 가이드라인 3-3절을 따릅니다.

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

단일 모델(x1, x2, x3 헤드, x7 스태커)은 결정 단계에서 후보로만 비교했고, 최종 결합에는 쓰이지 않습니다.

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
| H6 서명 투표 (선택) | 팀원 설계: 같은 서명(페이지·링크·링크 종류·alt 관계·이미지 파일명 패턴)의 공개 이웃 2건 이상 100% 일치 + 확신도 0.9 이하 → 교체 | 기본 꺼짐 (LB 0.93) |

![하네스 모듈별 결정 행 수 (평가 600행)](figures/harness_modules.png)

핵심 원리(검수 일치):
- 주최 측은 평가 600건 전체를 독립 재판정했고, 원 라벨과 재판정이 갈린 건을 경계사례로 공개했습니다.
- 공개되지 않은 491건은 10/13 라벨 재검증에서 정정된 5건을 빼면 두 판정이 일치한 행입니다.
- 이런 행의 정답은 P(y | x, 일치) ∝ P원라벨러(y | x) × P검수자(y | x)를 따릅니다.
- 공개 쌍둥이의 재판정 라벨을 옮기는 것(H2·H4)은 검수자의 일관성을 이용하는 것입니다.
- 원 라벨러가 그 구조에 그 라벨을 거의 주지 않는다면(학습 데이터 기준) 옮기지 않습니다(H5).
- H5는 16번 제출 결과(H4 9행 중 3행 오답)를 본 뒤 설계했습니다.
  - 기준은 학습 라벨과 모델 점수만 씁니다. 행 단위 리더보드 정보는 쓰지 않았습니다.
  - 기준을 정한 뒤 한 번 제출해 확인했습니다(3/3 정답).

검증 하네스(학습 데이터 시뮬레이션), conformal 분석, H6 실험의 결과는 `docs/method_report_harness.txt`에 있습니다(분석 코드는 최종 파일에 쓰이지 않아 제출본에서 제외).

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

![단계별 리더보드 점수](figures/lb_progression.png)

효과가 없었던 대표 시도:
- 판정기를 검수자 대리로 쓴 재순위(PoE): 0.92
- LoRA 5-fold OOF 스태커(X19): 0.93
- H6 서명 투표: 0.93
- 링크 안 사진 '중복형 금지' 규칙: 0.94 구간, 갱신 없음
- 판정기 단독 배정: 0.80

### 4-3. 검증 성능 (학습 데이터 OOF)

각 모델의 fold 밖 예측을 학습 라벨과 비교한 값입니다(`model/runs_frozen/*/metrics.json`, 노트북 01의 4절).
대회 지표는 0.5 × 이진 F1 + 0.5 × macro-F1입니다.

| 모델 | 검증 행 | 대회 지표 | 이진 F1 | macro-F1 |
|---|---:|---:|---:|---:|
| x1 규칙·구조 특징 + TF-IDF → HGB | 2,400 | 0.5907 | 0.7026 | 0.4789 |
| x2 OCR 특징 → HGB | 2,400 | 0.6435 | 0.7793 | 0.5077 |
| x3 Qwen3.5-9B 임베딩 → LR | 2,400 | 0.5853 | 0.6943 | 0.4762 |
| x3 InternVL3.5-14B 임베딩 → LR | 2,400 | 0.5863 | 0.6939 | 0.4787 |
| x15 판정기 LR (결합 비중 0.1) | 2,400 | 0.6881 | 0.7342 | 0.6421 |
| x7 LightGBM 스태커, 판정기 없음 (특징 264개) | 2,400 | 0.6095 | 0.7918 | 0.4271 |
| x7j 같은 스태커 + 판정기 특징 (364개, 비교 후보) | 2,400 | 0.7763 | 0.8361 | 0.7165 |
| **x17 최종 스태커 (결합 비중 0.9)** | 2,400 | 0.7697 | 0.8357 | 0.7036 |
| x6 Qwen3.5-9B LoRA | 458 (fold 0) | 0.5948 | 0.6891 | 0.5004 |
| x6 Qwen3.8-27B LoRA | 458 (fold 0) | 0.7227 | 0.7802 | 0.6652 |
| 결합: 스태커 + 판정기 LR | 458 (fold 0) | 0.7622 | 0.8289 | 0.6955 |
| **결합: + LoRA 0.4 (최종 결합)** | 458 (fold 0) | **0.7691** | **0.8561** | 0.6822 |

- **판정기 특징이 핵심입니다.** 같은 LightGBM 스태커에 판정기 특징을 더하면 0.6095 → 0.7763으로 오릅니다. macro-F1은 0.43 → 0.72로, 희소 클래스에서 효과가 큽니다.
  - 판정기만 따로 쓰면(x15 0.69, 판정기 단독 배정 LB 0.80) 약하므로, 스태커 특징으로만 씁니다.
- **결합 규칙 선택**
  - argmax OOF로는 x7j(0.7763)가 x17(0.7697)보다 조금 높습니다.
  - 하지만 실제 제출 방식인 할당량 배정까지 흉내 낸 fold별 정직 시뮬레이션에서는 x17이 가장 높습니다: x17 0.8811 > x7j 0.8779 > Caruana 앙상블 0.8669 > 단일 최고 0.8660.
  - 그래서 x17을 골랐습니다(`outputs/report.json`의 `perfold_quota`).
- **LoRA 비중**
  - fold 0 격자: 0.0 0.8709 / 0.2 0.8817 / 0.4 0.8809 / 0.6 0.8719. 0.2와 0.4가 사실상 같습니다.
  - 리더보드 A/B 비교로 0.4를 골랐습니다: 0.2 → 0.87, 0.4 → 0.88, 0.6 → 0.91(당시 최고 0.9194보다 낮음), 27B LoRA 단독 → 0.88.
- 평가 데이터의 실제 점수(0.9357)가 OOF보다 높은 이유는 두 가지입니다.
  - 평가 600행 중 공개 109행과 규칙 확정 124행이 정답 근거를 가집니다.
  - 나머지 행은 공식 베이스라인 문서의 평가셋 클래스 분포(적절 409 · 무의미형 5 · 파일명형 8 · 중복형 35 · 장식오용형 6 · 불충분형 133 · 무관형 4)에 맞춰 배정됩니다.
- 남은 주요 오류는 가이드라인에서도 판단이 개입하는 **적절 ↔ 불충분형** 경계입니다(아래 혼동행렬). 하네스 H2·H4·H5가 이 경계를 겨냥합니다.

![x17 스태커 5-fold OOF 혼동행렬](figures/confusion_x17_oof.png)

![최종 제출의 클래스 개수와 평가셋 클래스 분포](figures/class_counts.png)

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

- 필요한 입력은 `data/koaltq_bundle/`과 `model/runs_frozen/`뿐입니다.
- 환경변수 `KOALTQ_BUNDLE`(데이터 번들), `KOALTQ_RUNS`(모델 출력 폴더), `PYTHONPATH=src`는 스크립트가 설정합니다.
- 기존 플래그 조합(`KOALTQ_SOLE_NOT_DECO=1 KOALTQ_SIB_K2_OK=1 KOALTQ_K2_VETO=1`)도 같은 파일을 만듭니다.

### 5-2. 데모 (CPU)

`bash demo/run_demo.sh` (인터프리터는 `PYTHON=...`로 지정)

1. 5-1의 결정 단계를 실행해 `outputs/reproduced_submission.csv`를 만들고 sha256을 대조합니다.
2. 생성된 CSV의 앞 10행과 라벨별 개수를 출력합니다.
3. `scripts/make_result_viewer.py`로 `outputs/result_viewer.html`을 만들고, CSV와 뷰어를 기본 앱으로 엽니다(`DEMO_NO_OPEN=1`이면 열지 않음).
   - 뷰어는 600행 각각의 이미지·alt·페이지·판정 라벨과, 하네스가 정하거나 바꾼 행의 근거 모듈을 보여 줍니다. 라벨별 필터와 검색을 지원합니다.

데모영상은 `demo/`에 있습니다(`demo/README.md`).

### 5-3. 전체 학습·추론 (GPU, Linux SSH 서버)

저장소 루트에서 실행합니다. 기존 가상환경이 있으면 덮어쓰지 말고 별도 경로를 씁니다.

```bash
cd /path/to/KoAltQ_submission_0.9357

python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-gpu.txt

# 물리 GPU 2번을 쓰는 예시입니다. 사용 가능한 GPU 번호로 바꿉니다.
export CUDA_VISIBLE_DEVICES=2
# 한 장치만 노출하면 프로세스 안에서는 cuda:0으로 보입니다.
export KOALTQ_DEVICE=cuda:0
export KOALTQ_CONTENT="$PWD/data"
export KOALTQ_BUNDLE="$PWD/data/koaltq_bundle"
export KOALTQ_RUNS="$PWD/runs_rebuild"
export PYTHONPATH="$PWD/src"
# OCR 격리 venv 위치 (기본값은 Colab의 /content)
export KOALTQ_OCR_VENV_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/koaltq/ocr_venvs"
mkdir -p "$KOALTQ_RUNS"

python -u -m koaltq2.run x0
```

`x0`가 `manifest ok: 3007 files …`와 `data ok: train 2400 test 600 fold0 458 boundary_test 109 …`를 출력하면 데이터 계약 검사가 통과한 것입니다. 팀원이 48GB 서버에서 출력을 확인한 단계는 다음과 같습니다.

```bash
# 기초 특징과 OCR
python -u -m koaltq2.run x1
python -u -m koaltq2.run x2

# 고정 VLM 임베딩
python -u -m koaltq2.run x3 --model qwen35_9b
python -u -m koaltq2.run x3 --model internvl35_14b

# 9B LoRA fold 0 검증, 속성 질의
python -u -m koaltq2.run x6 --mode fold --backbone qwen35_9b
python -u -m koaltq2.run x12 --model qwen35_9b

# 9B LoRA 전체 학습
python -u -m koaltq2.run x6 --mode full --backbone qwen35_9b
```

| 단계 | 주요 입력 | 주요 출력 (`$KOALTQ_RUNS`) | 완료 확인 |
|---|---|---|---|
| x0 | bundle, manifest | 없음 (검사만) | `manifest ok`, `data ok` |
| x1 | train/test JSONL | `x1_*` OOF·test 확률 | `metrics.json`, `test.npy` |
| x2 | 3,000장 이미지 | `x2_ocr/ocr.jsonl`, OCR 특징, `x2_hgb_ocr/` | OCR 3,000행, `ocr_engine.txt` |
| x3 | bundle + x2 OCR | `x3_<model>/feats.npy`, LR/MLP 헤드 | `meta.json`, 헤드별 `metrics.json` |
| x6 fold | bundle + x2 OCR | fold 0 OOF·test 확률, `adapter_ep*` | `metrics.json`, `oof.npy` |
| x12 | bundle | `x12_<model>/attrs.npy` | 3,000행 `attrs.npy`, `meta.json` |
| x6 full | train 2,400 + 공개 109, x2 OCR | 전체 학습 test 확률, `adapter_ep*` | `test.npy`, `meta.json`의 `reload_check: pass` |

- x2는 PyTorch 환경과 Paddle CUDA 패키지가 충돌하지 않도록 `$KOALTQ_OCR_VENV_ROOT/venv_ocr_paddle`을 자동으로 만듭니다.
  - 첫 실행 때 `paddlepaddle-gpu==3.4.0`, `paddleocr==3.7.0`과 PP-OCRv5 korean 모델을 내려받으므로 네트워크와 쓰기 권한이 필요합니다.
  - OCR은 중간 JSONL을 행 단위로 기록합니다. 다시 실행하면 정상 행은 두고 빠진 행만 다시 시도합니다. Paddle이 실패하면 EasyOCR로 다시 시도합니다.
- 모델 가중치는 Hugging Face 기본 캐시를 씁니다. `HF_HOME`을 설정하지 않으면 사용자 홈 캐시에 저장됩니다.
- **x6가 저장하는 `adapter_ep*`(LoRA 가중치)는 실행이 끝나면 바로 별도 위치로 복사해 보존합니다.**
- 메모리: 48GB 서버에서 Qwen3.5-9B x3는 최대 19.95GB, x6 fold/full은 최대 19.06GB를 썼습니다. 원래 Colab 로그에서 27B 판정기는 약 54GB, 27B LoRA는 약 56.8GB, Qwen3-VL-32B 속성 질의는 약 62.6GB를 썼으므로, 27B·32B 단계는 48GB 장치에서 그대로 실행하지 않습니다.
- 전체 단계의 정확한 순서·플래그·소요 시간은 `docs/pipeline_provenance.md`와 노트북 02를 따릅니다.
- SSH 서버에서 Jupyter를 쓸 때는 같은 가상환경에 `requirements-notebook.txt`를 설치하고 그 커널을 선택합니다. 노트북 02의 `/content` 경로는 위 환경 변수로 바꿉니다.

### 5-4. 전체 학습·추론 (GPU, Google Colab A100)

`notebooks/02_full_pipeline_colab.ipynb`를 Colab(A100 80GB)에서 순서대로 실행합니다.
- 명령과 순서는 원래 실행 로그와 작업 큐에서 복원했습니다(`docs/pipeline_provenance.md`).
- 준비·x0·x1 셀은 Colab에서 실행해 확인했습니다. GPU 단계 전체를 한 번에 다시 돌려 보지는 않았습니다.

1. 준비(제출물 폴더 배치, 패키지 설치, x0 데이터 계약 검사)
2. 기초 특징: x1, x2(OCR), x3(Qwen3.5-9B, InternVL3.5-14B 임베딩)
3. 9B LoRA(fold 0 검증·전체 학습), x12 속성 질의 3종
4. 판정기 x14·x16 (Qwen3.8-27B 변형 8개 + 다른 모델 판정기 + x12 27B)
5. **체크포인트(`x14_*` 13개) → x7·x15·x17 스태커**
6. 27B LoRA(fold 0 검증), 판정기 rot1·rot6, 27B LoRA 전체 학습
7. x17fx (`x14_*` 15개 시점)
8. 결정 단계 + 하네스 → 제출 파일과 비교

유의점:
- A100 기준 GPU 시간은 약 15시간(대부분 27B 판정기와 LoRA), CPU 단계는 약 20분입니다.
  - 노트북은 끝난 단계를 건너뛰는 이어하기와, 출력을 Google Drive에 두는 옵션(`USE_DRIVE`)을 지원합니다.
  - Colab 런타임이 끝나면 `/content`가 지워지므로, `adapter_ep*` 가중치는 Drive 등에 따로 보존합니다.
- **순서 제약**: 스태커는 runs 폴더의 판정기 결과를 모두 읽어 특징을 만듭니다.
  - x7·x15·x17은 `x14_*`가 정확히 13개일 때(특징 364·124개), x17fx는 15개일 때(378개) 실행합니다.
  - 노트북에 assert가 있습니다.
- **재현 범위**
  - 최종 제출에 쓰인 LoRA 전체 학습본(27B·9B)의 어댑터 가중치는 보존되지 못했고, 그 test 확률만 `model/runs_frozen/`에 있습니다.
  - VLM 임베딩 원본(`x3_*/feats.npy`, 약 0.45GB)도 용량 때문에 넣지 않았습니다. 그래서 `model/runs_frozen/`만으로 스태커를 다시 학습하면 특징이 달라집니다(argmax 97.5% 일치).
  - GPU 학습은 비결정적 요소가 있어, 처음부터 다시 학습하면 확률이 조금 달라지고 최종 파일도 몇 행 다를 수 있습니다.
  - 제출 파일과의 정확한 일치는 5-1(저장된 모델 출력)로 확인합니다.

Linux SSH와 Colab의 차이:

| 항목 | Linux SSH | Google Colab |
|---|---|---|
| 작업 경로 | 저장소 루트 | `/content/KoAltQ_submission_0.9357` |
| 데이터 | `$PWD/data/koaltq_bundle` | 노트북이 지정한 경로 |
| 출력 | 사용자 지정 `$KOALTQ_RUNS` | `/content/runs_rebuild` 또는 Google Drive |
| GPU 선택 | `CUDA_VISIBLE_DEVICES`로 물리 GPU 선택 | 런타임이 제공하는 A100 1장 |
| OCR venv | `$KOALTQ_OCR_VENV_ROOT/venv_ocr_*` | `/content/venv_ocr_*` (기본값) |
| 지속성 | 서버 디스크에 캐시·출력 유지 | 런타임 종료 시 `/content` 소실, Drive 사용 가능 |

---

## 6. 외부 자료 활용 및 이용 조건

| 자료 | 용도 | 출처 · 라이선스 (Hugging Face 모델 카드, 2026-11-04 확인) | 학습에 쓴 리비전 |
|---|---|---|---|
| Qwen/Qwen3.8-27B | 판정기, 27B LoRA, 속성 질의 | Apache-2.0 | `1d4bf0f2ff60…` |
| Qwen/Qwen3.5-9B | 임베딩, 9B LoRA, 속성 질의, 판정기 | Apache-2.0 | `c20223623576…` |
| Qwen/Qwen3-VL-32B-Instruct, Qwen3-VL-8B-Instruct | 속성 질의, 판정기 | Apache-2.0 | 기록 없음 (현재 `0cfaf48183f5…`, `0c351dd01ed8…`) |
| OpenGVLab/InternVL3_5-14B-HF | 임베딩 | Apache-2.0 | `226b96d5912e…` |
| PaddleOCR PP-OCRv5 (paddleocr 3.7.0) | 이미지 속 글자 인식 (가중치는 PaddleX 공식 저장소에서 자동 다운로드) | Apache-2.0 | - |
| klue/roberta-large | 개발 중 비교 후보(교차 인코더). 최종 파일에 쓰이지 않아 제출본에서 제외 | 모델 카드에 라이선스 표기 없음 | 기록 없음 |
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
| Claude Code (Anthropic) | CLI 2.1.311~2.1.318, 모델 Claude Opus 5.5 (`claude-opus-5-5`), 2026-10-27~11-06 | 문제 분석, 아이디어 탐색, 데이터 처리, 모델 설계, 코드 작성, 실험 운영, 결과 분석, 문서 작성 | `src/koaltq2` 모듈 구현, Colab 작업 큐 운영, 리더보드 점수 역산 분석(개발용, 제출본 제외), 하네스·검증 하네스·conformal 분석, README·보고서 초안, 제출본 정리와 재현 검증(11-06) |
| Codex CLI (OpenAI, oh-my-codex) | GPT-6-Astra (주 에이전트), GPT-6-Luna (하위 에이전트), 일부 GPT-6.1-Sol | 문제 분석, 실험 (2026-10-27, 11-01) | 대회 데이터로 VLM 베이스라인을 돌려 보는 초기 탐색, Colab A100에서 다른 VLM 후보(Kanana·Qwen 계열 등) 비교 실험 시도 |
| Codex CLI (OpenAI) | GPT-6.1-Sol, reasoning high, 읽기 전용 | 계획 (2026-11-02) | Claude 계획(Plan A)과 독립적으로 작성한 실험 계획(Plan B). 팀이 비교 후 혼합안 선택 |
| Codex CLI (OpenAI) | GPT-6-Astra, 읽기 전용 | 계획 (2026-11-05) | 제출본 정리·데모영상 계획(Plan B). Claude 계획과 비교 |
| OpenAI Codex (팀원 사용) | 모델명 기록 없음 (팀원 로컬 환경) | 문서 작성 (2026-11-05) | 기존 문서·스크립트·CLI 대조, 연구실 Linux 서버 실행 환경 확인, README 통합 초안(2-3·5-3절, 부록 A) |
| Google Colab CLI | 0.7.5 | 실험 실행 | Claude Code가 Colab A100 세션에서 학습·추론 작업을 실행하고 결과를 내려받음 |
| Orca computer use (화면 제어) | - | 리더보드 제출 | 사용자 승인 하에 Claude Code가 사용자 Chrome에서 결과 파일 제출·점수 확인. 로그인은 사용자가 직접 함 |
| Claude Code 멀티 에이전트 리뷰 | 리뷰어 3 + 반박 검증 7 (11/02), 리뷰어 3 + 검증 1 (11/04, 2회) | 코드·결과물 검토 | 사전 코드 결함 7건 수정, 하네스 코드 동치성 확인, 결과물 점검(본 README 반영) |

**AI 사용과 관련해 밝혀 둘 사항**
- 초기 분석 단계(2026-11-03 13:00 이전)에 Claude Code가 대화 중 일부 학습·평가 데이터 레코드를 직접 열람·분석했습니다.
  - Q&A 618 검토 후에는 평가 레코드를 개별로 열람하지 않고 코드·집계 수준에서만 분석했습니다.
  - 평가 데이터를 사람이나 AI가 수작업으로 라벨링하지는 않았습니다.
  - 예측은 모두 공개 가중치 모델로만 했습니다.
- 개발 중에 리더보드 표시 점수로 "제출 간에 바뀐 행 중 몇 개가 맞았는지"를 역산하는 분석 도구를 썼습니다(규칙 효과 확인용).
  - 이 분석에서 리더보드는 10/13 정정 전 정답(적절 405 · 불충분형 132 · 무관형 9)으로 채점되는 것으로 보였습니다. 정정 후 정답은 적절 409 · 불충분형 133 · 무관형 4입니다.
  - 이 도구는 최종 파일을 만드는 데 쓰이지 않아 제출본에서 뺐습니다. 행 단위로 정답을 알아내 파일에 넣는 데에는 쓰지 않았습니다.
- 화면 제어 제출 중 2026-11-04 08:47에 창 인식 오류로 의도하지 않은 제출 2건이 기록됐습니다. 최종 점수에는 영향이 없습니다.
- 리더보드 제출 횟수: 대회 플랫폼 표시(일일 제출 횟수 제한 0)와 팀 확인에 따라 제한 없이 제출했습니다.
- 최종 LoRA 어댑터 가중치는 백업 과정에서 제외되어 보존되지 못했습니다(1절). 제출물에는 출력만 있습니다.

**참가자 주요 기여 및 검증**
- 제출 여부, 최종 제출본, 최종 방법은 참가자가 결정했습니다.
- 계획 단계에서 두 계획(Claude·Codex)을 비교해 혼합안을 선택했습니다.
- 팀원이 하네스 설계(H6: 공개 이웃 2건 이상 100% 일치 + 확신도 0.9)를 제안했고, 이를 구현해 리더보드로 검증했습니다(0.93).
- 팀원이 연구실 Linux GPU 서버에서 실행 환경과 9B 단계까지의 학습·추론을 확인했습니다(2-3·5-3절).
- 규정 해석은 Q&A 535·618·678을 기준으로 참가자가 판단했습니다.
- 리더보드 채점 정답에 대한 의문은 참가자가 주최 측 문의 여부를 판단했습니다.
- 모든 규칙 변경은 리더보드 결과로 확인했습니다.
- 제출 파일은 재현 스크립트로 sha256까지 대조했습니다. 제출본을 정리한 뒤에도 다시 대조했습니다.

**주요 의사결정과 근거** (AI가 분석·후보를 만들고, 참가자가 근거를 확인해 채택 여부를 정함)

| 결정 | 검토한 대안 | 근거 | 결과 |
|---|---|---|---|
| 공개 경계사례 109건을 학습·후처리에 사용 | 쓰지 않음 | Q&A 678에서 사용 허용을 확인 | H0 109행, 스태커·LoRA 전체 학습에 포함 |
| 판정기는 스태커 특징으로만 사용 | 판정기 단독 배정 | 단독 배정 LB 0.80. 스태커 특징으로 쓰면 OOF 0.61 → 0.78 (4-3절) | x7j·x17 스태커 |
| 할당량 배정 | argmax | 공식 베이스라인 문서의 평가셋 클래스 분포를 사용. 결합 규칙은 할당량을 흉내 낸 fold별 시뮬레이션으로 선택 | S3 결정 규칙, x17 + 판정기 LR |
| 희소 클래스 전용 판정(X13) 제거 | 유지 | 리더보드 A/B에서 제거 쪽이 높음 (0.87 → 0.89) | 제거 |
| LoRA 비중 0.4 | 0.2, 0.6, 27B 단독 | fold 0에서 0.2와 0.4가 동률이고, 리더보드 A/B에서 0.4가 가장 높음 (4-3절) | 0.4 |
| H5 검수 일치 거부 | H4를 그대로 적용 | 16번 제출에서 H4 9행 중 3행 오답. 학습 라벨·모델 점수만으로 거부 기준을 정한 뒤 1회 확인 | 3행 거부, 0.9357 |
| H6 서명 투표 미채택 | 채택 | 학습 시뮬레이션 정밀도는 0.70이었지만 리더보드 0.93으로 하락. 평가 공개 행이 '판정이 갈린 행'이라는 구조 차이로 설명 | 기본 꺼짐 |
| 의사 라벨 미사용 | 확신 높은 평가 행 244개의 의사 라벨로 스태커 추가 학습 | OOF(사이트 가중) 0.7721로 비슷하고, 같은 개수 재배정 시 2행만 바뀌어 근거 대비 위험이 큼 | 최종 미사용 |

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

## 8. 제출 전 확인사항

| 항목 | 상태 | 위치 |
|---|---|---|
| README에 결과물 구성 및 실행 방법 | ✅ | 0·1·5절 |
| 실행환경·라이브러리 버전 | ✅ | 2절, requirements*.txt, docs/colab_pip_freeze_full.txt |
| 학습 및 평가(추론) 과정 확인 | ✅ (범위: 5-4) | 재현·평가: notebooks/01 (실행 결과 포함) / 학습 절차: notebooks/02, docs/pipeline_provenance.md |
| 데모영상 | ✅ | demo/ (demo/README.md) |
| 외부 데이터·코드·모델 이용조건 | ✅ | 6절 (모델 카드 확인; klue/roberta-large는 표기 없음, 제출본 미포함) |
| 개인정보·인증정보 미포함 | ✅ | 6절 (팀원 정보 비식별화, 키·비밀번호 없음) |
| AI 활용범위 및 주요 검토 과정 | ✅ | 7절 |
| (참고) 학습된 모델 파일 | 출력만 포함 | 1절, model/README.md (LoRA 가중치 미보존) |

---

## 부록 A. 문제 해결

| 증상 | 확인 순서 |
|---|---|
| 재현 결과 sha256 불일치 | 새 출력 경로에서 다시 실행하고 `model/runs_frozen/`이 바뀌지 않았는지 확인합니다. GPU 전체 재학습은 비결정적 요소 때문에 바이트 단위 일치 대상이 아닙니다. |
| 원하지 않는 GPU를 씀 | `nvidia-smi`로 여유 장치를 확인한 뒤 `CUDA_VISIBLE_DEVICES=<물리 번호>`를 설정합니다. 한 장치만 노출하면 `KOALTQ_DEVICE`는 `cuda:0`입니다. |
| CUDA 메모리 부족 | 공유 서버라면 다른 작업을 종료하기 전에 관리자·작업자와 확인합니다. 27B·32B 단계는 80GB A100이 필요합니다. |
| Hugging Face 모델 다운로드 실패 | 네트워크, 디스크 여유, 캐시 쓰기 권한을 확인합니다. 공개 모델은 토큰이 필요 없습니다. |
| OCR 환경 생성·import 실패 | `KOALTQ_OCR_VENV_ROOT`의 쓰기 권한, 네트워크, `x2_ocr/ocr_paddle.summary.json`의 `errors_head`를 확인합니다. |
| `/content` 경로 오류 | Colab 밖에서는 `KOALTQ_CONTENT`, `KOALTQ_BUNDLE`, `KOALTQ_RUNS`, `KOALTQ_OCR_VENV_ROOT`를 설정합니다(5-3절). |
| Jupyter와 가상환경이 다름 | 노트북 첫 셀에서 `sys.executable`을 확인하고 해당 커널을 선택합니다. |
| Qwen3.5/3.8이 매우 느리거나 시작 실패 | `flash-linear-attention==0.6.0`의 `fla` import를 확인합니다. `KOALTQ_ALLOW_SLOW=1`은 느린 참조 구현을 허용하지만 원래 실행 조건이 아닙니다. |
