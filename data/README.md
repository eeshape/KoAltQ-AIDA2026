# data/ — 학습·평가 데이터 (대회 제공 데이터)

`koaltq_bundle/`은 대회 배포본을 정리한 번들입니다.
- 배포본: `공개_데이터셋_20260917`, 10/13 라벨 재검증 후 버전
- 경계사례 자료: `경계사례_20261013.csv`. 10/26 Q&A 678에서 사용이 공식화됨

**레코드와 이미지**
- 레코드(jsonl)는 배포본 그대로입니다.
- 이미지는 모델 입력용으로 정규화한 사본입니다(원본 파일은 바꾸지 않음).
  - 확장자 대신 파일 시그니처로 형식 판별
  - SVG → PNG, GIF/MPO는 첫 프레임, CMYK → RGB
  - 투명 배경은 흰색으로 합성
  - 최장변 1536px 상한
  - 사진 계열은 JPEG q90, 그래픽 계열은 PNG(640×640 초과 시 JPEG q92)
  - 학습 이미지는 iCCP 색 프로필 오류를 고친 사본에서 만들었습니다.
- 이미지 파일명은 `image_meta.jsonl`의 `out_file`로 연결됩니다. 원본 크기 `src_w`/`src_h`도 기록되어 있습니다.

| 파일 | 건수 | 내용 |
|---|---:|---|
| data/train.jsonl | 2,400 | 학습 레코드 + 라벨 (record_id, page_url, page_title, image_url, image_file, alt_text, context_text, in_link, link_dest, heading_path, label …) |
| data/test.jsonl | 600 | 평가 레코드 (라벨 없음) |
| data/image_meta.jsonl | 3,000 | 이미지 메타데이터 (원본 크기, 형식, out_file 등) |
| data/folds.jsonl | 2,400 | 학습 5-fold (fold, group). 같은 페이지·템플릿 묶음(277개)은 같은 fold. 코드가 sha256을 고정 확인 |
| data/boundary_test_labels.jsonl | 109 | 공개 경계사례 중 평가 데이터: 데이터셋_라벨(공식 정답) |
| data/boundary_test_rejudge.jsonl | 109 | 같은 109건의 dataset_label / rejudge_label / 경계 유형 |
| data/boundary_train_ids.jsonl | 17 | 학습 데이터 중 경계사례 id와 유형 |
| data/pseudo_test_v1.jsonl, v2 | 199, 244 | 모델이 만든 확신 의사 라벨 (비교 실험용, 최종 제출에는 미사용) |
| images/ | 3,000 | 정규화한 이미지 |
| docs/guideline_v1.1.md | - | 대회 라벨링 가이드라인 (판정기 프롬프트에 사용) |

## 무결성
- `MANIFEST.json`은 `scripts/build_bundle.py`가 만든 3,007개 파일의 sha256 목록입니다(Colab에서 전체 일치 확인).
- 그 뒤에 추가한 세 파일의 sha256:

| 파일 | sha256 | 만든 방법 |
|---|---|---|
| data/boundary_test_rejudge.jsonl | `beb77f6696fc0b4ae0eade7c2ca240e9d8362ab32db6bb63f72fb68716bd0070` | `경계사례_20261013.csv`에서 분할=test인 109행의 record_id, 데이터셋_라벨 → dataset_label, 재판정_라벨 → rejudge_label, 유형 → type |
| data/pseudo_test_v1.jsonl | `0fe085ec87707634f87b8fb40e96dd5d1f1243b47a89111b5adf5c4f185db214` | 실험용 의사 라벨 (모델 출력) |
| data/pseudo_test_v2.jsonl | `7eec28a964f8264825439e915628ce89caad5d470663019e97b264c85da685e8` | 실험용 의사 라벨 (모델 출력) |

## 번들 다시 만들기 (`scripts/build_bundle.py`)

이 스크립트는 원래 팀 작업 폴더 구조를 전제로 하며, 투명성을 위해 포함했습니다. 필요한 입력은 다음과 같습니다.
- 대회 배포본 폴더
- 이전 작업에서 만든 fold 파일(번들의 `data/folds.jsonl`과 같은 내용)
- iCCP를 고친 학습 이미지 사본

작업 폴더를 `KOALTQ_RAW_ROOT`로 지정합니다.
- 출력은 zip과 10MB 미만 조각이며, Colab 업로드용입니다.
- SVG 변환에 macOS `sips`, 이미지 처리에 Pillow가 필요합니다.
- `boundary_test_rejudge.jsonl`과 의사 라벨 파일은 이 스크립트가 만들지 않습니다(위 표).

## 근거와 유의사항
- 경계사례 라벨 활용 근거(Q&A 678, 2026-10-26)
  - 원문: "활용하셔도 됩니다. 학습, 예측 결과 후처리, 최종 제출 어디에 쓰셔도 제한하지 않습니다."
- 대회 제공 데이터 외 외부 데이터는 사용하지 않았습니다.
- 레코드는 공공기관 웹페이지에서 수집된 데이터입니다. 웹에 공개된 기관 연락처 등이 포함될 수 있습니다.
