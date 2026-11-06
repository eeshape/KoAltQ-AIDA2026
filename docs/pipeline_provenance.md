> 이 문서는 최종 제출(21번, LB 0.9357)에 쓰인 모든 모델 출력이 어떤 명령으로 만들어졌는지 실행 로그·작업 큐 기록에서 복원한 출처 기록입니다.
> 표의 `SP/…`(qedit*.py, queue*.json 등)는 실험 중 쓰던 내부 작업 파일로 제출물에 포함하지 않았습니다.
> `RL Lnnn`은 `docs/experiment_log.md`의 줄 번호입니다.
> 경로 대응: `cells/*.py` → `scripts/colab/*.py`, `RUN_LOG.md` → `docs/experiment_log.md`. `bootstrap.sh`·`queue.json`·`qedit*.py`는 제출물에 없음(내부 작업 파일).
> 실행 순서는 `notebooks/02_full_pipeline_colab.ipynb`에 그대로 옮겼습니다.
> **제출본 정리(2026-11-06)**: 최종 파일에 영향이 없는 실험 모듈(x10 보고서, x11 교차 인코더, x13 희소 클래스 실험, x18 추론형 판정기, x19 LoRA 스태커, pick, harness_val)과 리더보드 분석 도구는 제출본에서 뺐습니다.
> 아래의 x10·x11·x13·x19·pick 관련 행은 원래 실행 기록으로 남겨 둡니다. x11·x13 출력을 뺀 `model/runs_frozen/`으로도 최종 파일의 sha256이 같음을 확인했고, 노트북 02에서도 x11 단계를 뺐습니다.

# KoAltQ best submission (21번, LB 0.9357): commands that produced every run directory

Read-only reconstruction, written 2026-11-04. Paths abbreviated below:
`SP` = `<작업 폴더>`,
`EXP` = `<실험 코드 폴더>`, `RL` = `docs/experiment_log.md`.

## Sources and conventions

- Frozen runs dir `SP/repro_runs`: 135 of its arrays and json files (excluding `x8_*`) are byte-identical to the Colab light snapshot
  `results/20261102_colab/snap_final/runs` (taken 2026-11-03 11:02 KST). Three more files were added later:
  `x17_stack_lgbm_b109fx/test.npy` (= `SP/x17fxdl/test.npy`, downloaded 11:38 KST, sha256 `a5899901…`) and
  `x6_qwen38_27b_lora_full/{test.npy,metrics.json}` (= `SP/loradl/...`, sha256 `371f5461…`, downloaded about 13:00 KST).
  I checked this with sha256 on 138 files.
- Time zones: run logs (`[run] <cell> start HH:MM:SS args=[...]`), `logs/queue_runner.log` and `metrics.json` `time` use the
  Colab clock, which is UTC. RUN_LOG.md and the mtimes of `qedit*.py` are KST (UTC+9).
- Command prefix. Every job was started as
  `cd /content && PYTHONPATH=/content/code python -u -m koaltq2.run <args>`. The queue jobs used `cells/queue_runner.py`:
  `cmd = j.get("shell") or f"cd /content && PYTHONPATH=/content/code python -u -m koaltq2.run {j['args']}"`. The notebook cells
  used `cells/launch.py`: `nohup env PYTHONPATH=/content/code python -u -m koaltq2.run …`.
  `KOALTQ_BUNDLE` and `KOALTQ_RUNS` kept their defaults, `/content/koaltq_bundle` and `/content/runs` (`koaltq2/common.py`).
  The env vars in the table were per-process prefixes. No other `KOALTQ_*` variable was set.
- Hardware. One Colab A100-SXM4-80GB High-RAM with Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2 and
  bitsandbytes 0.50.2 (`logs/x0.log`, RL L13). flash-linear-attention was 0.6.0 (in the logs) and torchao was uninstalled.
  The queue runner ran several GPU jobs at once when memory allowed.
  VRAM labels used below:
  - **peak**: `peak_gb` from meta.json (torch `max_memory_allocated`).
  - **obs**: nvidia-smi used memory that `queue_runner.log` printed while that job ran alone.
  - **res**: the `gpu_gb` admission budget in the queue spec. It is a budget, not a measurement.
- Runtime is the `[run] … DONE in Xs` value from the log unless noted otherwise.

## 0. How the final x8 reads the runs dir (this decides "used")

The final command is `python -m koaltq2.run x8 --objective lbq --wlora 0.4` with `KOALTQ_HARNESS=1` (`koaltq2/x8_blend.py`).

- `load_runs` reads every sub-directory of `RUNS` except names starting with `x8`, the name `submissions`, and names
  ending in `_blend`. It also skips any directory without `test.npy`. The rest are grouped by `oof_mask.npy`:
  - **full5**: mask all True.
  - **fold0**: mask equals fold 0 (458 rows).
  - **test_only**: has `test.npy` but no `oof.npy`.
- **Rule selection (lbq).** The candidates are `caruana` (over all full5 runs), `single`, and `blend:<c>` + `x15_judge_lr`
  for each full5 `c` whose name starts with `x7_`, `x7j_`, `x17_` or `x19_`. Each candidate is scored by an honest per-fold
  quota simulation that uses only OOF.
  The frozen report (`x8_blend_lbq/report.json`) selected `blend:x17_stack_lgbm_b109` with a = 0.1, scoring .8811.
  The runner-up was `blend:x7j_stack_lgbm` at .8779, then `caruana` .8669 and `single` .8660.
- **Final logits** are 0.9·log P(x17) + 0.1·log P(x15). Then 0.6·(that) + 0.4·log(LoRA), where LoRA is the normalized
  geometric mean of the `_full` twins' `test.npy` for every fold0 run starting with `x6_`. Bias is zero under lbq.
- After that come the class quotas, then the harness (H3–H5), then `apply_l0`. The harness reads:
  - `x12_{qwen35_9b,qwen38_27b,qwen3vl_32b,qwen3vl_8b}/attrs.npy` (column 2, q_photo, for the photo vote),
  - `x2_ocr/ocr_features.npy` + `ocr_feature_names.json` (`ocr_chars`),
  - `x14_qwen38_27b_long*/probs.npy`: all 10 directories, mean log-prob, used for the H5b judge veto.
- Data: L0, R4′, H4 and the quotas need `koaltq_bundle/data/boundary_test_rejudge.jsonl`.

## 1. Ordered stages

Order = original execution order, rearranged only where a dependency forces it.
**Stages 29–31 (x7/x15/x17) must run while exactly 13 `x14_*` dirs exist, that is before rot1 and rot6 exist.**
That was the original state at 23:58–00:06 UTC (queue chain "c4"). Their logs and metrics list the blocks
`… rot2, rot3, rot4, rot5 …, avg27b_long[8]`. `x17_stack_lgbm_b109fx` (stage 36) was built after rot1 and rot6 existed:
15 `x14` dirs, `avg27b_long[10]`, 378 features.

Status: **C** = command certain (log args or queue spec + meta agree). **C-nolog** = certain from the queue spec and meta,
but the producing log is not available locally (see §2).

| # | Run dir(s) written | Used by final x8? | Command (prefix `cd /content && PYTHONPATH=/content/code`) | Hardware (VRAM) | Runtime | Depends on | Source / original start (UTC) / status |
|---|---|---|---|---|---|---|---|
| 0 | none (`/content/koaltq_bundle`, `/content/code`) | prerequisite | unzip `koaltq2_code.zip` to `/content/code`; put the 22 parts `koaltq_bundle.zip.part000…021` + `parts.json` in `/content`; `python -m koaltq2.run x0`; `pip install -q 'transformers==5.22.1' 'peft==0.21.2' 'bitsandbytes==0.50.2' accelerate flash-linear-attention`; `pip uninstall -y torchao`; **upload `EXP/build/koaltq_bundle/data/boundary_test_rejudge.jsonl` to `/content/koaltq_bundle/data/`** (it is not in `koaltq_bundle.zip`; bundle zip sha `96a04b7e…`) | GPU check only | 6.8 s (x0) | bundle parts | cells/c00_setup.py, c01_install.py, c01b_fix_torchao.py; logs/x0.log 14:31; bootstrap.sh (rejudge upload); RL L14-15, L330. C |
| 1 | x1_ctx_lr, x1_hgb, x1_blend, x1_rules, submissions/S0_x1_l0 | x1_ctx_lr, x1_hgb: **selection only** (full5). x1_blend: no (skipped, `_blend`). x1_rules: no (no test.npy) | `python -m koaltq2.run x1` | CPU | 119.7 s | bundle | cells/c02_x1.py; logs/x1.log 14:31. C |
| 2 | x2_ocr (ocr.jsonl sha256 `3cb586c8…`, ocr_paddle.jsonl, ocr_features.npy 3000×12 (v1), ocr_feature_names.json, ocr_engine.txt), x2_hgb_ocr | x2_ocr: **YES (harness: ocr_chars)**; also upstream of everything through the evidence cards and OCR features. x2_hgb_ocr: **selection only** | `python -m koaltq2.run x2 paddle` (creates `/content/venv_ocr_paddle` with `paddlepaddle-gpu==3.4.0` from the cu130 index + `paddleocr==3.7.0`) | GPU (Paddle, VRAM not logged) | 806.6 s (OCR 270.8 s) | bundle | cells/c03_x2.py; logs/x2.log 14:35. C |
| 3 | x3_qwen35_9b (feats.npy 3000×16384, absent from the frozen dir; meta.json), x3_qwen35_9b_lr, _mlp_ce, _mlp_bs | heads: **selection only**. feats.npy: upstream of x7, x7j and x17 (PCA48 block) | `python -m koaltq2.run x3 --model qwen35_9b` | GPU, peak 19.95 GB | 1253.4 s (extraction 682.3 s) | x2_ocr/ocr.jsonl | cells/c04_x3_qwen35.py; logs/x3_qwen35_9b.log 14:49; meta cards_hash `7f50b370a8578720`. C |
| 4 | x3_internvl35_14b (feats.npy 3000×20480, absent; meta.json), x3_internvl35_14b_lr, _mlp_ce, _mlp_bs | same as row 3 | `python -m koaltq2.run x3 --model internvl35_14b`. The original then ran `rm -rf /root/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-14B-HF` (queue tag cleanup_internvl) to free disk | GPU, peak 31.69 GB | 2430.3 s (extraction 1812.6 s) | x2_ocr | cells/c06_x3_internvl.py; logs/x3_internvl35_14b.log 15:11; cells/queue.json. C |
| 5 | x6_qwen35_9b_lora_fold | **YES, LoRA gate**: its fold-0 mask makes it a LoRA candidate. Its OOF only feeds the LoRA grid, which `--wlora 0.4` overrides. Its test.npy is used only if the `_full` twin is missing | `python -m koaltq2.run x6 --mode fold --backbone qwen35_9b` (defaults: epochs 2.0, lr 1e-4, accum 16, rf-t 0.06, tau 1.0, r 16, seed 20261102) | GPU, peak 19.06 GB | 5103.6 s | x2_ocr | cells/c05_x6_fold.py; logs/x6_qwen35_9b_fold.log 14:52; meta (n_train 1942, train_cards_hash `a553c8ba144af259`). C |
| 6 | x11_xenc_klue_large | **selection only** | `python -m koaltq2.run x11` (defaults: `--model klue/roberta-large --epochs 4 --lr 2e-5 --seed 20261102 --tag klue_large`) | GPU (VRAM not logged) | 1230.8 s | x2_ocr | cells/c11_x11.py; logs/x11_xenc.log 15:15. C |
| 7 | x12_qwen35_9b (attrs.npy 3000×5) | **YES (harness photo vote)** + upstream of x7, x15, x17 | `python -m koaltq2.run x12 --model qwen35_9b` | GPU, obs ≈18.3 GB (res 22) | 2859.6 s | bundle only (no OCR) | SP/qedit.py; logs/x12_qwen35_9b.log 15:36. C |
| 8 | x6_qwen35_9b_lora_full (test.npy only, no oof) | **YES, final logits** (9B half of the LoRA test component) | `python -m koaltq2.run x6 --mode full --backbone qwen35_9b` (train 2400 + 109 boundary rows) | GPU, peak 19.06 GB (res 30) | 4408.9 s | x2_ocr | cells/c09_x6_full.py, cells/queue.json, qedit2.py; logs/x6_qwen35_9b_full.log 16:20; meta n_train 2509, reload_check pass. C |
| 9 | x12_qwen3vl_8b | **YES (harness)** + upstream | `python -m koaltq2.run x12 --model qwen3vl_8b` | GPU (res 22) | 2353.8 s | bundle | qedit3.py; logs/x12_qwen3vl_8b.log 16:41. C |
| 10 | x12_qwen3vl_32b | **YES (harness)** + upstream | `python -m koaltq2.run x12 --model qwen3vl_32b` (bf16, no quantization in x12) | GPU, obs ≈62.6 GB (res 70) | 2770.3 s | bundle | qedit3.py; logs/x12_qwen3vl_32b.log 17:33. C |
| 11 | x14_qwen3vl_32b | no (upstream of x7j, x15, x17 only) | `python -m koaltq2.run x14 --model qwen3vl_32b` | GPU (res 66; same weights as row 10) | 1422.3 s | x2_ocr | qedit6.py; logs/x14_qwen3vl_32b.log 18:20. C |
| 12 | x14_qwen3vl_32b_long | no (upstream only) | `python -m koaltq2.run x14 --model qwen3vl_32b --guide long` (default prefix tolerance 0.03; prefix cache stayed on) | GPU (res 68) | 1234.4 s | x2_ocr + docs/guideline_v1.1.md | qedit9.py; logs/x14_qwen3vl_32b_long.log 18:44. C |
| 13 | x14_qwen38_27b | no (upstream only) | `python -m koaltq2.run x14 --model qwen38_27b` | GPU (res 62; 27B judge obs ≈54 GB) | 1148.7 s | x2_ocr | qedit7.py; logs/x14_qwen38_27b.log 19:24. C |
| 14 | x14_qwen3vl_8b | no (upstream only) | `python -m koaltq2.run x14 --model qwen3vl_8b` | GPU, obs ≈17.5 GB (res 24) | 839.8 s | x2_ocr | qedit13.py; logs/x14_qwen3vl_8b.log 19:43. C |
| 15 | x14_qwen35_9b | no (upstream only) | `python -m koaltq2.run x14 --model qwen35_9b` | GPU (res 26) | 835.7 s | x2_ocr | qedit13.py; logs/x14_qwen35_9b.log 19:45. C |
| 16 | x14_qwen38_27b_long | **YES (harness H5b)** + upstream | Two processes wrote this dir. **(a)** `python -m koaltq2.run x14 --model qwen38_27b --guide long` (queue tag x14_qwen38_27b_long). Its prefix check at rec 0 gave dP .0448 > .03, so it ran uncached. It was killed (rc -15) after `probs_partial.npy` held 800 rows. **(b)** `KOALTQ_PREFIX_TOL=0.08 python -m koaltq2.run x14 --model qwen38_27b --guide long` (tag x14_qwen38_27b_long2) resumed rows 800–2999 with the cache. See §2 for an exact replay | GPU, obs ≈54.2 GB (res 64) | (a) 20:03:53→20:24:23, about 1230 s at 1.22 s/rec; (b) 704.2 s | x2_ocr + guideline | qedit14.py + qedit16.py; logs/x14_qwen38_27b_long.log, x14_qwen38_27b_long2.log; queue_runner.log. C (2 invocations) |
| 17 | x12_qwen38_27b (q_match and q_empty columns are NaN) | **YES (harness)** + upstream | `python -m koaltq2.run x12 --model qwen38_27b --questions q_decor,q_photo,q_info` | GPU (res 60) | 2020.6 s | bundle | qedit15.py; logs/x12_qwen38_27b.log 20:36. C |
| 18 | x14_qwen38_27b_long_hires | **YES (harness)** + upstream | `KOALTQ_PREFIX_TOL=0.08 KOALTQ_MAX_IMAGE_TOKENS=2048 python -m koaltq2.run x14 --model qwen38_27b --guide long --tag hires` | GPU (res 68) | 970.5 s | x2_ocr + guideline | qedit15 → qedit19 (final spec); log 21:14 (args `--guide long --tag hires`, pixel cap 2048); meta max_image_tokens 2048. C |
| 19 | x14_qwen38_27b_long_revctx | **YES (harness)** + upstream | `KOALTQ_PREFIX_TOL=0.08 KOALTQ_CTX_CHARS=800 KOALTQ_OCR_CHARS=800 python -m koaltq2.run x14 --model qwen38_27b --guide long --order rev --tag revctx` | GPU (res 62) | 930.9 s | x2_ocr + guideline | qedit17 → qedit19; log 21:30; meta order rev, ctx 800, ocr 800. C |
| 20 | x16_insuff_qwen38_27b_long (score.npy) | no (upstream of x7j, x15, x17) | `KOALTQ_PREFIX_TOL=0.08 python -m koaltq2.run x14 --model qwen38_27b --guide long --ask insuff` | GPU (res 62) | 937.1 s | x2_ocr + guideline | qedit20.py (tag x16_insuff_27b); logs/x16_insuff_27b.log 21:46. C |
| 21 | x16_ok_qwen38_27b_long | no (upstream only) | `KOALTQ_PREFIX_TOL=0.08 python -m koaltq2.run x14 --model qwen38_27b --guide long --ask ok` | GPU (res 62) | 915.9 s | same as row 20 | qedit20.py (tag x16_ok_27b); logs/x16_ok_27b.log 22:02. C |
| 22 | x14_qwen38_27b_long_ctx | **YES (harness)** + upstream | `KOALTQ_PREFIX_TOL=0.08 KOALTQ_CTX_CHARS=800 KOALTQ_OCR_CHARS=800 python -m koaltq2.run x14 --model qwen38_27b --guide long --tag ctx` | GPU (res 62) | 950.5 s | x2_ocr + guideline | qedit21.py; logs/x14_qwen38_27b_ctx.log 22:22; meta. C |
| 23 | x14_qwen38_27b_long_rot2 | **YES (harness)** + upstream | `KOALTQ_PREFIX_TOL=0.08 python -m koaltq2.run x14 --model qwen38_27b --guide long --order rot2 --tag rot2` | GPU (res 62) | 950.6 s | same | qedit23.py; log 22:44. C |
| 24 | x14_qwen38_27b_long_rot4 | **YES (harness)** + upstream | same command with `--order rot4 --tag rot4` | GPU (res 62) | 932.7 s | same | qedit23.py; log 23:03. C |
| 25 | x14_qwen38_27b_long_rot3 | **YES (harness)** + upstream | same command with `--order rot3 --tag rot3` | GPU (res 62) | 943.4 s | same | qedit24.py; log 23:23. C |
| 26 | x14_qwen38_27b_long_rot5 | **YES (harness)** + upstream | same command with `--order rot5 --tag rot5` | GPU (res 62) | 921.4 s | same | qedit24.py; log 23:42. C |
| 27 | **checkpoint "c4"**: RUNS now holds exactly 13 `x14_*`, 2 `x16_*`, 4 `x12_*`, 2 `x3_*/feats.npy` and `x2_ocr` | no | (no command) | | | | logs/x7_stack_c4.log `[x7] blocks:` line |
| 28 | (the original also ran `x6 --mode smoke --backbone qwen38_27b …` at this point; it is unused, see the unused table) | | | | | | |
| 29 | x7_stack_hgb (+ blocks.json), x7_stack_lgbm, x7j_stack_hgb, x7j_stack_lgbm | **selection only** (Caruana pool and the blend:x7_/x7j_ rule candidates) | `python -m koaltq2.run x7` | CPU (sklearn HGB + LightGBM) | 351.1 s | x2_ocr features, x3 feats ×2, x12 ×4, x14 ×13 (+avg27b_long[8]), x16 ×2 | qedit24.py (tag x7_stack_c4); logs/x7_stack_c4.log 23:58; metrics n_features 264 (x7) / 364 (x7j). C |
| 30 | x15_judge_lr | **YES, final logits**: weight a = 0.1. Its OOF also chooses a and the rule | `python -m koaltq2.run x15` | CPU | 20.7 s | x14 ×13 + avg27b_long[8] + x16 ×2 + x12 ×4 (124 features) | qedit24.py (tag x15_c4); logs/x15_c4.log 00:00; metrics `sources`. C |
| 31 | x17_stack_lgbm_b109 | **YES, final logits**: OOF kept (it selects the rule and a); **test.npy is replaced by the x17fx test.npy** | `python -m koaltq2.run x17` (no env vars; trains on train + 109 boundary labels) | CPU (LightGBM) | 123.5 s | x7j feature set (364 features) + boundary_test_labels | qedit24.py (tag x17_c4); logs/x17_c4.log 00:04; metrics `blocks`, OOF .7697. C |
| 32 | x6_qwen38_27b_lora_fold | **YES, LoRA gate** (as row 5; 27B fold-0 OOF, fold0 comp .7227) | `python -m koaltq2.run x6 --mode fold --backbone qwen38_27b --accum 16 --epochs 1.5` | GPU, peak 56.85 GB (obs 53.6; res 74) | 5118.1 s | x2_ocr | cells/queue.json, qedit3.py (after-deps later moved by qedit7…24); logs/x6_qwen38_27b_fold.log 00:02. C |
| 33 | x14_qwen38_27b_long_rot1 | **YES (harness)** + x17fx features | `KOALTQ_PREFIX_TOL=0.08 python -m koaltq2.run x14 --model qwen38_27b --guide long --order rot1 --tag rot1` | GPU (res 62) | 920.2 s | x2_ocr + guideline | qedit26.py; log 01:31. C |
| 34 | x14_qwen38_27b_long_rot6 | **YES (harness)** + x17fx features | same command with `--order rot6 --tag rot6` | GPU (res 62) | 917.9 s | same | qedit26.py; log 01:47 (finished about 02:02). C |
| 35 | x6_qwen38_27b_lora_full (test.npy, metrics.json) | **YES, final logits** (27B half of the LoRA test component) | `python -m koaltq2.run x6 --mode full --backbone qwen38_27b --accum 16 --epochs 1.5` | GPU, peak 56.83 GB (res 74) | train 5928.3 s + infer 256.0 s (meta), plus model load and the reload check; no log | x2_ocr | qedit29.py (tag x6_qwen38_27b_full, queued 11:15 KST); metrics time 04:00:09 UTC; meta: full, 27B, epochs 1.5, accum 16, r 16, seed 20261102, n_train 2509 (so no X6_EXTRA); RL L403/L407. **C-nolog** |
| 36 | x17_stack_lgbm_b109fx (in the frozen dir only test.npy; a re-run also writes oof.npy and metrics) | **YES (only its test.npy)**. It is copied over `x17_stack_lgbm_b109/test.npy`, then the directory is deleted. If left in place with oof.npy it would join full5 and the rule candidates | `KOALTQ_X17_EXTRA=1 python -m koaltq2.run x17` (EXTRA_SRC defaults to R2,R4, giving suffix `fx`: train + 109 + 133 rule-fixed test rows (R2 116, R4′ 17)) | CPU (LightGBM) | not logged (x17 takes ≈2 min) | row 31's inputs **plus rot1 and rot6** (15 x14 dirs, avg27b_long[10], 378 features) + `boundary_test_rejudge.jsonl` (R4′) | qedit30.py (tag x17_fx, queued 11:34 KST = 02:34 UTC, after rot6 finished); RL L409 (OOF .7684); `SP/x17fx_new/metrics.json` (bit-identical re-run, see §2). **C-nolog** |
| 37 | (assembly, no new dir) | prerequisite | on a copy of the runs dir: `cp x17_stack_lgbm_b109fx/test.npy x17_stack_lgbm_b109/test.npy && rm -rf x17_stack_lgbm_b109fx x6_qwen38_27b_fx_lora_full x6_qwen38_27b_fx3_lora_full` | CPU | seconds | rows 31, 36 | EXP/reproduce_best.sh. C |
| 38 | x8_blend_lbq (report.json, honest_preds.npz, test_logits_final.npy, harness_in_{pred,fixed,logits}.npy, harness_log.jsonl, harness_summary.json) + submissions/S1/S2/**S3_ensemble_quota_l0_lbq**.{csv,json} | **FINAL** | `KOALTQ_HARNESS=1 KOALTQ_BUNDLE=<bundle> KOALTQ_RUNS=<runs> PYTHONPATH=<code> python -m koaltq2.run x8 --objective lbq --wlora 0.4`. Equivalent: `KOALTQ_SOLE_NOT_DECO=1 KOALTQ_SIB_K2_OK=1 KOALTQ_K2_VETO=1` instead of `KOALTQ_HARNESS=1`. Expected S3 sha256 `87c3de688bd20937…` | CPU | Colab `x8 --objective lbq` took 495.3 s (x8_blend_c_lb.log); the final local run was not logged | all of the above + bundle incl. boundary_test_rejudge.jsonl | EXP/reproduce_best.sh header and command; koaltq2/x8_blend.py, harness.py. C |

Rough total GPU time if run one job at a time: about 54,000 s, or **≈15 GPU-hours** (sum of the runtimes above).
CPU stages add about 20 min. RL L474 gives the Colab rate for this machine as 8.88 CU/h.

### Present in `repro_runs` but not read by the final x8

| Dir / file | Producing command (source) | Why unused |
|---|---|---|
| x1_blend, x1_rules | `x1` (row 1) | `_blend` names are skipped; x1_rules has no test.npy |
| x3_qwen35_9b, x3_internvl35_14b | `x3 --model …` (rows 3–4) | the frozen dir holds only meta.json: `cells/pack_light.py` excluded `feats*`. In a re-run, x7 and x17 **need** their feats.npy (`x7_stack.vlm_blocks` globs `x3_*/feats.npy`) |
| x6_qwen38_27b_smoke | `x6 --mode smoke --backbone qwen38_27b --limit 24 --epochs 1` (queue.json/qedit3; log 19:13 UTC, 450.1 s, peak 55.86 GB) | connection test; no test.npy |
| x10_report | `x10` (queue tag x10_report; log 01:42 UTC, 122.3 s, CPU) | report figures and zip; no test.npy |
| x13_spec | `x13` (tags x13_b … x13_c_nj; about 15 s each, CPU) | builds the S4 specialists. The final file is S3; no test.npy |
| x8_blend / x8_blend_lbobj / x8_blend_lbq / x8_blend_lbq_nojudge | `x8` (x8_blend_b) / `x8 --objective lb` (x8_blend_b_lb) / `x8 --objective lbq` (x8_blend_c_lb, overwritten by row 38) / `x8 --objective lbq --exclude '^(x14\|x15\|x16\|x17\|x7j)' --suffix _nojudge` (x8_blend_c_nj2) | `x8*` names are skipped |
| x14_qwen35_9b, x14_qwen38_27b, x14_qwen3vl_32b, x14_qwen3vl_32b_long, x14_qwen3vl_8b, x16_* | rows 11–15, 20–21 | not read by x8 or the harness (the harness glob is `x14_qwen38_27b_long*`); needed upstream by x7j, x15 and x17 |
| submissions/, logs/, lb_estimates.json (`lb`), FINAL_CANDIDATES.md (`pick`), local_leaderboard.csv, queue_status.json | various | not runs (x8 skips `submissions` and non-directories) |
| (not in repro_runs) x6_qwen38_27b_fx_lora_full, x6_qwen38_27b_fx3_lora_full | `KOALTQ_X6_EXTRA=1 … x6 --mode full --backbone qwen38_27b --accum 16 --epochs 1.5 --seed 7 --tag fx` (qedit31) and `… --epochs 2.0 --seed 11 --tag fx3` (qedit32) | LoRA v2/v3 candidates; reproduce_best.sh deletes them; not needed |
| (2nd Colab session, not in repro_runs) *_v2, *_ocr2, x6_*_c2f*, x6_*_f1..f4, x19_* | SP/queue_new.json, queue_v2…v6.json | later experiments; none feed the 21번 file |

## 2. Uncertain or not-log-verified items

**No stage has an unknown command or env var.** Two runs have no producing log (marked UNCERTAIN, minor) and one run
was written by two invocations.

| Run | Status | Best evidence |
|---|---|---|
| x6_qwen38_27b_lora_full | **UNCERTAIN (minor): no producing log** (`/content/runs/logs/x6_qwen38_27b_full.log` was never downloaded) | Command from qedit29.py: `x6 --mode full --backbone qwen38_27b --accum 16 --epochs 1.5`, gpu_gb 74. The metrics.json meta matches every CLI value: mode full, epochs 1.5, accum 16, lr 1e-4, r 16, seed 20261102, rf_t .06, tau 1. n_train 2509 = 2400 + 109, so neither KOALTQ_X6_EXTRA nor X6_PSEUDO was set. train_cards_hash `3a2298a5e582e082` equals the 9B full run's. Timing fits: queued 11:15 KST; train+infer 6184 s; metrics time 04:00:09 UTC = 13:00 KST. test.npy sha `371f5461…` = SP/loradl copy. RL L403/L407. |
| x17_stack_lgbm_b109fx | **UNCERTAIN (minor): no producing log**; feature context inferred | Command from qedit30.py: `KOALTQ_X17_EXTRA=1 … run x17`, tag x17_fx, queued 11:34 KST = 02:34 UTC, after rot6 finished at about 02:02 UTC, so 10 `x14_qwen38_27b_long*` dirs existed. Supporting evidence: (1) `SP/x17fx_new` is a re-run of the same effective command, written 17:17 UTC Nov 3 by the x17ps job on old code that ignored `KOALTQ_X17_PSEUDO` (RL L634-635). Its test.npy is **bit-identical** (sha `a5899901…`) and its metrics show 378 features, avg27b_long[10] and rot1/rot6 among the blocks, OOF .7684. (2) RL L409 reports OOF .7684 for the original. (3) RL L730: re-training the deployed setting in the original Colab runs gave bit-identical output. (4) RL L731: re-training from the reduced repro_runs (no x3 feats) matched only 97.5 % of argmaxes. |
| x14_qwen38_27b_long | CERTAIN, but two processes | Rows 0–799 were computed **uncached** (default `KOALTQ_PREFIX_TOL` .03; check dP .0448 disabled the cache). Rows 800–2999 were computed **with the prefix cache** under `KOALTQ_PREFIX_TOL=0.08` (checks at rec 800–802: .0032/.0332/.0458). Exact replay (my reconstruction from x14_judge.py: with `--limit`, the partial file is saved every 200 rows and kept on return): `python -m koaltq2.run x14 --model qwen38_27b --guide long --limit 800`, then `KOALTQ_PREFIX_TOL=0.08 python -m koaltq2.run x14 --model qwen38_27b --guide long`. |

Other caveats (facts, not command uncertainty):

- **Code revisions.** Some stages ran on earlier revisions of their module:
  - `x3_frozen.py` mtime 11-03 00:12 KST; x3 runs started 23:49 and 00:11 KST.
  - `x12_attr.py` 05:03 KST, after 3 of the 4 x12 runs.
  - `x14_judge.py` 07:42 KST, after all non-rot x14/x16 runs.
  - `x6_lora.py` 17:45 KST, `x7_stack.py` 23:29 KST, `x17_boundary.py` 11-04 07:50 KST.

  The later edits I found are env-gated options whose defaults keep the old behaviour. Evidence of equivalence:
  - the prompts printed in the x14 logs match the current prompt builders;
  - re-extracted x3 features had identical metadata and cards hash `7f50b370` (RL L560, L590);
  - x17fx re-ran bit-identically (above).

  I could not verify the x3 MLP/LR head code as it was at run time. Those heads are selection-only.
- **Bit-exactness of GPU stages is not guaranteed** (LoRA training, x11, x3 heads, VLM inference). The versions of
  scikit-learn, LightGBM, numpy and Paddle were not recorded. `causal_conv1d` was **not** installed: the 27B logs show the
  "falling back to its reference PyTorch implementation" warning.
- **Rule-selection margin.** blend:x17 scored .8811 against .8779 for blend:x7j_stack_lgbm. If upstream re-runs drift,
  x8 could pick another rule. `x8 --rule blend:x17_stack_lgbm_b109` (FORCE_RULE, in x8_blend.cli) pins it. That flag was
  **not** part of the original command.
- **Fixed anchors to check in a re-run:**
  - OCR sha `3cb586c8…`
  - x3 cards_hash `7f50b370a8578720`
  - x6 train_cards_hash `a553c8ba144af259` (fold runs) and `3a2298a5e582e082` (full runs)
  - x7j/x17 = 364 features, x15 = 124, x17fx = 378
  - x8 report rule `blend:x17_stack_lgbm_b109` with a 0.1
  - final S3 sha `87c3de68…`

## 3. Base models downloaded by each GPU stage (from the code)

| Stage | Code location | HF repo id (key used) | Load | Revision recorded in meta |
|---|---|---|---|---|
| x3 (rows 3–4) | `x3_frozen.MODELS` | `Qwen/Qwen3.5-9B` (qwen35_9b); `OpenGVLab/InternVL3_5-14B-HF` (internvl35_14b) | bf16, `AutoModelForImageTextToText` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (9B); `226b96d5912e69159abc0384cefcbd51487fdce0` (InternVL) |
| x6 (rows 5, 8, 32, 35) | `x6_lora.BACKBONES` | `Qwen/Qwen3.5-9B` (qwen35_9b); `Qwen/Qwen3.8-27B` (qwen38_27b) | bf16 `Qwen3_5ForSequenceClassification` + LoRA r16 | 9B `c2022362…`; 27B `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| x11 (row 6) | `x11_xenc.cli --model` default | `klue/roberta-large` | `AutoModelForSequenceClassification`, bf16 autocast | not recorded |
| x12 / x14 / x16 (rows 7, 9–26, 33–34) | `x12_attr.MODELS` (x14 imports it) | `Qwen/Qwen3.5-9B` (qwen35_9b); `Qwen/Qwen3.8-27B` (qwen38_27b); `Qwen/Qwen3-VL-8B-Instruct` (qwen3vl_8b); `Qwen/Qwen3-VL-32B-Instruct` (qwen3vl_32b) | bf16, `AutoModelForImageTextToText`, no quantization | not recorded (same snapshots as above where shared) |
| x2 (row 2) | `x2_ocr.ensure_ocr_venv` / `ocr_worker.py` | not HF: PaddleOCR `PP-OCRv5` (`lang="korean"`), auto-downloads `PP-OCRv5_server_det` + `korean_PP-OCRv5_mobile_rec` from PaddleX official model hosters (x2.log) | paddlepaddle-gpu 3.4.0 (cu130) + paddleocr 3.7.0 in a separate venv | n/a |

Models listed in the dicts but **not** used by this pipeline:
- `x3_frozen.MODELS`: Qwen3.5-0.8B, Qwen3-VL-32B-Instruct (the planned x5 extraction was dropped in qedit3; no x3_qwen3vl_32b),
  Qwen3-VL-8B, gemma-4-31B-it, gemma-4-12B-it, Qwen3.8-27B.
- `x6_lora.BACKBONES`: Qwen3.5-4B, 0.8B, Qwen3.6-27B.
- `x12_attr.MODELS`: Qwen3.5-4B, 0.8B.

Prefetch lists in the original: `cells/c02b_prefetch.py` (Qwen/Qwen3.5-9B, OpenGVLab/InternVL3_5-14B-HF,
Qwen/Qwen3-VL-32B-Instruct) and the queue job prefetch_27b (Qwen/Qwen3.8-27B).
The qwen35-family stages (x3 qwen35_9b, x6, x12/x14 on 9B and 27B) also need `flash-linear-attention`:
`require_fast_deltanet` raises an error otherwise unless `KOALTQ_ALLOW_SLOW=1`.

## 4. Facts from the code that matter for a notebook

- **Globs that make directory contents part of the input:**
  - `x7`, `x17`: `x3_*/feats.npy`, `x12_*`, `x14_*`, `x14_qwen38_27b_long*` (avg block), `x16_*`.
  - `x15`: `x14_*`, `x14_qwen38_27b_long*`, `x16_*`, `x12_*`.
  - harness: `x14_qwen38_27b_long*`.
  - x8 `load_runs`: every full5 / fold0 / test_only directory. Extra `x7_`/`x7j_`/`x17_`/`x19_` dirs with full OOF, or extra
    fold-0 `x6_*` dirs, change x8's candidates.
  - Do not create extra dirs matching these patterns, and delete `x17_stack_lgbm_b109fx` before x8 (row 37).
- Set `KOALTQ_*` env vars **per command** (the queue used `VAR=… python …`). Do not set them kernel-wide: RL L727 records
  a leaked kernel-level `KOALTQ_STACK_SEEDS=3`. Keep `KOALTQ_STACK_SEEDS`, `KOALTQ_X17_SUFFIX`, `KOALTQ_OCR_V2`,
  `KOALTQ_CARD_IMGINFO`, `KOALTQ_X6_FOLD`, `KOALTQ_L0_EXTRA` and `KOALTQ_R4_LABEL` unset.
- `boundary_test_rejudge.jsonl` must be in `koaltq_bundle/data/` before row 36 (x17fx R4′ rows) and row 38
  (L0/R4′, K2/H4, quotas). It is not inside `koaltq_bundle.zip`. Without it, `propagate_from_boundary` silently falls back
  to dataset labels.
- `Data()` asserts the folds sha `4154b923…`, 2400/600 rows, fold0 = 458 and 109 boundary rows.
- Disk: the original deleted the InternVL weights after x3 (cleanup_internvl) and later removed orphan HF blobs
  (`cells/orphan_blobs.py`). RL L79 records a 99 % disk crisis.
- Colab reclaimed idle CLI sessions after about 15–20 min without client activity (RL L556-559). Keep a keep-alive for the
  long GPU stages.
