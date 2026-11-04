"""X12: guideline attribute QA with a VLM (zero-shot yes/no probabilities) -> features targeting the rare classes.

For every record and question, P(예) = softmax over the first-token logits of '예' vs '아니오' after the assistant
prompt (Qwen3.5 template, thinking disabled). Questions map to guideline rules:
  q_match  alt relates to what the image shows / where it links      -> 무관형 (1 - p)
  q_decor  image is pure decoration                                  -> 장식오용형
  q_photo  image is a camera photo (vs graphic/poster/scan)          -> thumbnail rule 불충분형 vs 중복형
  q_info   image contains informative text (dates/place/target/fee)  -> 불충분형
  q_empty  alt is a contentless stock word/symbol                    -> 무의미형
Output: RUNS/x12_<model>/attrs.npy (3000 x 5, NaN for empty-alt questions that need an alt) + train AUC report.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import L2I, Data, run_dir

QUESTIONS = {
    "q_match": "alt 텍스트: \"{alt}\"\n이 alt 텍스트가 이미지에 실제로 보이는 내용 또는 이 이미지가 연결된 링크 목적지와 관련이 있습니까? '예' 또는 '아니오'로만 답하세요.",
    "q_decor": "이 이미지는 정보를 전혀 전달하지 않는 순수 장식(구분선, 배경 무늬, 꾸밈용 아이콘 등)입니까? '예' 또는 '아니오'로만 답하세요.",
    "q_photo": "이 이미지는 카메라로 촬영한 사진입니까? 그래픽·아이콘·로고·포스터·문서 스캔이면 아니오입니다. '예' 또는 '아니오'로만 답하세요.",
    "q_info": "이 이미지 안에 날짜·기간·장소·대상·금액·신청방법 같은 정보성 글자가 보입니까? '예' 또는 '아니오'로만 답하세요.",
    "q_empty": "alt 텍스트: \"{alt}\"\n이 alt 텍스트는 '이미지', '사진', '배너', 숫자나 기호처럼 아무 내용도 없는 상투어입니까? '예' 또는 '아니오'로만 답하세요.",
}
NEEDS_ALT = {"q_match", "q_empty"}
MODELS = {"qwen35_9b": ("Qwen/Qwen3.5-9B", "qwen35"), "qwen38_27b": ("Qwen/Qwen3.8-27B", "qwen35"),
          "qwen35_4b": ("Qwen/Qwen3.5-4B", "qwen35"), "qwen3vl_8b": ("Qwen/Qwen3-VL-8B-Instruct", "qwen3vl"),
          "qwen3vl_32b": ("Qwen/Qwen3-VL-32B-Instruct", "qwen3vl"), "qwen35_08b": ("Qwen/Qwen3.5-0.8B", "qwen35")}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = ~np.isnan(s)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def cli(argv: list[str]) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .prompting import load_image
    from .x3_frozen import build_inputs, check_pixel_cap, configure_pixels, require_fast_deltanet

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_9b", choices=list(MODELS))
    ap.add_argument("--questions", default="", help="comma-separated subset of question keys (others stay NaN)")
    a = ap.parse_args(argv)
    ask = set(a.questions.split(",")) if a.questions else set(QUESTIONS)
    assert ask <= set(QUESTIONS), ask
    dev = os.environ.get("KOALTQ_DEVICE", "cuda:0")
    data = Data()
    out = run_dir(f"x12_{a.model}")
    repo, fam = MODELS[a.model]
    require_fast_deltanet(fam)
    local = snapshot_download(repo)
    proc = AutoProcessor.from_pretrained(local)
    configure_pixels(proc, fam)
    model = AutoModelForImageTextToText.from_pretrained(local, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa")
    model.eval()
    check_pixel_cap(model, proc, fam)
    tok = proc.tokenizer
    yes_id = tok.encode("예", add_special_tokens=False)[0]
    no_id = tok.encode("아니오", add_special_tokens=False)[0]
    assert yes_id != no_id
    keys = list(QUESTIONS)
    partial = out / "attrs_partial.npy"
    A = list(np.load(partial)) if partial.exists() else []
    t0 = time.time()
    for i in range(len(A), len(data.all)):
        r = data.all[i]
        img = load_image(data.image_path(r["record_id"]))
        alt = (r.get("alt_text") or "").strip()
        row = []
        for k in keys:
            if (k in NEEDS_ALT and not alt) or k not in ask:
                row.append(np.nan)
                continue
            inputs = build_inputs(proc, fam, img, QUESTIONS[k].format(alt=alt[:200]))
            inputs = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in inputs.items()}
            with torch.inference_mode():
                lg = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            row.append(float(torch.softmax(lg[[yes_id, no_id]], -1)[0]))
        A.append(np.array(row, dtype=np.float32))
        if (i + 1) % 200 == 0:
            np.save(out / "attrs_partial.tmp.npy", np.stack(A))
            os.replace(out / "attrs_partial.tmp.npy", partial)
            print(f"[x12 {a.model}] {i + 1}/3000 {(time.time() - t0) / (i + 1 - 0):.2f}s/rec", flush=True)
    A = np.stack(A)
    np.save(out / "attrs.npy", A)
    partial.unlink(missing_ok=True)
    y = data.y
    tr = A[: data.n_train]
    rep = {
        "auc_무관형_vs_rest(1-q_match)": auc((y == L2I["무관형"]).astype(int), 1 - tr[:, 0]),
        "auc_장식오용형_vs_rest(q_decor)": auc((y == L2I["장식오용형"]).astype(int), tr[:, 1]),
        "auc_불충분형_vs_중복형(q_photo)": auc((y[np.isin(y, [3, 5])] == 5).astype(int), tr[np.isin(y, [3, 5]), 2]),
        "auc_불충분형_vs_적절(q_info)": auc((y[np.isin(y, [0, 5])] == 5).astype(int), tr[np.isin(y, [0, 5]), 3]),
        "auc_무의미형_vs_rest(q_empty)": auc((y == L2I["무의미형"]).astype(int), tr[:, 4]),
        "seconds": round(time.time() - t0, 1), "model": a.model, "questions": QUESTIONS,
    }
    (out / "meta.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[x12] train AUCs:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()
                                           if k.startswith("auc")}, ensure_ascii=False), flush=True)


def main() -> None:
    cli([])
