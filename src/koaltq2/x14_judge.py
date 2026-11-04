"""X14: zero-shot guideline judge (organizer-baseline style) -> 7-class probabilities from a large VLM.

The organizer's zero-shot VLM baseline (image + alt + context + guideline as instruction) reached 무관형 F1 0.889 on
2,790 records, far above what our narrow yes/no attribute questions (X12) give. X14 reproduces that setup with an open
VLM: the evidence card (guideline summary + record + OCR) plus an extended rule note, answer constrained to one letter
A-G, and P(label) = softmax over the 7 letter logits at the first answer position (no generation, no training).
Output: RUNS/x14_<model>/probs.npy (3000 x 7, LABELS order) + meta.json with train AUCs / argmax scores.
The probabilities are label-free (zero-shot), so they are valid features for every train fold and the test set.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import LABELS, Data, run_dir, scores
from .x12_attr import MODELS, auc

LETTERS = "ABCDEFG"
RULE_NOTE = (
    "[추가 기준]\n"
    "무관형: alt가 이미지에 실제로 보이는 대상과 다른 것을 말함(예: 오시는 길 지도 이미지에 alt '회사 로고', "
    "행사 사진에 다른 행사명, 다른 이미지의 설명이 잘못 붙음). 내용이 일부만 빠진 것은 무관형이 아니라 불충분형.\n"
    "장식오용형: 구분선·배경 무늬·꾸밈 아이콘처럼 제거해도 정보 손실이 없는 이미지에 내용성 alt가 붙음. "
    "링크 안에서 목적지를 알리는 아이콘이나 링크 텍스트에 없는 정보를 주는 아이콘은 장식이 아님.\n"
    "무의미형: '이미지','사진','배너','아이콘','*', 숫자처럼 이미지 유형만 말하거나 의미가 없는 alt."
)
ASK = ("답: 다음 중 하나의 기호로만 답하세요.\n"
       "A=적절 B=무의미형 C=파일명형 D=중복형 E=장식오용형 F=불충분형 G=무관형")
ORDER = list(range(7))  # ORDER[i] = label index shown with letter LETTERS[i]; "rev" reverses it (position-bias check)
BINARY_ASKS = {  # X16: targeted yes/no judgements under the same guideline prompt; score = P(A)
    "insuff": ("질문: 이미지 안에만 있고 alt·링크 텍스트·인접 텍스트 어디에도 없는 핵심 정보(사진에 보이는 장면·인물·대상, "
               "포스터·안내물의 기간·장소·대상·금액·참여방법 등)가 있습니까? 등록일·제목 반복·기관명·로고는 핵심 정보가 아닙니다.\n"
               "답: A=있다(정보 누락) B=없다 중 하나의 기호로만 답하세요."),
    "ok": ("질문: 가이드라인에 따르면 이 alt는 그 맥락에서 필요한 정보를 빠짐없이, 반복 없이 전달하는 '적절'입니까?\n"
           "답: A=적절 B=부적절 중 하나의 기호로만 답하세요."),
}


def set_order(kind: str) -> None:
    global ASK
    if kind == "rev":
        ORDER[:] = list(range(7))[::-1]
    elif kind.startswith("rot"):  # cyclic rotation: letter A shows label k, B shows k+1, ...
        k = int(kind[3:])
        ORDER[:] = [(i + k) % 7 for i in range(7)]
    else:
        ORDER[:] = list(range(7))
    ASK = "답: 다음 중 하나의 기호로만 답하세요.\n" + " ".join(f"{LETTERS[i]}={LABELS[ORDER[i]]}" for i in range(7))


def long_guide() -> str:
    """Guideline v1.1 sections 2-3 (label table, procedure, boundary rules) - the organizer baseline's instruction."""
    from .common import BUNDLE

    g = (BUNDLE / "docs" / "guideline_v1.1.md").read_text(encoding="utf-8")
    a, b = g.index("## 2. 라벨 체계"), g.index("### 3-4. 보류 조건")
    return ("당신은 한국어 웹 대체텍스트(alt) 품질 판정 전문가입니다. 아래 라벨링 가이드라인에 따라 레코드 하나를 판정합니다.\n\n"
            + g[a:b].strip() + "\n\n" + RULE_NOTE)


def long_user(r: dict) -> str:
    from .prompting import evidence_card

    card = evidence_card(r)
    body = card.split("\n[레코드]", 1)[1]
    if ASK.startswith("질문:"):  # binary asks carry their own question
        body = body.replace("\n질문: 이 이미지의 alt 품질 라벨은 7범주 중 무엇인가?", "")
    return "[레코드]" + body + "\n" + ASK


def prompt_for(r: dict) -> str:
    from .prompting import evidence_card

    card = evidence_card(r)
    assert "\n[레코드]" in card
    return card.replace("\n[레코드]", "\n" + RULE_NOTE + "\n[레코드]", 1) + "\n" + ASK


def chat_inputs(proc, fam: str, img, user: str, system: str | None):
    msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
           [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
    kw = {"enable_thinking": False} if fam == "qwen35" else {}
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    return prompt, proc(text=[prompt], images=[img], return_tensors="pt")


class PrefixRunner:
    """Reuses the KV cache of the shared system prefix (the long guideline) across records.
    Positions for the suffix come from get_rope_index on the full sequence (M-RoPE), the cache is deep-copied per
    record. Verified against the uncached forward on the first records; falls back to uncached on any mismatch."""

    def __init__(self, model, proc, dev: str, prefix_text: str):
        import torch

        self.model, self.dev, self.ok = model, dev, True
        self.ids = proc.tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        self.P = self.ids.shape[1]
        with torch.inference_mode():
            self.cache = model(input_ids=self.ids, use_cache=True).past_key_values
        print(f"[x14] prefix cache built: {self.P} tokens", flush=True)

    def logits(self, inputs: dict):
        import copy

        import torch

        ids = inputs["input_ids"]
        if not torch.equal(ids[0, : self.P], self.ids[0]):
            raise RuntimeError("prefix token mismatch")
        core = self.model.model
        pos, _ = core.get_rope_index(ids, mm_token_type_ids=inputs["mm_token_type_ids"], image_grid_thw=inputs["image_grid_thw"],
                                     attention_mask=inputs["attention_mask"])
        cache = copy.deepcopy(self.cache)
        with torch.inference_mode():
            out = self.model(input_ids=ids[:, self.P:], pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                             mm_token_type_ids=inputs["mm_token_type_ids"][:, self.P:], attention_mask=inputs["attention_mask"],
                             position_ids=pos[..., self.P:], past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float()


def cli(argv: list[str]) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .prompting import load_image, require_ocr
    from .x3_frozen import build_inputs, check_pixel_cap, configure_pixels, require_fast_deltanet

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_32b", choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--guide", default="short", choices=["short", "long"])
    ap.add_argument("--order", default="fwd", choices=["fwd", "rev"] + [f"rot{k}" for k in range(1, 7)])
    ap.add_argument("--ask", default="label7", choices=["label7"] + list(BINARY_ASKS))
    ap.add_argument("--tag", default="", help="suffix for the run directory (e.g. hires when KOALTQ_MAX_IMAGE_TOKENS is raised)")
    a = ap.parse_args(argv)
    set_order(a.order)
    global ASK
    if a.ask != "label7":
        ASK = BINARY_ASKS[a.ask]
    ocr_sha = require_ocr()  # asserts the full 3000-row OCR file (returns its sha256)
    dev = os.environ.get("KOALTQ_DEVICE", "cuda:0")
    data = Data()
    stem = "x14" if a.ask == "label7" else f"x16_{a.ask}"
    out = run_dir(f"{stem}_{a.model}" + ("_long" if a.guide == "long" else "") + (f"_{a.tag}" if a.tag else ""))
    repo, fam = MODELS[a.model]
    require_fast_deltanet(fam)
    local = snapshot_download(repo)
    proc = AutoProcessor.from_pretrained(local)
    configure_pixels(proc, fam)
    model = AutoModelForImageTextToText.from_pretrained(local, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa")
    model.eval()
    check_pixel_cap(model, proc, fam)
    tok = proc.tokenizer
    letters = LETTERS if a.ask == "label7" else "AB"
    ids = [tok.encode(c, add_special_tokens=False) for c in letters]
    assert all(len(i) == 1 for i in ids), ids
    ids = [i[0] for i in ids]
    assert len(set(ids)) == len(letters)
    n = len(data.all) if a.limit is None else min(a.limit, len(data.all))
    partial = out / "probs_partial.npy"
    P = list(np.load(partial)) if partial.exists() else []
    t0, start = time.time(), len(P)
    system = long_guide() if a.guide == "long" else None
    runner = None
    if system:
        p0, _ = chat_inputs(proc, fam, load_image(data.image_path(data.all[0]["record_id"])), long_user(data.all[0]), system)
        cut = p0.index("<|im_start|>user")
        try:
            runner = PrefixRunner(model, proc, dev, p0[:cut])
        except Exception as e:  # noqa: BLE001
            print(f"[x14] prefix cache unavailable ({e!r}) -> uncached", flush=True)
    ex_user = long_user(data.all[0]) if system else prompt_for(data.all[0])
    print(f"[x14 {a.model}] guide={a.guide} system_chars={len(system or '')}\nuser example:\n{ex_user}\n---", flush=True)
    n_checked = 0
    for i in range(start, n):
        r = data.all[i]
        img = load_image(data.image_path(r["record_id"]))
        if system:
            _, inputs = chat_inputs(proc, fam, img, long_user(r), system)
        else:
            inputs = build_inputs(proc, fam, img, prompt_for(r))
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        lg = None
        if runner is not None and runner.ok:
            try:
                lg = runner.logits(inputs)
            except Exception as e:  # noqa: BLE001
                print(f"[x14] prefix path failed ({e!r}) -> uncached from now on", flush=True)
                runner.ok = False
        if lg is None or n_checked < 3:
            with torch.inference_mode():
                full = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            if lg is not None:
                d_max = float((torch.softmax(lg[ids], -1) - torch.softmax(full[ids], -1)).abs().max())
                print(f"[x14] prefix-cache check rec {i}: max |dP| = {d_max:.4f}", flush=True)
                n_checked += 1
                if d_max > float(os.environ.get("KOALTQ_PREFIX_TOL", "0.03")):
                    print("[x14] prefix cache disagrees with the full forward -> uncached from now on", flush=True)
                    runner.ok = False
            lg = full
        pl = torch.softmax(lg[ids], -1).cpu().numpy().astype(np.float32)  # letter order
        if a.ask != "label7":
            P.append(pl)  # [P(A), P(B)]
        else:
            pr = np.zeros(7, np.float32)
            pr[ORDER] = pl  # label order
            P.append(pr)
        if (i + 1) % 200 == 0:
            np.save(out / "probs_partial.tmp.npy", np.stack(P))
            os.replace(out / "probs_partial.tmp.npy", partial)
            print(f"[x14 {a.model}] {i + 1}/{n} {(time.time() - t0) / (i + 1 - start):.2f}s/rec", flush=True)
    P = np.stack(P)
    if a.limit is not None:
        print("[x14] smoke probs:", np.round(P[:5], 3).tolist(), flush=True)
        return
    y = data.y
    if a.ask != "label7":
        sc = P[:, 0].astype(np.float32)
        np.save(out / "score.npy", sc)
        partial.unlink(missing_ok=True)
        t = sc[: data.n_train]
        m = np.isin(y, [0, 5])
        rep = {"ask": a.ask, "model": a.model, "guide": a.guide, "seconds": round(time.time() - t0, 1),
               "auc_불충분형_vs_적절": auc((y[m] == 5).astype(int), t[m]), "auc_부적절_vs_적절": auc((y != 0).astype(int), t),
               "auc_적절_vs_rest": auc((y == 0).astype(int), t)}
        (out / "meta.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[x16]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()}, ensure_ascii=False), flush=True)
        return
    np.save(out / "probs.npy", P)
    partial.unlink(missing_ok=True)
    tr = P[: data.n_train]
    rep = {f"auc_{lab}": auc((y == c).astype(int), tr[:, c]) for c, lab in enumerate(LABELS)}
    rep["train_argmax"] = scores(y, tr.argmax(1))
    rep["seconds"] = round(time.time() - t0, 1)
    rep["model"] = a.model
    rep["guide"] = a.guide
    rep["order"] = a.order
    rep["ctx_chars"] = int(os.environ.get("KOALTQ_CTX_CHARS", "400"))
    rep["ocr_chars"] = int(os.environ.get("KOALTQ_OCR_CHARS", "300"))
    rep["max_image_tokens"] = int(os.environ.get("KOALTQ_MAX_IMAGE_TOKENS", "1024"))
    rep["prefix_cache"] = bool(runner is not None and runner.ok)
    rep["ocr_sha256"] = ocr_sha
    (out / "meta.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[x14] train AUCs:", json.dumps({k: round(v, 3) for k, v in rep.items() if k.startswith("auc")}, ensure_ascii=False),
          "| argmax", {k: round(v, 4) for k, v in rep["train_argmax"].items() if isinstance(v, float)}, flush=True)


def main() -> None:
    cli([])
