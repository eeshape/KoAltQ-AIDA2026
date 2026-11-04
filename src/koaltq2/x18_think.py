"""X18: reasoning ("thinking") guideline judge on the rows that matter for the leaderboard.

Why: the non-boundary test labels agree with the organizer's independent audit (all 600 test rows were audited; only the
109 rows where audit and original label disagreed kept the original label). Zero-shot guideline judges match that audit
far better than train-label models (109-row re-judgment AUC .90 vs .38). A judge that reasons before answering may match it
better still. Generation is expensive, so only the test residual rows (not fixed by L0) and the 109 boundary rows (scored
against 재판정_라벨 as a validation of audit agreement) are judged.

Method: Qwen3.5-family VLM with enable_thinking=True, guideline (v1.1 sections 2-3) as system prompt, record card + A-G
question as user turn; greedy generation with a short thinking budget; then the letter distribution is read at the answer
position after '</think>' (if the budget runs out, '</think>' is appended and the letter logits are read from a forward).
Output: RUNS/x18_<model>/probs_test.npy (600 x 7, NaN rows for unjudged test rows) + meta.json (109-row audit AUC).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import L2I, Data, run_dir
from .x12_attr import MODELS
from .x14_judge import LETTERS, chat_inputs, long_guide, long_user


def cli(argv: list[str]) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .lbest import fixed_rows
    from .prompting import load_image, require_ocr
    from .x3_frozen import configure_pixels, require_fast_deltanet

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen38_27b", choices=list(MODELS))
    ap.add_argument("--think_tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    require_ocr()
    dev = os.environ.get("KOALTQ_DEVICE", "cuda:0")
    data = Data()
    out = run_dir(f"x18_{a.model}")
    repo, fam = MODELS[a.model]
    assert fam == "qwen35", "thinking template is implemented for the Qwen3.5 family"
    require_fast_deltanet(fam)
    local = snapshot_download(repo)
    proc = AutoProcessor.from_pretrained(local)
    configure_pixels(proc, fam)
    model = AutoModelForImageTextToText.from_pretrained(local, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa")
    model.eval()
    tok = proc.tokenizer
    ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    end_think = tok.encode("</think>", add_special_tokens=False)
    fx = fixed_rows(data)
    todo = [i for i in range(len(data.test)) if i not in fx or data.test[i]["record_id"] in data.boundary_test]
    todo.sort(key=lambda i: data.test[i]["record_id"] not in data.boundary_test)  # boundary rows first (validation)
    if a.limit:
        todo = todo[: a.limit]
    system = long_guide() + "\n\n판정 전에 핵심 근거만 짧게 생각하고, 마지막에 기호 하나로 답하세요."
    partial = out / "partial.json"
    done = json.loads(partial.read_text()) if partial.exists() else {}
    t0, n0 = time.time(), len(done)
    for k, i in enumerate(todo):
        rid = data.test[i]["record_id"]
        if rid in done:
            continue
        r = data.test[i]
        img = load_image(data.image_path(rid))
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": long_user(r)}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        inputs = proc(text=[prompt], images=[img], return_tensors="pt")
        inputs = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in inputs.items()}
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=a.think_tokens, do_sample=False)
        new = gen[0, inputs["input_ids"].shape[1]:].tolist()
        # cut after '</think>' if present, else force it; then read the letter distribution at the next position
        cut = None
        for j in range(len(new) - len(end_think) + 1):
            if new[j:j + len(end_think)] == end_think:
                cut = j + len(end_think)
                break
        prefix = new[:cut] if cut is not None else new + end_think
        tail = tok.encode("\n\n", add_special_tokens=False)
        seq = torch.cat([inputs["input_ids"], torch.tensor([prefix + tail], device=dev)], dim=1)
        full = {kk: v for kk, v in inputs.items() if kk not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        full["input_ids"] = seq
        full["attention_mask"] = torch.ones_like(seq)
        if "mm_token_type_ids" in inputs:
            pad = torch.zeros((1, seq.shape[1] - inputs["mm_token_type_ids"].shape[1]), dtype=inputs["mm_token_type_ids"].dtype, device=dev)
            full["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
        with torch.inference_mode():
            lg = model(**full, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(lg[ids], -1).cpu().numpy().tolist()
        done[rid] = {"p": p, "thought_tokens": len(prefix), "closed": cut is not None}
        if (len(done) - n0) % 20 == 0:
            partial.write_text(json.dumps(done))
            el = time.time() - t0
            print(f"[x18] {len(done)}/{len(todo)} {el / max(1, len(done) - n0):.1f}s/row", flush=True)
    partial.write_text(json.dumps(done))
    P = np.full((len(data.test), 7), np.nan, dtype=np.float32)
    for i, r in enumerate(data.test):
        if r["record_id"] in done:
            P[i] = done[r["record_id"]]["p"]
    np.save(out / "probs_test.npy", P)
    rep = {"model": a.model, "think_tokens": a.think_tokens, "rows": len(done),
           "closed_think_frac": float(np.mean([v["closed"] for v in done.values()])) if done else None}
    try:
        from sklearn.metrics import roc_auc_score

        bm = [i for i, r in enumerate(data.test) if r["record_id"] in data.boundary_rejudge and r["record_id"] in done]
        yr = np.array([L2I[data.boundary_rejudge[data.test[i]["record_id"]]] for i in bm])
        rep["auc109_auditor_inappropriate"] = float(roc_auc_score((yr != 0).astype(int), 1 - P[bm, 0]))
        rep["acc109_auditor"] = float((P[bm].argmax(1) == yr).mean())
    except Exception as e:  # noqa: BLE001
        rep["auc_error"] = repr(e)
    (out / "meta.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[x18]", json.dumps(rep, ensure_ascii=False), flush=True)


def main() -> None:
    cli([])
