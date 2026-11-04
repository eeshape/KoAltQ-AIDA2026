"""X11: Korean text cross-encoder (KLUE-RoBERTa-large) over (alt, structured evidence incl. OCR) with
logit-adjusted CE + square-root repeat-factor sampling. 5-fold grouped OOF + full-train test.

This is the only model family that previously caught caption-duplicate 중복형 (pairwise alt-vs-context reading);
adding OCR text gives it the 'information only inside the image' signal.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from .common import N_CLASSES, Data, evaluate_and_save
from .prompting import ocr_lookup, require_ocr
from .rules import link_kind

DEVICE = os.environ.get("KOALTQ_DEVICE", "cuda:0")


def pair(r: dict) -> tuple[str, str]:
    alt = r.get("alt_text") or "(빈 alt)"
    kind = link_kind(r)
    ocr = ocr_lookup().get(r["record_id"], "")
    b = " [SEP] ".join([
        f"링크:{'예' if r.get('in_link') else '아니오'}({kind}) {(r.get('link_dest') or '')[:120]}",
        f"문맥: {(r.get('context_text') or '')[:300]}",
        f"이미지글자: {ocr[:200] if ocr else '없음'}",
        f"제목: {(r.get('page_title') or '')[:80]} / {(r.get('heading_path') or '')[:80]}",
    ])
    return alt, b


def train_fold(model_id, tok, tr_recs, tr_y, ev_recs, epochs, lr, seed):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=N_CLASSES).to(DEVICE)
    freq = np.bincount(tr_y, minlength=N_CLASSES) / len(tr_y)
    rc = np.maximum(1.0, np.sqrt(0.06 / np.maximum(freq, 1e-9)))
    sampled = freq * rc
    sampled /= sampled.sum()
    log_prior = torch.tensor(np.log(np.maximum(sampled, 1e-6)), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bs = 16
    steps_per_ep = int(math.ceil(len(tr_y) * rc[tr_y].mean() / bs))
    total = steps_per_ep * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * total)) * max(0.0, (total - s) / total))
    enc_cache = {}

    def enc(recs):
        a, b = zip(*[pair(r) for r in recs])
        return tok(list(a), list(b), truncation="longest_first", max_length=384, padding=True, return_tensors="pt",
                   return_token_type_ids=False).to(DEVICE)  # RoBERTa: type_vocab_size=1 (segment-B ids would overflow)

    model.train()
    for ep in range(epochs):
        reps = rc[tr_y]
        cnt = np.floor(reps).astype(int) + (rng.random(len(tr_y)) < (reps - np.floor(reps))).astype(int)
        idx = np.repeat(np.arange(len(tr_y)), cnt)
        rng.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc([tr_recs[i] for i in b])).logits.float()
            loss = F.cross_entropy(logits + log_prior, torch.tensor(tr_y[b], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    outs = []
    with torch.inference_mode():
        for recs in ev_recs:
            P = []
            for s in range(0, len(recs), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    P.append(torch.softmax(model(**enc(recs[s:s + 64])).logits.float(), -1).cpu().numpy())
            outs.append(np.concatenate(P) if P else np.zeros((0, N_CLASSES)))
    del model, opt
    torch.cuda.empty_cache()
    return outs


def cli(argv: list[str]) -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klue/roberta-large")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--tag", default="klue_large")
    a = ap.parse_args(argv)
    require_ocr()
    data = Data()
    tok = AutoTokenizer.from_pretrained(a.model)
    oof = np.zeros((data.n_train, N_CLASSES))
    t0 = time.time()
    for k in range(5):
        tr = data.folds != k
        trr = [r for r, m in zip(data.train, tr) if m]
        (p,) = train_fold(a.model, tok, trr, data.y[tr], [[r for r, m in zip(data.train, ~tr) if m]], a.epochs, a.lr, a.seed + k)
        oof[~tr] = p
        print(f"[x11] fold {k} done {time.time() - t0:.0f}s", flush=True)
    (te,) = train_fold(a.model, tok, list(data.train), data.y, [data.test], a.epochs, a.lr, a.seed + 99)
    evaluate_and_save(f"x11_xenc_{a.tag}", data, oof=oof, test=te,
                      note=f"{a.model} cross-encoder (alt | link/context/OCR/title), logit-adjusted CE, {a.epochs} ep")


def main() -> None:
    cli([])
