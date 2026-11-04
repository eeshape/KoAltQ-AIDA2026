"""X6/X9: discriminative LoRA fine-tuning of a VLM with a 7-way classification head.

Default backbone: Qwen/Qwen3.5-9B via transformers' native Qwen3_5ForSequenceClassification (image + evidence card).
Objective (differs from the colleague's generative label-token QLoRA):
  * classification head on the last (non-pad) token, bf16 LoRA r16 on the language model only (vision frozen)
  * square-root repeat-factor sampling for rare classes (LVIS-style, stochastic rounding per epoch)
  * logit-adjusted cross-entropy w.r.t. the *sampled* class prior (no double correction)
Modes:
  fold  : train on folds 1-4 (1,942) -> predict fold 0 (458) + test (600)          [selection / comparison]
  full  : train on all 2,400 + the 109 officially released test labels -> predict test [final model]
  smoke : tiny run (connection test: forward/backward/save/reload) on N records
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

from .common import L2I, N_CLASSES, Data, evaluate_and_save, run_dir

DEVICE = os.environ.get("KOALTQ_DEVICE", "cuda:0")
BACKBONES = {"qwen35_9b": "Qwen/Qwen3.5-9B", "qwen35_4b": "Qwen/Qwen3.5-4B", "qwen35_08b": "Qwen/Qwen3.5-0.8B",
             "qwen38_27b": "Qwen/Qwen3.8-27B", "qwen36_27b": "Qwen/Qwen3.6-27B"}
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(backbone: str, r: int = 16):
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForSequenceClassification

    from .x3_frozen import configure_pixels, require_fast_deltanet

    require_fast_deltanet("qwen35")
    local = snapshot_download(BACKBONES[backbone])
    processor = AutoProcessor.from_pretrained(local)
    configure_pixels(processor, "qwen35")
    dtype = torch.bfloat16
    model = Qwen3_5ForSequenceClassification.from_pretrained(local, num_labels=N_CLASSES, dtype=dtype,
                                                              attn_implementation="sdpa", device_map=DEVICE)
    pad = processor.tokenizer.pad_token_id
    model.config.pad_token_id = pad
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.pad_token_id = pad
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():  # keep the trainable head in fp32 so AdamW updates are not rounded away
        if p.requires_grad and "score" in n:
            p.data = p.data.float()
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[x6] backbone={backbone} commit={local.rstrip('/').split('/')[-1]} lora_modules={n_lora} trainable={trainable/1e6:.1f}M",
          flush=True)
    assert n_lora > 0, "LoRA target regex matched nothing"
    return model, processor, local.rstrip("/").split("/")[-1]


def encode(processor, r: dict, data: Data):
    from .prompting import evidence_card, load_image
    from .x3_frozen import build_inputs

    inputs = build_inputs(processor, "qwen35", load_image(data.image_path(r["record_id"])), evidence_card(r))
    return {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}


def repeat_factor_epoch(labels: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    freq = np.bincount(labels, minlength=N_CLASSES) / len(labels)
    rc = np.maximum(1.0, np.sqrt(t / np.maximum(freq, 1e-9)))
    reps = rc[labels]
    counts = np.floor(reps).astype(int) + (rng.random(len(labels)) < (reps - np.floor(reps))).astype(int)
    idx = np.repeat(np.arange(len(labels)), counts)
    rng.shuffle(idx)
    return idx


def train_and_predict(mode: str, backbone: str, epochs: float, lr: float, accum: int, rf_t: float, tau: float,
                      limit: int | None, seed: int, r: int, tag: str = "") -> dict:
    import torch
    import torch.nn.functional as F

    data = Data()
    name = (f"x6_{backbone}{('_' + tag) if tag else ''}_lora_{mode}" if mode != "smoke" else f"x6_{backbone}_smoke")
    out = run_dir(name)
    set_seed(seed)
    hold = int(os.environ.get("KOALTQ_X6_FOLD", "0"))  # held-out fold for mode=fold (default 0)
    if mode == "fold":
        train_recs = [r for r, f in zip(data.train, data.folds) if f != hold]
        train_y = data.y[data.folds != hold]
        eval_sets = {"fold0": [r for r, f in zip(data.train, data.folds) if f == hold], "test": data.test}
    elif mode == "full":
        bt = [r for r in data.test if r["record_id"] in data.boundary_test]
        train_recs = list(data.train) + bt
        train_y = np.concatenate([data.y, np.array([L2I[data.boundary_test[r["record_id"]]] for r in bt])])
        if os.environ.get("KOALTQ_X6_EXTRA") == "1":  # also the rule-fixed test rows (R2 role markers, R4' siblings)
            from .rules import l0_fixed

            ex = [(i, lab) for i, (lab, src) in l0_fixed(data).items() if src in ("R2_restoration_role_marker", "R4_sibling_propagation")]
            train_recs += [data.test[i] for i, _ in ex]
            train_y = np.concatenate([train_y, np.array([lab for _, lab in ex])])
            print(f"[x6] + {len(ex)} rule-fixed test rows as training rows", flush=True)
        pseudo = os.environ.get("KOALTQ_X6_PSEUDO")  # optional confident pseudo-labels for unfixed test rows
        if pseudo:
            pl = [json.loads(x) for x in open(pseudo, encoding="utf-8") if x.strip()]
            byid = {r["record_id"]: r for r in data.test}
            have = {r["record_id"] for r in train_recs}
            pl = [x for x in pl if x["record_id"] in byid and x["record_id"] not in have]
            train_recs += [byid[x["record_id"]] for x in pl]
            train_y = np.concatenate([train_y, np.array([L2I[x["label"]] for x in pl])])
            print(f"[x6] + {len(pl)} pseudo-labelled test rows from {pseudo}", flush=True)
        eval_sets = {"test": data.test}
    else:  # smoke
        n = limit or 16
        train_recs, train_y = list(data.train[:n]), data.y[:n]
        eval_sets = {"fold0": [r for r, f in zip(data.train, data.folds) if f == 0][:8], "test": data.test[:8]}
    if limit and mode != "smoke":
        train_recs, train_y = train_recs[:limit], train_y[:limit]

    from .prompting import cards_hash, require_ocr
    from .x3_frozen import check_pixel_cap

    ocr_sha = require_ocr()
    model, processor, commit = build_model(backbone, r=r)
    check_pixel_cap(model, processor, "qwen35")
    rng = np.random.default_rng(seed)
    # sampled prior for logit adjustment = class distribution after repeat-factor sampling (expected)
    freq = np.bincount(train_y, minlength=N_CLASSES) / len(train_y)
    rc = np.maximum(1.0, np.sqrt(rf_t / np.maximum(freq, 1e-9)))
    sampled = freq * rc
    sampled = sampled / sampled.sum()
    log_prior = torch.tensor(np.log(np.maximum(sampled, 1e-6)), dtype=torch.float32, device=DEVICE)

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "score" in n]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "score" not in n]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lr}, {"params": head_params, "lr": lr * 5}], weight_decay=0.01)
    n_ep_full = int(math.ceil(epochs))
    est_steps = int(len(train_recs) * float(np.mean(rc[train_y])) * epochs / accum) + 1
    warm = max(1, int(0.05 * est_steps))

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, est_steps - warm))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.train()
    t0 = time.time()
    step, micro, losses = 0, 0, []
    max_micro = int(len(train_recs) * float(np.mean(rc[train_y])) * epochs)
    for ep in range(n_ep_full):
        order = repeat_factor_epoch(train_y, rf_t, rng)
        for i in order:
            if micro >= max_micro:
                break
            inputs = encode(processor, train_recs[i], data)
            with torch.autocast("cuda" if DEVICE.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                logits = model(**inputs).logits.float()
            loss = F.cross_entropy(logits + tau * log_prior, torch.tensor([int(train_y[i])], device=DEVICE)) / accum
            loss.backward()
            losses.append(loss.item() * accum)
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)  # clip separately: the head's large grads must not
                torch.nn.utils.clip_grad_norm_(head_params, 1.0)  # dominate the norm and starve the LoRA updates
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or mode == "smoke":
                    el = time.time() - t0
                    eta = el / micro * (max_micro - micro)
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    print(f"[x6 {mode}] ep{ep} step {step}/{est_steps} micro {micro}/{max_micro} loss {np.mean(losses[-accum*25:]):.4f}"
                          f" {el/micro:.2f}s/sample eta {eta/60:.1f}m peak {mem:.1f}GB", flush=True)
        model.save_pretrained(out / f"adapter_ep{ep}")
    train_sec = time.time() - t0

    model.eval()
    probs: dict[str, np.ndarray] = {}
    t1 = time.time()
    for key, recs in eval_sets.items():
        P = np.zeros((len(recs), N_CLASSES))
        for j, rec in enumerate(recs):
            with torch.inference_mode():
                inputs = encode(processor, rec, data)
                with torch.autocast("cuda" if DEVICE.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                    lg = model(**inputs).logits.float()
            P[j] = torch.softmax(lg, -1)[0].cpu().numpy()
        probs[key] = P
        np.save(out / f"probs_{key}.npy", P)
    infer_sec = time.time() - t1
    meta = {"mode": mode, "backbone": backbone, "commit": commit, "epochs": epochs, "lr": lr, "accum": accum,
            "rf_t": rf_t, "tau": tau, "r": r, "seed": seed, "n_train": len(train_recs), "steps": step,
            "train_sec": round(train_sec, 1), "infer_sec": round(infer_sec, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "sampled_prior": sampled.round(4).tolist(), "final_loss": float(np.mean(losses[-200:])) if losses else None,
            "boundary_rows_in_training": mode == "full", "ocr_sha": ocr_sha,
            "train_cards_hash": cards_hash(train_recs), "max_image_tokens": int(os.environ.get("KOALTQ_MAX_IMAGE_TOKENS", "1024"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("[x6] meta", meta, flush=True)

    if mode == "fold":
        oof = np.zeros((data.n_train, N_CLASSES))
        oof[data.folds == hold] = probs["fold0"]
        evaluate_and_save(name, data, oof=oof, test=probs["test"], oof_mask=(data.folds == hold), extra={"meta": meta, "held_out_fold": hold},
                          note="discriminative LoRA, trained on folds 1-4")
    elif mode == "full":
        evaluate_and_save(name, data, oof=None, test=probs["test"], extra={"meta": meta},
                          note="discriminative LoRA on 2,400 + 109 boundary test rows (boundary diag is in-sample)")
    # reload contract (Plan B): fresh base + saved adapter must reproduce probabilities on a few rows
    if mode in ("smoke", "full"):
        import gc

        del opt, sched, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            reload_check(backbone, out / f"adapter_ep{n_ep_full - 1}", processor, data, eval_sets["test"][:4], probs["test"][:4])
            meta["reload_check"] = "pass"
        except Exception as e:  # noqa: BLE001
            meta["reload_check"] = f"FAILED: {e!r}"[:300]
            print("[x6] reload check failed (predictions already saved):", meta["reload_check"], flush=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def reload_check(backbone, adapter_dir, processor, data, recs, ref_probs) -> None:
    import gc

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import Qwen3_5ForSequenceClassification

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    local = snapshot_download(BACKBONES[backbone])
    base = Qwen3_5ForSequenceClassification.from_pretrained(local, num_labels=N_CLASSES, dtype=torch.bfloat16,
                                                             attn_implementation="sdpa", device_map=DEVICE)
    base.config.pad_token_id = processor.tokenizer.pad_token_id
    if getattr(base.config, "text_config", None) is not None:
        base.config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    m = PeftModel.from_pretrained(base, str(adapter_dir))
    m.eval()
    diffs = []
    for rec, ref in zip(recs, ref_probs):
        with torch.inference_mode():
            inputs = encode(processor, rec, data)
            with torch.autocast("cuda" if DEVICE.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                p = torch.softmax(m(**inputs).logits.float(), -1)[0].cpu().numpy()
        diffs.append(float(np.abs(p - ref).max()))
    print(f"[x6] reload check: max |Δprob| over {len(diffs)} rows = {max(diffs):.5f}", flush=True)
    assert max(diffs) < 0.02, "reloaded adapter does not reproduce predictions"
    del m, base
    gc.collect()


def cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fold", "full", "smoke"], required=True)
    ap.add_argument("--backbone", default="qwen35_9b", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--rf-t", type=float, default=0.06, help="repeat-factor threshold (class freq)")
    ap.add_argument("--tau", type=float, default=1.0, help="logit-adjustment strength")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    accum = 4 if a.mode == "smoke" else a.accum
    train_and_predict(a.mode, a.backbone, a.epochs, a.lr, accum, a.rf_t, a.tau, a.limit, a.seed, a.r, a.tag)


def main() -> None:
    cli(["--mode", "smoke", "--backbone", "qwen35_08b", "--limit", "8", "--epochs", "1"])
