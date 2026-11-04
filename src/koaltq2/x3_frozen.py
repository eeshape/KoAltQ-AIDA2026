"""X3-X5: frozen VLM feature extraction (evidence-card prompt + image) and classification heads.

Backbones (forward-only, no weight updates):
  qwen35_9b      Qwen/Qwen3.5-9B                 bf16
  internvl35_14b OpenGVLab/InternVL3_5-14B-HF    bf16
  qwen3vl_32b    Qwen/Qwen3-VL-32B-Instruct      4-bit NF4 (bf16 compute)
  qwen35_08b     Qwen/Qwen3.5-0.8B               bf16 (smoke / connection test only)

Feature per record = concat(last-token hidden state at layers L/2, 3L/4, L) + mean of image-token states at layer L.
Heads on cached features (5-fold grouped OOF + full-train test):
  lr      : standardized LogisticRegression(class_weight=balanced)
  mlp_ce  : LayerNorm-Linear(256)-GELU-Dropout(0.1)-Linear(7), cross-entropy           (Plan B head)
  mlp_bs  : same head, Balanced Softmax (logits + log fold-prior during training)       (Plan B E2)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .common import N_CLASSES, Data, evaluate_and_save, run_dir

MODELS = {
    "qwen35_9b": {"repo": "Qwen/Qwen3.5-9B", "quant": None, "family": "qwen35"},
    "qwen35_08b": {"repo": "Qwen/Qwen3.5-0.8B", "quant": None, "family": "qwen35"},
    "internvl35_14b": {"repo": "OpenGVLab/InternVL3_5-14B-HF", "quant": None, "family": "internvl"},
    "qwen3vl_32b": {"repo": "Qwen/Qwen3-VL-32B-Instruct", "quant": "auto", "family": "qwen3vl"},
    "qwen3vl_8b": {"repo": "Qwen/Qwen3-VL-8B-Instruct", "quant": None, "family": "qwen3vl"},
    "gemma4_31b": {"repo": "google/gemma-4-31B-it", "quant": "auto", "family": "gemma4"},
    "gemma4_12b": {"repo": "google/gemma-4-12B-it", "quant": None, "family": "gemma4"},
    "qwen38_27b": {"repo": "Qwen/Qwen3.8-27B", "quant": "auto", "family": "qwen35"},
}
DEVICE = os.environ.get("KOALTQ_DEVICE", "cuda:0")
MAX_IMAGE_TOKENS = int(os.environ.get("KOALTQ_MAX_IMAGE_TOKENS", "1024"))  # Qwen: 32x32 px per visual token -> ~1 MP cap (small poster text stays legible)
MIN_IMAGE_TOKENS = 64


# ----------------------------------------------------------------------------- model loading
def load_backbone(name: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    spec = MODELS[name]
    local = snapshot_download(spec["repo"])
    kw = {"dtype": torch.bfloat16, "device_map": DEVICE, "attn_implementation": "sdpa"}
    quant = spec["quant"]
    if quant == "auto":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        quant = None if total >= 75 else "nf4"
        spec = {**spec, "quant": quant}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(local, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(local)
    configure_pixels(processor, spec["family"])
    commit = local.rstrip("/").split("/")[-1]
    return model, processor, spec, commit


def configure_pixels(processor, family: str) -> None:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    if family in ("qwen35", "qwen3vl"):
        px = 32 * 32
        lo, hi = MIN_IMAGE_TOKENS * px, MAX_IMAGE_TOKENS * px
        ip.size = {"shortest_edge": lo, "longest_edge": hi}  # v5 converts the plain dict back to SizeDict
        for attr, val in (("min_pixels", lo), ("max_pixels", hi)):
            if hasattr(ip, attr):
                setattr(ip, attr, val)
    elif family == "internvl":
        for attr, val in (("max_patches", 4), ("min_patches", 1)):
            if hasattr(ip, attr):
                setattr(ip, attr, val)


def check_pixel_cap(model, processor, family: str) -> int:
    """Process a 1536x1536 dummy image and assert the visual-token cap really holds."""
    from PIL import Image

    tok = image_token_id(model, processor)
    inp = build_inputs(processor, family, Image.new("RGB", (1536, 1536), (200, 200, 200)), "test")
    n = int((inp["input_ids"][0] == tok).sum()) if tok is not None else -1
    print(f"[pixel-cap] family={family} 1536x1536 -> {n} image tokens (cap {MAX_IMAGE_TOKENS})", flush=True)
    if family in ("qwen35", "qwen3vl"):
        assert 0 < n <= MAX_IMAGE_TOKENS + 8, f"image-token cap not applied: {n}"
    return n


def require_fast_deltanet(family: str) -> None:
    """Qwen3.5 DeltaNet silently falls back to a slow torch loop without flash-linear-attention."""
    import torch

    if family != "qwen35" or not torch.cuda.is_available() or os.environ.get("KOALTQ_ALLOW_SLOW") == "1":
        return
    try:
        import importlib.metadata as md

        import fla.ops.gated_delta_rule  # noqa: F401
        print("[fla] flash-linear-attention", md.version("flash-linear-attention"), flush=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"flash-linear-attention unavailable ({e!r}); install it or set KOALTQ_ALLOW_SLOW=1") from e


def build_inputs(processor, family: str, image, text: str):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    kw = {"enable_thinking": False} if family == "qwen35" else {}
    try:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    except TypeError:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return processor(text=[prompt], images=[image], return_tensors="pt")


def image_token_id(model, processor) -> int | None:
    for obj in (model.config, getattr(model.config, "text_config", None)):
        if obj is not None and getattr(obj, "image_token_id", None) is not None:
            return int(obj.image_token_id)
    tok = getattr(processor, "image_token", None)
    if tok:
        return int(processor.tokenizer.convert_tokens_to_ids(tok))
    return None


# ----------------------------------------------------------------------------- extraction
def extract(name: str, limit: int | None = None, force: bool = False) -> dict:
    import torch

    from .prompting import cards_hash, evidence_card, load_image, require_ocr

    data = Data()
    out = run_dir(f"x3_{name}")
    records = data.all if limit is None else data.all[:limit]
    ocr_sha = require_ocr()
    chash = cards_hash(records)
    sig = {"model": name, "max_image_tokens": MAX_IMAGE_TOKENS, "ocr_sha": ocr_sha, "cards_hash": chash}
    if limit is None and not force and (out / "feats.npy").exists() and (out / "meta.json").exists():
        old = json.loads((out / "meta.json").read_text())
        if all(old.get(k) == v for k, v in sig.items()):
            print("[x3] feats.npy up to date for", sig, "- skipping extraction (use --force to redo)", flush=True)
            return old
    partial = out / "feats_partial.npy"
    sidecar = out / "feats_partial.sig.json"
    done: list[np.ndarray] = []
    if partial.exists() and limit is None and sidecar.exists() and json.loads(sidecar.read_text()) == sig:
        done = list(np.load(partial))
        print(f"[x3] resuming from {len(done)} cached rows", flush=True)
    model, processor, spec, commit = load_backbone(name)
    require_fast_deltanet(spec["family"])
    check_pixel_cap(model, processor, spec["family"])
    img_tok = image_token_id(model, processor)
    t0 = time.time()
    n_tok, n_img = [], []
    prompt_hash = hashlib.sha256(evidence_card(records[0]).encode()).hexdigest()[:16]
    for i in range(len(done), len(records)):
        r = records[i]
        inputs = build_inputs(processor, spec["family"], load_image(data.image_path(r["record_id"])), evidence_card(r))
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outp = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = outp.hidden_states
        L = len(hs) - 1
        vecs = [hs[l][0, -1, :] for l in (L // 2, (3 * L) // 4, L)]
        ids = inputs["input_ids"][0]
        mask = (ids == img_tok) if img_tok is not None else torch.zeros_like(ids, dtype=torch.bool)
        n_tok.append(int(ids.numel()))
        n_img.append(int(mask.sum()))
        vecs.append(hs[L][0, mask].mean(0) if mask.any() else torch.zeros_like(vecs[-1]))
        done.append(torch.cat(vecs).float().cpu().numpy().astype(np.float16))
        del outp, hs
        if (i + 1) % 200 == 0 and limit is None:
            np.save(out / "feats_partial.tmp.npy", np.stack(done))
            os.replace(out / "feats_partial.tmp.npy", partial)
            sidecar.write_text(json.dumps(sig))
            rate = (time.time() - t0) / max(1, i + 1 - 0)
            print(f"[x3 {name}] {i + 1}/{len(records)} {rate:.2f}s/rec tok_mean={np.mean(n_tok):.0f} img_tok_mean={np.mean(n_img):.0f}",
                  flush=True)
    feats = np.stack(done)
    meta = {**sig, "repo": spec["repo"], "commit": commit, "quant": spec["quant"], "n": int(len(feats)),
            "dim": int(feats.shape[1]), "prompt_sha16": prompt_hash, "max_image_tokens": MAX_IMAGE_TOKENS,
            "tokens_mean": float(np.mean(n_tok)) if n_tok else None, "image_tokens_mean": float(np.mean(n_img)) if n_img else None,
            "image_tokens_min": int(np.min(n_img)) if n_img else None, "seconds": round(time.time() - t0, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
    if limit is None:
        np.save(out / "feats.tmp.npy", feats)
        os.replace(out / "feats.tmp.npy", out / "feats.npy")
        partial.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    (out / ("meta.json" if limit is None else "smoke_meta.json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("[x3] extract meta", meta, flush=True)
    return meta


# ----------------------------------------------------------------------------- heads
def _mlp_head_predict(Xtr, ytr, Xva_list, groups_tr, loss: str, seed: int = 0):
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupShuffleSplit

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    itr, ies = next(gss.split(Xtr, ytr, groups_tr))
    d = Xtr.shape[1]
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, N_CLASSES)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    prior = np.bincount(ytr[itr], minlength=N_CLASSES).astype(np.float64) + 1.0
    log_prior = torch.tensor(np.log(prior / prior.sum()), dtype=torch.float32, device=dev)
    xt = torch.tensor(Xtr[itr], dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr[itr], dtype=torch.long, device=dev)
    xe = torch.tensor(Xtr[ies], dtype=torch.float32, device=dev)
    ye = torch.tensor(ytr[ies], dtype=torch.long, device=dev)
    ce = nn.CrossEntropyLoss()
    best, best_state, bad = 1e9, None, 0
    for _ in range(100):
        head.train()
        perm = torch.randperm(len(xt), device=dev)
        for s in range(0, len(xt), 64):
            b = perm[s:s + 64]
            logits = head(xt[b])
            if loss == "bs":
                logits = logits + log_prior
            l = ce(logits, yt[b])
            opt.zero_grad()
            l.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            le = head(xe)
            vl = ce(le + log_prior if loss == "bs" else le, ye).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    head.load_state_dict(best_state)
    head.eval()
    outs = []
    with torch.no_grad():
        for Xva in Xva_list:
            outs.append(torch.softmax(head(torch.tensor(Xva, dtype=torch.float32, device=dev)), -1).cpu().numpy())
    return outs


def heads(name: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = Data()
    feats = np.load(run_dir(f"x3_{name}") / "feats.npy").astype(np.float32)
    assert feats.shape[0] == 3000, feats.shape
    Xtr_all, Xte = feats[: data.n_train], feats[data.n_train:]
    y, folds, groups = data.y, data.folds, data.groups
    for head in ("lr", "mlp_ce", "mlp_bs"):
        oof = np.zeros((data.n_train, N_CLASSES))
        te_list = []
        for k in list(range(5)) + ["full"]:
            tr = (folds != k) if k != "full" else np.ones(data.n_train, bool)
            sc = StandardScaler().fit(Xtr_all[tr])
            Xa = sc.transform(Xtr_all[tr])
            if head == "lr":
                m = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
                m.fit(Xa, y[tr])

                def pp(X):
                    p = np.zeros((len(X), N_CLASSES))
                    p[:, m.classes_] = m.predict_proba(X)
                    return p

                if k == "full":
                    te_list.append(pp(sc.transform(Xte)))
                else:
                    oof[folds == k] = pp(sc.transform(Xtr_all[folds == k]))
            else:
                loss = "bs" if head == "mlp_bs" else "ce"
                if k == "full":
                    (p,) = _mlp_head_predict(Xa, y[tr], [sc.transform(Xte)], groups[tr], loss)
                    te_list.append(p)
                else:
                    (p,) = _mlp_head_predict(Xa, y[tr], [sc.transform(Xtr_all[folds == k])], groups[tr], loss)
                    oof[folds == k] = p
        evaluate_and_save(f"x3_{name}_{head}", data, oof=oof, test=te_list[0],
                          note=f"frozen {name} features -> {head}")


def cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None, help="smoke: extract first N records only, no heads")
    ap.add_argument("--heads-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.heads_only:
        extract(a.model, a.limit, a.force)
    if a.limit is None:
        heads(a.model)


def main() -> None:
    cli(["--model", "qwen35_08b", "--limit", "8"])
