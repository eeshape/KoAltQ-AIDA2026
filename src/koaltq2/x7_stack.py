"""X7: feature-level stacking — guideline rule features + OCR evidence features + PCA of frozen VLM features
-> gradient boosting (sklearn HGB and LightGBM), 5-fold grouped OOF + full-train test.

PCA is fit label-free on all 3,000 rows. VLM feature sets are taken from every finished X3 run (feats.npy).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .common import RUNS, Data, evaluate_and_save
from .rules import feature_matrix
from .x1_gbm import SEED, cv_predict, tfidf_matrix


def vlm_blocks(n_comp: int = 48) -> tuple[list[np.ndarray], list[str]]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    blocks, names = [], []
    for d in sorted(RUNS.glob("x3_*")):
        f = d / "feats.npy"
        if not f.exists():
            continue
        X = np.load(f).astype(np.float32)
        if X.shape[0] != 3000:
            continue
        Z = PCA(n_components=n_comp, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
        blocks.append(Z.astype(np.float32))
        names.append(d.name)
    return blocks, names


def build_features(data: Data):
    """(base matrix, judge blocks, base names, judge names) for all 3,000 rows."""
    from sklearn.decomposition import TruncatedSVD

    Xf, _ = feature_matrix(data)
    Xs = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(tfidf_matrix(data)).astype(np.float32)
    ocr_path = RUNS / "x2_ocr" / ("ocr_features_v2.npy" if os.environ.get("KOALTQ_OCR_V2") == "1" else "ocr_features.npy")
    if os.environ.get("KOALTQ_OCR_V2") == "1":
        assert ocr_path.exists(), "run `koaltq2.run x2 v2` first"
    Xo = np.load(ocr_path) if ocr_path.exists() else np.zeros((3000, 0), np.float32)
    vb, vnames = vlm_blocks()
    attrs = []
    for d in sorted(RUNS.glob("x12_*")):
        if (d / "attrs.npy").exists():
            a = np.load(d / "attrs.npy").astype(np.float32)
            attrs.append(np.nan_to_num(a, nan=-1.0))
            vnames.append(d.name)
    judge, jnames = [], []
    for d in sorted(RUNS.glob("x14_*")):  # zero-shot guideline judge: 7-class log-probabilities (label-free)
        if (d / "probs.npy").exists():
            judge.append(np.log(np.load(d / "probs.npy").astype(np.float32) + 1e-6))
            jnames.append(d.name)
    avg = [np.log(np.load(d / "probs.npy").astype(np.float32) + 1e-6) for d in sorted(RUNS.glob("x14_qwen38_27b_long*"))
           if (d / "probs.npy").exists()]
    if len(avg) >= 2:  # option-order-debiased 27B judge: mean log-prob over the long-guide variants (fwd/rev/rot*)
        judge.append(np.mean(avg, axis=0))
        jnames.append(f"avg27b_long[{len(avg)}]")
    for d in sorted(RUNS.glob("x16_*")):  # targeted yes/no judges (label-free)
        if (d / "score.npy").exists():
            v = np.clip(np.load(d / "score.npy").astype(np.float32), 1e-6, 1 - 1e-6)
            judge.append(np.log(v / (1 - v))[:, None])
            jnames.append(d.name)
    print(f"[x7] blocks: rules {Xf.shape[1]}, svd 64, ocr {Xo.shape[1]}, vlm/attr {vnames}, judges {jnames}", flush=True)
    base = np.hstack([Xf, Xs, Xo] + vb + attrs)
    return base, judge, vnames, jnames


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    data = Data()
    base, judge, vnames, jnames = build_features(data)
    tag = "+".join(n.replace("x3_", "") for n in vnames) or "none"
    # two stackers: without the X14 judges (complements X15, which uses only the judges) and with them
    variants = [("x7", base, vnames)] + ([("x7j", np.hstack([base] + judge), vnames + jnames)] if judge else [])
    for prefix, X, names in variants:
        oof, te = cv_predict(
            lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=10,
                                                   l2_regularization=1.0, class_weight="balanced", random_state=SEED),
            X, data)
        evaluate_and_save(f"{prefix}_stack_hgb", data, oof=oof, test=te, note=f"rules+svd+ocr+PCA48({tag}) [{prefix}] -> HGB",
                          extra={"vlm_blocks": names, "n_features": int(X.shape[1])})
        try:
            import lightgbm as lgb

            oof2, te2 = cv_predict(
                lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                           class_weight="balanced", random_state=SEED, verbose=-1),
                X, data)
            evaluate_and_save(f"{prefix}_stack_lgbm", data, oof=oof2, test=te2, note=f"rules+svd+ocr+PCA48({tag}) [{prefix}] -> LightGBM",
                              extra={"vlm_blocks": names, "n_features": int(X.shape[1])})
        except ImportError:
            print("[x7] lightgbm unavailable; skipped", flush=True)
    (RUNS / "x7_stack_hgb" / "blocks.json").write_text(json.dumps(vnames), encoding="utf-8")
