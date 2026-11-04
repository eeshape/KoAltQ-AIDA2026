"""X15: calibrated judge - multinomial logistic regression on the label-free VLM outputs only.

Inputs (all zero-shot, no label leakage across folds): log-probabilities of every X14 guideline-judge run and the X12
attribute answers (NaN -> 0.5 plus a missing flag). A small linear model turns the judges' letter preferences
(which carry option-order / prior bias) into calibrated 7-class probabilities. Grouped 5-fold OOF + full-train test,
saved as a regular run so X8 can weight it against the other base models.
"""

from __future__ import annotations

import numpy as np

from .common import RUNS, Data, evaluate_and_save
from .x1_gbm import cv_predict


def judge_features() -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for d in sorted(RUNS.glob("x14_*")):
        if (d / "probs.npy").exists():
            blocks.append(np.log(np.load(d / "probs.npy").astype(np.float64) + 1e-6))
            names.append(d.name)
    avg = [np.log(np.load(d / "probs.npy").astype(np.float64) + 1e-6) for d in sorted(RUNS.glob("x14_qwen38_27b_long*"))
           if (d / "probs.npy").exists()]
    if len(avg) >= 2:
        blocks.append(np.mean(avg, axis=0))
        names.append(f"avg27b_long[{len(avg)}]")
    for d in sorted(RUNS.glob("x16_*")):
        if (d / "score.npy").exists():
            v = np.clip(np.load(d / "score.npy").astype(np.float64), 1e-6, 1 - 1e-6)
            blocks.append(np.log(v / (1 - v))[:, None])
            names.append(d.name)
    for d in sorted(RUNS.glob("x12_*")):
        if (d / "attrs.npy").exists():
            a = np.load(d / "attrs.npy").astype(np.float64)
            blocks.append(np.nan_to_num(a, nan=0.5))
            blocks.append(np.isnan(a[:, [0]]).astype(np.float64))
            names.append(d.name)
    return (np.hstack(blocks) if blocks else np.zeros((3000, 0))), names


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = Data()
    X, names = judge_features()
    assert X.shape[1] > 0, "no X14/X12 outputs found"
    print(f"[x15] judge features {X.shape} from {names}", flush=True)
    for C in (0.05,):
        oof, te = cv_predict(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
                             X, data)
        evaluate_and_save("x15_judge_lr", data, oof=oof, test=te, note=f"LR(C={C}) on X14 log-probs + X12 attrs",
                          extra={"sources": names, "n_features": int(X.shape[1])})
