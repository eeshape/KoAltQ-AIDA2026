"""X17: judge stacker (x7j features, LightGBM) that also trains on the 109 officially released test labels.

The 109 boundary labels are allowed for training (organizer Q&A nttId=678, column 데이터셋_라벨). They come from the
test sites, where our binary errors concentrate (불충분형 <-> 적절). Train-OOF simulation cannot see that benefit, so it is
checked directly: grouped 5-fold over the 109 rows (by page) - a model trained on train + the other 109 folds vs. a model
trained on train only, both scored on the held-out 109 rows. The run is saved only as an extra base model:
OOF on train rows (each fold model sees the other train folds + all 109 rows) and test predictions (train + 109).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .common import N_CLASSES, RUNS, Data, evaluate_and_save, scores
from .x1_gbm import SEED
from .x7_stack import build_features


def lgbm(seed: int = SEED):
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced",
                              random_state=seed, verbose=-1)


def n_seeds() -> int:
    return max(1, int(os.environ.get("KOALTQ_STACK_SEEDS", "1")))


def bag_tag() -> str:
    return f"_bag{n_seeds()}" if n_seeds() > 1 else ""


def fit_proba(Xfit: np.ndarray, yfit: np.ndarray, *Xev: np.ndarray, w: np.ndarray | None = None) -> list[np.ndarray]:
    """LightGBM fit + class probabilities, averaged over KOALTQ_STACK_SEEDS seeds (row/column subsampling differs per
    seed). The default of 1 seed is exactly the original single model (random_state=SEED). w: optional sample weights."""
    n = n_seeds()
    outs = [np.zeros((len(X), N_CLASSES)) for X in Xev]
    for s in range(n):
        m = lgbm(SEED + s).fit(Xfit, yfit, sample_weight=w)
        for o, X in zip(outs, Xev):
            o += proba(m, X) / n
    return outs


def proba(m, X) -> np.ndarray:
    p = np.zeros((len(X), N_CLASSES))
    p[:, m.classes_] = m.predict_proba(X)
    return p


def main() -> None:
    from sklearn.model_selection import GroupKFold

    data = Data()
    base, judge, vnames, jnames = build_features(data)
    X = np.hstack([base] + judge)
    ntr = data.n_train
    Xtr, ytr, Xte = X[:ntr], data.y, X[ntr:]
    bm = data.boundary_mask_test()
    by = data.boundary_y_test()
    bidx = np.flatnonzero(bm)
    Xb, yb = Xte[bidx], by[bidx]
    gb = np.array([data.test[i]["page_url"] for i in bidx])
    # diagnostic on the 109 rows
    p_without, p_with = np.zeros((len(bidx), N_CLASSES)), np.zeros((len(bidx), N_CLASSES))
    m0 = lgbm().fit(Xtr, ytr)
    p_without[:] = proba(m0, Xb)
    for tri, tei in GroupKFold(n_splits=5).split(Xb, yb, gb):
        m = lgbm().fit(np.vstack([Xtr, Xb[tri]]), np.r_[ytr, yb[tri]])
        p_with[tei] = proba(m, Xb[tei])
    d0, d1 = scores(yb, p_without.argmax(1)), scores(yb, p_with.argmax(1))
    diag = {"heldout109_without": {k: round(v, 4) for k, v in d0.items() if isinstance(v, float)},
            "heldout109_with_other109": {k: round(v, 4) for k, v in d1.items() if isinstance(v, float)},
            "acc_without": round(float((p_without.argmax(1) == yb).mean()), 4),
            "acc_with": round(float((p_with.argmax(1) == yb).mean()), 4)}
    print("[x17] held-out 109 diag:", json.dumps(diag, ensure_ascii=False), flush=True)
    base_name = "x17_stack_lgbm_b109"
    if os.environ.get("KOALTQ_X17_NO_B109") == "1":  # residual rows are the audit-agreed ones; the 109 are the disputed ones
        Xb, yb, gb = Xb[:0], yb[:0], gb[:0]
        base_name = "x17_stack_lgbm_nob109"
        print("[x17] the 109 boundary rows are NOT used as training rows", flush=True)
    extra_name = ""
    if os.environ.get("KOALTQ_X17_EXTRA") == "1":  # also train on the rule-fixed test rows (R2 role markers, R4' siblings)
        from .rules import l0_fixed

        # KOALTQ_X17_EXTRA_SRC=R2 keeps only the guideline-rule rows: the R4' rows carry labels propagated from 재판정_라벨,
        # which this pipeline otherwise uses only in post-processing (sibling inference), never for training
        srcs = {"R2": "R2_restoration_role_marker", "R4": "R4_sibling_propagation"}
        use = [srcs[k] for k in os.environ.get("KOALTQ_X17_EXTRA_SRC", "R2,R4").split(",") if k in srcs]
        ex = [(i, lab) for i, (lab, src) in l0_fixed(data).items() if src in use]
        Xb = np.vstack([Xb, Xte[[i for i, _ in ex]]])
        yb = np.r_[yb, [lab for _, lab in ex]]
        gb = np.r_[gb, [data.test[i]["page_url"] for i, _ in ex]]
        extra_name = "fx" if len(use) == 2 else "fx" + "".join(k for k in srcs if srcs[k] in use).lower()
        print(f"[x17] + {len(ex)} rule-fixed test rows as training rows", flush=True)
    pseudo = os.environ.get("KOALTQ_X17_PSEUDO")  # confident pseudo-labels of unfixed test rows (jsonl record_id/label)
    if pseudo:
        from .common import L2I
        from .rules import l0_fixed

        pos = {r["record_id"]: i for i, r in enumerate(data.test)}
        taken = set(bidx.tolist()) | set(l0_fixed(data))
        pl = [json.loads(x) for x in open(pseudo, encoding="utf-8") if x.strip()]
        pl = [(pos[x["record_id"]], L2I[x["label"]]) for x in pl if x["record_id"] in pos and pos[x["record_id"]] not in taken]
        Xb = np.vstack([Xb, Xte[[i for i, _ in pl]]])
        yb = np.r_[yb, [lab for _, lab in pl]].astype(int)
        gb = np.r_[gb, [data.test[i]["page_url"] for i, _ in pl]]
        extra_name += "ps"
        print(f"[x17] + {len(pl)} pseudo-labelled test rows as training rows", flush=True)
    # base model: OOF on train with all 109 rows added to every fold, test from train + 109
    # optional audit-agreement importance weights for the train rows (residual test rows are the rows where the original
    # label and the independent audit agreed: P(y|x, agreed) ∝ P_orig(y|x) * P_audit(y|x); weighting train rows by the
    # guideline judge's probability of their label, P_judge(y_i|x_i)^gamma, targets that population). Extra rows: weight 1.
    gamma = float(os.environ.get("KOALTQ_X17_AUDITW", "0") or 0)
    wtr = np.ones(ntr)
    if gamma > 0:
        J = np.mean([np.log(np.load(p / "probs.npy") + 1e-6) for p in sorted(RUNS.glob("x14_qwen38_27b_long*"))], axis=0)[:ntr]
        J = J - np.log(np.exp(J).sum(1, keepdims=True))
        wtr = np.exp(gamma * J[np.arange(ntr), ytr])
        wtr = np.clip(wtr / wtr.mean(), 0.02, None)
        extra_name += f"_aw{gamma:g}"
        print(f"[x17] audit-agreement weights gamma={gamma}: min {wtr.min():.3f} median {np.median(wtr):.3f} max {wtr.max():.2f}", flush=True)
    oof = np.zeros((ntr, N_CLASSES))
    for k in range(5):
        tr, va = data.folds != k, data.folds == k
        oof[va], = fit_proba(np.vstack([Xtr[tr], Xb]), np.r_[ytr[tr], yb], Xtr[va], w=np.r_[wtr[tr], np.ones(len(yb))])
    te, = fit_proba(np.vstack([Xtr, Xb]), np.r_[ytr, yb], Xte, w=np.r_[wtr, np.ones(len(yb))])
    evaluate_and_save(base_name + extra_name + bag_tag() + os.environ.get("KOALTQ_X17_SUFFIX", ""), data, oof=oof, test=te, note="x7j features -> LightGBM, trained on train + 109 test labels",
                      extra={"diag_109": diag, "blocks": vnames + jnames, "n_features": int(X.shape[1])})
