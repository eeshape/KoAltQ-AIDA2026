"""X19: stacker that also sees the 27B LoRA classifier's out-of-fold probabilities as features.

Until now the LoRA was mixed with a fixed weight (0.4) after the stacker. Here the 5-fold LoRA runs
(x6_qwen38_27b_lora_fold = held-out fold 0, x6_qwen38_27b_f{k}_lora_fold = held-out fold k) give an honest
LoRA prediction for every train row; the stacker learns per class how much to trust it next to the other features.
Test rows (including the extra rule-fixed / boundary rows used as training rows) get the geometric mean of the five
fold models' test predictions, which matches the OOF distribution (each fold model saw 80% of the train rows).
Same training recipe as X17fx: train + 109 boundary rows + rule-fixed test rows, LightGBM.
"""

from __future__ import annotations

import json

import numpy as np

from .common import N_CLASSES, RUNS, Data, evaluate_and_save, scores
from .x17_boundary import bag_tag, fit_proba
from .x7_stack import build_features

import os

# Fold runs in held-out-fold order 0..4. KOALTQ_X19_FOLDS overrides with a pattern containing "{k}"
# (e.g. "x6_qwen38_27b_c2f{k}_lora_fold" for the improved-card folds).
_PAT = os.environ.get("KOALTQ_X19_FOLDS", "")  # several sources may be given separated by ';'
SOURCES = ([[p.format(k=k) for k in range(5)] for p in _PAT.split(";") if p.strip()] if _PAT else
           [["x6_qwen38_27b_lora_fold"] + [f"x6_qwen38_27b_f{k}_lora_fold" for k in (1, 2, 3, 4)]])
FOLD_RUNS = SOURCES[0]
NO_B109 = os.environ.get("KOALTQ_X19_NO_B109", "") == "1"  # train on train + rule-fixed rows only (no 109 boundary rows)
SUFFIX = os.environ.get("KOALTQ_X19_SUFFIX", "")


def lora_features(data: Data, fold_runs: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full((data.n_train, N_CLASSES), np.nan)
    tests = []
    for k, name in enumerate(fold_runs or FOLD_RUNS):
        d = RUNS / name
        o, m, t = np.load(d / "oof.npy"), np.load(d / "oof_mask.npy"), np.load(d / "test.npy")
        assert (m == (data.folds == k)).all(), f"{name}: held-out rows are not fold {k}"
        oof[m] = o[m]
        tests.append(np.log(t + 1e-9))
    assert not np.isnan(oof).any(), "missing LoRA OOF rows"
    te = np.mean(tests, axis=0)
    te = np.exp(te - te.max(1, keepdims=True))
    te /= te.sum(1, keepdims=True)
    return np.log(oof + 1e-6), np.log(te + 1e-6)


def main() -> None:
    from .rules import l0_fixed

    data = Data()
    base, judge, vnames, jnames = build_features(data)
    blocks = [lora_features(data, src) for src in SOURCES]
    lo_tr, lo_te = blocks[0]
    X = np.hstack([base] + judge)
    for b_tr, b_te in blocks:
        X = np.hstack([X, np.vstack([b_tr, b_te])])
    print(f"[x19] LoRA OOF sources: {[s[0] for s in SOURCES]}", flush=True)
    ntr = data.n_train
    Xtr, ytr, Xte = X[:ntr], data.y, X[ntr:]
    print(f"[x19] features {X.shape} (+7 LoRA OOF log-probs) | LoRA OOF comp "
          f"{scores(ytr, lo_tr.argmax(1))['composite']:.4f}", flush=True)
    bidx = np.flatnonzero(data.boundary_mask_test())
    by = data.boundary_y_test()
    ex = [(i, lab) for i, (lab, src) in l0_fixed(data).items() if src in ("R2_restoration_role_marker", "R4_sibling_propagation")]
    assert len(ex) >= 120, f"only {len(ex)} rule-fixed rows: is boundary_test_rejudge.jsonl missing?"
    if NO_B109:
        bidx = bidx[:0]
    Xb = np.vstack([Xte[bidx], Xte[[i for i, _ in ex]]])
    yb = np.r_[by[bidx], [lab for _, lab in ex]].astype(int)
    pseudo = os.environ.get("KOALTQ_X19_PSEUDO")  # confident pseudo-labels of unfixed test rows (as in x17 KOALTQ_X17_PSEUDO)
    ps_tag = ""
    if pseudo:
        from .common import L2I

        pos = {r["record_id"]: i for i, r in enumerate(data.test)}
        taken = set(np.flatnonzero(data.boundary_mask_test()).tolist()) | set(l0_fixed(data))
        pl = [json.loads(x) for x in open(pseudo, encoding="utf-8") if x.strip()]
        pl = [(pos[x["record_id"]], L2I[x["label"]]) for x in pl if x["record_id"] in pos and pos[x["record_id"]] not in taken]
        Xb = np.vstack([Xb, Xte[[i for i, _ in pl]]])
        yb = np.r_[yb, [lab for _, lab in pl]].astype(int)
        ps_tag = "_ps"
        print(f"[x19] + {len(pl)} pseudo-labelled test rows as training rows", flush=True)
    print(f"[x19] folds {FOLD_RUNS[0]}..{FOLD_RUNS[-1]} | extra training rows {len(yb)} (boundary {len(bidx)})", flush=True)
    oof = np.zeros((ntr, N_CLASSES))
    for k in range(5):
        tr, va = data.folds != k, data.folds == k
        oof[va], = fit_proba(np.vstack([Xtr[tr], Xb]), np.r_[ytr[tr], yb], Xtr[va])
    te, = fit_proba(np.vstack([Xtr, Xb]), np.r_[ytr, yb], Xte)
    evaluate_and_save("x19_stack_lgbm_lora" + ("_nob109" if NO_B109 else "") + ps_tag + bag_tag() + SUFFIX, data, oof=oof, test=te,
                      note="x17fx features + 5-fold 27B LoRA OOF log-probs -> LightGBM (train + 109 + rule-fixed test rows)",
                      extra={"blocks": vnames + jnames + ["lora27b_oof"], "n_features": int(X.shape[1]),
                             "lora_oof_comp": scores(ytr, lo_tr.argmax(1))["composite"]})
    print(json.dumps({"x19_oof": scores(ytr, oof.argmax(1))["composite"]}), flush=True)
