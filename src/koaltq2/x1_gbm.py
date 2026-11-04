"""X1: L0 rule table + two cheap supervised models with 5-fold grouped OOF.

  * x1_ctx_lr : TF-IDF (word + char) over a structured text view -> multinomial LR (class_weight balanced)
  * x1_hgb    : HistGradientBoosting over guideline rule features + image metadata + SVD text factors

TF-IDF/SVD are fit label-free on all 3,000 texts (train+test). Folds = the fixed manifest (sha 4154b923).
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .common import N_CLASSES, Data, evaluate_and_save, run_dir
from .rules import apply_l0, feature_matrix, link_kind, rule_precision_table
from .submit import write_submission

SEED = 20261102


def text_view(r: dict) -> str:
    title = (r.get("page_title") or "").replace("\n", " ")[:200]
    kind = link_kind(r)
    return " | ".join([
        f"ALT {r.get('alt_text') or '<EMPTY>'}",
        f"LINK[{kind}] {r.get('link_dest') or ''}",
        f"CTX {r.get('context_text') or ''}",
        f"TITLE {title}",
        f"HEAD {r.get('heading_path') or ''}",
    ])


def tfidf_matrix(data: Data):
    texts = [text_view(r) for r in data.all]
    alts = [(r.get("alt_text") or "<EMPTY>") for r in data.all]
    v_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=120000)
    v_alt = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=1, sublinear_tf=True, max_features=40000)
    X = sparse.hstack([v_word.fit_transform(texts), v_char.fit_transform(texts), v_alt.fit_transform(alts)]).tocsr()
    return X


def cv_predict(make_model, X, data: Data, fit_kwargs_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """5-fold grouped OOF on the 2,400 train rows + a full-train fit for the 600 test rows."""
    Xtr, Xte = X[: data.n_train], X[data.n_train:]
    oof = np.zeros((data.n_train, N_CLASSES))
    for k in range(5):
        tr = data.folds != k
        va = data.folds == k
        m = make_model()
        kw = fit_kwargs_fn(tr) if fit_kwargs_fn else {}
        m.fit(Xtr[tr], data.y[tr], **kw)
        p = np.zeros((va.sum(), N_CLASSES))
        p[:, m.classes_] = m.predict_proba(Xtr[va])
        oof[va] = p
    m = make_model()
    kw = fit_kwargs_fn(np.ones(data.n_train, bool)) if fit_kwargs_fn else {}
    m.fit(Xtr, data.y, **kw)
    te = np.zeros((Xte.shape[0], N_CLASSES))
    te[:, m.classes_] = m.predict_proba(Xte)
    return oof, te


def main(data: Data | None = None) -> dict:
    t0 = time.time()
    data = data or Data()
    out = run_dir("x1_rules")
    table = rule_precision_table(data)
    (out / "rule_precision.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[X1] L0 hard-rule precision on train / fires on test:")
    for t in table:
        print("   ", t)

    # --- model A: TF-IDF + LR
    Xt = tfidf_matrix(data)
    oof_lr, te_lr = cv_predict(
        lambda: LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"), Xt, data)
    res_lr = evaluate_and_save("x1_ctx_lr", data, oof=oof_lr, test=te_lr, note="TF-IDF word+char + LR balanced")

    # --- model B: rule/meta features + SVD text factors -> HGB
    Xf, names = feature_matrix(data)
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    Xs = svd.fit_transform(Xt)
    Xd = np.hstack([Xf, Xs.astype(np.float32)])
    oof_hgb, te_hgb = cv_predict(
        lambda: HistGradientBoostingClassifier(learning_rate=0.06, max_iter=350, max_leaf_nodes=31, min_samples_leaf=8,
                                               l2_regularization=1.0, class_weight="balanced", random_state=SEED),
        Xd, data)
    res_hgb = evaluate_and_save("x1_hgb", data, oof=oof_hgb, test=te_hgb, note="rule+meta features + SVD64 -> HGB balanced",
                                extra={"n_features": len(names) + 64})
    np.save(run_dir("x1_hgb") / "feature_matrix_rules.npy", Xf)
    (run_dir("x1_hgb") / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")

    # --- simple geometric blend of the two
    blend_oof = np.exp(0.5 * np.log(oof_lr + 1e-9) + 0.5 * np.log(oof_hgb + 1e-9))
    blend_oof /= blend_oof.sum(1, keepdims=True)
    blend_te = np.exp(0.5 * np.log(te_lr + 1e-9) + 0.5 * np.log(te_hgb + 1e-9))
    blend_te /= blend_te.sum(1, keepdims=True)
    res_bl = evaluate_and_save("x1_blend", data, oof=blend_oof, test=blend_te, note="geo-mean(ctx_lr, hgb)")

    # --- S0 submission: best X1 model by OOF composite + L0 (rules + 109 official labels)
    best = max([("x1_ctx_lr", res_lr, te_lr), ("x1_hgb", res_hgb, te_hgb), ("x1_blend", res_bl, te_bl := blend_te)],
               key=lambda t: t[1]["oof"]["composite"])
    pred, log = apply_l0(data, best[2].argmax(1))
    write_submission(data, pred, "S0_x1_l0", {"base_model": best[0], "l0_log": log})
    print(f"[X1] S0 base={best[0]} l0={log}  ({time.time() - t0:.1f}s)")
    return {"best": best[0], "l0": log}


if __name__ == "__main__":
    main()
