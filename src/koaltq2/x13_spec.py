"""X13: rare-class specialists + quota decision (submission S4).

For 무관형 / 무의미형 / 장식오용형 the quota picks (k = official residual count) are chosen by the rank-average of
  (a) the ensemble probability for that class and (b) zero-shot VLM attribute scores averaged over all X12 runs:
      무관형 <- 1 - q_match (alt unrelated to image),  무의미형 <- q_empty,  장식오용형 <- q_decor (only non-empty alts)
All other rows follow the class-quota decision on the ensemble logits.
The same rule is simulated on pseudo test pools drawn from honest OOF rows (MC, test-site weights) for an honest estimate.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata

from .common import L2I, LABELS, N_CLASSES, RUNS, Data, run_dir, test_shift_weights
from .lbest import f1s_from_confusion, fixed_rows, quota_assign, residual_counts
from .rules import apply_l0
from .submit import write_submission

RARE = {"무관형": 0, "무의미형": 4, "장식오용형": 1}  # class -> X12 attribute column (q_match, q_empty, q_decor)
# weight of the specialist rank vs the ensemble rank, derived per class from standalone train AUCs (see alpha_rule):
#   first run (X12 only): 무관형 .92 vs .80 -> 99 (specialist only); 무의미형 .94 vs .85 -> 4; 장식오용형 .81 vs .91 -> 0
ALPHA = {"무관형": 99.0, "무의미형": 4.0, "장식오용형": 0.0}
TOP_SOURCES = 2  # rank-average the two best label-free sources per class (single best-by-AUC picks were unstable)


def alpha_rule(auc_spec: float, auc_ens: float) -> float:
    d = auc_spec - auc_ens
    return 99.0 if d >= 0.10 else 4.0 if d >= 0.03 else 1.0 if d > -0.03 else 0.0


def train_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def spec_candidates(data: Data) -> dict[str, dict[int, np.ndarray]]:
    """Label-free specialist scores per rare class: X12 attribute mean and every X14 guideline-judge run."""
    cands = {}
    a = attr_scores(data)
    if a:
        cands["x12_mean"] = a
    has_alt0 = np.array([bool((r.get("alt_text") or "").strip()) for r in data.all])
    for d in sorted(RUNS.glob("x12_*")):  # single attribute runs too (a strong model can beat the mean)
        if (d / "attrs.npy").exists():
            A = np.load(d / "attrs.npy")
            one = {}
            for lab, col in RARE.items():
                v = A[:, col].astype(float)
                if np.isnan(v).all():
                    continue
                v = 1 - v if lab == "무관형" else v
                one[L2I[lab]] = np.where(has_alt0, np.nan_to_num(v, nan=0.0), 0.0)
            if len(one) == len(RARE):
                cands[d.name] = one
            elif one:
                cands[d.name] = {c: one.get(c, a[c] if a else np.zeros(len(data.all))) for c in [L2I[lab] for lab in RARE]}
    has_alt = np.array([bool((r.get("alt_text") or "").strip()) for r in data.all])
    for d in sorted(RUNS.glob("x14_*")):
        if (d / "probs.npy").exists():
            P = np.load(d / "probs.npy")
            cands[d.name] = {L2I[lab]: np.where(has_alt, P[:, L2I[lab]], 0.0) for lab in RARE}
    if len(cands) > 1:  # rank-average of all sources
        n = len(data.all)
        srcs = list(cands.values())
        cands["rankavg_all"] = {L2I[lab]: np.mean([rankdata(v[L2I[lab]]) / n for v in srcs], axis=0) for lab in RARE}
    return cands


def choose_specialists(data: Data, P_oof: np.ndarray) -> tuple[dict[int, np.ndarray] | None, dict]:
    """Per rare class pick the specialist source with the best train AUC, then set alpha from AUC(spec) - AUC(ensemble)."""
    cands = spec_candidates(data)
    if not cands:
        return None, {}
    y, ntr = data.y, data.n_train
    spec, info = {}, {}
    for lab in RARE:
        c = L2I[lab]
        yc = (y == c).astype(int)
        aucs = {name: train_auc(yc, v[c][:ntr]) for name, v in cands.items() if name != "rankavg_all"}
        top = sorted(aucs, key=lambda k: -1 if np.isnan(aucs[k]) else -aucs[k])[:TOP_SOURCES]
        n_all = len(data.all)
        spec[c] = np.mean([rankdata(cands[k][c]) / n_all for k in top], axis=0)  # rank average of the best sources
        a_spec = train_auc(yc, spec[c][:ntr])
        a_ens = train_auc(yc, P_oof[:, c])
        ALPHA[lab] = alpha_rule(a_spec, a_ens)
        info[lab] = {"source": "+".join(top), "auc_spec": round(a_spec, 4), "auc_ensemble": round(a_ens, 4), "alpha": ALPHA[lab],
                     "all_aucs": {k: round(v, 4) for k, v in aucs.items()}}
    return spec, info


def attr_scores(data: Data) -> dict[int, np.ndarray] | None:
    mats = [np.load(d / "attrs.npy") for d in sorted(RUNS.glob("x12_*")) if (d / "attrs.npy").exists()]
    if not mats:
        return None
    A = np.nanmean(np.stack(mats), axis=0)  # (3000, 5)
    has_alt = np.array([bool((r.get("alt_text") or "").strip()) for r in data.all])
    out = {}
    for lab, col in RARE.items():
        s = A[:, col].copy()
        if lab == "무관형":
            s = 1 - s
        s = np.nan_to_num(s, nan=0.0)
        if lab in ("무관형", "장식오용형", "무의미형"):
            s = np.where(has_alt, s, 0.0)  # these classes need a non-empty alt
        out[L2I[lab]] = s
    return out


def decide(L: np.ndarray, target: np.ndarray, spec: dict[int, np.ndarray] | None) -> np.ndarray:
    """Rare classes first: top-k rows by rank(ensemble logit) + alpha * rank(specialist score); the remaining rows
    are assigned by class quotas over the non-rare classes (so their predicted counts match the targets exactly)."""
    if not spec:
        return quota_assign(L, target)
    pred = np.full(len(L), -1)
    taken = np.zeros(len(L), bool)
    rest_target = target.astype(float).copy()
    for c in spec:
        k = int(target[c])
        rest_target[c] = 0
        if k <= 0:
            continue
        a = ALPHA.get(LABELS[c], 1.0)
        score = rankdata(L[:, c]) + a * rankdata(spec[c])
        score[taken] = -np.inf
        top = np.argsort(-score)[:k]
        pred[top] = c
        taken[top] = True
    rest = ~taken
    pred[rest] = quota_assign(L[rest], rest_target)
    return pred


def simulate(data: Data, P_oof: np.ndarray, spec_tr: dict | None, n_sim: int = 300, seed: int = 0,
             mask: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    y, w = data.y, test_shift_weights(data)
    if mask is not None:
        y = np.where(mask, y, -1)  # rows outside the mask are never drawn
    L = np.log(P_oof + 1e-9)
    fixed = fixed_rows(data)
    res = residual_counts(data, fixed).astype(int)
    fixed_lab = np.array(list(fixed.values()), dtype=int)
    by = {c: np.flatnonzero(y == c) for c in range(N_CLASSES)}
    out = {"quota": [], "quota_spec": [], "per_class_spec": []}
    for _ in range(n_sim):
        idx, yt = [], []
        for c in range(N_CLASSES):
            if res[c] and len(by[c]):
                p = w[by[c]] / w[by[c]].sum()
                idx.extend(rng.choice(by[c], size=res[c], replace=True, p=p))
                yt.extend([c] * res[c])
        idx, yt = np.array(idx), np.array(yt)
        for key, sp in (("quota", None), ("quota_spec", spec_tr)):
            spi = {c: s[idx] for c, s in sp.items()} if sp else None
            pr = decide(L[idx], res.astype(float), spi)
            M = np.zeros((N_CLASSES, N_CLASSES))
            np.add.at(M, (np.concatenate([yt, fixed_lab]), np.concatenate([pr, fixed_lab])), 1.0)
            pc, binf = f1s_from_confusion(M)
            out[key].append(0.5 * pc.mean() + 0.5 * binf)
            if key == "quota_spec":
                out["per_class_spec"].append(pc)
    return {"quota": float(np.mean(out["quota"])), "quota_spec": float(np.mean(out["quota_spec"])),
            "quota_spec_sd": float(np.std(out["quota_spec"])),
            "per_class_spec": dict(zip(LABELS, np.mean(out["per_class_spec"], axis=0).round(3).tolist()))}


def main() -> None:
    data = Data()
    srcs = [RUNS / n for n in ("x8_blend_lbq", "x8_blend_lbq_nojudge", "x8_blend_lbobj", "x8_blend")
            if (RUNS / n / "honest_preds.npz").exists()]
    for src in srcs:
        run_source(data, src)


def run_source(data: Data, src) -> None:
    z = np.load(src / "honest_preds.npz")
    Lh = z["L_ensemble_bias"]
    P_oof = np.exp(Lh - Lh.max(1, keepdims=True))
    P_oof /= P_oof.sum(1, keepdims=True)
    L_te = np.load(src / "test_logits_final.npy")
    spec_all, spec_info = choose_specialists(data, P_oof)
    print(f"[x13] specialists: {json.dumps(spec_info, ensure_ascii=False)}", flush=True)
    spec_tr = {c: s[: data.n_train] for c, s in spec_all.items()} if spec_all else None
    if "L_fold0" in z.files:  # per-fold honest simulation (selection never saw the simulated fold)
        parts = []
        for k in range(5):
            Lk = z[f"L_fold{k}"]
            Pk = np.exp(Lk - Lk.max(1, keepdims=True))
            Pk /= Pk.sum(1, keepdims=True)
            parts.append(simulate(data, Pk, spec_tr, n_sim=200, mask=data.folds == k))
        sim = {"quota": float(np.mean([p["quota"] for p in parts])), "quota_spec": float(np.mean([p["quota_spec"] for p in parts])),
               "quota_spec_sd": float(np.mean([p["quota_spec_sd"] for p in parts])),
               "per_fold_quota_spec": [round(p["quota_spec"], 4) for p in parts],
               "per_class_spec": {lab: round(float(np.mean([p["per_class_spec"][lab] for p in parts])), 3) for lab in LABELS},
               "estimator": "per-fold"}
    else:
        sim = simulate(data, P_oof, spec_tr)
        sim["estimator"] = "pooled"
    sim["specialists"] = spec_info
    print(f"[x13] source={src.name} | MC LB estimate: quota {sim['quota']:.4f} -> quota+specialists "
          f"{sim['quota_spec']:.4f} (sd {sim['quota_spec_sd']:.4f}) | per-class {sim['per_class_spec']}", flush=True)
    # test decision on the residual pool
    fixed = fixed_rows(data)
    res = residual_counts(data, fixed)
    resid = np.array([i not in fixed for i in range(len(data.test))])
    spec_te = {c: s[data.n_train:][resid] for c, s in spec_all.items()} if spec_all else None
    pred = L_te.argmax(1)
    pred[resid] = decide(L_te[resid], np.round(res * resid.sum() / max(res.sum(), 1)), spec_te)
    p4, log = apply_l0(data, pred)
    write_submission(data, p4, f"S4_quota_specialists_l0_{src.name.replace('x8_blend', 'x8')}",
                     {"source": src.name, "sim": sim, "l0": log, "x12_runs": [d.name for d in sorted(RUNS.glob("x12_*"))],
                      "x14_runs": [d.name for d in sorted(RUNS.glob("x14_*"))]})
    out = run_dir("x13_spec")
    (out / f"report_{src.name}.json").write_text(json.dumps(sim, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(sim, ensure_ascii=False, indent=1), encoding="utf-8")
