"""Leaderboard-style score estimate ("LB 추정") for every run, using the official formula
    composite = 0.5 * binary F1 (부적절 = any non-'적절' label is positive) + 0.5 * 7-class macro-F1
on a SIMULATED 600-row test confusion matrix (test labels are hidden):

  * fixed rows (what our submissions overwrite): the 109 officially released test labels (exact) and the
    L0 guideline rules (R1 filename 8, R2 restoration role-marker 116, R3 digit-only) - counted as correct
  * residual rows: true class counts = official derived test totals minus fixed labels; each class is spread over
    predicted classes with the model's conditional confusion P(pred | true) measured on its out-of-fold rows,
    re-weighted to the test site mix
  * expected TP/FP/FN -> per-class F1 -> macro-F1, binary F1 -> composite

It is an ESTIMATE (assumes OOF error rates transfer to the test residual); real scores need a leaderboard submission.
"""

from __future__ import annotations

import json

import numpy as np

from .common import DERIVED_TEST_COUNTS, L2I, LABELS, N_CLASSES, RUNS, Data, test_shift_weights
from .rules import l0_rule_label

def fixed_rows(data: Data) -> dict[int, int]:
    from .rules import l0_fixed

    return {i: lab for i, (lab, _src) in l0_fixed(data).items()}


def residual_counts(data: Data, fixed: dict[int, int]) -> np.ndarray:
    total = np.array([DERIVED_TEST_COUNTS[lab] for lab in LABELS], dtype=float)
    fc = np.bincount(np.array(list(fixed.values()), dtype=int), minlength=N_CLASSES).astype(float)
    return np.maximum(total - fc, 0.0)


def f1s_from_confusion(M: np.ndarray) -> tuple[np.ndarray, float]:
    tp = np.diag(M)
    fp = M.sum(0) - tp
    fn = M.sum(1) - tp
    den = 2 * tp + fp + fn
    pc = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    # binary: positive = any non-appropriate class
    t_pos = M[1:, :].sum(1)  # rows: true inappropriate
    btp = M[1:, 1:].sum()
    bfp = M[0, 1:].sum()
    bfn = M[1:, 0].sum()
    bden = 2 * btp + bfp + bfn
    return pc, float(2 * btp / bden) if bden > 0 else 0.0


def lb_estimate(data: Data, oof_pred: np.ndarray, mask: np.ndarray, weighted: bool = True) -> dict:
    y = data.y[mask]
    p = np.asarray(oof_pred)[mask]
    w = test_shift_weights(data)[mask] if weighted else np.ones(mask.sum())
    C = np.zeros((N_CLASSES, N_CLASSES))
    for t in range(N_CLASSES):
        sel = y == t
        if w[sel].sum() > 0:
            C[t] = np.bincount(p[sel], weights=w[sel], minlength=N_CLASSES) / w[sel].sum()
        else:
            C[t, t] = 1.0
    fixed = fixed_rows(data)
    res = residual_counts(data, fixed)
    M = res[:, None] * C
    for lab in fixed.values():
        M[lab, lab] += 1.0
    pc, binf = f1s_from_confusion(M)
    macro = float(pc.mean())
    return {"composite": 0.5 * macro + 0.5 * binf, "macro_f1": macro, "binary_f1": binf,
            "per_class_f1": {LABELS[c]: round(float(pc[c]), 3) for c in range(N_CLASSES)},
            "n_fixed": len(fixed), "residual_counts": dict(zip(LABELS, res.astype(int).tolist()))}


def table() -> list[dict]:
    data = Data()
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not (d / "oof.npy").exists():
            continue
        oof, mask = np.load(d / "oof.npy"), np.load(d / "oof_mask.npy")
        est = lb_estimate(data, oof.argmax(1), mask)
        sim = lb_sim(data, oof, mask)
        met = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
        rows.append({"run": d.name, "lb_est": round(est["composite"], 4), "lb_est_macro": round(est["macro_f1"], 4),
                     "lb_est_binary": round(est["binary_f1"], 4), "sim_argmax": round(sim["argmax"][0], 4),
                     "sim_quota": round(sim["quota"][0], 4), "sim_quota_sd": round(sim["quota"][1], 4),
                     "oof_comp": round(met.get("oof", {}).get("composite", float("nan")), 4),
                     "rows": "fold0" if mask.sum() < data.n_train else "5fold", "per_class": est["per_class_f1"]})
    for hp in sorted(RUNS.glob("x8_blend*/honest_preds.npz")):
        tagx = {"x8_blend": "x8", "x8_blend_lbobj": "x8lb", "x8_blend_lbq": "x8lbq"}.get(hp.parent.name, hp.parent.name)
        z = np.load(hp)
        for k in z.files:
            if k.startswith("L_fold"):
                continue
            arr = z[k]
            full = arr.shape[0] == data.n_train
            m = np.ones(data.n_train, bool) if full else (data.folds == 0)
            if arr.ndim == 2:  # honest nested log-scores -> probabilities -> argmax + quota simulation
                P = np.zeros((data.n_train, N_CLASSES))
                e = np.exp(arr - arr.max(1, keepdims=True))
                P[m] = e / e.sum(1, keepdims=True)
                pred = P.argmax(1)
                sim = lb_sim(data, P, m)
                sa, sq, sd = sim["argmax"][0], sim["quota"][0], sim["quota"][1]
            else:
                pred = arr if full else np.zeros(data.n_train, int)
                if not full:
                    pred[data.folds == 0] = arr
                sa = sq = sd = float("nan")
            est = lb_estimate(data, pred, m)
            rows.append({"run": f"{tagx}:{k}", "lb_est": round(est["composite"], 4), "lb_est_macro": round(est["macro_f1"], 4),
                         "lb_est_binary": round(est["binary_f1"], 4), "oof_comp": float("nan"), "sim_argmax": round(sa, 4),
                         "sim_quota": round(sq, 4), "sim_quota_sd": round(sd, 4),
                         "rows": "nested" if full else "fold0-nested", "per_class": est["per_class_f1"]})
    rows.sort(key=lambda r: -r["lb_est"])
    return rows


def main() -> None:
    rows = table()
    (RUNS / "lb_estimates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("예상 리더보드 점수(LB 추정) — 공식: 0.5*이진F1 + 0.5*macroF1, test 600건 시뮬레이션 (실제 제출 점수 아님)")
    print(f"{'순위':>3} {'실험':34s} {'LB추정':>7} {'macro':>7} {'binary':>7} {'시뮬argmax':>9} {'시뮬quota':>9} {'OOFcomp':>8} {'근거':>8}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3} {r['run']:34s} {r['lb_est']:7.4f} {r['lb_est_macro']:7.4f} {r['lb_est_binary']:7.4f} "
              f"{r['sim_argmax']:9.4f} {r['sim_quota']:9.4f} {r['oof_comp']:8.4f} {r['rows']:>8}")


def quota_assign(L: np.ndarray, target: np.ndarray, iters: int = 120, eta: float = 0.4) -> np.ndarray:
    b = np.zeros(L.shape[1])
    b[target <= 0] = -30.0
    for _ in range(iters):
        cnt = np.bincount((L + b).argmax(1), minlength=L.shape[1])
        if np.abs(cnt - target).sum() <= 1:
            break
        upd = eta * np.log((target + 0.5) / (cnt + 0.5))
        upd[target <= 0] = 0.0
        b = b + upd
    return (L + b).argmax(1)


def lb_sim(data: Data, oof_prob: np.ndarray, mask: np.ndarray, n_sim: int = 200, seed: int = 0) -> dict:
    """Monte-Carlo LB: draw pseudo test-residual pools from OOF rows (class counts = official residual counts,
    rows sampled with test-site weights), decide by argmax or by class quotas, add the fixed rows, score."""
    rng = np.random.default_rng(seed)
    y = data.y[mask]
    L = np.log(np.asarray(oof_prob)[mask] + 1e-9)
    w = test_shift_weights(data)[mask]
    fixed = fixed_rows(data)
    res = residual_counts(data, fixed).astype(int)
    fixed_lab = np.array(list(fixed.values()), dtype=int)
    by_cls = {c: np.flatnonzero(y == c) for c in range(N_CLASSES)}
    out = {"argmax": [], "quota": []}
    for _ in range(n_sim):
        idx, yt = [], []
        for c in range(N_CLASSES):
            if res[c] == 0:
                continue
            pool = by_cls[c]
            if len(pool) == 0:
                continue
            pw = w[pool] / w[pool].sum() if w[pool].sum() > 0 else None
            pick = rng.choice(pool, size=res[c], replace=True, p=pw)
            idx.extend(pick)
            yt.extend([c] * res[c])
        idx, yt = np.array(idx), np.array(yt)
        for strat in ("argmax", "quota"):
            pr = L[idx].argmax(1) if strat == "argmax" else quota_assign(L[idx], res.astype(float))
            yy = np.concatenate([yt, fixed_lab])
            pp = np.concatenate([pr, fixed_lab])
            M = np.zeros((N_CLASSES, N_CLASSES))
            np.add.at(M, (yy, pp), 1.0)
            pc, binf = f1s_from_confusion(M)
            out[strat].append(0.5 * pc.mean() + 0.5 * binf)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}
