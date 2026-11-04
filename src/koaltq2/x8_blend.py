"""X8: ensemble selection + class-bias tuning + LoRA mixing + test prior (quota) calibration -> submissions S1-S3.

Honesty rules: every tuned quantity (ensemble weights, class biases, LoRA mix weight) is also evaluated with a
nested split so the printed 'honest' scores are not selected on the rows they are reported on.
Selection objective = mean(composite on OOF, composite on OOF re-weighted to the test site mix).
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold

from .common import DERIVED_TEST_COUNTS, L2I, LABELS, N_CLASSES, RUNS, Data, run_dir, scores, test_shift_weights
from .rules import apply_l0, l0_rule_label
from .submit import write_submission

EPS = 1e-9
LORA_OVERRIDE: str | None = None


OBJECTIVE = "oof"  # "oof": mean(OOF composite, test-weighted OOF composite); "lb": simulated leaderboard composite
_LB_CTX: dict = {}


def _lb_composite(y, pred, w) -> float:
    """Expected leaderboard composite: residual class counts x P(pred|true) (weighted) + L0-fixed rows as correct."""
    res, fixed_lab = _LB_CTX["res"], _LB_CTX["fixed_lab"]
    M = np.zeros((N_CLASSES, N_CLASSES))
    for t in range(N_CLASSES):
        sel = y == t
        if w[sel].sum() > 0:
            M[t] = res[t] * np.bincount(pred[sel], weights=w[sel], minlength=N_CLASSES) / w[sel].sum()
        else:
            M[t, t] = res[t]
    np.add.at(M, (fixed_lab, fixed_lab), 1.0)
    tp = np.diag(M)
    fp, fn = M.sum(0) - tp, M.sum(1) - tp
    pc = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    btp, bfp, bfn = M[1:, 1:].sum(), M[0, 1:].sum(), M[1:, 0].sum()
    binf = 2 * btp / max(2 * btp + bfp + bfn, 1e-12)
    return 0.5 * pc.mean() + 0.5 * binf


_DATA: list = []  # [Data] set in main (needed by the quota simulation)
QSIM_N = 40


def softmax_rows(L: np.ndarray) -> np.ndarray:
    e = np.exp(L - L.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def score_logits(L: np.ndarray, y: np.ndarray, w: np.ndarray, rows: np.ndarray) -> float:
    """Selection score of full-length (2400-row) logits restricted to `rows`.
    lbq: Monte-Carlo leaderboard composite under class quotas (the decision rule of S3/S4), common random numbers;
    otherwise: the argmax objective."""
    if OBJECTIVE == "lbq":
        from .lbest import lb_sim

        return lb_sim(_DATA[0], softmax_rows(L), rows, n_sim=QSIM_N)["quota"][0]
    return objective(y[rows], L[rows].argmax(1), w[rows])


def objective(y, pred, w) -> float:
    if OBJECTIVE == "lb":
        return _lb_composite(np.asarray(y), np.asarray(pred), np.asarray(w, dtype=float))
    return 0.5 * scores(y, pred)["composite"] + 0.5 * scores(y, pred, w)["composite"]


WLORA_OVERRIDE: float | None = None
EXCLUDE: str | None = None  # regex of run names to leave out (e.g. judge-based runs for a judge-free candidate)
OUT_SUFFIX = ""
FORCE_RULE: str | None = None  # e.g. "blend:x19_stack_lgbm_lora" to pin the combination rule


def load_runs(data: Data) -> dict:
    import re

    full5, fold0, test_only = {}, {}, {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("x8") or d.name == "submissions" or d.name.endswith("_blend"):
            continue  # derived combinations are excluded so no model is double counted
        if EXCLUDE and re.search(EXCLUDE, d.name):
            continue
        te = d / "test.npy"
        if not te.exists():
            continue
        t = np.load(te)
        if (d / "oof.npy").exists():
            o, m = np.load(d / "oof.npy"), np.load(d / "oof_mask.npy")
            if m.all():
                full5[d.name] = (o, t)
            elif (m == (data.folds == 0)).all():
                fold0[d.name] = (o, t)
        else:
            test_only[d.name] = t
    return {"full5": full5, "fold0": fold0, "test_only": test_only}


def caruana(logps: list[np.ndarray], y: np.ndarray, w: np.ndarray, rows: np.ndarray, n_iter: int = 25) -> np.ndarray:
    counts = np.zeros(len(logps))
    cur = None
    for it in range(n_iter):
        best = (-1.0, None)
        for j, lp in enumerate(logps):
            cand = lp if cur is None else (cur * it + lp) / (it + 1)
            s = score_logits(cand, y, w, rows)
            if s > best[0] + 1e-12:
                best = (s, j)
        j = best[1]
        cur = logps[j] if cur is None else (cur * it + logps[j]) / (it + 1)
        counts[j] += 1
    return counts / counts.sum()


def mix(logps: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(wt * lp for wt, lp in zip(weights, logps))


def tune_bias(L: np.ndarray, y: np.ndarray, w: np.ndarray, bound: float = 2.0, passes: int = 2) -> np.ndarray:
    if OBJECTIVE == "lbq":
        return np.zeros(N_CLASSES)  # class quotas set the per-class offsets themselves
    grid = np.round(np.arange(-bound, bound + 1e-9, 0.1), 2)
    b = np.zeros(N_CLASSES)
    base = objective(y, (L + b).argmax(1), w)
    for _ in range(passes):
        for c in range(N_CLASSES):
            bestg, bests = b[c], base
            for g in grid:
                bb = b.copy()
                bb[c] = g
                s = objective(y, (L + bb).argmax(1), w)
                if s > bests + 1e-9:
                    bests, bestg = s, g
            b[c], base = bestg, bests
    return b


def quota_bias(L: np.ndarray, target: np.ndarray, iters: int = 400, eta: float = 0.3) -> np.ndarray:
    b = np.zeros(N_CLASSES)
    big = -30.0
    b[target <= 0] = big
    for _ in range(iters):
        cnt = np.bincount((L + b).argmax(1), minlength=N_CLASSES)
        if np.abs(cnt - target).sum() <= 2:
            break
        upd = eta * np.log((target + 0.5) / (cnt + 0.5))
        upd[target <= 0] = 0.0
        b = b + upd
    return b


def main() -> None:
    data = Data()
    y, folds, groups = data.y, data.folds, data.groups
    w = test_shift_weights(data)
    _DATA[:] = [data]
    if OBJECTIVE in ("lb", "lbq"):
        from .lbest import fixed_rows, residual_counts

        fx = fixed_rows(data)
        _LB_CTX["res"] = residual_counts(data, fx)
        _LB_CTX["fixed_lab"] = np.array(list(fx.values()), dtype=int)
    runs = load_runs(data)
    names = list(runs["full5"])
    assert names, "no 5-fold runs found"
    logp_oof = [np.log(runs["full5"][n][0] + EPS) for n in names]
    logp_te = [np.log(runs["full5"][n][1] + EPS) for n in names]
    allrows = np.ones(data.n_train, bool)
    report: dict = {"singles": {}}
    for n, lp in zip(names, logp_oof):
        report["singles"][n] = {"oof": scores(y, lp.argmax(1))["composite"], "oof_tw": scores(y, lp.argmax(1), w)["composite"],
                                "objective": score_logits(lp, y, w, allrows)}
    best_single = max(names, key=lambda n: report["singles"][n]["objective"])

    # --- honest (nested) estimate of ensemble selection + bias tuning
    honest_pred_ens = np.zeros(data.n_train, int)
    honest_pred_bias = np.zeros(data.n_train, int)
    honest_pred_single = np.zeros(data.n_train, int)
    honest_L = np.zeros((data.n_train, N_CLASSES))
    honest_L_single = np.zeros((data.n_train, N_CLASSES))
    fold_L: dict[str, list] = {"caruana": [], "single": []}
    for k in range(5):
        tr = folds != k
        j = max(range(len(names)), key=lambda q: score_logits(logp_oof[q], y, w, tr))
        honest_pred_single[folds == k] = logp_oof[j][folds == k].argmax(1)
        honest_L_single[folds == k] = logp_oof[j][folds == k]
        wts = caruana(logp_oof, y, w, tr)
        Lk = mix(logp_oof, wts)
        fold_L["caruana"].append(Lk)
        fold_L["single"].append(logp_oof[j])
        honest_pred_ens[folds == k] = Lk[folds == k].argmax(1)
        bk = 0.5 * tune_bias(Lk[tr], y[tr], w[tr])
        honest_pred_bias[folds == k] = (Lk[folds == k] + bk).argmax(1)
        honest_L[folds == k] = Lk[folds == k] + bk
    rule, rule_a, rule_c = "caruana", None, None
    if OBJECTIVE == "lbq":
        from .lbest import lb_sim

        grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        judge = "x15_judge_lr" if "x15_judge_lr" in names else None
        for cname in [n for n in names if n.startswith(("x7_", "x7j_", "x17_", "x19_"))] if judge else []:
            A, B = logp_oof[names.index(cname)], logp_oof[names.index(judge)]
            fold_L[f"blend:{cname}"] = []
            for k in range(5):
                tr = folds != k
                a = max(grid, key=lambda a: score_logits((1 - a) * A + a * B, y, w, tr))
                fold_L[f"blend:{cname}"].append((1 - a) * A + a * B)
        # honest per-fold estimate: selection on the other folds, quota simulation on the held-out fold's rows only
        # (per-fold weights make logits incomparable across folds, so the folds are simulated separately)
        perfold = {}
        for key, Ls in fold_L.items():
            vals = [lb_sim(data, softmax_rows(Ls[k]), folds == k, n_sim=200)["quota"][0] for k in range(5)]
            perfold[key] = {"mean": float(np.mean(vals)), "folds": [round(v, 4) for v in vals]}
        report["perfold_quota"] = perfold
        rule = max(perfold, key=lambda k: perfold[k]["mean"])
        if FORCE_RULE:
            assert FORCE_RULE in perfold, f"{FORCE_RULE} not among {list(perfold)}"
            rule = FORCE_RULE
        print("  per-fold honest quota sim:", {k: round(v["mean"], 4) for k, v in perfold.items()}, "-> rule", rule, flush=True)
        if rule.startswith("blend:"):
            rule_c = rule.split(":", 1)[1]
            A, B = logp_oof[names.index(rule_c)], logp_oof[names.index(judge)]
            rule_a = max(grid, key=lambda a: score_logits((1 - a) * A + a * B, y, w, allrows))
    report["honest_best_single"] = scores(y, honest_pred_single)
    report["honest_best_single_tw"] = scores(y, honest_pred_single, w)
    report["honest_ensemble"] = scores(y, honest_pred_ens)
    report["honest_ensemble_bias"] = scores(y, honest_pred_bias)
    report["honest_ensemble_tw"] = scores(y, honest_pred_ens, w)
    report["honest_ensemble_bias_tw"] = scores(y, honest_pred_bias, w)

    # --- final ensemble on all OOF rows (falls back to the best single if the nested ensemble is worse)
    if OBJECTIVE == "lbq":  # honest nested logits under the quota simulation (more sims for the decision)
        from .lbest import lb_sim

        ens_obj = lb_sim(data, softmax_rows(honest_L), allrows, n_sim=200)["quota"][0]
        sgl_obj = lb_sim(data, softmax_rows(honest_L_single), allrows, n_sim=200)["quota"][0]
    else:
        ens_obj = objective(y, honest_pred_bias, w)  # same objective as the selection (oof-mean or simulated LB)
        sgl_obj = objective(y, honest_pred_single, w)
    report["s2_choice_objective"] = {"ensemble_bias": round(float(ens_obj), 4), "best_single": round(float(sgl_obj), 4),
                                     "objective": OBJECTIVE}
    use_single = sgl_obj > ens_obj
    if use_single:
        honest_L = honest_L_single  # keep the honest logits consistent with the submitted rule
    report["s2_base"] = "best_single" if use_single else "ensemble"
    wts = caruana(logp_oof, y, w, allrows)
    if OBJECTIVE == "lbq":
        use_single = rule == "single"
        if rule_c is not None:
            wts = np.array([(1 - rule_a) if n == rule_c else rule_a if n == "x15_judge_lr" else 0.0 for n in names])
        report["rule"] = {"name": rule, "a": rule_a}
        honest_L = np.zeros((data.n_train, N_CLASSES))
        for k in range(5):
            honest_L[folds == k] = fold_L[rule][k][folds == k]
    if use_single:
        wts = np.array([1.0 if n == best_single else 0.0 for n in names])
    L_oof = mix(logp_oof, wts)
    L_te = mix(logp_te, wts)
    bias = 0.5 * tune_bias(L_oof, y, w)
    report["ensemble_weights"] = {n: round(float(x), 3) for n, x in zip(names, wts) if x > 0}
    report["bias"] = dict(zip(LABELS, bias.round(2).tolist()))

    # --- optional LoRA mix (fold-0 predictions from x6 fold mode; test from x6 full if present)
    cands = [n for n in runs["fold0"] if n.startswith("x6_")]
    rank = lambda n: next((i for i, k in enumerate(("27b", "9b", "4b", "08b")) if f"_{k}_" in n or n.endswith(k)), 9)  # noqa: E731
    for n in cands:
        o, _ = runs["fold0"][n]
        print(f"  lora candidate {n}: fold0 comp {scores(y[folds == 0], o[folds == 0].argmax(1))['composite']:.4f}")
    if LORA_OVERRIDE:
        lora_keys = [k for k in LORA_OVERRIDE.split(",") if k in cands]
    else:
        lora_keys = sorted(cands, key=rank)
    lora_key = ",".join(lora_keys) if lora_keys else None
    w_lora = 0.0
    if lora_key:
        # LoRA component = mean log-prob over all selected fold-mode LoRA runs; test side prefers each run's _full twin
        lo_oof = np.exp(np.mean([np.log(runs["fold0"][k][0] + EPS) for k in lora_keys], axis=0))
        lo_te = np.exp(np.mean([np.log(runs["test_only"].get(k.replace("_fold", "_full"), runs["fold0"][k][1]) + EPS)
                                for k in lora_keys], axis=0))
        lo_te = lo_te / lo_te.sum(1, keepdims=True)
        f0 = folds == 0
        if OBJECTIVE == "lbq":
            E0 = fold_L[rule][0][f0]  # chosen rule, selected on folds 1-4 only
        else:
            wts_14 = caruana(logp_oof, y, w, folds != 0)  # ensemble weights that never saw fold 0
            E0 = mix(logp_oof, wts_14)[f0]
        Lo0 = np.log(lo_oof[f0] + EPS)
        y0, w0, g0 = y[f0], w[f0], groups[f0]
        grid = np.round(np.arange(0, 1.0001, 0.1), 2)
        honest = np.zeros(f0.sum(), int)
        honest_mix_L = np.zeros((f0.sum(), N_CLASSES))
        idx0 = np.flatnonzero(f0)

        def sub_score(Lsub: np.ndarray, part: np.ndarray) -> float:
            if OBJECTIVE != "lbq":
                return objective(y0[part], Lsub[part].argmax(1), w0[part])
            full = np.zeros((data.n_train, N_CLASSES))
            full[idx0] = Lsub
            m = np.zeros(data.n_train, bool)
            m[idx0[part]] = True
            return score_logits(full, y, w, m)

        if OBJECTIVE == "lbq":
            grid = np.round(np.arange(0, 0.61, 0.2), 2)
        for tri, tei in GroupKFold(n_splits=5).split(E0, y0, g0):
            bw = max(grid, key=lambda a: sub_score((1 - a) * E0 + a * Lo0, tri))
            honest[tei] = ((1 - bw) * E0[tei] + bw * Lo0[tei]).argmax(1)
            honest_mix_L[tei] = (1 - bw) * E0[tei] + bw * Lo0[tei]
        gscores = {float(a): sub_score((1 - a) * E0 + a * Lo0, np.ones(len(y0), bool)) for a in grid}
        w_lora = float(max(gscores, key=gscores.get))
        if OBJECTIVE == "lbq" and gscores[w_lora] < gscores[0.0] + 0.01:
            w_lora = 0.0  # fold-0-only evidence: require a clear gain (>= +0.01 quota composite) before mixing the LoRA in
        if WLORA_OVERRIDE is not None:
            w_lora = WLORA_OVERRIDE  # leaderboard-driven override (e.g. after an A/B submission)
        report["lora_grid_fold0"] = {str(a): round(v, 4) for a, v in gscores.items()}
        print("  lora grid (fold0):", report["lora_grid_fold0"], "-> w_lora", w_lora, flush=True)
        report["lora"] = {"run": lora_key, "test_from": {k: ("full" if k.replace("_fold", "_full") in runs["test_only"] else "fold")
                                                        for k in lora_keys},
                          "w_lora": w_lora, "fold0_honest_mix": scores(y0, honest),
                          "fold0_ensemble_only": scores(y0, E0.argmax(1)), "fold0_lora_only": scores(y0, Lo0.argmax(1))}
        if w_lora > 0:
            M0 = (1 - w_lora) * E0 + w_lora * Lo0
            nested_new = np.zeros(f0.sum(), int)
            for tri, tei in GroupKFold(n_splits=5).split(M0, y0, g0):
                bb = 0.5 * tune_bias(M0[tri], y0[tri], w0[tri], bound=1.0)
                nested_new[tei] = (M0[tei] + bb).argmax(1)
            keep_old = objective(y0, (M0 + bias).argmax(1), w0)
            use_new = objective(y0, nested_new, w0) > keep_old
            report["lora"]["fold0_nested_bias_new"] = scores(y0, nested_new)
            report["lora"]["fold0_bias_2400rows"] = scores(y0, (M0 + bias).argmax(1))
            if use_new:
                bias = 0.5 * tune_bias(M0, y0, w0, bound=1.0)
            L_te = (1 - w_lora) * L_te + w_lora * np.log(lo_te + EPS)
            report["bias_after_lora"] = {"used_fold0_bias": bool(use_new), **dict(zip(LABELS, bias.round(2).tolist()))}

    # --- submissions
    out = run_dir({"oof": "x8_blend", "lb": "x8_blend_lbobj", "lbq": "x8_blend_lbq"}[OBJECTIVE] + OUT_SUFFIX)
    hp = {"best_single": honest_pred_single, "ensemble": honest_pred_ens, "ensemble_bias": honest_pred_bias}
    hp["L_ensemble_bias"] = honest_L
    if OBJECTIVE == "lbq":
        for k in range(5):
            hp[f"L_fold{k}"] = fold_L[rule][k]
    if "lora" in report:
        hp["lora_mix_fold0"] = honest
        hp["L_lora_mix_fold0"] = honest_mix_L
    np.savez(out / "honest_preds.npz", **hp)
    single_te = runs["full5"][best_single][1]
    p1, log1 = apply_l0(data, single_te.argmax(1))
    sfx = {"oof": "", "lb": "_lbobj", "lbq": "_lbq"}[OBJECTIVE] + OUT_SUFFIX
    write_submission(data, p1, "S1_best_single_l0" + sfx, {"model": best_single, "l0": log1})
    p2_raw = (L_te + bias).argmax(1)
    p2, log2 = apply_l0(data, p2_raw)
    write_submission(data, p2, "S2_ensemble_bias_l0" + sfx, {"weights": report["ensemble_weights"], "w_lora": w_lora, "l0": log2})

    # S3: prior (quota) calibration on rows not fixed by L0, targets = derived test counts minus L0-fixed labels
    from .rules import l0_fixed

    fixed = np.zeros(len(data.test), bool)
    fixed_counts = np.zeros(N_CLASSES)
    for i, (lab_idx, _src) in l0_fixed(data).items():
        fixed[i] = True
        fixed_counts[lab_idx] += 1
    target = np.array([DERIVED_TEST_COUNTS[lab] for lab in LABELS]) - fixed_counts
    target = np.maximum(target, 0)
    scale = (~fixed).sum() / max(1, target.sum())
    target = np.round(target * scale)
    qb = quota_bias((L_te + bias)[~fixed], target)
    p3_raw = p2_raw.copy()
    p3_raw[~fixed] = ((L_te + bias)[~fixed] + qb).argmax(1)
    if os.environ.get("KOALTQ_HARNESS") == "1":
        # Structure-Signature Consistency Harness (koaltq2.harness): H3 link constraint, H4 structure twins, H5 agreement
        # veto, optional H6 signature vote; replaces the KOALTQ_SOLE_NOT_DECO / SIB_K2_OK / K2_VETO blocks below
        from .harness import Harness

        np.save(out / "harness_in_pred.npy", p3_raw)
        np.save(out / "harness_in_fixed.npy", fixed)
        np.save(out / "harness_in_logits.npy", L_te + bias)
        h = Harness(data, L_te + bias, s6=os.environ.get("KOALTQ_HARNESS_S6") == "1",
                    gate=float(os.environ.get("KOALTQ_HARNESS_GATE", "0.9")), s6_label=os.environ.get("KOALTQ_HARNESS_LABEL", "rejudge"))
        p3_raw = h.run(p3_raw, fixed)
        h.write(out)
        print("[x8] harness:", json.dumps(h.summary(), ensure_ascii=False), flush=True)
    if os.environ.get("KOALTQ_SOLE_NOT_DECO") == "1" and os.environ.get("KOALTQ_HARNESS") != "1":
        # guideline 3-3: an image that is the only content of a link is functional, never decoration (train: 0 of the 667 rows with this structure)
        from .rules import link_kind

        sole = np.array([bool(r.get("in_link")) and link_kind(r) in ("path", "url", "empty", "anchor", "js") for r in data.test])
        hit = (~fixed) & sole & (p3_raw == L2I["장식오용형"])
        p3_raw[hit] = L2I["적절"]
        print(f"[x8] sole-link rows moved 장식오용형 -> 적절: {int(hit.sum())}", flush=True)
    if os.environ.get("KOALTQ_SIB_K2_OK") == "1" and os.environ.get("KOALTQ_HARNESS") != "1":
        from .rules import k2_sibling_to_ok, k2_veto

        hit = k2_sibling_to_ok(data, p3_raw, fixed)
        if os.environ.get("KOALTQ_K2_VETO") == "1":  # 21번: residual rows are audit-agreed rows (rules.k2_veto)
            veto = k2_veto(data, hit, L_te + bias)
            hit &= ~veto
            print(f"[x8] K2 veto (train structure >= 95% 불충분형 / blend and judge both >= .95): {int(veto.sum())} rows stay 불충분형",
                  flush=True)
        p3_raw[hit] = L2I["적절"]
        print(f"[x8] K2 sibling rows moved 불충분형 -> 적절: {int(hit.sum())}", flush=True)
    p3, log3 = apply_l0(data, p3_raw)
    write_submission(data, p3, "S3_ensemble_quota_l0" + sfx, {"residual_target": dict(zip(LABELS, target.tolist())),
                                                        "quota_bias": qb.round(2).tolist(), "l0": log3})
    report["residual_target"] = dict(zip(LABELS, target.astype(int).tolist()))
    report["best_single"] = best_single
    np.save(out / "test_logits_final.npy", L_te + bias)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best_single", "ensemble_weights", "bias", "residual_target")}, ensure_ascii=False, indent=1))
    print("  S2/S3 base:", report["s2_base"])
    for key in ("honest_best_single", "honest_best_single_tw", "honest_ensemble", "honest_ensemble_bias",
                "honest_ensemble_tw", "honest_ensemble_bias_tw"):
        s = report[key]
        print(f"  {key:26s} comp {s['composite']:.4f} macro {s['macro_f1']:.4f} bin {s['binary_f1']:.4f}")
    if "lora" in report:
        print("  lora:", json.dumps({k: (round(v["composite"], 4) if isinstance(v, dict) and "composite" in v else v)
                                    for k, v in report["lora"].items()}, ensure_ascii=False, default=str))


def cli(argv: list[str]) -> None:
    global LORA_OVERRIDE, OBJECTIVE
    if "--lora" in argv:
        LORA_OVERRIDE = argv[argv.index("--lora") + 1]
    if "--objective" in argv:
        OBJECTIVE = argv[argv.index("--objective") + 1]
    global EXCLUDE, OUT_SUFFIX, FORCE_RULE
    if "--rule" in argv:
        FORCE_RULE = argv[argv.index("--rule") + 1]
    if "--exclude" in argv:
        EXCLUDE = argv[argv.index("--exclude") + 1]
    if "--suffix" in argv:
        OUT_SUFFIX = argv[argv.index("--suffix") + 1]
    global WLORA_OVERRIDE
    if "--wlora" in argv:
        WLORA_OVERRIDE = float(argv[argv.index("--wlora") + 1])
    main()
