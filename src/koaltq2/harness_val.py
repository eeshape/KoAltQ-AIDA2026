"""Evaluation harness for the twin rules of koaltq2.harness, run on TRAIN rows (labels known, nothing from test).

What it measures (and what not): in every page ~18% of the rows (109/600 on test) are drawn at RANDOM as 'public' rows
with their ORIGINAL train labels; the other rows get the honest out-of-fold prediction of the stacker blend
(honest_preds.npz L_ensemble_bias = per-fold x17 + judge-LR blend, i.e. before the LoRA mix and before the class quota).
Each twin rule then corrects the non-public rows from their public twins; per rule: rows changed, precision of the
changes, net corrected rows (fixed - broken), averaged over R draws. Rules are scored standalone and in harness order
(H2 first; H4 only on rows H2 did not decide; H6 only on rows neither touched).
This is NOT a validation of the deployed transfer: on test the public rows are exactly the 109 rows where the original
label and the audit DISAGREED (0/109 have 데이터셋_라벨 == 재판정_라벨) and H2/H4/H6 transfer the audit label. Train has no
audit labels, so this only measures how often random page twins share original labels and what a confidence gate does.
Also reports, for the test rows H4 acted on, what a 0.9 confidence gate or a 90% conformal set would have allowed.
usage: python -m koaltq2.run hval <x8 output dir> [--reps 30] [--public 0.18]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .common import L2I, LABELS, Data
from .harness import INSTANCE_LABELS, _softmax, file_pattern, page_of
from .rules import _struct_helpers, alt_template, link_kind


def cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("x8_dir")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--public", type=float, default=0.18)
    a = ap.parse_args(argv)
    d = Data()
    n = d.n_train
    x8 = Path(a.x8_dir)
    Loof = np.load(x8 / "honest_preds.npz")["L_ensemble_bias"]
    P = _softmax(Loof)
    pred0 = P.argmax(1)
    y = d.y
    rel, img = _struct_helpers(d)
    keys = {
        "H2_template_twins": [(page_of(r), bool(r.get("in_link")), link_kind(r), alt_template(r.get("alt_text"))) for r in d.train],
        "H4_structure_twins": [(page_of(r), bool(r.get("in_link")), link_kind(r), rel(r), img(g, r)) for g, r in enumerate(d.train)],
        "H6_signature": [(page_of(r), bool(r.get("in_link")), link_kind(r), rel(r), file_pattern(r)) for r in d.train],
    }
    rules = {  # name: (key, min neighbours, agreement, gate, direction)
        "H2_template_twins(>=1 nb, 80%)": ("H2_template_twins", 1, 0.8, None, None),
        "H4_structure_twins(불충분->적절, >=1 nb, 80%)": ("H4_structure_twins", 1, 0.8, None, ("불충분형", "적절")),
        "H6_signature(>=2 nb, 100%, no gate)": ("H6_signature", 2, 1.0, None, None),
        "H6_signature(>=2 nb, 100%, gate .9)": ("H6_signature", 2, 1.0, 0.9, None),
    }
    pages = collections.defaultdict(list)
    for g, r in enumerate(d.train):
        pages[page_of(r)].append(g)
    rng = np.random.default_rng(20261104)
    res = {k: collections.defaultdict(list) for k in rules}
    def apply_rule(name: str, public: np.ndarray, skip: np.ndarray) -> tuple[int, int, int, np.ndarray]:
        """-> (changed, fixed, broke, decided): decided = rows the rule claims (twin group found and the rule's own
        conditions met), whether or not the label changes; skip = rows an earlier module already decided."""
        kname, min_nb, agree, gate, direction = rules[name]
        K = keys[kname]
        groups = collections.defaultdict(list)
        for g in np.flatnonzero(public):
            if LABELS[y[g]] not in INSTANCE_LABELS:
                groups[K[g]].append(int(y[g]))
        decided = np.zeros(n, bool)
        fixed_n = broke = changed = 0
        for g in np.flatnonzero(~public & ~skip):
            labs = groups.get(K[g])
            if not labs or len(labs) < min_nb:
                continue
            vals, cnts = np.unique(labs, return_counts=True)
            if cnts.max() / len(labs) < agree:
                continue
            new, cur = int(vals[cnts.argmax()]), int(pred0[g])
            if direction and (LABELS[cur], LABELS[new]) != direction:
                continue
            if kname != "H6_signature":
                decided[g] = True  # H2 fixes every row with a twin group; H4 claims every K2 hit (veto or not)
            if new == cur or (gate is not None and P[g, cur] > gate):
                continue
            decided[g] = True
            changed += 1
            fixed_n += int(new == y[g])
            broke += int(cur == y[g])
        return changed, fixed_n, broke, decided

    order = ["H2_template_twins(>=1 nb, 80%)", "H4_structure_twins(불충분->적절, >=1 nb, 80%)", "H6_signature(>=2 nb, 100%, gate .9)"]
    for name in order:
        res["ordered: " + name] = collections.defaultdict(list)
    for _ in range(a.reps):
        public = np.zeros(n, bool)
        for gs in pages.values():
            k = int(round(a.public * len(gs)))
            if k:
                public[rng.choice(gs, size=k, replace=False)] = True
        for name in rules:  # standalone: every non-public row
            changed, fixed_n, broke, _ = apply_rule(name, public, np.zeros(n, bool))
            res[name]["changed"].append(changed)
            res[name]["precision"].append(fixed_n / changed if changed else np.nan)
            res[name]["net"].append(fixed_n - broke)
        taken = np.zeros(n, bool)  # harness order: H2, then H4 on rows H2 left, then H6 on rows neither touched
        for name in order:
            changed, fixed_n, broke, decided = apply_rule(name, public, taken)
            taken |= decided
            res["ordered: " + name]["changed"].append(changed)
            res["ordered: " + name]["precision"].append(fixed_n / changed if changed else np.nan)
            res["ordered: " + name]["net"].append(fixed_n - broke)
    out = {}
    for name, v in res.items():
        out[name] = {k: (round(float(np.nanmean(x)), 3), round(float(np.nanstd(x)), 3)) for k, x in v.items()}
        print(f"[hval] {name:45s} changed {out[name]['changed'][0]:6.1f}  precision {out[name]['precision'][0]:.3f}  "
              f"net {out[name]['net'][0]:+.1f}", flush=True)
    # conformal prediction sets (LAC score 1 - p_true): cross-conformal coverage on train OOF, then sets for test rows
    alpha = 0.1
    s = 1.0 - P[np.arange(n), y]
    cov, size = [], []
    for k in range(5):
        cal, ev = d.folds != k, d.folds == k
        q = np.quantile(s[cal], min(1.0, np.ceil((cal.sum() + 1) * (1 - alpha)) / cal.sum()), method="higher")
        sets = (1.0 - P[ev]) <= q
        cov.append(float(sets[np.arange(ev.sum()), y[ev]].mean()))
        size.append(float(sets.sum(1).mean()))
    q_all = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    Pt = _softmax(np.load(x8 / "test_logits_final.npy"))
    tsets = (1.0 - Pt) <= q_all
    fixed = np.load(x8 / "harness_in_fixed.npy")
    tsize = tsets.sum(1)
    out["conformal"] = {"alpha": alpha, "oof_coverage_per_fold": [round(c, 3) for c in cov], "oof_mean_set_size": round(float(np.mean(size)), 3),
                        "q_hat": round(float(q_all), 4), "test_residual_rows": int((~fixed).sum()),
                        "test_residual_singletons": int(((tsize == 1) & ~fixed).sum()),
                        "test_residual_set_size_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(tsize[~fixed], return_counts=True))}}
    print(f"[hval] conformal (alpha {alpha}): OOF coverage {np.mean(cov):.3f} (per fold {[round(c, 3) for c in cov]}), mean set size {np.mean(size):.2f}; "
          f"test residual rows {(~fixed).sum()}: singletons {((tsize == 1) & ~fixed).sum()}, set sizes {out['conformal']['test_residual_set_size_hist']}", flush=True)
    # what a confidence gate would have done to the rows H4 changed on test
    logp = x8 / "harness_log.jsonl"
    if logp.exists():
        pos = {r["record_id"]: j for j, r in enumerate(d.test)}
        h4 = [e for e in map(json.loads, open(logp, encoding="utf-8")) if e["module"].startswith("H4")]
        conf = [e["blend_p_current"] for e in h4]
        out["test_H4_rows_blend_confidence"] = sorted(conf)
        out["test_H4_rows_blocked_by_gate_0.9"] = int(sum(c > 0.9 for c in conf))
        out["test_H4_applied_rows_blocked_by_gate_0.9"] = int(sum(e["blend_p_current"] > 0.9 for e in h4 if e["applied"]))
        ok_in_set = [bool(tsets[pos[e["record_id"]], L2I["적절"]]) for e in h4]
        ok_in_set_applied = [x for x, e in zip(ok_in_set, h4) if e["applied"]]
        out["test_H4_rows_conformal_set_contains_적절"] = int(sum(ok_in_set))
        out["test_H4_applied_rows_conformal_set_contains_적절"] = int(sum(ok_in_set_applied))
        print(f"[hval] test H4 rows: blend confidence {sorted(conf)} -> a 0.9 gate would block {out['test_H4_rows_blocked_by_gate_0.9']} of {len(h4)} "
              f"(of the {sum(e['applied'] for e in h4)} applied: {out['test_H4_applied_rows_blocked_by_gate_0.9']}); "
              f"conformal 90% set contains 적절 for {sum(ok_in_set)}/{len(h4)} (applied: {sum(ok_in_set_applied)}/{len(ok_in_set_applied)})", flush=True)
    (x8 / "harness_val.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
