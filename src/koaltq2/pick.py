"""Rank the submission CSVs by their leaderboard-style estimate and write FINAL_CANDIDATES.md.

Mapping from submission file to the estimate that describes its decision rule:
  S0_x1_l0                       -> x1 best model argmax (lb_est of that run)
  S1_best_single_l0[_lbobj]      -> lb_est of the chosen single model
  S2_ensemble_bias_l0[_lbobj]    -> honest nested ensemble+bias (x8 / x8lb : L_ensemble_bias, analytic lb_est)
  S3_ensemble_quota_l0[_lbobj]   -> honest nested ensemble logits under class quotas (Monte-Carlo sim_quota)
"""

from __future__ import annotations

import json

from .common import RUNS
from .lbest import LEADERBOARD


def main() -> None:
    rows = {r["run"]: r for r in json.loads((RUNS / "lb_estimates.json").read_text(encoding="utf-8"))}
    subs = sorted((RUNS / "submissions").glob("*.csv"))
    out = []
    for p in subs:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) if p.with_suffix(".json").exists() else {}
        name = p.stem
        tag = ("x8lbqnj" if name.endswith("_nojudge") else "x8lbq" if name.endswith("_lbq") else
               "x8lb" if name.endswith("_lbobj") else "x8")
        est, basis = None, ""
        if name.startswith("S0"):
            r = rows.get(meta.get("base_model", "x1_hgb"))
            est, basis = (r or {}).get("lb_est"), f"{meta.get('base_model')} argmax"
        elif name.startswith("S1"):
            r = rows.get(meta.get("model", ""))
            est, basis = (r or {}).get("lb_est"), f"{meta.get('model')} argmax"
        elif name.startswith("S2"):
            r = rows.get(f"{tag}:L_ensemble_bias")
            est, basis = (r or {}).get("lb_est"), f"{tag} nested ensemble+bias"
        elif name.startswith("S4"):
            src = {"x8lbq": "x8_blend_lbq", "x8lbqnj": "x8_blend_lbq_nojudge", "x8lb": "x8_blend_lbobj", "x8": "x8_blend"}[tag]
            rep = RUNS / "x13_spec" / f"report_{src}.json"
            if not rep.exists():
                rep = RUNS / "x13_spec" / "report.json"
            sim = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else {}
            est = round(sim["quota_spec"], 4) if "quota_spec" in sim else None
            basis = f"{tag} + quota + rare-class VLM specialists ({sim.get('estimator', 'pooled')} MC sim)"
        elif name.startswith("S3") and tag in ("x8lbq", "x8lbqnj"):
            rdir = RUNS / ("x8_blend_lbq" if tag == "x8lbq" else "x8_blend_lbq_nojudge")
            rep = json.loads((rdir / "report.json").read_text(encoding="utf-8")) if (rdir / "report.json").exists() else {}
            rule = (rep.get("rule") or {}).get("name", "caruana")
            pf = (rep.get("perfold_quota") or {}).get(rule, {})
            est = round(pf["mean"], 4) if "mean" in pf else None
            basis = f"{tag} rule={rule} a={(rep.get('rule') or {}).get('a')} + quota (per-fold honest MC sim)"
        elif name.startswith("S3"):
            r = rows.get(f"{tag}:L_ensemble_bias")
            est, basis = (r or {}).get("sim_quota"), f"{tag} nested ensemble + quota (MC sim)"
        out.append({"file": p.name, "lb_est": est, "basis": basis, "counts": meta.get("counts")})
    out.sort(key=lambda r: -(r["lb_est"] or 0))
    lines = ["# 제출 후보 (예상 리더보드 점수 순)", "",
             "예상 점수 = 리더보드 공식(0.5×이진F1 + 0.5×macroF1)을 test 600건 시뮬레이션에 적용한 추정치. **실제 점수는 제출해야 확인됨.**",
             "참고 실제 리더보드: " + ", ".join(f"{k} {v}" for k, v in LEADERBOARD.items()), "",
             "| 순위 | 파일 | 예상 점수 | 근거 | 예측 분포 |", "|---:|---|---:|---|---|"]
    for i, r in enumerate(out, 1):
        lines.append(f"| {i} | `{r['file']}` | {r['lb_est'] if r['lb_est'] is not None else '-'} | {r['basis']} | {r['counts']} |")
    (RUNS / "FINAL_CANDIDATES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
