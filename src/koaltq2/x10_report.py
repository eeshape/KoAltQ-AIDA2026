"""X10: report artifacts (summary table, figures, error examples) + one ZIP of /content/runs for download."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .common import LABELS, RUNS, Data, confusion, run_dir


def summary_rows() -> list[dict]:
    rows = []
    for d in sorted(RUNS.iterdir()):
        m = d / "metrics.json"
        if not m.exists():
            continue
        r = json.loads(m.read_text(encoding="utf-8"))
        row = {"run": r["name"], "note": r.get("note", "")}
        for key in ("oof", "oof_test_weighted", "oof_fold0", "boundary109_diag"):
            s = r.get(key)
            if s:
                row[f"{key}.composite"] = round(s["composite"], 4)
                row[f"{key}.macro"] = round(s["macro_f1"], 4)
                row[f"{key}.binary"] = round(s["binary_f1"], 4)
        if r.get("oof_ci"):
            row["oof.composite_ci95"] = [round(x, 3) for x in r["oof_ci"]["composite"]]
        rows.append(row)
    return rows


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = Data()
    out = run_dir("x10_report")
    rows = summary_rows()
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = ["run", "oof.composite", "oof.macro", "oof.binary", "oof_test_weighted.composite", "oof_fold0.composite",
            "boundary109_diag.composite"]
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
    full = [r for r in rows if "oof.composite" in r]
    if full:
        full.sort(key=lambda r: r["oof.composite"])
        fig, ax = plt.subplots(figsize=(9, 0.4 * len(full) + 1.5))
        ax.barh([r["run"] for r in full], [r["oof.composite"] for r in full], color="#0B7A85")
        ax.set_xlabel("pooled 5-fold OOF composite (0.5 macro + 0.5 binary)")
        fig.tight_layout()
        fig.savefig(out / "oof_composite.png", dpi=150)
        plt.close(fig)
    for d in sorted(RUNS.iterdir()):
        if (d / "oof.npy").exists() and (d / "oof_mask.npy").exists():
            oof, mask = np.load(d / "oof.npy"), np.load(d / "oof_mask.npy")
            cm = np.array(confusion(data.y[mask], oof.argmax(1)[mask]))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(LABELS)), [str(i) for i in range(len(LABELS))])
            ax.set_yticks(range(len(LABELS)), [f"{i} {lab}" for i, lab in enumerate(LABELS)])
            for i in range(len(LABELS)):
                for j in range(len(LABELS)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            ax.set_xlabel("predicted")
            ax.set_title(d.name)
            fig.tight_layout()
            fig.savefig(out / f"cm_{d.name}.png", dpi=130)
            plt.close(fig)
    zpath = Path(shutil.make_archive(str(RUNS.parent / "koaltq_runs"), "zip", root_dir=str(RUNS.parent), base_dir=RUNS.name))
    print(f"[x10] archive {zpath} {zpath.stat().st_size} bytes")
