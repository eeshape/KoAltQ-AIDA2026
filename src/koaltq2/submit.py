"""Submission CSV writing + validation (record_id,label; 600 rows; 7 labels; test order)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .common import LABELS, RUNS, Data


def write_submission(data: Data, pred_idx: np.ndarray, name: str, info: dict | None = None) -> Path:
    pred_idx = np.asarray(pred_idx)
    assert pred_idx.shape == (len(data.test),)
    out_dir = RUNS / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "label"])
        for r, p in zip(data.test, pred_idx):
            w.writerow([r["record_id"], LABELS[int(p)]])
    validate_submission(data, path)
    counts = {lab: int((pred_idx == i).sum()) for i, lab in enumerate(LABELS)}
    meta = {"name": name, "counts": counts, **(info or {})}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[submission] {path} counts={counts}")
    return path


def validate_submission(data: Data, path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["record_id", "label"], rows[0]
    body = rows[1:]
    assert len(body) == 600, len(body)
    ids = [r[0] for r in body]
    assert ids == [r["record_id"] for r in data.test], "IDs must match test order exactly"
    assert len(set(ids)) == 600
    assert all(r[1] in LABELS for r in body), "unknown label"
    # Boundary labels must be present verbatim (we always apply them in L0 when requested).
