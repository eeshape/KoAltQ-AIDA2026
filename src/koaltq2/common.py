"""Shared data access, official metrics, test-shift weights and the run registry."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

LABELS = ["적절", "무의미형", "파일명형", "중복형", "장식오용형", "불충분형", "무관형"]
L2I = {lab: i for i, lab in enumerate(LABELS)}
N_CLASSES = len(LABELS)
APPROPRIATE = 0

# Test-set label totals derived from the official CHANGELOG (2026-10-13 corrected overall totals)
# minus the 2,400 train counts in README_학습데이터.md. Used ONLY for prior-shift calibration (S3).
DERIVED_TEST_COUNTS = {"적절": 409, "무의미형": 5, "파일명형": 8, "중복형": 35, "장식오용형": 6, "불충분형": 133, "무관형": 4}

BUNDLE = Path(os.environ.get("KOALTQ_BUNDLE", "/content/koaltq_bundle"))
RUNS = Path(os.environ.get("KOALTQ_RUNS", "/content/runs"))
FOLDS_SHA = "4154b923c1a693caf95807590e6c8493abd37f14258c7d0fc836fee982d7e2fe"


# ----------------------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    """Split ONLY on '\n': the train JSONL contains literal U+2028 inside strings."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Data:
    """All records in a fixed order: train (2,400) followed by test (600)."""

    def __init__(self, bundle: Path = BUNDLE):
        self.bundle = Path(bundle)
        d = self.bundle / "data"
        self.train = read_jsonl(d / "train.jsonl")
        self.test = read_jsonl(d / "test.jsonl")
        folds = read_jsonl(d / "folds.jsonl")
        self.fold = {f["record_id"]: int(f["fold"]) for f in folds}
        self.group = {f["record_id"]: f["group"] for f in folds}
        self.boundary_test = {r["record_id"]: r["label"] for r in read_jsonl(d / "boundary_test_labels.jsonl")}
        self.boundary_train_ids = {r["record_id"] for r in read_jsonl(d / "boundary_train_ids.jsonl")}
        rj = d / "boundary_test_rejudge.jsonl"  # official 재판정_라벨 (independent audit view) of the 109 test rows
        self.boundary_rejudge = {r["record_id"]: r["rejudge_label"] for r in read_jsonl(rj)} if rj.exists() else {}
        self.image_meta = {m["record_id"]: m for m in read_jsonl(d / "image_meta.jsonl")}
        self.all = self.train + self.test
        self.ids = [r["record_id"] for r in self.all]
        self.idx = {rid: i for i, rid in enumerate(self.ids)}
        self.n_train = len(self.train)
        self.y = np.array([L2I[r["label"]] for r in self.train], dtype=np.int64)
        self.folds = np.array([self.fold[r["record_id"]] for r in self.train], dtype=np.int64)
        self.groups = np.array([self.group[r["record_id"]] for r in self.train])
        self._check()

    def _check(self) -> None:
        assert len(self.train) == 2400 and len(self.test) == 600
        assert len(set(self.ids)) == 3000
        assert sha256_file(self.bundle / "data" / "folds.jsonl") == FOLDS_SHA, "fold manifest changed"
        assert len(self.boundary_test) == 109 and set(self.boundary_test) <= {r["record_id"] for r in self.test}
        assert all(v in L2I for v in self.boundary_test.values())
        fold0 = int((self.folds == 0).sum())
        assert fold0 == 458, fold0

    def image_path(self, rid: str) -> Path:
        return self.bundle / "images" / self.image_meta[rid]["out_file"]

    def boundary_mask_test(self) -> np.ndarray:
        return np.array([r["record_id"] in self.boundary_test for r in self.test])

    def boundary_y_test(self) -> np.ndarray:
        """Labels for the 109 known test rows (in test order, -1 where unknown)."""
        return np.array([L2I[self.boundary_test[r["record_id"]]] if r["record_id"] in self.boundary_test else -1
                         for r in self.test], dtype=np.int64)


def host(record: dict) -> str:
    h = urlparse(record.get("page_url") or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


def test_shift_weights(data: Data, clip: float = 10.0) -> np.ndarray:
    """Importance weights p_test(host)/p_train(host) for train rows (mimics the test site mix)."""
    tr = Counter(host(r) for r in data.train)
    te = Counter(host(r) for r in data.test)
    w = np.array([(te.get(host(r), 0) / len(data.test)) / (tr[host(r)] / len(data.train)) for r in data.train])
    return np.clip(w, 0.0, clip)


# ----------------------------------------------------------------------------- metrics
def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    out = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        tp = w[(y_pred == c) & (y_true == c)].sum()
        fp = w[(y_pred == c) & (y_true != c)].sum()
        fn = w[(y_pred != c) & (y_true == c)].sum()
        den = 2 * tp + fp + fn
        out[c] = 0.0 if den == 0 else 2 * tp / den
    return out


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    t = y_true != APPROPRIATE
    p = y_pred != APPROPRIATE
    tp = w[t & p].sum()
    fp = w[~t & p].sum()
    fn = w[t & ~p].sum()
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else float(2 * tp / den)


def scores(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> dict:
    """Official composite = 0.5 * binary F1 (부적절 positive) + 0.5 * 7-class macro-F1 (sklearn 'macro')."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pc = f1_per_class(y_true, y_pred, sample_weight)
    macro = float(pc.mean())
    binf = binary_f1(y_true, y_pred, sample_weight)
    return {"macro_f1": macro, "binary_f1": binf, "composite": 0.5 * macro + 0.5 * binf,
            "per_class_f1": {LABELS[c]: float(pc[c]) for c in range(N_CLASSES)},
            "n": int(len(y_true))}


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    m = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m.tolist()


def group_bootstrap_ci(y_true, y_pred, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Group-cluster bootstrap 95% CI for composite / macro / binary."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    vals = {"composite": [], "macro_f1": [], "binary_f1": []}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([pos[g] for g in pick])
        s = scores(y_true[ix], y_pred[ix])
        for k in vals:
            vals[k].append(s[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


# ----------------------------------------------------------------------------- registry
def run_dir(name: str) -> Path:
    p = RUNS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def evaluate_and_save(name: str, data: Data, *, oof: np.ndarray | None, test: np.ndarray | None,
                      oof_mask: np.ndarray | None = None, extra: dict | None = None, note: str = "") -> dict:
    """Save probabilities + metrics for one experiment and append to the local leaderboard.

    oof:  (2400, 7) probabilities; rows outside oof_mask are ignored (e.g. fold-0-only runs).
    test: (600, 7) probabilities from the model trained on all available labels.
    """
    out = run_dir(name)
    res: dict = {"name": name, "note": note, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if oof is not None:
        oof = np.asarray(oof, dtype=np.float64)
        mask = np.ones(data.n_train, bool) if oof_mask is None else np.asarray(oof_mask, bool)
        np.save(out / "oof.npy", oof)
        np.save(out / "oof_mask.npy", mask)
        yp = oof.argmax(1)
        y = data.y
        res["oof"] = scores(y[mask], yp[mask])
        res["oof_test_weighted"] = scores(y[mask], yp[mask], test_shift_weights(data)[mask])
        res["oof_fold0"] = scores(y[data.folds == 0], yp[data.folds == 0]) if mask[data.folds == 0].all() else None
        res["confusion_oof"] = confusion(y[mask], yp[mask])
        res["oof_ci"] = group_bootstrap_ci(y[mask], yp[mask], data.groups[mask], n_boot=300)
    if test is not None:
        test = np.asarray(test, dtype=np.float64)
        np.save(out / "test.npy", test)
        bm = data.boundary_mask_test()
        by = data.boundary_y_test()
        res["boundary109_diag"] = scores(by[bm], test.argmax(1)[bm])
        res["test_pred_counts"] = {LABELS[c]: int((test.argmax(1) == c).sum()) for c in range(N_CLASSES)}
    if extra:
        res.update(extra)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    _append_leaderboard(res)
    print_summary(res)
    return res


def _append_leaderboard(res: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    lb = RUNS / "local_leaderboard.csv"
    row = {"name": res["name"], "time": res["time"]}
    for key in ("oof", "oof_test_weighted", "oof_fold0", "boundary109_diag"):
        s = res.get(key)
        row[f"{key}_composite"] = None if not s else round(s["composite"], 4)
        row[f"{key}_macro"] = None if not s else round(s["macro_f1"], 4)
        row[f"{key}_binary"] = None if not s else round(s["binary_f1"], 4)
    new = not lb.exists()
    with open(lb, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def print_summary(res: dict) -> None:
    def fmt(s):
        return "-" if not s else f"comp {s['composite']:.4f} | macro {s['macro_f1']:.4f} | bin {s['binary_f1']:.4f} (n={s['n']})"

    print(f"=== {res['name']} ===")
    for key, title in (("oof", "OOF(pooled)"), ("oof_test_weighted", "OOF(test-site-weighted)"),
                       ("oof_fold0", "fold0(458)"), ("boundary109_diag", "test boundary-109 diag")):
        if key in res:
            print(f"  {title:26s} {fmt(res[key])}")
    if res.get("oof"):
        pc = res["oof"]["per_class_f1"]
        print("  per-class F1 (OOF): " + ", ".join(f"{k} {v:.2f}" for k, v in pc.items()))
    if res.get("test_pred_counts"):
        print("  test pred counts: " + ", ".join(f"{k} {v}" for k, v in res["test_pred_counts"].items()))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
