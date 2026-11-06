"""X2: Korean OCR evidence (PaddleOCR PP-OCRv5 korean in an isolated venv; EasyOCR fallback) -> OCR features -> HGB.

The canonical OCR file RUNS/x2_ocr/ocr.jsonl is written ONLY after an engine produced 3,000 healthy rows.
It is injected into the evidence card used by X3-X6 (prompting.require_ocr / ocr_lookup).
Feature idea (guideline 3-3): information that exists ONLY inside the image (dates/periods, place, target,
amount, how to apply) and is absent from alt + link text + context => '불충분형' signal.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import BUNDLE, RUNS, Data, evaluate_and_save, run_dir
from .rules import char_ngrams, feature_matrix

_DATE = re.compile(r"\d{4}\s*[.\-/년]\s*\d{1,2}|\d{1,2}\s*[./]\s*\d{1,2}\s*[(~∼\-]|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}:\d{2}")
_KW = ["기간", "일시", "장소", "대상", "신청", "접수", "문의", "모집", "참가", "시간", "위치", "주소", "요금", "무료", "원",
       "마감", "발표", "자격", "방법", "운영", "개최", "전시", "행사", "교육", "공모"]


def ensure_ocr_venv(engine: str) -> Path:
    """Per-engine venv created WITHOUT ensurepip (Colab's python3.13 lacks it); packages via `pip --python`."""
    venv = Path(os.environ.get("KOALTQ_OCR_VENV_ROOT", "/content")) / f"venv_ocr_{engine}"
    py = venv / "bin" / "python"
    probe = "import paddleocr" if engine == "paddle" else "import easyocr"
    if py.exists() and subprocess.run([str(py), "-c", probe], capture_output=True).returncode == 0:
        return py
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    pip = [sys.executable, "-m", "pip", "--python", str(py), "install", "-q"]
    if engine == "paddle":
        subprocess.run(pip + ["paddlepaddle-gpu==3.4.0", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu130/",
                              "--extra-index-url", "https://pypi.org/simple"], check=True)
        subprocess.run(pip + ["paddleocr==3.7.0"], check=True)
    else:
        subprocess.run(pip + ["easyocr"], check=True)
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, f"{engine} import failed in venv: {r.stderr[-500:]}"
    return py


def run_ocr(engine: str) -> Path:
    out_dir = run_dir("x2_ocr")
    eng_out = out_dir / f"ocr_{engine}.jsonl"
    py = ensure_ocr_venv(engine)
    worker = Path(__file__).with_name("ocr_worker.py")
    t0 = time.time()
    p = subprocess.run([str(py), str(worker), str(BUNDLE), str(eng_out), engine], text=True)
    print(f"[x2] {engine} worker exit {p.returncode} in {time.time() - t0:.1f}s", flush=True)
    rows = {}
    for x in eng_out.read_text(encoding="utf-8").split("\n") if eng_out.exists() else []:
        try:
            d = json.loads(x)
            rows[d["record_id"]] = d
        except Exception:  # noqa: BLE001
            pass
    with_text = sum(1 for d in rows.values() if d["texts"])
    print(f"[x2] {engine}: {len(rows)}/3000 rows, with text {with_text}", flush=True)
    assert p.returncode == 0, f"{engine} worker failed (exit {p.returncode})"
    assert len(rows) == 3000, f"{engine} OCR incomplete: {len(rows)}/3000"
    assert with_text / 3000 > 0.2, f"{engine} OCR implausible: only {with_text} images with text"
    shutil.copyfile(eng_out, out_dir / "ocr.jsonl")
    (out_dir / "ocr_engine.txt").write_text(engine, encoding="utf-8")
    return out_dir / "ocr.jsonl"


def ocr_features(data: Data, ocr_path: Path) -> tuple[np.ndarray, list[str]]:
    from .prompting import read_ocr_rows

    ocr = read_ocr_rows(ocr_path)
    rows = []
    for r in data.all:
        d = ocr.get(r["record_id"], {"texts": [], "scores": []})
        text = " ".join(d["texts"])
        ctx = " ".join([r.get("alt_text") or "", r.get("link_dest") or "", r.get("context_text") or "",
                        (r.get("page_title") or "")[:300], r.get("heading_path") or ""])
        og, cg, ag = char_ngrams(text), char_ngrams(ctx), char_ngrams(r.get("alt_text") or "")
        dates_o = set(m.group(0).replace(" ", "") for m in _DATE.finditer(text))
        dates_c = set(m.group(0).replace(" ", "") for m in _DATE.finditer(ctx))
        kw_o = {k for k in _KW if k in text}
        kw_c = {k for k in _KW if k in ctx}
        nums_o = set(re.findall(r"\d{2,}", text))
        nums_c = set(re.findall(r"\d{2,}", ctx))
        rows.append({
            "ocr_lines": float(len(d["texts"])),
            "ocr_chars": float(len(text)),
            "ocr_has_text": float(len(text) >= 4),
            "ocr_score": float(np.mean(d["scores"])) if d["scores"] else 0.0,
            "ocr_novel_ratio": (len(og - cg) / len(og)) if og else 0.0,
            "ocr_alt_cover": (len(og & ag) / len(og)) if og else 0.0,
            "alt_in_ocr": (len(ag & og) / len(ag)) if ag else 0.0,
            "ocr_dates": float(len(dates_o)),
            "ocr_dates_novel": float(len(dates_o - dates_c)),
            "ocr_kw": float(len(kw_o)),
            "ocr_kw_novel": float(len(kw_o - kw_c)),
            "ocr_nums_novel": float(len(nums_o - nums_c)),
        })
    names = list(rows[0])
    return np.array([[row[k] for k in names] for row in rows], dtype=np.float32), names


def _norm_dates(s: str) -> set[str]:
    """Canonical date/time tokens: '2026.7', '2026-07', '2026년 7월' -> '2026-7'; '7.13(', '7월 13일' -> '7-13'; '14:00'."""
    out = set()
    for m in re.finditer(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})(?:\s*[.\-/월]\s*(\d{1,2}))?", s):
        out.add(f"{m.group(1)}-{int(m.group(2))}")
        if m.group(3):
            out.add(f"{int(m.group(2))}-{int(m.group(3))}")
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*[./월]\s*(\d{1,2})(?!\d)", s):
        mo, dd = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= dd <= 31:
            out.add(f"{mo}-{dd}")
    for m in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2})", s):
        out.add(f"{int(m.group(1))}:{m.group(2)}")
    return out


def _line_cover(lines: list[str], ref: str, thr: float = 0.6) -> tuple[float, float]:
    """Share of OCR lines whose character bigrams are mostly (>= thr) present in ref, and the chars left uncovered."""
    refg = char_ngrams(ref)
    good = [ln for ln in lines if len(ln.strip()) >= 2]
    if not good:
        return 0.0, 0.0
    cov = []
    for ln in good:
        g = char_ngrams(ln)
        cov.append(len(g & refg) / max(1, len(g)) >= thr)
    return float(np.mean(cov)), float(sum(len(ln) for ln, c in zip(good, cov) if not c))


def ocr_features_v2(data: Data, ocr_path: Path) -> tuple[np.ndarray, list[str]]:
    """v1 features with ocr_dates_novel computed on normalized dates, plus line-level coverage of the OCR text by
    alt / alt+link+context (guideline 3-3: 불충분형 only when image-only information is missing from all of them)."""
    from .prompting import read_ocr_rows

    X, names = ocr_features(data, ocr_path)
    ocr = read_ocr_rows(ocr_path)
    k = names.index("ocr_dates_novel")
    extra = []
    for i, r in enumerate(data.all):
        d = ocr.get(r["record_id"], {"texts": [], "scores": []})
        lines = list(d["texts"])
        text = " ".join(lines)
        alt = r.get("alt_text") or ""
        ctx = " ".join([alt, r.get("link_dest") or "", r.get("context_text") or "", (r.get("page_title") or "")[:300],
                        r.get("heading_path") or ""])
        X[i, k] = float(len(_norm_dates(text) - _norm_dates(ctx)))
        cov_all, unc_all = _line_cover(lines, ctx)
        cov_alt, _ = _line_cover(lines, alt)
        extra.append([cov_all, cov_alt, unc_all])
    names2 = names + ["ocr_line_cover_all", "ocr_line_cover_alt", "ocr_uncovered_chars"]
    return np.hstack([X, np.array(extra, dtype=np.float32)]), names2


def main(engine: str = "paddle") -> None:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .prompting import read_ocr_rows
    from .x1_gbm import SEED, cv_predict, tfidf_matrix

    data = Data()
    path = RUNS / "x2_ocr" / "ocr.jsonl"
    if len(read_ocr_rows(path)) < 3000:
        try:
            path = run_ocr(engine)
        except Exception as e:  # noqa: BLE001
            other = "easyocr" if engine == "paddle" else "paddle"
            print(f"[x2] {engine} OCR failed ({e!r}); falling back to {other}", flush=True)
            path = run_ocr(other)
    Xo, onames = ocr_features(data, path)
    np.save(run_dir("x2_ocr") / "ocr_features.npy", Xo)
    (run_dir("x2_ocr") / "ocr_feature_names.json").write_text(json.dumps(onames), encoding="utf-8")
    has = Xo[:, onames.index("ocr_has_text")]
    print(f"[x2] images with OCR text: train {has[:data.n_train].mean():.2%} test {has[data.n_train:].mean():.2%}", flush=True)
    Xf, _ = feature_matrix(data)
    Xs = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(tfidf_matrix(data)).astype(np.float32)
    X = np.hstack([Xf, Xs, Xo])
    oof, te = cv_predict(
        lambda: HistGradientBoostingClassifier(learning_rate=0.06, max_iter=350, max_leaf_nodes=31, min_samples_leaf=8,
                                               l2_regularization=1.0, class_weight="balanced", random_state=SEED),
        X, data)
    evaluate_and_save("x2_hgb_ocr", data, oof=oof, test=te, note="X1 HGB features + OCR evidence features",
                      extra={"ocr_engine": (RUNS / "x2_ocr" / "ocr_engine.txt").read_text().strip()})


def cli(argv: list[str]) -> None:
    if argv and argv[0] == "v2":  # features only, from the existing OCR output (no OCR re-run)
        data = Data()
        Xo, onames = ocr_features_v2(data, RUNS / "x2_ocr" / "ocr.jsonl")
        np.save(run_dir("x2_ocr") / "ocr_features_v2.npy", Xo)
        (run_dir("x2_ocr") / "ocr_feature_names_v2.json").write_text(json.dumps(onames), encoding="utf-8")
        print(f"[x2] v2 OCR features {Xo.shape} saved", flush=True)
        return
    main(argv[0] if argv else "paddle")
