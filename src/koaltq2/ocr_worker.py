"""Standalone OCR worker (runs inside an isolated venv so paddle's CUDA libs never touch the torch process).

usage: python ocr_worker.py <bundle_dir> <out_jsonl> [paddle|easyocr]
* one JSON line per successfully processed record: {record_id, texts[], scores[], n_raw}; flushed per row
* failed images are NOT written (a resume retries them); a summary JSON is written next to the output
* exits 3 if the first 20 images all fail or more than 1% of images fail (so the caller can fall back)
* tolerant resume: a truncated/garbled last line is dropped and the file is rewritten with the good rows
"""

import json
import sys
import time
from pathlib import Path


def read_good_rows(path: Path) -> dict:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            rows[d["record_id"]] = d
        except Exception:  # noqa: BLE001
            continue
    return rows


def main() -> None:
    bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
    engine = sys.argv[3] if len(sys.argv) > 3 else "paddle"
    meta = [json.loads(x) for x in (bundle / "data" / "image_meta.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    good = read_good_rows(out)
    with open(out, "w", encoding="utf-8") as f:  # rewrite only parseable rows
        for d in good.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if engine == "paddle":
        from paddleocr import PaddleOCR

        kw = dict(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, device="gpu:0")
        try:
            ocr = PaddleOCR(ocr_version="PP-OCRv5", **kw)
        except Exception as e:  # noqa: BLE001
            print("PP-OCRv5 explicit init failed, retry default:", repr(e)[:300], flush=True)
            ocr = PaddleOCR(**kw)

        def run(p: Path):
            texts, scores = [], []
            for r in ocr.predict(str(p)):
                d = r if hasattr(r, "get") else {}
                if not d.get("rec_texts") and hasattr(r, "json"):
                    d = (r.json or {}).get("res", {})
                texts += list(d.get("rec_texts") or [])
                scores += [float(s) for s in (d.get("rec_scores") or [])]
            return texts, scores
    else:
        import easyocr

        reader = easyocr.Reader(["ko", "en"], gpu=True)

        def run(p: Path):
            res = reader.readtext(str(p))
            return [t for _, t, _ in res], [float(c) for _, _, c in res]

    t0 = time.time()
    n_ok = n_err = n_text = attempted = 0
    errors = []
    with open(out, "a", encoding="utf-8", buffering=1) as f:
        for m in meta:
            if m["record_id"] in good:
                continue
            attempted += 1
            try:
                texts, scores = run(bundle / "images" / m["out_file"])
            except Exception as e:  # noqa: BLE001
                n_err += 1
                errors.append(f"{m['record_id']}: {e!r}"[:300])
                print("ERR", errors[-1], flush=True)
                if (attempted >= 20 and n_ok == 0) or (attempted >= 100 and n_err / attempted > 0.01):
                    break
                continue
            keep = [(t.strip(), s) for t, s in zip(texts, scores) if s >= 0.5 and t.strip()]
            f.write(json.dumps({"record_id": m["record_id"], "texts": [t for t, _ in keep],
                                "scores": [round(s, 3) for _, s in keep], "n_raw": len(texts)}, ensure_ascii=False) + "\n")
            n_ok += 1
            n_text += int(bool(keep))
            if n_ok % 200 == 0:
                print(f"[ocr] {len(good) + n_ok}/{len(meta)} {(time.time() - t0) / n_ok:.2f}s/img with_text={n_text / n_ok:.0%}",
                      flush=True)
    summary = {"engine": engine, "attempted": attempted, "ok": n_ok, "err": n_err, "with_text": n_text,
               "resumed_rows": len(good), "seconds": round(time.time() - t0, 1), "errors_head": errors[:10]}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[ocr] summary", json.dumps(summary, ensure_ascii=False), flush=True)
    if attempted and (n_err / attempted > 0.01 or (attempted >= 20 and n_ok == 0)):
        sys.exit(3)


if __name__ == "__main__":
    main()
