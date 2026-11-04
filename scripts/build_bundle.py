"""Build the Colab upload bundle for KoAltQ round-2 (Plan C).

Local-only script. Reads the original competition data (never modifies it) and writes
experiment/colab_20261102/build/koaltq_bundle.zip plus <10MB parts for browser upload.

Image normalization (documented for the report):
  * format sniffed by magic bytes (extensions are unreliable: .do/.es/.php/none)
  * SVG -> PNG via macOS `sips` (Colab has no sips)
  * first frame for GIF/MPO; CMYK -> RGB
  * any alpha (RGBA/LA/PA/P+transparency) composited onto WHITE (not black)
  * longest side capped at 1536 px (LANCZOS); smaller images are kept as-is
  * photo-origin (JPEG/AVIF/WEBP) -> JPEG q90; graphic-origin (PNG/GIF/SVG/BMP) -> PNG when
    w*h <= 640*640 else JPEG q92
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# competition workspace that holds data/공개_데이터셋_20260917 (the organizer release) and prepared/ (iCCP-repaired
# copy of the train images); set KOALTQ_RAW_ROOT to that folder
ROOT = Path(os.environ.get("KOALTQ_RAW_ROOT", "."))
DATA = ROOT / "data" / "공개_데이터셋_20260917"
TRAIN_JSONL = DATA / "koaltq_train" / "records.jsonl"
TEST_JSONL = DATA / "koaltq_test_public" / "records.jsonl"
TRAIN_IMG = ROOT / "prepared" / "koaltq_train" / "images"  # iCCP-repaired copy of the train images
TEST_IMG = DATA / "koaltq_test_public" / "images"
FOLDS = ROOT / "results" / "20261027_weekly" / "context" / "folds.jsonl"
FOLDS_SHA = "4154b923c1a693caf95807590e6c8493abd37f14258c7d0fc836fee982d7e2fe"
BOUNDARY_CSV = DATA / "경계사례_20261013.csv"
GUIDELINE = DATA / "문서" / "라벨링-가이드라인-v1.1.md"

LABELS = ["적절", "무의미형", "파일명형", "중복형", "장식오용형", "불충분형", "무관형"]
MAX_SIDE = 1536
PART_BYTES = 9_500_000

OUT = Path(__file__).resolve().parent / "build"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Split ONLY on '\n' (records contain literal U+2028)."""
    text = path.read_text(encoding="utf-8-sig")
    rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    return rows


def sniff(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[4:12] in (b"ftypavif", b"ftypavis"):
        return "avif"
    if b[:2] == b"BM":
        return "bmp"
    head = b[:4096].lower()
    if b"<svg" in head:
        return "svg"
    return "other"


def svg_to_png_bytes(src: Path) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        tmp_svg = Path(td) / "in.svg"
        shutil.copyfile(src, tmp_svg)
        out_png = Path(td) / "out.png"
        r = subprocess.run(
            ["sips", "-s", "format", "png", str(tmp_svg), "--out", str(out_png)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not out_png.exists():
            raise RuntimeError(f"sips failed for {src}: {r.stderr[:300]}")
        return out_png.read_bytes()


def to_rgb_white(im: Image.Image) -> tuple[Image.Image, bool]:
    """Composite any transparency onto white. Returns (rgb_image, had_alpha)."""
    had_alpha = False
    if im.mode == "P":
        if "transparency" in im.info:
            im = im.convert("RGBA")
        else:
            im = im.convert("RGB")
    if im.mode in ("LA", "PA"):
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        alpha = im.getchannel("A")
        had_alpha = alpha.getextrema()[0] < 255
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=alpha)
        return bg, had_alpha
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im, had_alpha


def normalize_image(src: Path) -> tuple[bytes, str, dict]:
    raw = src.read_bytes()
    fmt = sniff(raw)
    meta = {"src_format": fmt, "src_bytes": len(raw), "src_sha256": sha256_bytes(raw)}
    data = raw
    if fmt == "svg":
        data = svg_to_png_bytes(src)
    im = Image.open(io.BytesIO(data))
    meta["n_frames"] = int(getattr(im, "n_frames", 1) or 1)
    try:
        im.seek(0)
    except EOFError:
        pass
    im.load()
    meta["src_w"], meta["src_h"] = im.size
    meta["src_mode"] = im.mode
    rgb, had_alpha = to_rgb_white(im)
    meta["had_alpha"] = had_alpha
    w, h = rgb.size
    if max(w, h) > MAX_SIDE:
        rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    meta["out_w"], meta["out_h"] = rgb.size
    buf = io.BytesIO()
    photo_like = fmt in ("jpeg", "avif", "webp", "other")
    if not photo_like and rgb.size[0] * rgb.size[1] <= 640 * 640:
        rgb.save(buf, "PNG", optimize=True)  # small graphics/icons/text crops stay lossless
        ext = "png"
    else:
        rgb.save(buf, "JPEG", quality=90 if photo_like else 92, optimize=True)
        ext = "jpg"
    return buf.getvalue(), ext, meta


def main() -> int:
    if OUT.exists():
        print(f"refusing to overwrite existing {OUT}; delete it first", file=sys.stderr)
        return 2
    stage = OUT / "koaltq_bundle"
    (stage / "data").mkdir(parents=True)
    (stage / "images").mkdir()
    (stage / "docs").mkdir()

    train = read_jsonl(TRAIN_JSONL)
    test = read_jsonl(TEST_JSONL)
    assert len(train) == 2400 and len(test) == 600, (len(train), len(test))
    assert all(r["label"] in LABELS for r in train)
    ids_train = {r["record_id"] for r in train}
    ids_test = {r["record_id"] for r in test}
    assert len(ids_train) == 2400 and len(ids_test) == 600 and not (ids_train & ids_test)

    assert sha256_file(FOLDS) == FOLDS_SHA, "fold manifest hash mismatch"
    folds = read_jsonl(FOLDS)
    assert {f["record_id"] for f in folds} == ids_train

    # Boundary cases: ONLY the 데이터셋_라벨 column is the answer-key label (organizer Q&A 678, 2026-10-26).
    with open(BOUNDARY_CSV, encoding="utf-8-sig", newline="") as f:
        brows = list(csv.DictReader(f))
    assert len(brows) == 126, len(brows)
    btest = [
        {"record_id": r["record_id"], "label": r["데이터셋_라벨"], "boundary_type": r["유형"]}
        for r in brows
        if r["분할"] == "test"
    ]
    btrain = [
        {"record_id": r["record_id"], "label": r["데이터셋_라벨"], "boundary_type": r["유형"]}
        for r in brows
        if r["분할"] == "train"
    ]
    assert len(btest) == 109 and len(btrain) == 17, (len(btest), len(btrain))
    assert all(b["record_id"] in ids_test and b["label"] in LABELS for b in btest)
    tr_label = {r["record_id"]: r["label"] for r in train}
    assert all(tr_label[b["record_id"]] == b["label"] for b in btrain), "train boundary labels must match"

    def write_jsonl(p: Path, rows: list[dict]) -> None:
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(stage / "data" / "train.jsonl", train)
    write_jsonl(stage / "data" / "test.jsonl", test)
    shutil.copyfile(FOLDS, stage / "data" / "folds.jsonl")  # byte-identical: sha must stay 4154b923...
    write_jsonl(stage / "data" / "boundary_test_labels.jsonl", btest)
    write_jsonl(stage / "data" / "boundary_train_ids.jsonl", btrain)
    shutil.copyfile(GUIDELINE, stage / "docs" / "guideline_v1.1.md")

    metas = []
    for split, rows, img_dir in (("train", train, TRAIN_IMG), ("test", test, TEST_IMG)):
        for i, r in enumerate(rows):
            src = img_dir / r["image_file"]
            out_bytes, ext, meta = normalize_image(src)
            out_name = f"{r['record_id']}.{ext}"
            (stage / "images" / out_name).write_bytes(out_bytes)
            meta.update({"record_id": r["record_id"], "split": split, "src_file": r["image_file"], "out_file": out_name})
            metas.append(meta)
            if (i + 1) % 300 == 0:
                print(f"{split}: {i + 1}/{len(rows)}", flush=True)
    write_jsonl(stage / "data" / "image_meta.jsonl", metas)
    return package(stage, len(train), len(test), len(btest), len(btrain), metas)


def package(stage: Path, n_train: int, n_test: int, n_btest: int, n_btrain: int, metas: list[dict]) -> int:
    """Write MANIFEST.json, the zip and the <10MB parts from an already-staged bundle directory."""
    assert sha256_file(stage / "data" / "folds.jsonl") == FOLDS_SHA, "staged fold manifest must be byte-identical"
    for stale in (stage / "MANIFEST.json", OUT / "koaltq_bundle.zip", OUT / "parts.json"):
        if stale.exists():
            stale.unlink()
    if (OUT / "parts").exists():
        shutil.rmtree(OUT / "parts")
    train, test, btest, btrain = range(n_train), range(n_test), range(n_btest), range(n_btrain)
    manifest = {
        "bundle": "koaltq_bundle_v2",
        "counts": {"train": len(train), "test": len(test), "boundary_test": len(btest), "boundary_train": len(btrain)},
        "folds_sha256": FOLDS_SHA,
        "image_norm": {"max_side": MAX_SIDE, "alpha": "white", "svg": "sips",
                       "save": "photo-origin jpeg q90; graphic-origin png<=640x640 else jpeg q92"},
        "files": {},
    }
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(stage))] = sha256_file(p)
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    zpath = OUT / "koaltq_bundle.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                arc = str(Path("koaltq_bundle") / p.relative_to(stage))
                comp = zipfile.ZIP_STORED if p.suffix in (".jpg", ".png") else zipfile.ZIP_DEFLATED
                z.write(p, arc, compress_type=comp)
    full_sha = sha256_file(zpath)
    parts_dir = OUT / "parts"
    parts_dir.mkdir()
    parts = []
    with open(zpath, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(PART_BYTES)
            if not chunk:
                break
            name = f"koaltq_bundle.zip.part{idx:03d}"
            (parts_dir / name).write_bytes(chunk)
            parts.append({"name": name, "bytes": len(chunk), "sha256": sha256_bytes(chunk)})
            idx += 1
    (OUT / "parts.json").write_text(
        json.dumps({"zip_sha256": full_sha, "zip_bytes": zpath.stat().st_size, "parts": parts}, indent=1),
        encoding="utf-8",
    )
    fmt_counts: dict[str, int] = {}
    for m in metas:
        fmt_counts[m["src_format"]] = fmt_counts.get(m["src_format"], 0) + 1
    print(json.dumps({"zip_bytes": zpath.stat().st_size, "zip_sha256": full_sha, "n_parts": len(parts), "formats": fmt_counts,
                      "had_alpha": sum(m["had_alpha"] for m in metas)}, ensure_ascii=False))
    return 0


def repack() -> int:
    """Re-copy the byte-identical fold manifest into the existing stage and rebuild zip/parts (no image work)."""
    stage = OUT / "koaltq_bundle"
    shutil.copyfile(FOLDS, stage / "data" / "folds.jsonl")
    metas = read_jsonl(stage / "data" / "image_meta.jsonl")
    n = {k: len(read_jsonl(stage / "data" / f)) for k, f in
         (("train", "train.jsonl"), ("test", "test.jsonl"), ("bt", "boundary_test_labels.jsonl"), ("br", "boundary_train_ids.jsonl"))}
    return package(stage, n["train"], n["test"], n["bt"], n["br"], metas)


if __name__ == "__main__":
    raise SystemExit(repack() if "--repack" in sys.argv else main())
