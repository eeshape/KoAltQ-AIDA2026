"""Evidence-card prompt shared by the frozen-feature probes (X3-X5) and the discriminative LoRA (X6)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .common import RUNS
from .rules import filename_flags, link_kind, norm, strip_suffix

GUIDE = (
    "[판정 기준 요약: 웹 대체텍스트(alt) 품질 7범주]\n"
    "적절=이미지 역할에 맞는 정보 전달. 무의미형=내용 없는 상투어·기호('이미지','사진','배너','*'). "
    "파일명형=파일명·경로·확장자·'_수정/_원본' 같은 파일 관리 흔적. 중복형=인접 텍스트(캡션·링크 문구)를 그대로 반복해 정보 이득 0. "
    "장식오용형=순수 장식 이미지에 내용성 alt. 불충분형=방향은 맞으나 핵심 정보 누락. 무관형=이미지 내용과 불일치.\n"
    "우선순위: 파일명형>무관형>무의미형>장식오용형>중복형>불충분형. "
    "링크 안 이미지는 목적지를 알려야 함(GO/more 같은 CTA만 있으면 불충분형). "
    "썸네일 카드: 그래픽(아이콘·로고·배너)이 링크 텍스트를 반복하면 중복형, 촬영 사진·포스터가 제목만 반복하면 불충분형. "
    "포스터·카드뉴스 등 글자 포함 이미지는 기간·장소·대상·금액 등 실질 정보가 alt나 주변 텍스트에 있어야 함. "
    "빈 alt는 링크 텍스트·인접 텍스트가 목적지/정보를 주면 적절, 이미지에만 있는 정보가 빠지면 불충분형. "
    "본문이 대상을 충분히 서술하면 역할 표시 alt('원문자료','복원 전/후')도 적절."
)

_OCR_CACHE: dict | None = None
OCR_PATH = RUNS / "x2_ocr" / "ocr.jsonl"


def read_ocr_rows(path: Path) -> dict:
    """Tolerant reader: skips truncated/garbled lines."""
    rows = {}
    path = Path(path)
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


def use_ocr() -> bool:
    return os.environ.get("KOALTQ_NO_OCR", "") != "1"


def require_ocr() -> str | None:
    """X3/X6 must not silently run with partial OCR. Returns the OCR file sha256 (None if OCR disabled)."""
    if not use_ocr():
        return None
    rows = read_ocr_rows(OCR_PATH)
    assert len(rows) == 3000, (f"OCR file has {len(rows)}/3000 rows; run X2 first or set KOALTQ_NO_OCR=1 "
                               "to build evidence cards without OCR")
    return hashlib.sha256(OCR_PATH.read_bytes()).hexdigest()


def ocr_lookup() -> dict:
    global _OCR_CACHE
    if _OCR_CACHE is None:
        _OCR_CACHE = {rid: " ".join(d.get("texts") or []) for rid, d in read_ocr_rows(OCR_PATH).items()} if use_ocr() else {}
    return _OCR_CACHE


def _clip(s: str | None, n: int) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


_IMG_META: dict | None = None


def image_info_line(rid: str) -> str | None:
    """Original image size/format line (opt-in via KOALTQ_CARD_IMGINFO=1): the model sees an upscaled image, so a
    46x46 icon and a large photo look alike without it."""
    global _IMG_META
    if os.environ.get("KOALTQ_CARD_IMGINFO", "") != "1":
        return None
    if _IMG_META is None:
        from .common import BUNDLE
        _IMG_META = {}
        for line in (BUNDLE / "data" / "image_meta.jsonl").read_text(encoding="utf-8").split("\n"):
            if line.strip():
                d = json.loads(line)
                _IMG_META[d["record_id"]] = d
        assert len(_IMG_META) == 3000, f"image_meta has {len(_IMG_META)} rows"
    m = _IMG_META[rid]
    extra = (", 투명 배경" if m.get("had_alpha") else "") + (", 애니메이션" if (m.get("n_frames") or 1) > 1 else "")
    return f"이미지 원본: {m['src_w']}x{m['src_h']}px, {m.get('src_format') or '?'}{extra}"


def evidence_card(r: dict) -> str:
    alt = r.get("alt_text") or ""
    kind = link_kind(r)
    ld = r.get("link_dest") or ""
    ctx = r.get("context_text") or ""
    nas = norm(strip_suffix(alt))
    signals = []
    if kind == "text" and nas and nas in norm(ld):
        signals.append("alt가 링크 텍스트에 포함됨")
    if alt and norm(alt) and norm(alt) in norm(ctx):
        signals.append("alt가 인접 텍스트에 포함됨")
    if any(filename_flags(alt).values()):
        signals.append("파일명 흔적 패턴")
    if kind in ("url", "path", "anchor", "js"):
        signals.append(f"link_dest는 텍스트가 아닌 {kind}")
    lines = [
        GUIDE,
        "[레코드]",
        f"alt: {'(빈 문자열)' if alt == '' else _clip(alt, 300)}",
        f"링크 내부: {'예' if r.get('in_link') else '아니오'}" + (f" | 링크 텍스트({kind}): {_clip(ld, 160)}" if r.get("in_link") else ""),
        f"페이지 제목: {_clip(r.get('page_title'), 120)}",
        f"헤딩 경로: {_clip(r.get('heading_path'), 120)}",
        f"인접 텍스트: {_clip(ctx, int(os.environ.get('KOALTQ_CTX_CHARS', '400')))}",
        f"규칙 신호: {', '.join(signals) if signals else '없음'}",
    ]
    info = image_info_line(r["record_id"])
    if info:
        lines.append(info)
    if use_ocr():
        o = ocr_lookup().get(r["record_id"])
        lines.append(f"이미지 속 글자(OCR): {_clip(o, int(os.environ.get('KOALTQ_OCR_CHARS', '300'))) if o else '(없음)'}")
    lines.append("질문: 이 이미지의 alt 품질 라벨은 7범주 중 무엇인가?")
    return "\n".join(lines)


def cards_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(evidence_card(r).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def load_image(path: Path):
    from PIL import Image

    im = Image.open(path)
    im.load()
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if min(w, h) < 32:  # Qwen/InternVL processors need a minimal patch area
        scale = 32 / max(1, min(w, h))
        im = im.resize((max(32, int(w * scale)), max(32, int(h * scale))))
    return im
