"""Guideline-grounded rule signals (features) and the L0 hard-override rules.

Hard overrides are restricted to rules that are either (a) >= ~0.95 precise on the 2,400 train
labels or (b) a literal example in guideline v1.1. Everything else is only a model feature.
"""

from __future__ import annotations

import os

import re
import unicodedata

import numpy as np

from .common import L2I, LABELS, Data, host

_SUFFIX = re.compile(r"(이미지|아이콘|사진|그림|image|icon|img)\s*$", re.I)
_DATE = re.compile(r"(\d{4}\s*[.\-/년]\s*\d{1,2})|(\d{1,2}\s*[.\-/]\s*\d{1,2}\s*[~∼\-])|(\d{1,2}\s*월\s*\d{1,2}\s*일)")
_FN_EXT = re.compile(r"^[\w\-.() ]+\.(jpe?g|png|gif|bmp|svg|webp)$", re.I)
_FN_VERSION = re.compile(r"(_수정|_원본|_복제본|복제_|_v\d|_최종)")
_FN_TRAIL_NUM = re.compile(r"_\d+$")
_FN_CODE = re.compile(r"^[A-Za-z]\d+_\d+$")
_MEANINGLESS = {"이미지", "사진", "배너", "배너이미지", "로고", "아이콘", "메인이미지", "externalimage", "슬로건", "image", "img"}
_CTA = {"go", "more", "더보기", "전체보기", "상세안내", "다운로드", "바로가기", "상세보기", "자세히보기", "자세히"}
ROLE_MARKERS_GUIDELINE = {"원문자료", "복원 전", "복원 후"}  # guideline v1.1 3-3 example (국가기록원 복원 사례) -> 적절
ROLE_MARKERS_ALL = ROLE_MARKERS_GUIDELINE | {"원본", "복제본"}
HOSTS = ["museum.go.kr", "archives.go.kr", "seogu.go.kr", "gumi.go.kr", "korea.kr", "daedeok.go.kr",
         "yuseong.go.kr", "djjunggu.go.kr", "dong.daegu.kr", "daegu.go.kr", "mohw.go.kr"]
LD_KINDS = ["none", "empty", "anchor", "js", "url", "path", "text"]
SRC_FORMATS = ["jpeg", "png", "gif", "svg", "avif", "other"]


def norm(s: str | None) -> str:
    """NFKC, lowercase, drop whitespace and Unicode punctuation/symbol/separator/control chars."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in "PSZC")


def strip_suffix(alt: str) -> str:
    return _SUFFIX.sub("", (alt or "").strip())


def link_kind(r: dict) -> str:
    if not r.get("in_link"):
        return "none"
    s = (r.get("link_dest") or "").strip()
    if not s:
        return "empty"
    low = s.lower()
    if low.startswith("#"):
        return "anchor"
    if low.startswith("javascript"):
        return "js"
    if re.match(r"^(https?:)?//", low) or low.startswith("www."):
        return "url"
    has_hangul = re.search(r"[가-힣]", s) is not None
    if not has_hangul and " " not in s and any(c in s for c in "/.?=&"):
        return "path"
    return "text"


def char_ngrams(s: str, n: int = 2) -> set[str]:
    s = norm(s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: str, b: str) -> float:
    x, y = char_ngrams(a), char_ngrams(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def filename_flags(alt: str) -> dict:
    a = (alt or "").strip()
    return {
        "fn_ext": bool(_FN_EXT.match(a)),
        "fn_hwp": "원본 그림의 이름" in a,
        "fn_version": bool(_FN_VERSION.search(a)),
        "fn_trail_num": bool(_FN_TRAIL_NUM.search(a)),
        "fn_code": bool(_FN_CODE.match(a)),
    }


def record_features(r: dict, meta: dict) -> dict:
    alt = r.get("alt_text") or ""
    ld = r.get("link_dest") or ""
    ctx = r.get("context_text") or ""
    title = (r.get("page_title") or "")[:300]
    heading = r.get("heading_path") or ""
    kind = link_kind(r)
    na, nld, nctx = norm(alt), norm(ld), norm(ctx)
    nas = norm(strip_suffix(alt))
    f: dict = {}
    f["alt_empty"] = float(alt == "")
    f["in_link"] = float(bool(r.get("in_link")))
    for k in LD_KINDS:
        f[f"ld_{k}"] = float(kind == k)
    text_link = kind == "text"
    f["alt_in_link"] = float(text_link and bool(nas) and nas in nld)
    f["alt_eq_link"] = float(text_link and bool(na) and na == nld)
    f["link_is_alt_twice"] = float(text_link and bool(na) and nld == na + na)
    rest = ld.replace(alt, " ") if alt else ld
    f["link_rest_has_date"] = float(text_link and bool(_DATE.search(rest)))
    f["link_has_date"] = float(bool(_DATE.search(ld)))
    f["alt_in_context"] = float(bool(na) and na in nctx)
    f["alt_eq_context"] = float(bool(na) and na == nctx)
    f["alt_in_title"] = float(bool(na) and na in norm(title))
    f["alt_in_heading"] = float(bool(na) and na in norm(heading))
    f["jac_alt_link"] = jaccard(alt, ld) if text_link else 0.0
    f["jac_alt_ctx"] = jaccard(alt, ctx)
    ff = filename_flags(alt)
    f.update({k: float(v) for k, v in ff.items()})
    f["fn_any"] = float(any(ff.values()))
    f["meaningless"] = float(na in _MEANINGLESS or bool(re.fullmatch(r"비주얼\d*", na)))
    f["digit_only"] = float(bool(re.fullmatch(r"\d+", alt.strip())))
    f["symbols_only"] = float(bool(alt.strip()) and na == "")
    f["short_norm"] = float(0 < len(na) <= 1)
    f["html_artifact"] = float("<!--" in alt or "attr=" in alt)
    f["cta"] = float(alt.strip().lower() in _CTA)
    f["role_marker"] = float(alt.strip() in ROLE_MARKERS_ALL)
    f["restore_page"] = float("archivesrestore" in (r.get("page_url") or "").lower())
    f["alt_len"] = float(len(alt))
    f["alt_norm_len"] = float(len(na))
    f["alt_words"] = float(len(alt.split()))
    f["link_len"] = float(len(ld))
    f["ctx_len"] = float(len(ctx))
    f["title_len"] = float(min(len(r.get("page_title") or ""), 2000))
    f["heading_depth"] = float(heading.count(">") + (1 if heading else 0))
    f["kogl"] = float(bool(r.get("kogl_marker")))
    f["alt_hangul_ratio"] = (len(re.findall(r"[가-힣]", alt)) / len(alt)) if alt else 0.0
    f["alt_digit_ratio"] = (len(re.findall(r"\d", alt)) / len(alt)) if alt else 0.0
    f["alt_latin_ratio"] = (len(re.findall(r"[A-Za-z]", alt)) / len(alt)) if alt else 0.0
    h = host(r)
    for hh in HOSTS:
        f[f"host_{hh}"] = float(h == hh)
    fmt = meta.get("src_format", "other")
    for sf in SRC_FORMATS:
        f[f"fmt_{sf}"] = float(fmt == sf or (sf == "other" and fmt not in SRC_FORMATS[:-1]))
    w, hgt = float(meta.get("src_w", 0)), float(meta.get("src_h", 0))
    f["img_w"], f["img_h"] = w, hgt
    f["img_max_side"] = max(w, hgt)
    f["img_min_side"] = min(w, hgt)
    f["img_aspect"] = (w / hgt) if hgt else 0.0
    f["img_log_bytes"] = float(np.log1p(meta.get("src_bytes", 0)))
    f["img_alpha"] = float(bool(meta.get("had_alpha")))
    f["img_animated"] = float(int(meta.get("n_frames", 1)) > 1)
    small_graphic = fmt in ("png", "svg", "gif") and (meta.get("src_bytes", 0) < 20_000 or max(w, hgt) <= 120)
    f["small_graphic"] = float(small_graphic)
    f["alt_in_link_small_graphic"] = f["alt_in_link"] * f["small_graphic"]
    return f


def feature_matrix(data: Data) -> tuple[np.ndarray, list[str]]:
    rows = [record_features(r, data.image_meta[r["record_id"]]) for r in data.all]
    names = list(rows[0].keys())
    X = np.array([[row[k] for k in names] for row in rows], dtype=np.float32)
    return X, names


# ----------------------------------------------------------------------------- L0 hard rules
def l0_rule_label(r: dict) -> tuple[str | None, str | None]:
    """Return (label, rule_id) for records decided by a hard rule, else (None, None)."""
    alt = (r.get("alt_text") or "").strip()
    ff = filename_flags(alt) if alt else {}
    if alt and (ff["fn_ext"] or ff["fn_version"] or ff["fn_hwp"]):  # trail_num/code stay features only (1/2 precise)
        return "파일명형", "R1_filename_cascade"
    if alt in ROLE_MARKERS_GUIDELINE and "archives.go.kr" in (r.get("page_url") or ""):
        return "적절", "R2_restoration_role_marker"
    if re.fullmatch(r"\d+", alt or "x"):
        return "무의미형", "R3_digit_only"
    return None, None


def rule_precision_table(data: Data) -> list[dict]:
    """Precision/recall of each hard rule on the 2,400 labelled train rows + firing counts on test."""
    out = []
    for rule_id in ("R1_filename_cascade", "R2_restoration_role_marker", "R3_digit_only"):
        fired_tr = [(r, lab) for r in data.train for lab, rid in [l0_rule_label(r)] if rid == rule_id]
        fired_te = [r for r in data.test if l0_rule_label(r)[1] == rule_id]
        lab = None if not fired_tr and not fired_te else (l0_rule_label((fired_tr[0][0] if fired_tr else fired_te[0]))[0])
        tp = sum(1 for r, lb in fired_tr if r["label"] == lb)
        support = sum(1 for r in data.train if r["label"] == lab) if lab else 0
        out.append({"rule": rule_id, "label": lab, "train_fires": len(fired_tr), "train_tp": tp,
                    "train_precision": (tp / len(fired_tr)) if fired_tr else None,
                    "train_recall": (tp / support) if support else None, "test_fires": len(fired_te)})
    return out


def alt_template(alt: str | None) -> str:
    return re.sub(r"\d+", "#", (alt or "").strip())


def sibling_key(r: dict) -> tuple:
    return ((r.get("page_url") or "").split("#")[0], bool(r.get("in_link")), link_kind(r), alt_template(r.get("alt_text")))


def propagate_from_boundary(data: Data, min_agree: float = 0.8) -> dict[int, int]:
    """R4: unknown test rows inherit the label of officially released test rows with the same page, link context and
    alt template (digits -> #). Train leave-one-out precision of this sibling rule: 0.907 (coverage 39%).

    Which label is propagated matters: the organizer audited ALL 600 test rows; rows where the existing label and the
    independent audit disagreed became the 109 boundary rows (existing label kept). Every other test row therefore
    carries a label the audit agreed with, so a non-boundary sibling should follow the audit's view of the structure,
    i.e. the boundary row's 재판정_라벨, not its 데이터셋_라벨 (KOALTQ_R4_LABEL=dataset restores the old behaviour)."""
    use_rejudge = os.environ.get("KOALTQ_R4_LABEL", "rejudge") == "rejudge" and data.boundary_rejudge
    groups: dict[tuple, list[int]] = {}
    for r in data.test:
        lab = (data.boundary_rejudge if use_rejudge else data.boundary_test).get(r["record_id"])
        if use_rejudge and lab in ("무관형", "무의미형", "장식오용형", "파일명형"):
            lab = None  # instance-specific judgements (e.g. image/alt mismatch) do not transfer to siblings
        if lab is not None:
            groups.setdefault(sibling_key(r), []).append(L2I[lab])
    out = {}
    for i, r in enumerate(data.test):
        if r["record_id"] in data.boundary_test or l0_rule_label(r)[0] is not None:
            continue
        labs = groups.get(sibling_key(r))
        if labs:
            vals, cnts = np.unique(labs, return_counts=True)
            if cnts.max() / len(labs) >= min_agree:
                out[i] = int(vals[cnts.argmax()])
    return out


_CTA = {"go", "more", "더보기", "전체보기", "상세안내", "다운로드", "바로가기", "상세보기", "자세히보기", "자세히", "view", "viewmore",
        "more+", "+", "클릭", "이동", "바로보기", "보기", "신청하기"}
_THUMB_SUFFIX = re.compile(r"(이미지|아이콘|사진|그림|썸네일|image|icon|img|thumbnail)\s*$", re.I)


def extra_fixes(data: Data) -> dict[int, tuple[int, str]]:
    """Optional guideline 3-3 structural rules, opt-in via KOALTQ_L0_EXTRA (comma list). Lowest L0 precedence.
      A4X: alt (minus a trailing 이미지/사진/썸네일...) is contained in the link text and >=3 of 4 X12 VLMs say photo
           -> 불충분형 (photo thumbnail repeating the title; train 66/67)
      A9B: the image is the link's only content (url/path/anchor/js link), alt is a non-CTA text and OCR < 4 chars
           -> 적절 (functional icon/logo naming its destination; train 100/104; sole links are never 장식오용형 in train)
    """
    spec = {s.strip().upper() for s in os.environ.get("KOALTQ_L0_EXTRA", "").split(",") if s.strip()}
    if not spec:
        return {}
    import json as _json

    from .common import RUNS

    n = data.n_train
    votes = sum((np.load(RUNS / f"x12_{m}" / "attrs.npy")[:, 2] >= 0.5).astype(int)
                for m in ("qwen35_9b", "qwen38_27b", "qwen3vl_32b", "qwen3vl_8b"))
    names = _json.loads((RUNS / "x2_ocr" / "ocr_feature_names.json").read_text())
    ocr_chars = np.load(RUNS / "x2_ocr" / "ocr_features.npy")[:, names.index("ocr_chars")]
    assert len(votes) == len(ocr_chars) == n + len(data.test)
    out = {}
    for j, r in enumerate(data.test):
        alt = (r.get("alt_text") or "").strip()
        kind = link_kind(r)
        if "A4X" in spec and kind == "text" and alt:
            a2 = norm(_THUMB_SUFFIX.sub("", alt))
            if a2 and a2 in norm(r.get("link_dest")) and votes[n + j] >= 3:
                out[j] = (L2I["불충분형"], "A4x_photo_thumbnail")
                continue
        if ("A9B" in spec and r.get("in_link") and kind in ("path", "url", "empty", "anchor", "js") and alt
                and norm(alt) not in _CTA and ocr_chars[n + j] < 4):
            out[j] = (L2I["적절"], "A9b_sole_link_functional")
    return out


def _struct_helpers(data: Data):
    """Structure descriptors shared by the K2 sibling rule and its veto. rel: alt vs link text / context
    (empty, eq_link, in_link, eq_ctx, in_ctx, other). img(g, r) with g = global row index (train rows first, then test):
    photo (>= 3 of the 4 x12 VLM photo votes), text (OCR >= 20 chars), icon (<= 120 px), else graphic."""
    import json as _json

    from .common import RUNS

    votes = sum((np.load(RUNS / f"x12_{m}" / "attrs.npy")[:, 2] >= 0.5).astype(int)
                for m in ("qwen35_9b", "qwen38_27b", "qwen3vl_32b", "qwen3vl_8b"))
    names = _json.loads((RUNS / "x2_ocr" / "ocr_feature_names.json").read_text())
    ocr_chars = np.load(RUNS / "x2_ocr" / "ocr_features.npy")[:, names.index("ocr_chars")]

    def rel(r: dict) -> str:
        alt = (r.get("alt_text") or "").strip()
        k = link_kind(r)
        a, ld, c = norm(strip_suffix(alt)), norm(r.get("link_dest")), norm(r.get("context_text"))
        if not alt:
            return "empty"
        if k == "text" and a and a == ld:
            return "eq_link"
        if k == "text" and a and a in ld:
            return "in_link"
        if a and a == c:
            return "eq_ctx"
        if a and a in c:
            return "in_ctx"
        return "other"

    def img(g: int, r: dict) -> str:
        m = data.image_meta[r["record_id"]]
        if votes[g] >= 3:
            return "photo"
        if ocr_chars[g] >= 20:
            return "text"
        if max(m["src_w"], m["src_h"]) <= 120:
            return "icon"
        return "graphic"

    return rel, img


def k2_veto(data: Data, k2_mask: np.ndarray, test_logits: np.ndarray, min_n: int = 25, purity: float = 0.95,
            conf: float = 0.95) -> np.ndarray:
    """Veto for the K2 sibling rule (21번, LB 0.9304 -> 0.9357; all 3 vetoed rows were truly 불충분형).
    Residual test rows are rows where the original label and the independent audit AGREED, so a sibling's audit '적절'
    only transfers if the original labelers could also have said 적절. A K2 row keeps 불충분형 when
      (a) its page-free train structure group (in_link, link kind, alt/link/context relation, image type, alt length
          bucket) has >= min_n train rows and >= purity of them are 불충분형, or
      (b) both the supervised blend (test_logits, 600 x 7) and the 27B guideline judge (mean log-prob of the
          x14_qwen38_27b_long* runs) give P(불충분형) >= conf.
    Returns the boolean mask of vetoed rows (a subset of k2_mask)."""
    import collections

    from .common import RUNS

    n = data.n_train
    rel, img = _struct_helpers(data)

    def alen(r: dict) -> str:
        k = len((r.get("alt_text") or "").strip())
        return "0" if k == 0 else "1-5" if k <= 5 else "6-15" if k <= 15 else "16-40" if k <= 40 else "41+"

    def skey(g: int, r: dict) -> tuple:
        return (bool(r.get("in_link")), link_kind(r), rel(r), img(g, r), alen(r))

    tab: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for g, r in enumerate(data.train):
        tab[skey(g, r)][int(data.y[g])] += 1
    P = np.exp(test_logits - test_logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    J = np.mean([np.log(np.load(p / "probs.npy") + 1e-6) for p in sorted(RUNS.glob("x14_qwen38_27b_long*"))], axis=0)[n:]
    PJ = np.exp(J)
    PJ /= PJ.sum(1, keepdims=True)
    ins = L2I["불충분형"]
    veto = np.zeros(len(data.test), bool)
    for j, r in enumerate(data.test):
        if not k2_mask[j]:
            continue
        c = tab[skey(n + j, r)]
        t = sum(c.values())
        v_struct = t >= min_n and c[ins] / t >= purity
        v_conf = P[j, ins] >= conf and PJ[j, ins] >= conf
        veto[j] = v_struct or v_conf
    return veto


def k2_sibling_to_ok(data: Data, pred: np.ndarray, fixed_mask: np.ndarray, min_agree: float = 0.8) -> np.ndarray:
    """Post-processing used for 16번 (LB 0.9304): a residual row predicted 불충분형 becomes 적절 when the boundary rows
    of its looser sibling group (page, in_link, link kind, alt/link/context relation, image type) were re-judged 적절
    by >= min_agree of them (non-rare 재판정_라벨 only). Only this direction is applied: the R4 evidence and the LB
    arithmetic both contradict the opposite direction. Returns the boolean mask of rows that changed."""
    n = data.n_train
    rel, img = _struct_helpers(data)

    def key(j: int, r: dict) -> tuple:
        return ((r.get("page_url") or "").split("#")[0], bool(r.get("in_link")), link_kind(r), rel(r), img(n + j, r))

    groups: dict[tuple, list[int]] = {}
    for j, r in enumerate(data.test):
        lab = data.boundary_rejudge.get(r["record_id"])
        if lab and lab not in ("무관형", "무의미형", "장식오용형", "파일명형"):
            groups.setdefault(key(j, r), []).append(L2I[lab])
    hit = np.zeros(len(data.test), bool)
    for j, r in enumerate(data.test):
        if fixed_mask[j] or pred[j] != L2I["불충분형"]:
            continue
        labs = groups.get(key(j, r))
        if labs:
            vals, cnts = np.unique(labs, return_counts=True)
            if cnts.max() / len(labs) >= min_agree and int(vals[cnts.argmax()]) == L2I["적절"]:
                hit[j] = True
    return hit


def l0_fixed(data: Data) -> dict[int, tuple[int, str]]:
    """All test rows decided by L0: {test_idx: (label_idx, source)}; precedence boundary > rules > propagation > extras."""
    fixed = dict(extra_fixes(data))
    for i, lab in propagate_from_boundary(data).items():
        fixed[i] = (lab, "R4_sibling_propagation")
    for i, r in enumerate(data.test):
        lab, rid = l0_rule_label(r)
        if lab is not None:
            fixed[i] = (L2I[lab], rid)
        if r["record_id"] in data.boundary_test:
            fixed[i] = (L2I[data.boundary_test[r["record_id"]]], "boundary_109")
    return fixed


def apply_l0(data: Data, test_pred: np.ndarray, use_rules: bool = True, use_boundary: bool = True) -> tuple[np.ndarray, dict]:
    """Apply hard rules, sibling propagation (R4) and the officially permitted 109 boundary labels (데이터셋_라벨)."""
    pred = np.asarray(test_pred).copy()
    log = {"rules": {}, "boundary_overrides": 0, "boundary_changed": 0}
    if use_rules and use_boundary:
        for i, (lab, src) in extra_fixes(data).items():  # lowest precedence: overwritten below by R4 / rules / boundary
            log["rules"][src] = log["rules"].get(src, 0) + 1
            pred[i] = lab
        for i, lab in propagate_from_boundary(data).items():
            log["rules"]["R4_sibling_propagation"] = log["rules"].get("R4_sibling_propagation", 0) + 1
            pred[i] = lab
    for i, r in enumerate(data.test):
        if use_rules:
            lab, rid = l0_rule_label(r)
            if lab is not None:
                log["rules"][rid] = log["rules"].get(rid, 0) + 1
                pred[i] = L2I[lab]
        if use_boundary and r["record_id"] in data.boundary_test:
            new = L2I[data.boundary_test[r["record_id"]]]
            log["boundary_overrides"] += 1
            log["boundary_changed"] += int(pred[i] != new)
            pred[i] = new
    return pred, log


def label_names(idx: np.ndarray) -> list[str]:
    return [LABELS[int(i)] for i in idx]
