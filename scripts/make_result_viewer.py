"""제출 CSV(record_id,label)를 이미지·alt·문맥과 함께 보는 HTML 결과 뷰어를 만든다 (표준 라이브러리만 사용).

usage: python scripts/make_result_viewer.py [csv] [--out html] [--bundle data/koaltq_bundle] [--tour]
  csv   기본값 outputs/reproduced_submission.csv (scripts/reproduce_best.sh의 출력)
  같은 폴더에 harness_log.jsonl이 있으면 하네스가 정하거나 바꾼 행에 그 근거(모듈)를 함께 표시한다.
결과 HTML은 이미지를 data/koaltq_bundle/images/에서 상대 경로로 읽으므로 저장소 안에서 연다.
  --tour (또는 주소 끝 #tour): 페이지를 열면 필터·하네스 근거·검색을 약 40초 동안 자동으로 시연한다 (데모영상용).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["적절", "무의미형", "파일명형", "중복형", "장식오용형", "불충분형", "무관형"]
MODULES = {
    "H0_public_row": "H0 공개 경계사례 (공식 라벨)",
    "H1_R1_filename_cascade": "H1 가이드라인 규칙 R1 (파일명 흔적)",
    "H1_R2_restoration_role_marker": "H1 가이드라인 규칙 R2 (역할 표시 alt)",
    "H2_template_twins": "H2 템플릿 쌍둥이",
    "H3_sole_link_never_decoration": "H3 링크 제약 (유일 콘텐츠는 장식 아님)",
    "H4_structure_twins": "H4 구조 쌍둥이",
    "H4_structure_twins+H5a_original_labelers_never_give_it": "H5a 검수 일치 거부 (H4 보정 취소)",
    "H4_structure_twins+H5b_blend_and_judge_both_confident": "H5b 검수 일치 거부 (H4 보정 취소)",
}


def read_jsonl(path: Path) -> list[dict]:
    # split only on '\n': records may contain a literal U+2028 inside strings
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").split("\n") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", default=str(ROOT / "outputs" / "reproduced_submission.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundle", default=str(ROOT / "data" / "koaltq_bundle"))
    ap.add_argument("--tour", action="store_true", help="열면 자동 시연 (데모영상용)")
    a = ap.parse_args()

    csv_path, bundle = Path(a.csv).resolve(), Path(a.bundle).resolve()
    out = Path(a.out).resolve() if a.out else csv_path.with_name("result_viewer.html")
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        pred = {r["record_id"]: r["label"] for r in csv.DictReader(f)}
    test = read_jsonl(bundle / "data" / "test.jsonl")
    img = {m["record_id"]: m["out_file"] for m in read_jsonl(bundle / "data" / "image_meta.jsonl")}
    assert set(pred) == {r["record_id"] for r in test}, "CSV의 record_id가 평가 데이터 600건과 다릅니다"
    bad = sorted({v for v in pred.values() if v not in LABELS})
    assert not bad, f"알 수 없는 라벨: {bad}"

    log_path = csv_path.with_name("harness_log.jsonl")
    harness = {}
    if log_path.exists():
        for e in read_jsonl(log_path):
            harness[e["record_id"]] = e

    rel_img = os.path.relpath(bundle / "images", out.parent)
    rows = []
    for r in test:
        rid = r["record_id"]
        h = harness.get(rid)
        rows.append({
            "id": rid, "label": pred[rid], "alt": r.get("alt_text") or "", "title": r.get("page_title") or "",
            "url": r.get("page_url") or "", "ctx": (r.get("context_text") or "")[:220],
            "link": str(r.get("in_link")) == "True", "img": f"{rel_img}/{img[rid]}".replace(os.sep, "/"),
            "module": MODULES.get(h["module"], h["module"]) if h else "",
            "applied": bool(h and h.get("applied")), "before": h.get("before", "") if h else "",
        })
    counts = Counter(pred.values())
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    payload = json.dumps({"rows": rows, "labels": LABELS, "counts": {k: counts.get(k, 0) for k in LABELS}},
                         ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("__CSV__", html.escape(str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else csv_path.name))
    page = page.replace("__TOUR__", "true" if a.tour else "false").replace("__SHA__", sha).replace("__N__", str(len(rows))).replace("__DATA__", payload)
    out.write_text(page, encoding="utf-8")

    print(f"rows {len(rows)}  sha256 {sha}")
    for k in LABELS:
        print(f"  {k}{' ' * (10 - 2 * len(k))} {counts.get(k, 0):4d}")  # Hangul is double width in terminals
    print(f"harness rows {len(harness)} (log: {'yes' if harness else 'no'})")
    print(f"viewer: {out}")


TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KoAltQ 판정 결과</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--sub:#62625f;--line:#e2e2de;--accent:#2a6fdb;
--c0:#2e8b57;--c1:#8e44ad;--c2:#d35400;--c3:#2a6fdb;--c4:#b8860b;--c5:#c0392b;--c6:#5d6d7e}
@media (prefers-color-scheme:dark){:root{--bg:#141415;--card:#1f1f21;--ink:#ececea;--sub:#a3a3a0;--line:#333336;--accent:#6ea0ff;
--c0:#5cc28a;--c1:#c084e0;--c2:#f0904a;--c3:#6ea0ff;--c4:#e0b84a;--c5:#f07468;--c6:#9aa8b6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
header{padding:20px 16px 8px;max-width:1280px;margin:auto}h1{font-size:20px;margin:0 0 4px}
.meta{color:var(--sub);font-size:12px;word-break:break-all}
.bars{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:14px 0 6px}
.bar{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;cursor:pointer;text-align:left;color:var(--ink);font:inherit}
.bar[aria-pressed=true]{outline:2px solid var(--accent)}.bar b{font-size:18px}.bar i{display:block;height:4px;border-radius:2px;margin-top:6px}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
.tools input[type=search]{flex:1;min-width:200px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:inherit}
.tools label{color:var(--sub);font-size:13px}
main{max-width:1280px;margin:auto;padding:0 16px 40px;display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.img{height:150px;display:flex;align-items:center;justify-content:center;background:repeating-conic-gradient(var(--line) 0 25%,transparent 0 50%) 0 0/16px 16px}
.img img{max-width:100%;max-height:150px;object-fit:contain}
.body{padding:10px 12px;display:flex;flex-direction:column;gap:6px}
.badge{align-self:flex-start;color:#fff;border-radius:999px;padding:2px 10px;font-weight:600;font-size:13px}
.alt{font-size:14px}.alt q{quotes:"“" "”"}.empty{color:var(--sub);font-style:italic}
.small{color:var(--sub);font-size:12px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.why{font-size:12px;border-left:3px solid var(--accent);padding-left:8px}
.count{color:var(--sub);font-size:13px}
.cap{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:rgba(20,20,22,.88);color:#fff;padding:10px 18px;border-radius:999px;font-size:17px;font-weight:600;opacity:0;transition:opacity .3s;pointer-events:none;z-index:9}
.flash{outline:3px solid var(--accent)!important;outline-offset:2px;transition:outline .2s}
</style></head><body>
<header>
<h1>KoAltQ 판정 결과 뷰어</h1>
<div class="meta">파일 <b>__CSV__</b> · __N__행 · sha256 __SHA__</div>
<div class="meta">각 카드: 웹 이미지 · alt 텍스트 · 페이지 · 판정 라벨. 하네스(규칙)가 정하거나 바꾼 행은 그 근거를 함께 표시합니다.</div>
<div class="bars" id="bars"></div>
<div class="tools"><input type="search" id="q" placeholder="alt · 페이지 제목 · record_id 검색">
<label><input type="checkbox" id="h"> 하네스가 정한 행만</label><span class="count" id="count"></span></div>
</header>
<main id="grid"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const color=l=>`var(--c${D.labels.indexOf(l)})`;let sel=null;
const bars=document.getElementById('bars'),grid=document.getElementById('grid');
const mk=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
const total=D.rows.length;
[null,...D.labels].forEach(l=>{const b=mk('button','bar');b.setAttribute('aria-pressed',l===sel);
 const n=l?D.counts[l]:total;b.append(mk('span','',l||'전체'),document.createElement('br'),mk('b','',n));
 const i=mk('i');i.style.background=l?color(l):'var(--sub)';i.style.width=(100*n/total)+'%';b.append(i);
 b.onclick=()=>{sel=l;[...bars.children].forEach(x=>x.setAttribute('aria-pressed',x===b));draw()};bars.append(b)});
function card(r){const c=mk('article','card');const f=mk('div','img');const im=new Image();im.loading='lazy';im.alt='';im.src=r.img;f.append(im);
 const b=mk('div','body');const g=mk('span','badge',r.label);g.style.background=color(r.label);b.append(g);
 const a=mk('div','alt');a.append(mk('span','small','alt = '));if(r.alt){const q=mk('q','',r.alt);a.append(q)}else a.append(mk('span','empty','(빈 alt)'));b.append(a);
 b.append(mk('div','small',(r.link?'[링크 안 이미지] ':'')+r.title));if(r.ctx)b.append(mk('div','small','문맥: '+r.ctx));
 if(r.module){const w=mk('div','why',r.module+(r.applied?'':' → 적용 안 함')+(r.before&&r.before!==r.label?` (모델 제안 ${r.before} → ${r.label})`:''));b.append(w)}
 b.append(mk('div','small',r.id));c.append(f,b);return c}
function draw(){const q=document.getElementById('q').value.trim().toLowerCase(),h=document.getElementById('h').checked;
 const rs=D.rows.filter(r=>(!sel||r.label===sel)&&(!h||r.module)&&(!q||(r.alt+' '+r.title+' '+r.id).toLowerCase().includes(q)));
 grid.replaceChildren(...rs.map(card));document.getElementById('count').textContent=`${rs.length} / ${total}행 표시`}
document.getElementById('q').oninput=draw;document.getElementById('h').onchange=draw;draw();
// 자동 시연 (--tour 또는 #tour): 실제 버튼·체크박스·검색창을 순서대로 조작한다
const sleep=ms=>new Promise(r=>setTimeout(r,ms));const cap=mk('div','cap');document.body.append(cap);
const say=t=>{cap.textContent=t||'';cap.style.opacity=t?1:0};
const flash=el=>{el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),1100)};
const glide=(to,ms)=>new Promise(res=>{const s=window.scrollY,d=to-s,t0=performance.now();
 const f=n=>{const k=Math.min(1,(n-t0)/ms),e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;window.scrollTo(0,s+d*e);k<1?requestAnimationFrame(f):res()};requestAnimationFrame(f)});
const bar=l=>[...bars.children].find(b=>b.firstChild.textContent===l);
async function tour(){
 await sleep(1500);say('평가 데이터 600행의 판정 결과 · 카드마다 이미지, alt 텍스트, 판정 라벨');await sleep(3000);
 await glide(1500,4500);await sleep(1000);await glide(0,1500);
 for(const l of ['불충분형','중복형','파일명형']){const b=bar(l);say(`'${l}'로 판정된 행만 보기`);flash(b);b.click();await sleep(2200);await glide(700,2200);await sleep(700);await glide(0,900)}
 const all=bar('전체');say('전체 보기');flash(all);all.click();await sleep(1300);
 const h=document.getElementById('h');say('하네스(규칙)가 정하거나 바꾼 행과 그 근거 (H0~H5)');flash(h.parentElement);h.click();await sleep(2500);
 await glide(1800,5500);await sleep(1500);await glide(0,1300);
 const q=document.getElementById('q');h.click();say('alt 텍스트 검색: "QR"');flash(q);for(const ch of 'QR'){q.value+=ch;draw();await sleep(400)}await sleep(3000);
 q.value='';draw();say('');document.title='KoAltQ 판정 결과 (시연 끝)';
}
if(__TOUR__||location.hash==='#tour')tour();
</script></body></html>
"""

if __name__ == "__main__":
    main()
