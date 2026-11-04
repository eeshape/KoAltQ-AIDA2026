# [디스크 정리] 어떤 모델 스냅샷도 참조하지 않는 HF blob 파일만 제거 (blobs/ 하위 폴더까지 재귀 확인)
import os, subprocess
hub = "/root/.cache/huggingface/hub"
ref = set()
for m in os.listdir(hub):
    if m.startswith("models--"):
        for root, _d, files in os.walk(os.path.join(hub, m)):
            for f in files:
                p = os.path.join(root, f)
                if os.path.islink(p):
                    ref.add(os.path.realpath(p))
orph, keep_n = [], 0
for root, _d, files in os.walk(os.path.join(hub, "blobs")):
    for f in files:
        p = os.path.realpath(os.path.join(root, f))
        if p in ref:
            keep_n += 1
        else:
            orph.append(p)
size = sum(os.path.getsize(o) for o in orph)
print(f"referenced {len(ref)} | blob files kept {keep_n} | orphan files {len(orph)} = {size/1e9:.1f} GB")
inside = sum(1 for t in ref if t.startswith(hub + "/blobs"))
if len(ref) > 0 and all(os.path.exists(t) for t in ref) and keep_n == inside:
    for o in orph:
        os.remove(o)
    print("orphans removed")
else:
    print("SAFETY STOP: referenced/kept mismatch; nothing removed")
print(subprocess.run(["bash", "-lc", "df -h /root | tail -1"], capture_output=True, text=True).stdout)
