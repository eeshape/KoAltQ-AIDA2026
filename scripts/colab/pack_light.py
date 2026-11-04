# [결과 회수] /content/runs의 지표·예측·로그·제출 CSV만 압축(특징 행렬·어댑터 가중치 제외) -> 로컬로 다운로드해 독립 재채점
import os, zipfile
out = "/content/koaltq_runs_light.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("/content/runs"):
        dirs[:] = [d for d in dirs if not d.startswith("adapter")]
        for f in files:
            if f.startswith("feats") or f.endswith((".safetensors", ".bin", ".pt")):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, "/content"))
            n += 1
print("packed", n, "files", os.path.getsize(out), "bytes ->", out)
