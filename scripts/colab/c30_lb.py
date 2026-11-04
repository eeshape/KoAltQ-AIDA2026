# [LB 추정] 리더보드 방식(0.5*이진F1+0.5*macroF1) 예상 점수 순위표 — test 600건 시뮬레이션
import subprocess
r = subprocess.run(["bash","-lc","cd /content && PYTHONPATH=/content/code python -m koaltq2.run lb"], capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-1500:])
