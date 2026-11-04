# [X0] 환경 확인 + 데이터 번들 재조립(22조각, sha256) + 데이터 계약 검사
import subprocess
r = subprocess.run(["bash", "-lc", "cd /content && mkdir -p runs/logs && unzip -o -q koaltq2_code.zip -d /content/code && "
                    "PYTHONPATH=/content/code python -m koaltq2.run x0 2>&1 | tee /content/runs/logs/x0.log"],
                   capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
