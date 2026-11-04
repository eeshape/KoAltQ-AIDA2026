# [진행 확인] 실행 중 로그 tail + GPU 상태
import subprocess, glob, os
logs = sorted(glob.glob("/content/runs/logs/*.log"), key=os.path.getmtime)[-3:]
for lg in logs:
    print("=" * 20, os.path.basename(lg))
    print(subprocess.run(["bash", "-lc", f"tail -n 12 {lg}"], capture_output=True, text=True).stdout)
print(subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; "
                      "ps aux | grep 'koaltq2.run' | grep -v grep | awk '{print $2, $12, $13, $14, $15, $16}'"],
                     capture_output=True, text=True).stdout)
