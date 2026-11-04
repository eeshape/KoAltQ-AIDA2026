# [실행기] 셀 하나 = 실험 하나. 백그라운드(nohup)로 띄우고 로그로 진행 확인 (연결 끊김 대비, 중복 실행 방지)
import subprocess


def launch(cell_args: str, tag: str):
    log = f"/content/runs/logs/{tag}.log"
    running = subprocess.run(["bash", "-lc", f"pgrep -f 'koaltq2.run {cell_args}$' || true"],
                             capture_output=True, text=True).stdout.split()
    if running:
        print("already running (pid", running, ") ->", log)
        return
    cmd = f"cd /content && nohup env PYTHONPATH=/content/code python -u -m koaltq2.run {cell_args} > {log} 2>&1 &"
    subprocess.run(["bash", "-lc", "mkdir -p /content/runs/logs && " + cmd])
    print("launched:", cell_args, "->", log)
