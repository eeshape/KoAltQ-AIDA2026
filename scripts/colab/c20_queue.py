# [큐 실행기] 밤샘 실험 큐를 백그라운드로 실행 (GPU 메모리/선행작업 기반 자동 투입, queue.json 수정 가능)
import os, shutil, subprocess
def runner_pids():
    pids = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if len(args) >= 2 and args[0].endswith(b"python3") or (args and args[0].endswith(b"python")):
                if b"/content/cells/queue_runner.py" in args:
                    pids.append(int(pid))
    return pids
if not os.path.exists("/content/queue.json"):
    shutil.copy("/content/cells/queue.json", "/content/queue.json")
if runner_pids():
    print("already running:", runner_pids())
else:
    log = open("/content/runs/logs/queue_runner.log", "a")
    p = subprocess.Popen(["python", "-u", "/content/cells/queue_runner.py"], cwd="/content", stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print("started queue runner pid", p.pid)
