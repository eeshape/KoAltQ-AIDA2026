"""Overnight GPU job queue for the single Colab session (memory-aware, dependency-aware, editable while running).

/content/queue.json : list of jobs {tag, args | shell, gpu_gb, after:[tags], max_parallel}
/content/runs/queue_status.json : {done:[...], failed:[...], running:{tag:pid}, updated}
A job counts as done when its process exits 0; failed jobs are recorded and NOT retried automatically.
External (already running) jobs can be declared done by listing them in queue.json with "external": true.
"""

import json
import os
import re
import subprocess
import time

Q = "/content/queue.json"
S = "/content/runs/queue_status.json"
LOGS = "/content/runs/logs"
os.makedirs(LOGS, exist_ok=True)


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return float(out[0]) / 1024, float(out[1]) / 1024


def log_done(tag):
    p = f"{LOGS}/{tag}.log"
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()[-4000:]
    if re.search(r"\[run\] \S+ DONE in", txt):
        return True
    if re.search(r"\[run\] \S+ FAILED after", txt):
        return False
    return None


status = json.load(open(S)) if os.path.exists(S) else {"done": [], "failed": []}
done, failed = set(status.get("done", [])), set(status.get("failed", []))
running = {}
while True:
    try:
        q = json.load(open(Q))
    except Exception as e:  # noqa: BLE001
        print("queue read error", e, flush=True)
        time.sleep(30)
        continue
    for j in q:  # external jobs: completion inferred from their log file
        if j.get("external") and j["tag"] not in done | failed:
            st = log_done(j["tag"])
            if st is True:
                done.add(j["tag"])
            elif st is False:
                failed.add(j["tag"])
    for tag, p in list(running.items()):
        rc = p.poll()
        if rc is not None:
            (done if rc == 0 else failed).add(tag)
            running.pop(tag)
            print(time.strftime("%H:%M:%S"), "finished", tag, "rc", rc, flush=True)
    pending = [j for j in q if not j.get("external") and j["tag"] not in done | failed and j["tag"] not in running]
    used, total = gpu_mem()
    for j in pending:
        deps = j.get("after", [])
        if any(d in failed for d in deps) and not j.get("run_even_if_dep_failed"):
            failed.add(j["tag"])
            print("skip (dependency failed)", j["tag"], flush=True)
            continue
        soft = j.get("run_even_if_dep_failed")
        if not all(d in done or (soft and d in failed) for d in deps):  # soft deps: a failed dependency also unblocks
            continue
        if len(running) >= j.get("max_parallel", 3):
            continue
        if used + j.get("gpu_gb", 0) > total - 2.5:
            continue
        cmd = j.get("shell") or f"cd /content && PYTHONPATH=/content/code python -u -m koaltq2.run {j['args']}"
        p = subprocess.Popen(["bash", "-lc", f"{cmd} > {LOGS}/{j['tag']}.log 2>&1"])
        running[j["tag"]] = p
        print(time.strftime("%H:%M:%S"), "launched", j["tag"], "| gpu used", round(used, 1), "GB", flush=True)
        time.sleep(90)  # let the new job allocate before admitting another one
        break
    json.dump({"done": sorted(done), "failed": sorted(failed), "running": {t: p.pid for t, p in running.items()},
               "pending": [j["tag"] for j in pending if j["tag"] not in running], "gpu_used_gb": round(used, 1),
               "updated": time.strftime("%H:%M:%S")}, open(S, "w"), indent=1)
    time.sleep(30)
