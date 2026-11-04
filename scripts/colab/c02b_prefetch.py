# [가중치 사전 다운로드] 이후 셀(X3/X4/X5/X6)에서 쓸 모델을 미리 받아 GPU 대기 시간을 줄임
import subprocess
print(subprocess.run(["bash", "-lc", "df -h /content /root | tail -2; free -g | head -2"], capture_output=True, text=True).stdout)
code = ("from huggingface_hub import snapshot_download\n"
        "for rid in ['Qwen/Qwen3.5-9B', 'OpenGVLab/InternVL3_5-14B-HF', 'Qwen/Qwen3-VL-32B-Instruct']:\n"
        "    p = snapshot_download(rid)\n"
        "    print('done', rid, p, flush=True)\n")
open("/content/prefetch.py", "w").write(code)
subprocess.run(["bash", "-lc", "cd /content && nohup python -u prefetch.py > /content/runs/logs/prefetch.log 2>&1 &"])
print("prefetch launched -> /content/runs/logs/prefetch.log")
