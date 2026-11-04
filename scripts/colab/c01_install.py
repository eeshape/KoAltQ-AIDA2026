# [설치] transformers 5.22.1 / peft 0.21.2 / bitsandbytes 0.50.2 / accelerate / flash-linear-attention (Qwen3.5 DeltaNet 가속)
import subprocess
cmd = ("pip install -q 'transformers==5.22.1' 'peft==0.21.2' 'bitsandbytes==0.50.2' accelerate flash-linear-attention "
       "2>&1 | tail -5; python -c \"import transformers, peft, torch; print('tf', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)\"")
r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
