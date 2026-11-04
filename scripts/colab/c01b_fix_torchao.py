# [환경 수정] Colab 기본 torchao 0.10.0을 PEFT 0.21.2가 비호환으로 판단(ImportError) -> 사용하지 않으므로 제거
import subprocess
r = subprocess.run(["bash", "-lc", "pip uninstall -y torchao 2>&1 | tail -1; python -c \"import peft; from peft import LoraConfig, get_peft_model; print('peft import ok', peft.__version__)\""],
                   capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:])
