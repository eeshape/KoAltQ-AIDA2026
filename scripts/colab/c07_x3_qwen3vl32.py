# [X5] Qwen3-VL-32B(bf16, 80GB) 동결 특징 -> 3종 헤드
exec(open('/content/cells/launch.py').read())
launch("x3 --model qwen3vl_32b", "x3_qwen3vl_32b")
