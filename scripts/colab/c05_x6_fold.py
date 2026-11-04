# [X6] Qwen3.5-9B 판별 LoRA(분류헤드+logit adjustment+repeat-factor) folds1-4 학습 -> fold0+test
exec(open('/content/cells/launch.py').read())
launch("x6 --mode fold --backbone qwen35_9b", "x6_qwen35_9b_fold")
