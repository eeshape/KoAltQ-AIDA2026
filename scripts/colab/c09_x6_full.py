# [X9] 최종 LoRA: 2,400 + 공식 109건으로 재학습 -> test (재로딩 검증 포함)
exec(open('/content/cells/launch.py').read())
launch("x6 --mode full --backbone qwen35_9b", "x6_qwen35_9b_full")
