# Smoke test: caption the first 32 train images with LLaVA-NeXt.
# Verifies model load, prompt template, and JSON schema before the full run.

CUDA_VISIBLE_DEVICES=0 python -m src.generate_llava_captions \
    --debug \
    --out data/prompts/voc_llava_prompts_debug.json
