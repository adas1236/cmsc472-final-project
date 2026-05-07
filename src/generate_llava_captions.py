"""Generate LLaVA-NeXt captions for PASCAL VOC images.

Output schema matches what `gen_data.py` expects:
    [{"filename": "...", "caption": "...", "class_prompt": "..."}, ...]

For each image we issue all three HNU-VPAI instruction prompts, yielding
three records per image (suffixed `_p0/_p1/_p2`).

Usage examples
--------------
# Smoke test on the first 32 train samples:
python -m src.generate_llava_captions --debug --out data/prompts/voc_llava_prompts_debug.json

# Single-GPU full run (train + val ⇒ ~2913 images, ~8739 records):
python -m src.generate_llava_captions

# Two-GPU shard (split the global index range, 1464 train + 1449 val = 2913):
CUDA_VISIBLE_DEVICES=0 python -m src.generate_llava_captions \
    --start 0 --end 1456 --out data/prompts/voc_llava_prompts_shard0.json
CUDA_VISIBLE_DEVICES=1 python -m src.generate_llava_captions \
    --start 1456 --end 2913 --out data/prompts/voc_llava_prompts_shard1.json
# Then concatenate the shard JSONs into data/prompts/voc_llava_prompts.json.
"""

import argparse
import json
import os
import os.path as osp
import re

import torch
from tqdm.auto import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.pascal_voc_preprocess import derive_class_prompt, load_voc_split
from src.seed_utils import seed_everything

CAPTION_INSTRUCTIONS = [
    "Write a concise caption for the image presented",
    "Provide a short caption for the given image",
    "Provide a brief caption of the given image",
]

ASSISTANT_MARKERS = ("[/INST]", "ASSISTANT:")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="HF dataset splits to caption, in order. --start/--end span the concatenated stream.",
    )
    p.add_argument("--out", default="data/prompts/voc_llava_prompts.json")
    p.add_argument("--start", type=int, default=-1, help="Global start index across requested splits.")
    p.add_argument("--end", type=int, default=-1, help="Global end index (exclusive). -1 means all.")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--debug", action="store_true", help="Process only the first 32 samples after --start.")
    return p.parse_args()


def normalize_caption(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] in ".!?":
        text = text[:-1]
    return text.strip()


def generate_caption(processor, model, device, image, instruction, max_new_tokens) -> str:
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    full = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    for marker in ASSISTANT_MARKERS:
        if marker in full:
            full = full.rsplit(marker, 1)[-1]
    return normalize_caption(full)


def save(records, out_path):
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Loading {args.model} on {device} ...")
    processor = LlavaNextProcessor.from_pretrained(args.model)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    # Silence the "Setting pad_token_id to eos_token_id" warning that fires on every
    # generate() call when pad_token_id is unset (LLaVA-NeXt's default).
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    splits_cache = {}
    flat = []
    for split in args.splits:
        if split not in splits_cache:
            print(f"Loading split {split} ...")
            splits_cache[split] = load_voc_split(split)
        for i in range(len(splits_cache[split])):
            flat.append((split, i))
    print(f"Total samples across requested splits: {len(flat)}")

    start = max(0, args.start) if args.start != -1 else 0
    end = min(len(flat), args.end) if args.end != -1 else len(flat)
    if args.debug:
        end = min(start + 32, end)
    print(f"Processing global range [{start}, {end})  →  {end - start} images, up to {(end - start) * 3} records")

    records = []
    skipped = 0
    pbar = tqdm(range(start, end), desc="captioning", unit="img")
    for global_idx in pbar:
        split, local_idx = flat[global_idx]
        sample = splits_cache[split][local_idx]
        image = sample["image"].convert("RGB")
        class_prompt = derive_class_prompt(sample["mask"])
        if not class_prompt:
            skipped += 1
            pbar.set_postfix(records=len(records), skipped=skipped)
            continue
        for p_idx, instruction in enumerate(CAPTION_INSTRUCTIONS):
            caption = generate_caption(processor, model, device, image, instruction, args.max_new_tokens)
            if not caption:
                continue
            records.append({
                "filename": f"voc_{global_idx:05d}_p{p_idx}.jpg",
                "caption": f"{caption}; {class_prompt}",
                "class_prompt": class_prompt,
            })
        pbar.set_postfix(records=len(records), skipped=skipped)
        done = global_idx - start + 1
        if done % args.checkpoint_every == 0:
            save(records, args.out)

    save(records, args.out)
    print(f"Wrote {len(records)} records to {args.out} (skipped {skipped} images with empty class prompt)")


if __name__ == "__main__":
    main()
