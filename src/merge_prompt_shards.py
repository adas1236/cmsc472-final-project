"""Concatenate per-shard prompt JSONs into a single prompts file.

Usage:
    python -m src.merge_prompt_shards \
        --inputs data/prompts/voc_llava_prompts_shard0.json data/prompts/voc_llava_prompts_shard1.json \
        --out data/prompts/voc_llava_prompts.json
"""

import argparse
import json
import os
import os.path as osp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    merged = []
    seen = set()
    for path in args.inputs:
        with open(path) as f:
            shard = json.load(f)
        for r in shard:
            key = r["filename"]
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        print(f"  {path}: +{len(shard)} (total unique: {len(merged)})")

    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(merged)} records to {args.out}")


if __name__ == "__main__":
    main()
