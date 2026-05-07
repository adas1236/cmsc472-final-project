#!/usr/bin/env bash
# N-GPU sharded LLaVA-NeXt caption run over VOC train + val.
# 1464 train + 1449 val = 2913 images, sharded across all visible GPUs.
# After all shards finish, the per-shard JSONs are merged.
#
# GPU selection:
#   1. $CUDA_VISIBLE_DEVICES if set
#   2. otherwise: every GPU reported by `nvidia-smi -L`
#   3. cap with NUM_GPUS=K
# Reproducibility: SEED=K (default 1111).
#
# Usage: bash scripts/gen_llava_captions_ngpu.sh

set -e

SEED=${SEED:-1111}
total=2913
out=data/prompts/voc_llava_prompts.json

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t gpu_ids < <(nvidia-smi -L | awk -F'[ :]' '{print $2}')
else
    echo "no GPUs detected: nvidia-smi not on PATH and CUDA_VISIBLE_DEVICES not set" >&2
    exit 1
fi

if [[ -n "$NUM_GPUS" ]]; then
    gpu_ids=("${gpu_ids[@]:0:$NUM_GPUS}")
fi

n=${#gpu_ids[@]}
if [[ "$n" -lt 1 ]]; then
    echo "no GPUs detected" >&2
    exit 1
fi

unset CUDA_VISIBLE_DEVICES

per_gpu=$(( (total + n - 1) / n ))
echo "Sharding $total images across $n GPU(s) (~$per_gpu per shard, seed=$SEED)"

shards=()
pids=()
for i in "${!gpu_ids[@]}"; do
    gpu=${gpu_ids[$i]}
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))
    if [[ $end -gt $total ]]; then end=$total; fi
    if [[ $start -ge $total ]]; then break; fi
    shard_path="data/prompts/voc_llava_prompts_shard${i}.json"
    shards+=("$shard_path")
    echo "  GPU $gpu -> [$start, $end) -> $shard_path"
    CUDA_VISIBLE_DEVICES=$gpu python -m src.generate_llava_captions \
        --start $start --end $end \
        --out $shard_path \
        --seed $SEED &
    pids+=($!)
done

wait "${pids[@]}"

python -m src.merge_prompt_shards --inputs "${shards[@]}" --out $out
