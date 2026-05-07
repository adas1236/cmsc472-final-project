#!/usr/bin/env bash
# N-GPU sharded synthetic-image generation against the LLaVA prompts.
#
# GPU selection order:
#   1. $CUDA_VISIBLE_DEVICES if set (e.g. "0,2,3")
#   2. otherwise: every GPU reported by `nvidia-smi -L`
# Cap with `NUM_GPUS=K` to use only the first K of the selected GPUs.
#
# Usage: bash scripts/gen_data_voc_llava_ngpu.sh

set -e

SEED=${SEED:-1111}
sd_path="sd2-community/stable-diffusion-2-1-base"
batch_size=1
self_res=32
cross_res=16
work_dir=data/gen_voc_llava
json_path=data/prompts/voc_llava_prompts_debug.json

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

# Subprocesses set their own CUDA_VISIBLE_DEVICES, so unset the parent's mask
# to avoid double-mapping (e.g. parent CUDA_VISIBLE_DEVICES=2,3 + child =3).
unset CUDA_VISIBLE_DEVICES

total=$(python -c "import json; print(len(json.load(open('$json_path'))))")
per_gpu=$(( (total + n - 1) / n ))
echo "Sharding $total prompts across $n GPU(s) (~$per_gpu per shard)"

pids=()
for i in "${!gpu_ids[@]}"; do
    gpu=${gpu_ids[$i]}
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))
    if [[ $end -gt $total ]]; then end=$total; fi
    if [[ $start -ge $total ]]; then break; fi
    echo "  GPU $gpu -> [$start, $end)"
    CUDA_VISIBLE_DEVICES=$gpu python -m gen_data \
        --work-dir $work_dir \
        --json-path $json_path \
        --sd-path $sd_path \
        --batch-size $batch_size \
        --self-res $self_res \
        --cross-res $cross_res \
        --seed $SEED \
        --start $start --end $end &
    pids+=($!)
done

wait "${pids[@]}"
