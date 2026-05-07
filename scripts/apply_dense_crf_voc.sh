work_dir=${1:-data/gen_voc}
out_subdir=${2:-mask_crf}
num_workers=${3:-8}
SEED=${SEED:-1111}

python -m src.apply_dense_crf \
    --work-dir $work_dir \
    --out-subdir $out_subdir \
    --data-type voc \
    --num-workers $num_workers \
    --seed $SEED
