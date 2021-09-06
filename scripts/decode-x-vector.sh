#!/bin/bash
data_scp=/export/data/test/eval2000-sorted.scp

pynndir=/home/user/pynn2020/src  # pointer to pynn

# export environment variables
export PYTHONPATH=$pynndir

export CUDA_VISIBLE_DEVICES=3

pythonCMD="python -u -W ignore"

mkdir -p hypos
$pythonCMD $pynndir/pynn/bin/decode_x_vector.py \
           --data-scp $data_scp \
           --model-dic "model/epoch-avg.dic" \
           --batch-size 64 --downsample 1 --mean-sub --fp16 --space '▁' \
           --format "txt" \
           2>&1 | tee decode-x-vector.log
