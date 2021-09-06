#!/bin/bash

data_dir=/home/sebastian.wichmann/speaker_diarization/sys-all
pynndir=/home/sebastian.wichmann/projects/speaker_diarization/src  # pointer to pynn

# export environment variables
export PYTHONPATH=$pynndir

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=3
export CUDA_LAUNCH_BLOCKING=1


pythonCMD="python -u -W ignore"
mkdir -p model

$pythonCMD $pynndir/pynn/bin/train_x_vector.py \
           --train-scp $data_dir/01-Feats/tr.scp --train-target $data_dir/02-Speakers/tr.spk \
           --valid-scp $data_dir/01-Feats/vl.scp --valid-target $data_dir/02-Speakers/vl.spk \
           --n-classes 704 --d-input 20 --d-enc 1500 --n-enc 6 \
           --use-cnn --freq-kn 3 --freq-std 2 --downsample 1 \
           --d-dec 512 --n-dec 2 --d-emb 512 --d-project 256 --n-head 1 \
           --enc-dropout 0.0 --enc-dropconnect 0.3 --dec-dropout 0.0 --dec-dropconnect 0.2 --emb-drop 0.15 \
           --teacher-force 1. --fp16 \
           --n-epoch 100 --lr 0.002 --b-input 640000 --b-update 360 --n-warmup 300 --n-const 0 2>&1 \
           | tee run-x-vector.log