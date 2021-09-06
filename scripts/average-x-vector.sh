#!/bin/bash
data_scp=/export/data/test/eval2000-sorted.scp

pynndir=/home/user/pynn2020/src  # pointer to pynn

# export environment variables
export PYTHONPATH=$pynndir

pythonCMD="python -u -W ignore"

$pythonCMD $pynndir/pynn/bin/average_state.py \
           --model-path "model" --config "model.cfg" \
           --states "ALL" --save-all \
           2>&1 | tee average_x-vector.log
