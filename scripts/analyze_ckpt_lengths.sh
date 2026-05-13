#!/bin/bash
DATA_DIR=/data/wangzhichao/projects/SSD_test/AAAA/datasets/81012_layers_no_reply_0.01

python analyze_ckpt_lengths.py \
    --datadir $DATA_DIR \
    --target_context_len 5500 \
    --context_buckets 0,512,1024,2048,4096,5500,6144,8192,12288,32768 \
    --output_json ./outputs/ckpt_length_summary.json
