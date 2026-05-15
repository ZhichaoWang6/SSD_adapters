#!/bin/bash
# Online TwigVLM-style training: read jsonl + images directly, no .ckpt cache.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_PATH=/data/wangzhichao/datasets/combined_qwen_format.jsonl
OUTPUT_DIR=/data/wangzhichao/projects/SSD-twin/adapter_checkpoints/online_8_ce
EXIT_LAYER=8
NUM_ADAPTER_LAYERS=3

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_adapter_online.py \
    --basepath $MODEL_PATH \
    --data_path $DATA_PATH \
    --outdir $OUTPUT_DIR \
    --exit_layer $EXIT_LAYER \
    --num_adapter_layers $NUM_ADAPTER_LAYERS \
    --lr 1e-5 \
    --bs 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 10 \
    --num_warmup_steps 200 \
    --max_seq_len 8192 \
    --num_workers 2 \
    --skip_no_reply \
    --no_reply_keep_ratio 0.01 \
    --grad_clip 1.0 \
    --save_freq 1
