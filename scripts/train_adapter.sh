#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_DIR=/data/wangzhichao/projects/SSD_test/AAAA/datasets/2030_layers_no_reply_0.01
OUTPUT_DIR=/data/wangzhichao/projects/SSD_test/AAAA/adapter_checkpoints/twigvlm_30_ce
EXIT_LAYER=30
NUM_ADAPTER_LAYERS=3

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_adapter.py \
    --basepath $MODEL_PATH \
    --datadir $DATA_DIR \
    --outdir $OUTPUT_DIR \
    --exit_layer $EXIT_LAYER \
    --num_adapter_layers $NUM_ADAPTER_LAYERS \
    --lr 1e-5 \
    --bs 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 60 \
    --num_warmup_steps 200 \
    --grad_clip 1.0 \
    --save_freq 1
