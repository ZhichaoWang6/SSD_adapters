#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_DIR=/data/wangzhichao/projects/SSD_test/AAAA/datasets/2030_layers_no_reply_0.01
OUTPUT_DIR=/data/wangzhichao/projects/SSD_test/AAAA/adapter_checkpoints/offline_30_mrope_kl+ce_re
EXIT_LAYER=30

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_adapter.py \
    --basepath $MODEL_PATH \
    --datadir $DATA_DIR \
    --outdir $OUTPUT_DIR \
    --exit_layer $EXIT_LAYER \
    --lr 1e-5 \
    --bs 1 \
    --gradient_accumulation_steps 32 \
    --num_epochs 60 \
    --num_warmup_steps 30 \
    --kl_temperature 1.0 \
    --prefix_ce_tokens 6 \
    --prefix_ce_start 1 \
    --prefix_ce_decay 1.0 \
    --prefix_ce_weight 1.0 \
    --grad_clip 1.0 \
    --save_freq 1 

# !/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
# DATA_DIR=/data/wangzhichao/projects/SSD_test/datasets/81012_layers_no_reply_0.01
# OUTPUT_DIR=/data/wangzhichao/projects/SSD_test/adapter_checkpoints/test_offline_12_resume_weights_kl_prefix_ce
# EXIT_LAYER=12

# CUDA_VISIBLE_DEVICES=6 accelerate launch \
#     --num_processes 1 \
#     --num_machines 1 \
#     --mixed_precision bf16 \
#     train_adapter.py \
#     --basepath $MODEL_PATH \
#     --datadir $DATA_DIR \
#     --outdir $OUTPUT_DIR \
#     --exit_layer $EXIT_LAYER \
#     --lr 3e-6 \
#     --bs 1 \
#     --init_from_base_layer $EXIT_LAYER \
#     --gradient_accumulation_steps 32 \
#     --num_epochs 10 \
#     --resume_adapter /data/wangzhichao/projects/SSD_test/adapter_checkpoints/test_offline_12_resume_weights/epochs/epoch012_acc0.8179_accept0.4960_longconf0.2930_loss1.3326/adapter_model.bin \
#     --num_warmup_steps 30 \
#     --kl_temperature 1.0 \
#     --prefix_ce_tokens 4 \
#     --prefix_ce_start 1 \
#     --prefix_ce_decay 0.8 \
#     --prefix_ce_weight 1.0 \
#     --grad_clip 0.5 \
#     --save_freq 1 

       
