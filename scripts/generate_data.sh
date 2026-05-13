#!/bin/bash
# Step 1: Generate training data for the Kangaroo adapter
# This collects hidden states from the full model on MMDuet2 multimodal data.

MODEL_PATH=/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt
DATA_PATH=/data/wangzhichao/projects/SSD_test/AAAA/data/annotations/adapter/all.jsonl
OUTPUT_DIR=/data/wangzhichao/projects/SSD_test/AAAA/datasets/2030_layers_no_reply_0.01
EXIT_LAYERS=20,30  # Comma-separated list of exit layers to save hidden states for


CUDA_VISIBLE_DEVICES=6 python generate_training_data.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --exit_layers $EXIT_LAYERS \
    --no_reply_keep_ratio 0.01 \
    --gpu cuda:0
