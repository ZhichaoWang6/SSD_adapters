#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u inference_windows.py \
        --use_speculative_decoding \
        --compare_AR_SSD \
        --test_fname /data/wangzhichao/projects/SSD_test/AAAA/data/annotations/2fps/2fps_two_pic/ego_dataset.json \
        --output_fname ./outputs/window_1600_offline_12_mrope_0.6_60.jsonl \
        --device cuda:1 \
        --exit_layer 12 \
        --adapter_path /data/wangzhichao/projects/SSD_test/AAAA/adapter_checkpoints/offline_12_mrope_kl+ce_re/epochs/epoch060_top10.8985_draft11.0000_longdraft11.0000_overlap0.8407_loss0.3430 \
        --speculative_threshold 0.6 \
        --speculative_steps 6 \
        --window_max_context_tokens 1600 \
        --window_keep_user_turns 16 \
        --window_verbose \
    > ./logs/window_1600_offline_12_mrope_0.6_60.log 2>&1
