#!/bin/bash
# Standard inference (baseline, no speculative decoding)
python -u inference.py \
        --use_speculative_decoding \
        --compare_AR_SSD \
        --test_fname /data/wangzhichao/projects/SSD_test/AAAA/data/annotations/2fps/ego_dataset.json \
        --output_fname ./outputs/offline_30_resume_60_0.6.jsonl \
        --device cuda:6 \
        --exit_layer 30 \
        --adapter_path /data/wangzhichao/projects/SSD_test/AAAA/adapter_checkpoints/offline_30_mrope_kl+ce_re/epochs/epoch059_top10.9178_draft10.9973_longdraft10.9967_overlap0.8296_loss0.4537 \
        --speculative_threshold 0.6 \
    > ./logs/test_offline_30_resume_60_0.6.log 2>&1
