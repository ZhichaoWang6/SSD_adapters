#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt}
DATA_PATH=${DATA_PATH:-/data/wangzhichao/projects/SSD_test/AAAA/data/annotations/adapter/question_first_egoexolearn_ar.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/wangzhichao/projects/SSD_test/AAAA/datasets/question_first_window3500_layers}
EXIT_LAYERS=${EXIT_LAYERS:-8,10,12}
DEVICE=${DEVICE:-cuda:0}
WINDOW_MAX_CONTEXT_TOKENS=${WINDOW_MAX_CONTEXT_TOKENS:-3500}
WINDOW_KEEP_USER_TURNS=${WINDOW_KEEP_USER_TURNS:-0}
NO_REPLY_KEEP_RATIO=${NO_REPLY_KEEP_RATIO:-0.01}
START=${START:-0}
END=${END:-}

EXTRA_ARGS=(--start "$START")
if [ -n "$END" ]; then
    EXTRA_ARGS+=(--end "$END")
fi

python -u generate_training_data.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --exit_layers "$EXIT_LAYERS" \
    --no_reply_keep_ratio "$NO_REPLY_KEEP_RATIO" \
    --window_max_context_tokens "$WINDOW_MAX_CONTEXT_TOKENS" \
    --window_keep_user_turns "$WINDOW_KEEP_USER_TURNS" \
    --window_preserve_question \
    --gpu "$DEVICE" \
    "${EXTRA_ARGS[@]}"
