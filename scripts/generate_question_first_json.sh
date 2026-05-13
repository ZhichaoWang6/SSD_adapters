#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt}
INPUT_JSONL=${INPUT_JSONL:-/data/wangzhichao/projects/SSD_test/AAAA/data/egoexolearn-half_multi_half_single_question-2_sec_per_frame-sft.jsonl}
OUTPUT_JSON=${OUTPUT_JSON:-/data/wangzhichao/projects/SSD_test/AAAA/data/annotations/adapter/question_first_egoexolearn_ar.json}
IMAGE_ROOT=${IMAGE_ROOT:-/data/wangzhichao/datasets}
DEVICE=${DEVICE:-cuda:0}
EXIT_LAYER=${EXIT_LAYER:-12}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
START=${START:-0}
END=${END:-}
STRIP_EGO_TIME_SUFFIX=${STRIP_EGO_TIME_SUFFIX:-0}
FILL_MISSING_FRAMES=${FILL_MISSING_FRAMES:-0}
IMAGES_PER_USER_TURN=${IMAGES_PER_USER_TURN:-0}

EXTRA_ARGS=(--start "$START")
if [ -n "$END" ]; then
    EXTRA_ARGS+=(--end "$END")
fi
if [ "$STRIP_EGO_TIME_SUFFIX" = "1" ]; then
    EXTRA_ARGS+=(--strip_ego_time_suffix)
fi
if [ "$FILL_MISSING_FRAMES" = "1" ]; then
    EXTRA_ARGS+=(--fill_missing_frames)
fi
if [ "$IMAGES_PER_USER_TURN" != "0" ]; then
    EXTRA_ARGS+=(--images_per_user_turn "$IMAGES_PER_USER_TURN")
fi

python -u generate_streaming_ar_from_sft.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_json "$OUTPUT_JSON" \
    --model_path "$MODEL_PATH" \
    --generate_replies \
    --question_first \
    --split_questions \
    --image_root "$IMAGE_ROOT" \
    --strip_prefix ./data/datasets \
    --device "$DEVICE" \
    --manual_ar \
    --exit_layer "$EXIT_LAYER" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${EXTRA_ARGS[@]}"
