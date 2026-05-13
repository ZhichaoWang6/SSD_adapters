"""
Generate training data for the Kangaroo adapter by collecting hidden states
from the full Qwen2.5-VL model on multimodal dialogue data.

Each assistant turn becomes one training sample:
    input  = system + user_0 + asst_0 + ... + user_k + asst_k
    target = assistant content tokens in asst_k

Saved tensors:
- input_ids
- loss_mask
- hidden_state_layer{N}
- hidden_state
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adapter training data per assistant turn")
    parser.add_argument("--model_path", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    parser.add_argument("--data_path", type=str, default="/data/wangzhichao/projects/SSD/SSD3/datasets/egoexolearn_train.json")
    parser.add_argument("--output_dir", type=str, default="/data/wangzhichao/projects/SSD_RE/datasets/training_data/10_no_reply/")
    parser.add_argument("--exit_layers", type=str, default="2,3,4",
                        help="Comma-separated list of exit layers to save hidden states for")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=32768,
                        help="Skip turns whose tokenized length exceeds this value")
    parser.add_argument("--window_max_context_tokens", type=int, default=0,
                        help="If > 0, select a training context window at or below this token budget "
                             "before appending the target assistant turn.")
    parser.add_argument("--window_keep_user_turns", type=int, default=0,
                        help="If > 0, keep at most this many recent user turns, plus assistant turns between them, "
                             "before token-budget windowing.")
    parser.add_argument("--window_preserve_question", action="store_true",
                        help="Preserve the first text-bearing user turn as a fixed question anchor during windowing. "
                             "Use this for question-first streaming data.")
    parser.add_argument("--window_verbose", action="store_true",
                        help="Print selected window metadata for saved samples.")
    parser.add_argument("--skip_no_reply", action="store_true",
                        help='Skip assistant turns whose content is exactly "NO REPLY"')
    parser.add_argument("--no_reply_keep_ratio", type=float, default=0.1,
                        help="Keep ratio for NO REPLY samples. 1.0 keeps all, 0.0 keeps none.")
    parser.add_argument("--gpu", type=str, default="cuda:6")
    return parser.parse_args()


def load_data(data_path):
    if data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_assistant_text(turn: dict) -> str:
    content = turn.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return " ".join(texts).strip()
    return ""


def turn_has_text(turn: dict) -> bool:
    content = turn.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                return True
    return False


def build_loss_mask(full_input_ids, context_input_ids, tokenizer):
    """
    Mark only the current assistant reply content tokens with 1.

    The assistant chat-template header and trailing <|im_end|> are excluded.
    """
    loss_mask = torch.zeros_like(full_input_ids[0], dtype=torch.float32)

    full_ids = full_input_ids[0]
    full_len = full_ids.shape[0]
    context_len = context_input_ids.shape[1]
    if context_len >= full_len:
        return loss_mask

    ids = full_ids.tolist()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    start = context_len
    end = full_len

    content_start = start
    while content_start < end:
        prefix = tokenizer.decode(ids[start:content_start + 1], skip_special_tokens=False)
        if "\n" in prefix:
            content_start += 1
            break
        content_start += 1

    while end > content_start and ids[end - 1] == im_end_id:
        end -= 1

    for idx in range(content_start, end):
        loss_mask[idx] = 1.0

    return loss_mask


def slice_turns(messages: list):
    """
    Yield (context, assistant_turn, turn_idx) pairs.

    Context for turn k is:
        system + user_0 + asst_0 + ... + user_k
    """
    prefix = [messages[0]] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(prefix):]

    turn_idx = 0
    history = list(prefix)
    i = 0
    while i < len(rest) - 1:
        user_turn = rest[i]
        asst_turn = rest[i + 1]
        if user_turn["role"] != "user" or asst_turn["role"] != "assistant":
            i += 1
            continue

        history.append(user_turn)
        yield list(history), asst_turn, turn_idx

        history.append(asst_turn)
        turn_idx += 1
        i += 2


def split_system(history: list):
    if history and history[0].get("role") == "system":
        return [history[0]], history[1:]
    return [], history


def split_question_anchor(body: list, preserve_question: bool):
    if not preserve_question:
        return [], body, 0
    for idx, turn in enumerate(body):
        if turn.get("role") == "user" and turn_has_text(turn):
            return [turn], body[idx + 1:], idx
    return [], body, 0


def limit_by_user_turns(body: list, keep_user_turns: int):
    if keep_user_turns <= 0:
        return body, 0
    user_indices = [idx for idx, turn in enumerate(body) if turn.get("role") == "user"]
    if len(user_indices) <= keep_user_turns:
        return body, 0
    start = user_indices[-keep_user_turns]
    return body[start:], start


def candidate_start_indices(body: list):
    starts = [idx for idx, turn in enumerate(body) if turn.get("role") != "assistant"]
    if not starts and body:
        starts = [len(body) - 1]
    if not starts:
        starts = [0]
    return sorted(set(starts))


def build_context_inputs(processor, history: list):
    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs, int(inputs["input_ids"].shape[1])


def select_context_window(
    processor,
    context: list,
    max_context_tokens: int = 0,
    keep_user_turns: int = 0,
    preserve_question: bool = False,
    verbose: bool = False,
):
    system_turns, body = split_system(context)
    question_turns, body, question_prefix_dropped = split_question_anchor(body, preserve_question)
    body, user_turn_dropped = limit_by_user_turns(body, keep_user_turns)

    def make_candidate(start: int):
        candidate = system_turns + question_turns + body[start:]
        inputs, context_len = build_context_inputs(processor, candidate)
        return candidate, context_len

    if max_context_tokens <= 0:
        candidate = system_turns + question_turns + body
        _, context_len = build_context_inputs(processor, candidate)
        meta = {
            "window_context_len": context_len,
            "window_dropped_turns": question_prefix_dropped + user_turn_dropped,
            "window_question_prefix_dropped": question_prefix_dropped,
            "window_user_turn_dropped": user_turn_dropped,
            "window_budget_turn_dropped": 0,
            "window_total_turns": len(context),
            "window_selected_turns": len(candidate),
            "window_budget": 0,
            "window_keep_user_turns": keep_user_turns,
            "window_preserved_question": len(question_turns),
        }
        return candidate, meta

    starts = candidate_start_indices(body)
    cache = {}

    def get_candidate(start: int):
        if start not in cache:
            cache[start] = make_candidate(start)
        return cache[start]

    best_start = starts[-1]
    best_candidate, best_len = get_candidate(best_start)

    lo, hi = 0, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start = starts[mid]
        candidate, context_len = get_candidate(start)
        if context_len <= max_context_tokens:
            best_start = start
            best_candidate = candidate
            best_len = context_len
            hi = mid - 1
        else:
            lo = mid + 1

    dropped = question_prefix_dropped + user_turn_dropped + best_start
    meta = {
        "window_context_len": best_len,
        "window_dropped_turns": dropped,
        "window_question_prefix_dropped": question_prefix_dropped,
        "window_user_turn_dropped": user_turn_dropped,
        "window_budget_turn_dropped": best_start,
        "window_total_turns": len(context),
        "window_selected_turns": len(best_candidate),
        "window_budget": max_context_tokens,
        "window_keep_user_turns": keep_user_turns,
        "window_preserved_question": len(question_turns),
    }

    if verbose:
        print(
            f"  [window] context_len={best_len} budget={max_context_tokens} "
            f"selected_turns={len(best_candidate)}/{len(context)} dropped_turns={dropped} "
            f"preserved_question={len(question_turns)}"
        )
    if best_len > max_context_tokens:
        print(
            f"  [window] WARNING smallest window context_len={best_len} "
            f"> budget={max_context_tokens}"
        )

    return best_candidate, meta


@torch.no_grad()
def process_turn(model, processor, context: list, asst_turn: dict, exit_layers: list, max_seq_len: int,
                 window_meta: dict | None = None):
    full_history = context + [asst_turn]

    context_text = processor.apply_chat_template(context, tokenize=False, add_generation_prompt=False)
    full_text = processor.apply_chat_template(full_history, tokenize=False, add_generation_prompt=False)

    image_inputs, video_inputs = process_vision_info(full_history)

    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    context_inputs = processor(
        text=[context_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    seq_len = inputs.input_ids.shape[1]
    if seq_len > max_seq_len:
        print(f"  [SKIP] seq_len={seq_len} exceeds max_seq_len={max_seq_len}")
        return None

    forward_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "second_per_grid_ts": inputs.get("second_per_grid_ts"),
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
        "drop_method": "none",
        "drop_threshold": 1.0,
        "drop_absolute": True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    try:
        outputs = model(**forward_kwargs)
    except Exception as e:
        print(f"  [ERROR] forward pass failed: {e}")
        return None

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    loss_mask = build_loss_mask(inputs["input_ids"], context_inputs["input_ids"], tokenizer)
    if int(loss_mask.sum().item()) == 0:
        print("  [SKIP] loss_mask is empty")
        return None

    result = {
        "input_ids": inputs["input_ids"].cpu()[0],
        "loss_mask": loss_mask.cpu(),
        "context_len": int(context_inputs["input_ids"].shape[1]),
        "seq_len": int(seq_len),
        "answer_tokens": int(loss_mask.sum().item()),
    }
    if window_meta:
        result.update(window_meta)

    # Compute 3D mRoPE position_ids using base.get_rope_index.
    # Shape: (3, batch=1, seq_len) → save as (3, seq_len).
    try:
        position_ids, _ = model.get_rope_index(
            inputs["input_ids"],
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            inputs.get("second_per_grid_ts"),
            inputs.get("attention_mask"),
        )
        result["position_ids"] = position_ids.cpu()[:, 0]  # (3, seq_len)
    except Exception as e:
        print(f"  [WARN] get_rope_index failed: {e}; saving without position_ids")

    for layer in exit_layers:
        if layer < len(outputs.hidden_states):
            result[f"hidden_state_layer{layer}"] = outputs.hidden_states[layer].float().cpu()[0]

    result["hidden_state"] = outputs.hidden_states[-1].float().cpu()[0]

    for key, value in result.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"  [SKIP] {key} contains NaN/Inf")
                return None

    return result


def main():
    args = parse_args()
    exit_layers = [int(x) for x in args.exit_layers.split(",") if x.strip()]

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval().to(args.gpu)

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)

    end = args.end if args.end is not None else len(data)
    data = data[args.start:end]
    print(f"Processing {len(data)} samples (index {args.start}..{end})")

    os.makedirs(args.output_dir, exist_ok=True)

    total_turns = 0
    total_saved = 0
    total_skipped = 0

    for sample_i, example in enumerate(tqdm(data)):
        global_idx = args.start + sample_i
        messages = example.get("messages", example.get("conversation", []))
        if not messages:
            continue

        for context, asst_turn, turn_idx in slice_turns(messages):
            total_turns += 1

            is_no_reply = get_assistant_text(asst_turn) == "NO REPLY"
            if args.skip_no_reply and is_no_reply:
                total_skipped += 1
                continue
            if is_no_reply and random.random() > args.no_reply_keep_ratio:
                total_skipped += 1
                continue

            window_meta = None
            if args.window_max_context_tokens > 0 or args.window_keep_user_turns > 0 or args.window_preserve_question:
                context, window_meta = select_context_window(
                    processor=processor,
                    context=context,
                    max_context_tokens=args.window_max_context_tokens,
                    keep_user_turns=args.window_keep_user_turns,
                    preserve_question=args.window_preserve_question,
                    verbose=args.window_verbose,
                )

            result = process_turn(
                model=model,
                processor=processor,
                context=context,
                asst_turn=asst_turn,
                exit_layers=exit_layers,
                max_seq_len=args.max_seq_len,
                window_meta=window_meta,
            )

            if result is None:
                total_skipped += 1
                continue

            save_path = os.path.join(args.output_dir, f"data_{global_idx}_turn{turn_idx}.ckpt")
            torch.save(result, save_path)
            total_saved += 1

    print("\nDone.")
    print(f"  Total turns : {total_turns}")
    print(f"  Saved       : {total_saved}")
    print(f"  Skipped     : {total_skipped}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Exit layers : {exit_layers}")


if __name__ == "__main__":
    main()
