"""Concat two streaming-VL jsonl files and convert to Qwen2.5-VL structured format.

Input format (per jsonl line, your current format):
{
  "messages": [
    {"role": "user",      "content": "<image><image>"},
    {"role": "assistant", "content": "NO REPLY"},
    {"role": "user",      "content": "<image>What checks are done?"},
    ...
  ],
  "images": [
    {"path": "./data/datasets/Ego-Exo4D/frames/.../000001.jpg"},
    {"path": "./data/datasets/Ego-Exo4D/frames/.../000003.jpg"},
    ...
  ]
}

Output format (Qwen-friendly, what generate_training_data.py + qwen_vl_utils want):
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/abs/path/000001.jpg"},
        {"type": "image", "image": "/abs/path/000003.jpg"}
      ]
    },
    {"role": "assistant", "content": "NO REPLY"},
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/abs/path/000005.jpg"},
        {"type": "text",  "text": "What checks are done?"}
      ]
    },
    ...
  ]
}

Path rewriting:
  ./data/datasets/Ego-Exo4D/frames/...      -> {ego4d_root}/...
  ./data/datasets/EgoExoLearn/frames_time/  -> {egoexolearn_root}/...

System message (if present at index 0) is preserved as-is.
"""

import argparse
import json
import os
import re
import sys

IMG_PLACEHOLDER = "<image>"
PLACEHOLDER_RE = re.compile(re.escape(IMG_PLACEHOLDER))


def _rewrite_path(p: str, ego4d_root: str, egoexolearn_root: str) -> str:
    """Map dataset-relative paths to absolute paths on this machine."""
    # Common prefixes from the source jsonl.
    if "Ego-Exo4D/frames/" in p:
        rel = p.split("Ego-Exo4D/frames/", 1)[1]
        return os.path.join(ego4d_root, rel)
    if "EgoExoLearn/frames_time/" in p:
        rel = p.split("EgoExoLearn/frames_time/", 1)[1]
        return os.path.join(egoexolearn_root, rel)
    # Anything else: treat as already-absolute or as-is.
    return p


def _split_user_content(text: str, image_iter) -> list:
    """Split a user-message text on <image> placeholders, interleaving images."""
    parts = PLACEHOLDER_RE.split(text)
    blocks = []
    # parts has len = (#placeholders + 1); image goes between consecutive parts.
    for i, segment in enumerate(parts):
        if i > 0:
            try:
                img_path = next(image_iter)
            except StopIteration:
                raise ValueError(
                    "Sample has more <image> placeholders than entries in 'images' list."
                )
            blocks.append({"type": "image", "image": img_path})
        seg = segment.strip()
        if seg:
            blocks.append({"type": "text", "text": seg})
    return blocks


def convert_sample(sample: dict, ego4d_root: str, egoexolearn_root: str) -> dict:
    """Transform one sample from streaming-VL format to Qwen structured format."""
    if "messages" not in sample or "images" not in sample:
        raise ValueError(f"Sample missing 'messages' or 'images' key: keys={list(sample.keys())}")

    image_paths = [
        _rewrite_path(item["path"], ego4d_root, egoexolearn_root)
        for item in sample["images"]
    ]
    image_iter = iter(image_paths)

    new_messages = []
    for msg in sample["messages"]:
        role = msg.get("role")
        content = msg.get("content", "")

        if role in ("system", "assistant"):
            # Keep system / assistant content as plain string.
            new_messages.append({"role": role, "content": content})
            continue

        if role == "user":
            if isinstance(content, list):
                # Already structured — don't double-convert.
                new_messages.append({"role": role, "content": content})
                continue
            blocks = _split_user_content(content, image_iter)
            if not blocks:
                # Empty user turn (just whitespace); keep as empty text to avoid
                # losing the turn boundary.
                blocks = [{"type": "text", "text": ""}]
            new_messages.append({"role": role, "content": blocks})
            continue

        # Unknown role: keep as-is.
        new_messages.append(msg)

    remaining = list(image_iter)
    if remaining:
        raise ValueError(
            f"{len(remaining)} unused image(s) in sample: e.g. {remaining[:3]}"
        )

    return {"messages": new_messages}


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [skip] {path}:{line_no}: JSON decode error: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ego4d_jsonl", required=True,
                   help="Path to Ego-Exo4D jsonl.")
    p.add_argument("--egoexolearn_jsonl", required=True,
                   help="Path to EgoExoLearn jsonl.")
    p.add_argument("--output_jsonl", required=True,
                   help="Output combined jsonl path.")
    p.add_argument("--ego4d_root", default="/data/wangzhichao/datasets/Ego-Exo4D/frames",
                   help="Absolute root for Ego-Exo4D image frames.")
    p.add_argument("--egoexolearn_root", default="/data/wangzhichao/datasets/EgoExoLearn/frames_time",
                   help="Absolute root for EgoExoLearn image frames.")
    p.add_argument("--check_files", action="store_true",
                   help="Verify each rewritten image path exists on disk (slower).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    counts = {"ego4d_in": 0, "egoexolearn_in": 0, "ok": 0, "skipped": 0, "missing_images": 0}

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for src_label, src_path in (
            ("ego4d", args.ego4d_jsonl),
            ("egoexolearn", args.egoexolearn_jsonl),
        ):
            print(f"[{src_label}] reading {src_path}")
            for sample in iter_jsonl(src_path):
                counts[f"{src_label}_in"] += 1
                try:
                    converted = convert_sample(
                        sample,
                        ego4d_root=args.ego4d_root,
                        egoexolearn_root=args.egoexolearn_root,
                    )
                except Exception as e:
                    print(f"  [skip {src_label}] {e}", file=sys.stderr)
                    counts["skipped"] += 1
                    continue

                if args.check_files:
                    missing = []
                    for msg in converted["messages"]:
                        c = msg.get("content")
                        if isinstance(c, list):
                            for b in c:
                                if b.get("type") == "image" and not os.path.exists(b["image"]):
                                    missing.append(b["image"])
                    if missing:
                        print(f"  [skip {src_label}] missing image(s): {missing[:2]}",
                              file=sys.stderr)
                        counts["missing_images"] += 1
                        continue

                out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                counts["ok"] += 1

    print()
    print(f"Ego-Exo4D samples in    : {counts['ego4d_in']}")
    print(f"EgoExoLearn samples in  : {counts['egoexolearn_in']}")
    print(f"Wrote                   : {counts['ok']} → {args.output_jsonl}")
    print(f"Skipped (parse / format): {counts['skipped']}")
    if args.check_files:
        print(f"Skipped (missing image) : {counts['missing_images']}")


if __name__ == "__main__":
    main()
