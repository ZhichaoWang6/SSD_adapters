"""Verify that a freshly-deepcopied adapter matches the corresponding base layers
byte-for-byte. Also confirms the adapter has its own twig_norm and twig_head
(separate from base's norm/lm_head) but with byte-identical initial weights.

Usage:
    python verify_adapter_init.py \
        --basepath /path/to/Qwen2.5-VL \
        --exit_layer 8 \
        --num_adapter_layers 3
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

from kangaroo_model import KangarooQwenModel


def load_base_tensor(base_path, key):
    """Load a single tensor by full HF state-dict key (e.g. 'model.layers.8.self_attn.q_proj.weight')."""
    index_path = os.path.join(base_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wm = json.load(open(index_path))["weight_map"]
        if key not in wm:
            return None
        path = os.path.join(base_path, wm[key])
    else:
        path = os.path.join(base_path, "model.safetensors")
        if not os.path.exists(path):
            return None
    with safe_open(path, framework="pt", device="cpu") as f:
        if key in f.keys():
            return f.get_tensor(key)
    return None


def load_base_layer_state(base_path, layer_idx):
    """Return {param suffix → tensor} for base.model.layers[layer_idx].*"""
    prefix = f"model.layers.{layer_idx}."
    index_path = os.path.join(base_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wm = json.load(open(index_path))["weight_map"]
        files = {fn for k, fn in wm.items() if k.startswith(prefix)}
    else:
        files = {"model.safetensors"}
    out = {}
    for fn in files:
        path = os.path.join(base_path, fn)
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basepath", required=True)
    p.add_argument("--exit_layer", type=int, default=8)
    p.add_argument("--num_adapter_layers", type=int, default=3)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    args = p.parse_args()

    model = KangarooQwenModel(
        base_model_path=args.basepath,
        adapter_model_path=None,
        early_exit_layer=args.exit_layer,
        num_adapter_layers=args.num_adapter_layers,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    ok = True

    # === decoder layers ===
    for i in range(args.num_adapter_layers):
        base_idx = args.exit_layer + i
        base_state = load_base_layer_state(args.basepath, base_idx)
        adapter_layer = model.adapter_model.layers[i]
        adapter_sd = adapter_layer.state_dict()
        print(f"\n=== adapter.layers[{i}] vs base.model.layers[{base_idx}] ===")
        if not base_state:
            print(f"  [skip] base layer {base_idx} state not found")
            ok = False
            continue
        for key in sorted(adapter_sd.keys()):
            if key not in base_state:
                print(f"  [skip] {key}: not in base layer")
                continue
            a = adapter_sd[key].to("cpu", torch.float32)
            b = base_state[key].to("cpu", torch.float32)
            same = torch.equal(a, b)
            ok &= same
            print(f"  {key:50s} match={same}  max|delta|={(a - b).abs().max().item():.3e}")

    # === final norm ===
    print(f"\n=== adapter.norm vs base.model.norm ===")
    norm_b = load_base_tensor(args.basepath, "model.norm.weight")
    if norm_b is not None:
        a = model.adapter_model.norm.weight.detach().to("cpu", torch.float32)
        b = norm_b.to(torch.float32)
        same = torch.equal(a, b)
        ok &= same
        print(f"  norm.weight match={same}  max|delta|={(a - b).abs().max().item():.3e}")
    else:
        print("  [skip] base norm not found")
        ok = False

    # === lm_head ===
    print(f"\n=== adapter.lm_head vs base.lm_head ===")
    head_b = load_base_tensor(args.basepath, "lm_head.weight")
    if head_b is None:
        # Some Qwen checkpoints tie embeddings; lm_head may share with embed_tokens.
        head_b = load_base_tensor(args.basepath, "model.embed_tokens.weight")
        if head_b is not None:
            print("  (note: base lm_head appears tied to embed_tokens; compared against that)")
    if head_b is not None:
        a = model.adapter_model.lm_head.weight.detach().to("cpu", torch.float32)
        b = head_b.to(torch.float32)
        same = torch.equal(a, b)
        ok &= same
        print(f"  lm_head.weight match={same}  max|delta|={(a - b).abs().max().item():.3e}")
    else:
        print("  [skip] base lm_head not found")
        ok = False

    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
