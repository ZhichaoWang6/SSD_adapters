"""
Demonstrate the rope_deltas bug in adapter inference path.

Compares the position_ids that the adapter ACTUALLY uses (its 1D fallback)
vs what it SHOULD use (matching base model with rope_deltas applied) on a
realistic multimodal context.

Run on your machine:
    cd /data/wangzhichao/projects/SSD_adapters
    python demo_rope_delta_bug.py

Expected output: shows that adapter's RoPE positions are off by ~rope_deltas
(thousands) when used with multimodal data, causing every attention score
to be wrong.
"""

import torch


def base_model_position_ids(layer_past_length, seq_length, rope_deltas, batch_size=1):
    """Mimics earlyexit_qwen.py:191-199 - the CORRECT mRoPE position computation."""
    if rope_deltas is not None:
        delta = layer_past_length + rope_deltas
    else:
        delta = layer_past_length
    position_ids = torch.arange(seq_length)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1) + delta
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    return position_ids


def adapter_fallback_position_ids(past_key_values_length, seq_length, batch_size=1):
    """Mimics adapter.py:343-349 - the BUGGY adapter fallback (no rope_deltas)."""
    position_ids = torch.arange(
        past_key_values_length, past_key_values_length + seq_length,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)
    return position_ids


def simulate_round(scenario_name, context_length, rope_deltas, draft_step):
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"  context_length = {context_length}")
    print(f"  rope_deltas    = {rope_deltas}")
    print(f"  draft_step     = {draft_step}  (after prefill + {draft_step} draft tokens)")
    print('='*70)

    # At draft step k, both base and adapter have processed `context_length + k` tokens.
    # They process one new token, so layer_past_length = context_length + k.
    layer_past = context_length + draft_step
    seq_len = 1  # one new token per draft step

    base_pos = base_model_position_ids(layer_past, seq_len, rope_deltas)
    adapter_pos = adapter_fallback_position_ids(layer_past, seq_len)

    print(f"\nBase model position_ids (used by lower + verify layers):")
    print(f"  shape = {tuple(base_pos.shape)}   (3D mRoPE)")
    print(f"  value = {base_pos[:, 0, 0].tolist()}  (T, H, W channels)")

    print(f"\nAdapter fallback position_ids (used by adapter draft):")
    print(f"  shape = {tuple(adapter_pos.shape)}   (1D, will auto-broadcast to T=H=W)")
    print(f"  value = {adapter_pos[0, 0].item()}")

    base_t = int(base_pos[0, 0, 0])
    adapter_t = int(adapter_pos[0, 0])
    diff = base_t - adapter_t
    print(f"\n  DIFFERENCE = {diff}   ", end="")
    if diff == 0:
        print("✓ OK (text-only path)")
    else:
        print(f"✗ BUG! Adapter uses RoPE position {adapter_t} but verify uses {base_t}")
        print(f"  → Adapter computes attention with position {adapter_t},")
        print(f"  → Verify recomputes upper layers expecting position {base_t},")
        print(f"  → Same Q/K projections produce DIFFERENT logits → low accept rate")


if __name__ == "__main__":
    # Scenario A: pure text chat (e.g., ShareGPT warm-up) - no images
    # rope_deltas is None or 0 because there are no image/video tokens to skip over.
    simulate_round(
        "Pure text (ShareGPT warm-up)",
        context_length=500,
        rope_deltas=torch.tensor(0),
        draft_step=0,
    )

    # Scenario B: multimodal (e.g., 88-frame video, 200 image_pad per frame)
    # rope_deltas is large positive integer reflecting the visual-token expansion.
    # For Qwen2.5-VL, rope_deltas is roughly = (image_pixels_height/14) * (image_pixels_width/14) / 4
    # minus 1 per image, summed over all images. For our scenario it's around 17600.
    simulate_round(
        "Multimodal (88 frames @ 200 image_pad each, like ego_dataset.json)",
        context_length=5500,
        rope_deltas=torch.tensor(17600),  # realistic for 88-frame video
        draft_step=0,
    )

    # Scenario C: same multimodal context, 3 draft steps in
    simulate_round(
        "Multimodal, 3 draft tokens in",
        context_length=5500,
        rope_deltas=torch.tensor(17600),
        draft_step=3,
    )
