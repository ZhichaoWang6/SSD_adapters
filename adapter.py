"""
TwigVLM-style adapter for Qwen2.5-VL self-speculative decoding.

Architecture (TwigVLM on Qwen):
    hidden@exit_layer → N x Qwen2_5_VLDecoderLayer (native, flash-attn)
                       → twig_norm (Qwen2RMSNorm)
                       → twig_head (Linear)
                       → logits

All adapter parameters are trainable. At construction time we deep-copy from
the base model:
    adapter.layers[i] = base.model.layers[exit_layer + i]   (layer_idx remapped)
    adapter.norm      = base.model.norm
    adapter.lm_head   = base.lm_head
    adapter.rotary_emb = base.model.rotary_emb

This mirrors TwigVLM's approach: the draft path has its own lm_head and final
norm (trainable, not shared with base.lm_head). Training uses pure CE on
ground-truth next-token labels, not KL distillation against teacher logits.
"""

import copy
import json
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import DynamicCache


def _build_causal_4d_mask(
    seq_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a (1, 1, Q, K) additive causal mask compatible with HF attention layers."""
    total_len = seq_len + past_len
    q_pos = torch.arange(seq_len, device=device).view(-1, 1) + past_len
    k_pos = torch.arange(total_len, device=device).view(1, -1)
    neg_inf = torch.finfo(dtype).min
    mask = torch.where(
        k_pos <= q_pos,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )
    return mask[None, None, :, :]


class AdapterModel(nn.Module):
    """TwigVLM-style adapter built on top of Qwen2.5-VL native decoder layers."""

    def __init__(
        self,
        base_model: nn.Module,
        exit_layer: int,
        num_adapter_layers: int,
    ):
        super().__init__()
        self.exit_layer = exit_layer
        self.num_adapter_layers = num_adapter_layers
        qwen = base_model.model  # Qwen2_5_VLModel inside Qwen2_5_VLForConditionalGeneration
        total_layers = len(qwen.layers)
        if exit_layer + num_adapter_layers > total_layers:
            raise ValueError(
                f"exit_layer={exit_layer} + num_adapter_layers={num_adapter_layers} "
                f"exceeds base model layers ({total_layers})."
            )

        # Deep-copy N decoder layers from base and remap layer_idx for the
        # adapter's own DynamicCache (slots 0..N-1).
        layers = []
        for i in range(num_adapter_layers):
            src = qwen.layers[exit_layer + i]
            new_layer = copy.deepcopy(src)
            if hasattr(new_layer.self_attn, "layer_idx"):
                new_layer.self_attn.layer_idx = i
            layers.append(new_layer)
        self.layers = nn.ModuleList(layers)
        self.norm = copy.deepcopy(qwen.norm)
        self.lm_head = copy.deepcopy(base_model.lm_head)
        # rotary_emb has no learnable params (inv_freq is persistent=False).
        self.rotary_emb = copy.deepcopy(qwen.rotary_emb)

        n_params = sum(p.numel() for p in self.parameters())
        n_layer_params = sum(p.numel() for p in self.layers.parameters())
        n_head_params = sum(p.numel() for p in self.lm_head.parameters())
        print(
            f"[AdapterModel] {num_adapter_layers} layers from base "
            f"[{exit_layer}..{exit_layer + num_adapter_layers - 1}] + norm + lm_head | "
            f"total {n_params/1e6:.1f}M params "
            f"(layers {n_layer_params/1e6:.1f}M, head {n_head_params/1e6:.1f}M)"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[DynamicCache]]:
        """Forward through adapter stack.

        Args:
            hidden_states: (B, L, D), output of base.layers[exit_layer-1] /
                input expected by base.layers[exit_layer].
            position_ids: (3, B, L) mRoPE 3D positions OR (B, L) which will be
                broadcast to 3 channels.
            attention_mask: 4D additive mask `(B|1, 1, L, L+past_len)`. If None,
                a fresh causal mask is built (suitable for training and for
                inference when no padding is needed).
            past_key_value: DynamicCache or None.
            cache_position: (L,) absolute positions for KV cache write slots.
            use_cache: whether to return updated cache.

        Returns:
            logits: (B, L, V)
            past_key_value: same instance (mutated) or None.
        """
        B, L, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        past_len = past_key_value.get_seq_length() if past_key_value is not None else 0

        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + L, device=device)

        if attention_mask is None and L > 1:
            attention_mask = _build_causal_4d_mask(L, past_len, dtype, device)

        # Normalize position_ids to (3, B, L). Native mRoPE expects 3 channels.
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, past_key_value


def save_adapter_config(adapter_config_dict: dict, save_dir: str) -> None:
    """Persist adapter_config.json next to adapter_model.bin."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_config_dict, f, indent=2)
