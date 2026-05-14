"""
KangarooQwenModel: wraps Qwen2.5-VL base + TwigVLM-style adapter + LM head.

The adapter is a stack of N native Qwen2_5_VLDecoderLayer copies (default 3) +
its own norm + its own lm_head, deepcopied from
    base.model.layers[exit_layer..exit_layer + N - 1],
    base.model.norm,
    base.lm_head.

At inference time:
  - draft uses adapter (its own lm_head)
  - verify still uses base.lm_head (exposed as `self.head_model`)

Usage:
    model = KangarooQwenModel(
        base_model_path='Qwen/Qwen2.5-VL-3B-Instruct',
        adapter_model_path='path/to/adapter/checkpoint',  # optional
        early_exit_layer=8,
        num_adapter_layers=3,
        dtype=torch.bfloat16,
    )
"""

import json
import os

import torch
import torch.nn as nn

from adapter import AdapterModel
from earlyexit_qwen import EarlyExitQwen2_5_VLForConditionalGeneration


class KangarooQwenModel(nn.Module):

    def __init__(
        self,
        base_model_path: str,
        adapter_model_path: str = None,
        early_exit_layer: int = 8,
        num_adapter_layers: int = 3,
        dtype=torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        self.early_exit_layer = early_exit_layer

        from model import Qwen2_5_VLForConditionalGeneration

        raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).eval()

        self.base_model = EarlyExitQwen2_5_VLForConditionalGeneration(
            raw_model, early_exit_layer=early_exit_layer,
        )

        # Resolve num_adapter_layers + exit_layer from adapter_config.json if a
        # checkpoint is provided. Older checkpoints (no num_adapter_layers field)
        # are rejected — retrain with current code.
        adapter_meta = {}
        adapter_ckpt = None
        if adapter_model_path is not None:
            adapter_meta_path = os.path.join(adapter_model_path, "adapter_config.json")
            if os.path.exists(adapter_meta_path):
                with open(adapter_meta_path, "r", encoding="utf-8") as f:
                    adapter_meta = json.load(f)
            adapter_ckpt = os.path.join(adapter_model_path, "adapter_model.bin")

        if "exit_layer" in adapter_meta and adapter_meta["exit_layer"] != early_exit_layer:
            raise ValueError(
                f"Adapter checkpoint exit_layer={adapter_meta['exit_layer']} does not match "
                f"inference early_exit_layer={early_exit_layer}"
            )

        if adapter_ckpt is not None and os.path.exists(adapter_ckpt):
            if "num_adapter_layers" not in adapter_meta:
                raise ValueError(
                    f"Adapter checkpoint at {adapter_model_path} has no 'num_adapter_layers' in "
                    f"adapter_config.json. Please retrain with the current TwigVLM-style code."
                )
            ckpt_n = adapter_meta["num_adapter_layers"]
            if ckpt_n != num_adapter_layers:
                raise ValueError(
                    f"Adapter checkpoint num_adapter_layers={ckpt_n} does not match "
                    f"constructor argument num_adapter_layers={num_adapter_layers}"
                )

        # Build adapter by deepcopying from base. This installs the right
        # module types (Qwen2_5_VLDecoderLayer, Qwen2RMSNorm, Linear) with the
        # correct shapes and base-pretrained init. Loading a saved checkpoint
        # below simply overwrites these tensors.
        self.adapter_model = AdapterModel(
            base_model=raw_model,
            exit_layer=early_exit_layer,
            num_adapter_layers=num_adapter_layers,
        )

        if adapter_ckpt is not None and os.path.exists(adapter_ckpt):
            state_dict = torch.load(adapter_ckpt, map_location="cpu", weights_only=True)
            cleaned = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
                cleaned[new_key] = v
            missing, unexpected = self.adapter_model.load_state_dict(cleaned, strict=False)
            # rotary_emb buffers are non-persistent → they aren't in the saved
            # state_dict and that's fine; remove them from the "missing" report.
            missing = [k for k in missing if not k.startswith("rotary_emb.")]
            if missing:
                raise ValueError(
                    f"Adapter checkpoint missing {len(missing)} keys: {missing[:10]}"
                )
            if unexpected:
                raise ValueError(
                    f"Adapter checkpoint has {len(unexpected)} unexpected keys: {unexpected[:10]}"
                )
            print(f"Loaded adapter weights from {adapter_ckpt}")
        elif adapter_model_path is not None:
            print(
                f"Warning: adapter checkpoint not found at {adapter_ckpt}; "
                f"using base-layer-deepcopy initialization."
            )

        self.adapter_model = self.adapter_model.eval().to(raw_model.device).to(dtype)

        # Verify path still uses base.lm_head; draft path uses adapter.lm_head.
        self.head_model = raw_model.lm_head

    @property
    def device(self):
        return self.base_model.device

    @property
    def config(self):
        return self.base_model.config

    def to(self, device):
        self.base_model.model.to(device)
        self.adapter_model.to(device)
        return self

    def forward(self):
        raise NotImplementedError("Use speculative decoding inference loop instead of direct forward")

    def reset_status(self):
        self.base_model.past_key_values = None
        if hasattr(self.base_model.model, "rope_deltas"):
            self.base_model.model.rope_deltas = None
        if hasattr(self.base_model.model, "reset_status"):
            self.base_model.model.reset_status()


if __name__ == "__main__":
    model = KangarooQwenModel(
        base_model_path="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt",
        adapter_model_path=None,
        early_exit_layer=8,
        num_adapter_layers=3,
        dtype=torch.bfloat16,
    )
    print("KangarooQwenModel initialized successfully")
    print(model)
