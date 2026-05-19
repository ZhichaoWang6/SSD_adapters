"""
Online TwigVLM-style training for the Qwen2.5-VL self-speculative adapter.

Difference vs train_adapter.py (the offline variant):
    Offline: reads pre-computed .ckpt files (hidden_state_layer{K} cached on disk).
    Online (this file): reads raw jsonl + images, runs the frozen base model
                        layers[0..exit_layer-1] live each step, never caches.

Same architecture (TwigVLM on Qwen):
    base [0..K-1] (frozen)
        → hidden_states[K]
        → adapter.layers (N x Qwen2_5_VLDecoderLayer, trainable)
        → adapter.norm  (trainable)
        → adapter.lm_head (trainable)
        → logits
    Loss = CE on supervised assistant tokens.

Input format: a Qwen-structured jsonl produced by
    tools/concat_and_convert_jsonl.py. Each line:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": [{"type":"image","image":"/path/..."}, {"type":"text","text":"..."}]},
        {"role": "assistant", "content": "..."},
        ...
      ]
    }

For each conversation we yield one training sample per (context, assistant_turn)
pair (same slicing as generate_training_data.py), so the model is supervised on
one assistant reply at a time. NO REPLY turns can be filtered via
--skip_no_reply / --no_reply_keep_ratio.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from adapter import AdapterModel, save_adapter_config
from generate_training_data import build_loss_mask, get_assistant_text
from qwen_vl_utils import process_vision_info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basepath", required=True, help="Path to base Qwen2.5-VL checkpoint.")
    p.add_argument("--data_path", required=True, help="Path to Qwen-format jsonl.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--exit_layer", type=int, default=8)
    p.add_argument("--num_adapter_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=1, help="Only bs=1 supported (variable-length multimodal).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--num_warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=8192,
                   help="Skip samples whose tokenized length exceeds this.")
    p.add_argument("--num_workers", type=int, default=2,
                   help="DataLoader workers. Set 0 if processor fails to pickle.")
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--resume_adapter", type=str, default=None)
    p.add_argument("--skip_no_reply", action="store_true",
                   help="Drop assistant turns whose text is exactly 'NO REPLY'.")
    p.add_argument("--no_reply_keep_ratio", type=float, default=1.0,
                   help="Probability of keeping a NO REPLY sample (when not skipping all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return p.parse_args()


def _iter_assistant_turns(jsonl_path, skip_no_reply, no_reply_keep_ratio, rng):
    """Walk jsonl once; yield (conv, asst_idx) for every supervised assistant turn."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = conv.get("messages", [])
            has_prior_user = False
            for i, m in enumerate(messages):
                role = m.get("role")
                if role == "user":
                    has_prior_user = True
                    continue
                if role != "assistant":
                    continue
                if not has_prior_user:
                    continue
                is_no_reply = (get_assistant_text(m) == "NO REPLY")
                if skip_no_reply and is_no_reply:
                    continue
                if is_no_reply and rng.random() > no_reply_keep_ratio:
                    continue
                yield conv, i


class OnlineDataset(Dataset):
    def __init__(
        self, jsonl_path, processor,
        max_seq_len=8192, skip_no_reply=False, no_reply_keep_ratio=1.0, seed=42,
    ):
        rng = random.Random(seed)
        self.samples = list(_iter_assistant_turns(
            jsonl_path, skip_no_reply, no_reply_keep_ratio, rng,
        ))
        self.processor = processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.max_seq_len = max_seq_len
        print(f"[OnlineDataset] {len(self.samples)} (context, assistant) training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conv, asst_idx = self.samples[idx]
        messages = conv["messages"]
        context = messages[:asst_idx]
        asst_turn = messages[asst_idx]
        full_history = context + [asst_turn]

        try:
            context_text = self.processor.apply_chat_template(
                context, tokenize=False, add_generation_prompt=False,
            )
            full_text = self.processor.apply_chat_template(
                full_history, tokenize=False, add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(full_history)

            full_inputs = self.processor(
                text=[full_text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            context_inputs = self.processor(
                text=[context_text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        except Exception as e:
            print(f"[OnlineDataset] sample {idx} failed: {e}")
            return None

        full_ids = full_inputs["input_ids"]
        L = full_ids.shape[1]
        if L > self.max_seq_len:
            return None

        loss_mask = build_loss_mask(
            full_ids, context_inputs["input_ids"], self.tokenizer,
        )
        if loss_mask.sum().item() <= 0:
            return None

        item = {
            "input_ids": full_ids[0],
            "attention_mask": full_inputs["attention_mask"][0],
            "loss_mask": loss_mask,
        }
        for key in (
            "pixel_values", "image_grid_thw",
            "pixel_values_videos", "video_grid_thw",
            "second_per_grid_ts",
        ):
            if key in full_inputs and full_inputs[key] is not None:
                item[key] = full_inputs[key]
        return item


def collate_singleton(features):
    """bs=1 collator that drops None samples (oversize / unparseable / no-mask)."""
    features = [f for f in features if f is not None]
    if not features:
        return None
    f = features[0]
    out = {}
    for k, v in f.items():
        if isinstance(v, torch.Tensor) and v.dim() == 1:
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def compute_ce_loss(logits, input_ids, loss_mask):
    B, L, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().float()
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(B, L - 1)
    return (per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


@torch.no_grad()
def match_stats(logits, input_ids, loss_mask):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:].float()
    pred = shift_logits.argmax(dim=-1)
    correct = pred.eq(shift_labels).float()
    tc = (correct * shift_mask).sum().item()
    tt = shift_mask.sum().item()
    fc = ft = 0.0
    for b in range(shift_mask.shape[0]):
        positions = torch.nonzero(shift_mask[b] > 0, as_tuple=False).flatten()
        if positions.numel() > 0:
            ft += 1
            fc += correct[b, positions[0]].item()
    return tc, tt, fc, ft


def save_adapter(model, args, accelerator, tag):
    unwrapped = accelerator.unwrap_model(model)
    save_dir = os.path.join(args.outdir, tag)
    os.makedirs(save_dir, exist_ok=True)
    state_dict = {
        k: v for k, v in unwrapped.state_dict().items() if not k.startswith("rotary_emb.")
    }
    torch.save(state_dict, os.path.join(save_dir, "adapter_model.bin"))
    save_adapter_config({
        "exit_layer": args.exit_layer,
        "num_adapter_layers": args.num_adapter_layers,
        "training_recipe": "twigvlm_style_online_ce_on_labels",
    }, save_dir)
    print(f"  Saved [{tag}] to {save_dir}")


def _build_train_attention_mask(attention_mask_1d, dtype, device):
    """Combine causal mask + 1D padding mask into a 4D additive mask."""
    B, L = attention_mask_1d.shape
    neg_inf = torch.finfo(dtype).min
    q = torch.arange(L, device=device).view(-1, 1)
    k = torch.arange(L, device=device).view(1, -1)
    causal = torch.where(
        k <= q,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )[None, None, :, :]
    pad = (attention_mask_1d == 0).to(device=device)
    pad_mask = torch.where(
        pad,
        torch.full((), neg_inf, dtype=dtype, device=device),
        torch.zeros((), dtype=dtype, device=device),
    )[:, None, None, :]
    return causal + pad_mask


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.bs != 1:
        raise ValueError(
            f"--bs={args.bs} is not supported. Multimodal batching (variable-length "
            f"pixel_values / image_grid_thw) is not implemented in this script. "
            f"Use --bs 1 and raise --gradient_accumulation_steps to grow the effective "
            f"batch size (current effective batch = {args.gradient_accumulation_steps})."
        )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(args.outdir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.basepath)

    dataset = OnlineDataset(
        args.data_path, processor,
        max_seq_len=args.max_seq_len,
        skip_no_reply=args.skip_no_reply,
        no_reply_keep_ratio=args.no_reply_keep_ratio,
        seed=args.seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_singleton,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    if accelerator.is_main_process:
        print(f"Loading base model {args.basepath}")
    from model import Qwen2_5_VLForConditionalGeneration
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.basepath,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval()

    # Adapter is built by deepcopying from base BEFORE we truncate base layers.
    adapter = AdapterModel(
        base_model=base_model,
        exit_layer=args.exit_layer,
        num_adapter_layers=args.num_adapter_layers,
    )

    # Truncate base: only vision + embed + layers[0..K-1] + (unused) norm/lm_head remain.
    # We invoke base_model.model(...) (the inner Qwen2_5_VLModel) directly, which never
    # touches base_model.lm_head, so we can delete it for memory.
    del base_model.model.layers[args.exit_layer:]
    del base_model.lm_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_model.requires_grad_(False)
    base_model = base_model.eval().to(accelerator.device)

    if args.resume_adapter:
        sd = torch.load(args.resume_adapter, map_location="cpu", weights_only=True)
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()
        }
        missing, unexpected = adapter.load_state_dict(cleaned, strict=False)
        missing = [k for k in missing if not k.startswith("rotary_emb.")]
        if accelerator.is_main_process:
            print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")

    optimizer = optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9, 0.95))
    adapter, optimizer, train_loader = accelerator.prepare(adapter, optimizer, train_loader)

    updates_per_epoch = max(
        1,
        (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps,
    )
    total_steps = updates_per_epoch * args.num_epochs
    warmup_steps = min(args.num_warmup_steps, max(total_steps - 1, 0))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, 1),
    )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        eff_bs = args.bs * args.gradient_accumulation_steps * accelerator.num_processes
        print(f"Effective batch size: {eff_bs}")
        print(f"Updates per epoch: {updates_per_epoch}")
        print(f"Total optimizer steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print("Loss: CE on supervised next-token positions (online, no .ckpt cache)")

    global_optstep = 0
    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            print(f"=== Epoch {epoch} ===")
        epoch_loss = 0.0
        num_batches = 0
        tc = tt = fc = ft = 0.0
        adapter.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, data in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            if data is None:
                continue
            # Move tensors to device.
            data = {
                k: (v.to(accelerator.device) if isinstance(v, torch.Tensor) else v)
                for k, v in data.items()
            }

            with accelerator.accumulate(adapter):
                with torch.no_grad():
                    position_ids, _ = base_model.get_rope_index(
                        input_ids=data["input_ids"],
                        image_grid_thw=data.get("image_grid_thw"),
                        video_grid_thw=data.get("video_grid_thw"),
                        second_per_grid_ts=data.get("second_per_grid_ts"),
                        attention_mask=data.get("attention_mask"),
                    )
                    # Inner Qwen2_5_VLModel.forward does NOT accept
                    # second_per_grid_ts (only the outer ForCG class does).
                    # The temporal info already lives inside position_ids that
                    # we computed above, so we just don't pass it here.
                    base_kwargs = dict(
                        input_ids=data["input_ids"],
                        attention_mask=data.get("attention_mask"),
                        position_ids=position_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                    for k in ("pixel_values", "pixel_values_videos",
                              "image_grid_thw", "video_grid_thw"):
                        v = data.get(k)
                        if v is not None:
                            base_kwargs[k] = v
                    base_out = base_model.model(**base_kwargs)
                    # After del layers[K:], all_hidden_states has K+1 entries;
                    # index K = output of last remaining layer = input to (deleted) layer K.
                    hidden_states_early = base_out.hidden_states[args.exit_layer]

                attn_4d = _build_train_attention_mask(
                    data["attention_mask"],
                    hidden_states_early.dtype,
                    hidden_states_early.device,
                )

                logits, _ = adapter(
                    hidden_states=hidden_states_early,
                    position_ids=position_ids,
                    attention_mask=attn_4d,
                    use_cache=False,
                )
                loss = compute_ce_loss(logits, data["input_ids"], data["loss_mask"])

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                tc_, tt_, fc_, ft_ = match_stats(
                    logits.detach(), data["input_ids"], data["loss_mask"],
                )
                tc += tc_; tt += tt_; fc += fc_; ft += ft_
                epoch_loss += loss.detach().float().item()
                num_batches += 1

            if accelerator.sync_gradients:
                global_optstep += 1

            if accelerator.is_main_process and batch_idx % args.log_steps == 0:
                lr_now = scheduler.get_last_lr()[0]
                top1 = tc / tt if tt else 0.0
                first1 = fc / ft if ft else 0.0
                print(
                    f"Step: {batch_idx}\tOptStep: {global_optstep}\t"
                    f"LR: {lr_now:.3e}\tLoss: {loss.item():.4f}\t"
                    f"Top1: {top1:.4f}\tFirst: {first1:.4f}"
                )

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            top1 = tc / tt if tt else 0.0
            first1 = fc / ft if ft else 0.0
            print(
                f"Epoch {epoch} | avg_loss={avg_loss:.4f} | "
                f"top1={top1:.4f} | first_top1={first1:.4f}"
            )
            if (epoch + 1) % args.save_freq == 0:
                tag = f"epochs/epoch{epoch:03d}_top1{top1:.4f}_first1{first1:.4f}_loss{avg_loss:.4f}"
                save_adapter(adapter, args, accelerator, tag)


if __name__ == "__main__":
    main()
