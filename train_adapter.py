"""
TwigVLM-style training for the Qwen2.5-VL self-speculative adapter.

Loss: pure cross-entropy on ground-truth next-token labels, masked by
loss_mask (only supervised assistant tokens contribute). No teacher
distillation — the adapter is initialized by deep-copying base layers
[exit_layer..exit_layer + N - 1] + base.norm + base.lm_head, then trained
to predict labels directly. Mirrors TwigVLM's training recipe.

Input data: the .ckpt files produced by generate_training_data.py.
Required keys:
    - hidden_state_layer{exit_layer}  (B, L, D)  – frozen-base hidden state at the exit
    - input_ids                       (L,)       – token ids (used to build labels)
    - loss_mask                       (L,)       – 1 on supervised target positions
    - position_ids                    (3, L)     – 3D mRoPE positions (optional)
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from adapter import AdapterModel, save_adapter_config


def parse_args():
    parser = argparse.ArgumentParser(description="TwigVLM-style training for Qwen2.5-VL adapter")
    parser.add_argument("--basepath", type=str, required=True,
                        help="Path to base Qwen2.5-VL checkpoint.")
    parser.add_argument("--datadir", type=str, required=True,
                        help="Directory of training .ckpt files.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--exit_layer", type=int, default=8)
    parser.add_argument("--num_adapter_layers", type=int, default=3,
                        help="Number of stacked decoder layers in the adapter.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--total_steps", type=int, default=0,
                        help="Total optimizer steps. <=0 to auto-compute.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="Path to a previously saved adapter_model.bin to continue training from.")
    parser.add_argument("--min_mask_tokens", type=int, default=0,
                        help="Skip samples whose loss_mask.sum() <= this.")
    parser.add_argument("--min_context_len", type=int, default=None,
                        help="Skip samples whose first supervised token starts before this index.")
    parser.add_argument("--max_context_len", type=int, default=None,
                        help="Skip samples whose first supervised token starts after this index.")
    parser.add_argument("--filter_cache", type=str, default=None,
                        help="Optional path to cache the filtered file list.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def list_files(path):
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".ckpt"):
                out.append(os.path.join(root, f))
    return sorted(out)


def _sample_lengths(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    loss_mask = data["loss_mask"].float()
    answer_positions = torch.nonzero(loss_mask > 0, as_tuple=False).flatten()
    answer_tokens = int(answer_positions.numel())
    context_len = int(answer_positions[0].item()) if answer_tokens > 0 else -1
    return context_len, answer_tokens


def filter_by_sample_metadata(
    files, min_mask_tokens=0, min_context_len=None, max_context_len=None, cache_path=None,
):
    if min_mask_tokens <= 0 and min_context_len is None and max_context_len is None:
        return files

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        cache_matches = (
            cached.get("min_mask_tokens") == min_mask_tokens
            and cached.get("min_context_len") == min_context_len
            and cached.get("max_context_len") == max_context_len
            and set(cached.get("source_files", [])) == set(files)
        )
        if cache_matches:
            print(f"[filter] loaded {len(cached['kept_files'])} files from cache {cache_path}")
            return cached["kept_files"]

    kept = []
    print(
        f"[filter] scanning {len(files)} ckpts "
        f"(min_mask_tokens > {min_mask_tokens}, "
        f"min_context_len={min_context_len}, max_context_len={max_context_len})..."
    )
    for i, f in enumerate(files):
        try:
            context_len, answer_tokens = _sample_lengths(f)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        if answer_tokens <= min_mask_tokens:
            continue
        if min_context_len is not None and context_len < min_context_len:
            continue
        if max_context_len is not None and context_len > max_context_len:
            continue
        kept.append(f)
        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(files)}, kept {len(kept)}")
    print(f"[filter] kept {len(kept)}/{len(files)} ({100*len(kept)/max(len(files),1):.1f}%)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "min_mask_tokens": min_mask_tokens,
                "min_context_len": min_context_len,
                "max_context_len": max_context_len,
                "source_files": files,
                "kept_files": kept,
            }, f)
    return kept


class AdapterDataset(Dataset):
    def __init__(self, datapath, exit_layer):
        self.data = datapath
        self.exit_layer = exit_layer
        self._key = f"hidden_state_layer{exit_layer}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], map_location="cpu", weights_only=False)

        hidden_state_early = data[self._key][None, :]   # (1, L, D)
        input_ids = data["input_ids"]                    # (L,)
        loss_mask = data["loss_mask"].float()            # (L,)

        L = hidden_state_early.shape[1]
        attention_mask = [1] * L

        # 3D mRoPE position_ids if saved (shape: (3, L)); pseudo-batch (3, 1, L).
        position_ids = data.get("position_ids", None)
        if position_ids is not None:
            position_ids = position_ids[:, None, :]

        return {
            "hidden_states_early": hidden_state_early,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


class DataCollatorWithPadding:
    @staticmethod
    def pad_hidden(t, target_len):
        # t: (1, L, D)
        cur_len = t.shape[1]
        if cur_len >= target_len:
            return t
        pad = torch.zeros(1, target_len - cur_len, t.shape[2], dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    @staticmethod
    def pad_1d(t, target_len, value=0):
        cur_len = t.shape[0]
        if cur_len >= target_len:
            return t
        pad = torch.full((target_len - cur_len,), value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    @staticmethod
    def pad_position_ids(t, target_len):
        # t: (3, 1, L)
        cur_len = t.shape[2]
        if cur_len >= target_len:
            return t
        pad = torch.zeros(3, 1, target_len - cur_len, dtype=t.dtype)
        return torch.cat([t, pad], dim=2)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(item["hidden_states_early"].shape[1] for item in features)
        out = {
            "hidden_states_early": torch.cat(
                [self.pad_hidden(item["hidden_states_early"], max_len) for item in features], dim=0
            ),
            "input_ids": torch.stack(
                [self.pad_1d(item["input_ids"], max_len, value=0) for item in features], dim=0
            ),
            "loss_mask": torch.stack(
                [self.pad_1d(item["loss_mask"], max_len, value=0.0) for item in features], dim=0
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] + [0] * (max_len - len(item["attention_mask"])) for item in features]
            ),
        }
        if all(f.get("position_ids") is not None for f in features):
            pos = [self.pad_position_ids(f["position_ids"], max_len) for f in features]
            out["position_ids"] = torch.cat(pos, dim=1)  # (3, B, max_len)
        else:
            out["position_ids"] = None
        return out


def save_adapter(model, args, accelerator, tag):
    unwrapped = accelerator.unwrap_model(model)
    save_dir = os.path.join(args.outdir, tag)
    os.makedirs(save_dir, exist_ok=True)
    # Strip rotary_emb buffers (non-persistent) from saved tensors so checkpoints
    # are smaller and don't carry redundant data.
    state_dict = {
        k: v for k, v in unwrapped.state_dict().items() if not k.startswith("rotary_emb.")
    }
    torch.save(state_dict, os.path.join(save_dir, "adapter_model.bin"))

    adapter_config_dict = {
        "exit_layer": args.exit_layer,
        "num_adapter_layers": args.num_adapter_layers,
        "training_recipe": "twigvlm_style_ce_on_labels",
    }
    save_adapter_config(adapter_config_dict, save_dir)
    print(f"  Saved [{tag}] to {save_dir}")


def compute_ce_loss(logits, input_ids, loss_mask):
    """Standard next-token CE, masked to supervised positions (loss_mask == 1).

    logits:    (B, L, V)
    input_ids: (B, L)
    loss_mask: (B, L)
    """
    B, L, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().float()

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(B, L - 1)
    loss = (per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)
    return loss


@torch.no_grad()
def compute_match_stats(logits, input_ids, loss_mask):
    """Top1 and first-draft top1 on supervised positions.

    "First draft" = the first supervised position (token after prefill).
    """
    B, L, V = logits.shape
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:].float()

    pred = shift_logits.argmax(dim=-1)
    correct = pred.eq(shift_labels).float()

    token_correct = (correct * shift_mask).sum().item()
    token_total = shift_mask.sum().item()

    first_correct = 0.0
    first_total = 0.0
    for b in range(B):
        positions = torch.nonzero(shift_mask[b] > 0, as_tuple=False).flatten()
        if positions.numel() > 0:
            first_total += 1
            first_correct += correct[b, positions[0]].item()

    return {
        "token_correct": token_correct,
        "token_total": token_total,
        "first_correct": first_correct,
        "first_total": first_total,
    }


def _ratio(num, den):
    return float(num) / float(den) if den else 0.0


def main():
    args = parse_args()
    set_seed(42)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # Filter samples
    all_files = list_files(args.datadir)
    if not all_files:
        raise ValueError(f"No .ckpt files found in {args.datadir}")
    kept = filter_by_sample_metadata(
        all_files,
        min_mask_tokens=args.min_mask_tokens,
        min_context_len=args.min_context_len,
        max_context_len=args.max_context_len,
        cache_path=args.filter_cache,
    )
    if accelerator.is_main_process:
        print(f"Using {len(kept)} samples for training")

    traindataset = AdapterDataset(kept, args.exit_layer)
    train_loader = DataLoader(
        traindataset, batch_size=args.bs, shuffle=True,
        collate_fn=DataCollatorWithPadding(), num_workers=0, pin_memory=False,
    )

    if accelerator.is_main_process:
        os.makedirs(args.outdir, exist_ok=True)

    # Build base model on this process so we can deepcopy layers into adapter.
    # Keep base on CPU during init; we only need its parameters for the copy.
    if accelerator.is_main_process:
        print(f"Loading base model from {args.basepath} for adapter init...")
    from model import Qwen2_5_VLForConditionalGeneration
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.basepath,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval()

    model = AdapterModel(
        base_model=base_model,
        exit_layer=args.exit_layer,
        num_adapter_layers=args.num_adapter_layers,
    )

    # Free the base model — we only needed it for the deepcopy init.
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.resume_adapter:
        state_dict = torch.load(args.resume_adapter, map_location="cpu", weights_only=True)
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        missing = [k for k in missing if not k.startswith("rotary_emb.")]
        if missing or unexpected:
            print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        if accelerator.is_main_process:
            print(f"Resumed from {args.resume_adapter}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    updates_per_epoch = max(
        1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    )
    total_training_steps = args.total_steps if args.total_steps > 0 else updates_per_epoch * args.num_epochs
    warmup_steps = min(args.num_warmup_steps, max(total_training_steps - 1, 0))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1),
    )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        effective_bs = args.bs * args.gradient_accumulation_steps * accelerator.num_processes
        print(f"Effective batch size: {effective_bs}")
        print(f"Updates per epoch: {updates_per_epoch}")
        print(f"Total optimizer steps: {total_training_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print("Loss: pure CE on supervised next-token positions (TwigVLM-style)")

    if args.start_epoch > 0 and not args.resume_adapter:
        state_dir = os.path.join(args.outdir, "state", f"state_{args.start_epoch - 1}")
        if os.path.exists(state_dir):
            accelerator.load_state(state_dir)
            print(f"Resumed accelerator state from {state_dir}")

    global_optstep = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        if accelerator.is_main_process:
            print(f"=== Epoch {epoch} ===")
        epoch_loss = 0.0
        num_batches = 0
        token_correct_acc = 0.0
        token_total_acc = 0.0
        first_correct_acc = 0.0
        first_total_acc = 0.0
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, data in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            with accelerator.accumulate(model):
                hidden_states_early = data["hidden_states_early"]
                input_ids = data["input_ids"]
                loss_mask = data["loss_mask"]
                position_ids = data.get("position_ids")
                attention_mask_1d = data["attention_mask"]  # (B, L), padding-only

                if position_ids is None:
                    # Fallback: text-style positions (3 channels equal).
                    B, L = input_ids.shape
                    pos = torch.arange(L, device=input_ids.device).view(1, -1).expand(B, -1)
                    position_ids = pos.unsqueeze(0).expand(3, -1, -1).contiguous()

                # Pass the 2D padding mask straight to the adapter. The inner
                # Qwen FlashAttention2 reads this and constructs cu_seqlens for
                # the varlen path internally. A 4D additive mask would crash
                # flash-attn.
                logits, _ = model(
                    hidden_states=hidden_states_early,
                    position_ids=position_ids,
                    attention_mask=attention_mask_1d,
                    past_key_value=None,
                    use_cache=False,
                )

                loss = compute_ce_loss(logits, input_ids, loss_mask)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                stats = compute_match_stats(logits.detach(), input_ids, loss_mask)
                token_correct_acc += stats["token_correct"]
                token_total_acc += stats["token_total"]
                first_correct_acc += stats["first_correct"]
                first_total_acc += stats["first_total"]
                epoch_loss += loss.detach().float().item()
                num_batches += 1

            if accelerator.sync_gradients:
                global_optstep += 1

            if (
                accelerator.is_main_process
                and (batch_idx % args.log_steps == 0)
            ):
                lr_now = scheduler.get_last_lr()[0]
                top1 = _ratio(token_correct_acc, token_total_acc)
                first_top1 = _ratio(first_correct_acc, first_total_acc)
                print(
                    f"Step: {batch_idx}\tOptStep: {global_optstep}\t"
                    f"LR: {lr_now:.3e}\tLoss: {loss.item():.4f}\t"
                    f"Top1: {top1:.4f}\tFirst: {first_top1:.4f}"
                )

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            top1 = _ratio(token_correct_acc, token_total_acc)
            first_top1 = _ratio(first_correct_acc, first_total_acc)
            print(f"Epoch {epoch} | avg_loss={avg_loss:.4f} | top1={top1:.4f} | first_top1={first_top1:.4f}")

            if (epoch - args.start_epoch + 1) % args.save_freq == 0:
                tag = f"epochs/epoch{epoch:03d}_top1{top1:.4f}_first1{first_top1:.4f}_loss{avg_loss:.4f}"
                save_adapter(model, args, accelerator, tag)

        # Per-epoch accelerator state snapshot
        if accelerator.is_main_process and (epoch - args.start_epoch + 1) % args.save_freq == 0:
            state_dir = os.path.join(args.outdir, "state", f"state_{epoch}")
            os.makedirs(state_dir, exist_ok=True)
            accelerator.save_state(state_dir)


if __name__ == "__main__":
    main()
