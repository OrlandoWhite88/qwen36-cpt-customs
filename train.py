#!/usr/bin/env python3
"""Continued pre-training of Qwen3.6-27B on chapter-balanced US customs rulings.

Run this on a GPU VM after `prepare_dataset.py` has pushed the dataset to HF Hub:

    uv pip install -r requirements.txt
    huggingface-cli login          # or export HF_TOKEN=...
    python train.py --hf-dataset <yourname>/customs-rulings-cpt \
                    --push-to-hub <yourname>/qwen36-27b-cpt-customs

Default config is tuned for one 80GB GPU (H100/A100). For smaller GPUs:
  --load-in-4bit --max-seq-length 2048   # 24-48GB QLoRA
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import Counter
from typing import Any

# Eagerly import torch._inductor.config so that unsloth_zoo's import-time call to
# `inspect.getsource(torch._inductor.config)` works on torch>=2.4 where the submodule
# is no longer auto-attached to torch._inductor. Defensive: silently no-ops if torch
# isn't installed yet (e.g. when running --help on a fresh VM).
try:  # noqa: SIM105
    import torch._inductor.config  # noqa: F401
except Exception:
    pass


# ---------------- balancing --------------------------------------------------

def temperature_balanced_indices(
    rows: list[dict[str, Any]],
    alpha: float,
    max_repeat: int,
    budget_mult: float,
    seed: int,
) -> tuple[list[int], dict[str, float]]:
    """Sample (with replacement) document indices so per-bucket token counts match a
    temperature-scaled target distribution.

    target_b = B * t_b**alpha / sum_j t_j**alpha   (capped at max_repeat * t_b)

    Returns (shuffled list of indices into rows, dict of effective per-bucket tokens).
    """
    rng = random.Random(seed)

    bucket_to_indices: dict[str, list[int]] = {}
    bucket_tokens: Counter = Counter()
    for i, r in enumerate(rows):
        b = r["bucket"]
        bucket_to_indices.setdefault(b, []).append(i)
        bucket_tokens[b] += int(r["n_tokens"])

    total = sum(bucket_tokens.values())
    weights = {b: max(t, 1) ** alpha for b, t in bucket_tokens.items()}
    Z = sum(weights.values())
    B = total * budget_mult
    raw_targets = {b: B * weights[b] / Z for b in weights}

    # Apply per-bucket repetition cap and redistribute spillage proportionally.
    capped: dict[str, float] = {}
    spilled = 0.0
    for b, target in raw_targets.items():
        cap = bucket_tokens[b] * max_repeat
        if target > cap:
            spilled += target - cap
            capped[b] = float(cap)
        else:
            capped[b] = float(target)

    if spilled > 0:
        non_capped = [b for b in capped if capped[b] == raw_targets[b]]
        wnc = sum(weights[b] for b in non_capped)
        if wnc > 0:
            for b in non_capped:
                capped[b] += spilled * weights[b] / wnc

    sampled: list[int] = []
    for b, target in capped.items():
        idxs = bucket_to_indices[b][:]
        rng.shuffle(idxs)
        accumulated = 0.0
        i = 0
        while accumulated < target:
            if i >= len(idxs):
                rng.shuffle(idxs)
                i = 0
            sampled.append(idxs[i])
            accumulated += rows[idxs[i]]["n_tokens"]
            i += 1

    rng.shuffle(sampled)
    return sampled, capped


# ---------------- packing ----------------------------------------------------

def pack_token_lists(token_lists: list[list[int]], eos_id: int, max_seq_length: int) -> list[list[int]]:
    """Greedy EOS-joined packing into fixed max_seq_length blocks. Drops the trailing
    partial block (pads waste optimizer steps; the tail is small). Reset by EOS lets
    attention mask itself naturally."""
    packed: list[list[int]] = []
    buf: list[int] = []
    for ids in token_lists:
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= max_seq_length:
            packed.append(buf[:max_seq_length])
            buf = buf[max_seq_length:]
    return packed


# ---------------- main -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--hf-dataset", required=True,
                   help="HF dataset repo id with columns: ruling_id, bucket, n_tokens, text")
    g.add_argument("--alpha", type=float, default=1.0,
                   help="Balance temperature: 1.0 = use raw distribution (default), "
                        "0.3 = moderately flatten chapter buckets, 0.0 = uniform per bucket. "
                        "With the merged corpus (~37%% rulings have 0 or multi-chapter "
                        "classification) chapter weighting is mostly cosmetic; default 1.0.")
    g.add_argument("--max-repeat", type=int, default=50,
                   help="Maximum times any single document can be repeated when alpha<1.0 "
                        "(default 50).")
    g.add_argument("--budget-mult", type=float, default=1.0,
                   help="Total token budget = mult * raw corpus tokens (default 1.0).")
    g.add_argument("--seed", type=int, default=3407)

    g = p.add_argument_group("model")
    g.add_argument("--model-name", default="unsloth/Qwen3.6-27B")
    g.add_argument("--max-seq-length", type=int, default=4096)
    g.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA mode for 24-48GB GPUs (default off; 16-bit LoRA).")

    g = p.add_argument_group("lora")
    g.add_argument("--lora-r", type=int, default=128,
                   help="LoRA rank (default 128). Unsloth's published CPT recipe uses r=256 "
                        "on a 24GB L4; on an H200 r=128 is the sweet spot. Drop to 64 on 24-48GB.")
    g.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA alpha (default 32). With rsLoRA the effective scale is alpha/sqrt(r), "
                        "so 32/sqrt(128) ~= 2.83.")
    g.add_argument("--no-rslora", dest="use_rslora", action="store_false", default=True)
    g.add_argument("--no-embedding-tuning", dest="tune_embeddings", action="store_false", default=True,
                   help="Drop lm_head + embed_tokens from LoRA targets (saves VRAM, hurts CPT).")

    g = p.add_argument_group("optim")
    g.add_argument("--learning-rate", type=float, default=5e-5,
                   help="LoRA learning rate (default 5e-5; Unsloth's published CPT recipe).")
    g.add_argument("--embedding-learning-rate", type=float, default=5e-6,
                   help="10x smaller than LR for embed_tokens/lm_head (Unsloth recipe).")
    g.add_argument("--num-epochs", type=float, default=2.0)
    g.add_argument("--per-device-bs", type=int, default=1)
    g.add_argument("--grad-accum", type=int, default=16)
    g.add_argument("--warmup-ratio", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--lr-scheduler", default="cosine")
    g.add_argument("--max-steps", type=int, default=-1,
                   help="Cap optimizer steps (default -1 = use num-epochs).")

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="outputs/qwen36-27b-cpt-customs")
    g.add_argument("--push-to-hub", default=None,
                   help="HF repo id to push the trained LoRA adapter to (e.g. yourname/qwen36-27b-cpt-customs).")
    g.add_argument("--save-steps", type=int, default=500)
    g.add_argument("--logging-steps", type=int, default=10)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    # Lazy imports so --help works on a fresh VM before unsloth has been installed.
    from datasets import Dataset, load_dataset
    from unsloth import (
        FastLanguageModel,
        UnslothTrainer,
        UnslothTrainingArguments,
        is_bfloat16_supported,
    )

    print(f"=== Loading model {args.model_name} ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
    )

    # Qwen3.6-27B is a vision-language model; FastLanguageModel returns a
    # `Qwen3VLProcessor` whose `.tokenizer` attribute is the actual text
    # tokenizer. Fall back to `tokenizer` itself for plain (text-only) models.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    eos_id = text_tokenizer.eos_token_id
    print(f"  vocab={text_tokenizer.vocab_size}  eos={text_tokenizer.eos_token!r} ({eos_id})  "
          f"pad={text_tokenizer.pad_token!r}  bf16={is_bfloat16_supported()}")

    print(f"\n=== Loading HF dataset {args.hf_dataset} ===")
    raw = load_dataset(args.hf_dataset, split="train")
    print(f"  {len(raw):,} rows; columns={raw.column_names}")
    rows = raw.to_list()

    # Sanity: ensure required columns are present (forward-compatible with the old
    # processed-only schema and the new merged schema).
    required = {"ruling_id", "bucket", "n_tokens", "text"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}. "
                         f"Re-run prepare_dataset.py.")
    if args.alpha >= 1.0:
        print(f"  alpha={args.alpha} -> using raw distribution (no rebalancing).")

    print(f"\n=== Balanced sampling (alpha={args.alpha}, max_repeat={args.max_repeat}, "
          f"budget_mult={args.budget_mult}) ===")
    indices, eff_tokens = temperature_balanced_indices(
        rows, args.alpha, args.max_repeat, args.budget_mult, args.seed,
    )
    eff_total = sum(eff_tokens.values())
    eff_max, eff_min = max(eff_tokens.values()), min(eff_tokens.values())
    print(f"  Selected {len(indices):,} doc instances (with repeats)")
    print(f"  Effective budget: {eff_total:,.0f} tokens  (top:bottom ratio = {eff_max/eff_min:.1f}x)")

    print(f"  Top 8 buckets after balancing:")
    for b, t in sorted(eff_tokens.items(), key=lambda x: -x[1])[:8]:
        print(f"    {b:<22} {t:>14,.0f}  ({t/eff_total*100:5.2f}%)")
    print(f"  Bottom 4 buckets after balancing:")
    for b, t in sorted(eff_tokens.items(), key=lambda x: x[1])[:4]:
        print(f"    {b:<22} {t:>14,.0f}  ({t/eff_total*100:5.2f}%)")

    print(f"\n=== Tokenizing the sampled corpus ===")
    t0 = time.time()
    sampled_texts = [rows[i]["text"] for i in indices]
    BATCH = 2000
    token_lists: list[list[int]] = []
    for start in range(0, len(sampled_texts), BATCH):
        enc = text_tokenizer(
            sampled_texts[start:start + BATCH],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_lists.extend(enc["input_ids"])
    print(f"  {len(token_lists):,} docs tokenized in {time.time()-t0:.1f}s; "
          f"{sum(len(x) for x in token_lists):,} tokens")

    print(f"\n=== Packing into max_seq_length={args.max_seq_length} sequences ===")
    packed = pack_token_lists(token_lists, eos_id, args.max_seq_length)
    print(f"  {len(packed):,} packed sequences "
          f"({len(packed)*args.max_seq_length:,} usable tokens; "
          f"~{(1-len(packed)*args.max_seq_length/sum(len(x) for x in token_lists))*100:.1f}% trimmed at tail)")

    train_ds = Dataset.from_dict({
        "input_ids": packed,
        "labels":    [seq[:] for seq in packed],
    })

    print(f"\n=== Configuring LoRA "
          f"(r={args.lora_r}, alpha={args.lora_alpha}, rsLoRA={args.use_rslora}, "
          f"tune_embeddings={args.tune_embeddings}) ===")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    if args.tune_embeddings:
        target_modules += ["lm_head", "embed_tokens"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_rslora=args.use_rslora,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    n_train = len(train_ds)
    eff_bs = max(1, args.per_device_bs * args.grad_accum)
    n_steps_per_epoch = math.ceil(n_train / eff_bs)
    total_steps = (args.max_steps if args.max_steps > 0
                   else int(n_steps_per_epoch * args.num_epochs))
    print(f"\n=== Training ===")
    print(f"  Train sequences: {n_train:,}  effective batch: {eff_bs}")
    print(f"  Steps/epoch:     {n_steps_per_epoch:,}  total steps: {total_steps:,}")
    print(f"  LR={args.learning_rate}  embedding_LR={args.embedding_learning_rate}  "
          f"scheduler={args.lr_scheduler}  warmup_ratio={args.warmup_ratio}")

    train_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        embedding_learning_rate=args.embedding_learning_rate,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
    )
    if args.max_steps > 0:
        train_args_kwargs["max_steps"] = args.max_steps

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        max_seq_length=args.max_seq_length,
        args=UnslothTrainingArguments(**train_args_kwargs),
    )

    trainer.train()

    print(f"\n=== Saving LoRA adapter to {args.output_dir} ===")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"\n=== Pushing adapter to HF Hub: {args.push_to_hub} ===")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    print("\nDone.")


if __name__ == "__main__":
    main()
