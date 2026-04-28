# Qwen3.6-27B continued pre-training on US Customs rulings

Two scripts:

1. `prepare_dataset.py` — runs **once locally**, formats the merged 270,791-ruling NDJSON into CPT-ready text, computes the per-row token count under the Qwen3.6-27B tokenizer, and pushes the dataset to Hugging Face Hub.
2. `train.py` — runs **on the GPU VM**, pulls the dataset from HF, packs into 4096-token sequences, and continued-pretrains `Qwen/Qwen3.6-27B` with Unsloth.

## Source data

`us-rulings-merged.ndjson` (1.2 GB, 270,791 rulings). Each row has:

- `ruling_id`
- `processed`: bool — whether `processed_data` is present
- `raw`: object with `rulingNumber`, `docNo`, `title`, `references`, `classification` (list of HTS codes), `textContent` (the raw CBP ruling letter), and optionally `error`
- `processed_data` (only when `processed=true`): `date`, `short_product_description`, `full_description`, `reasoning`, `hts_code`

`prepare_dataset.py` uses `raw.textContent` as the body for **both** processed and unprocessed rows so the format is uniform across the corpus, and prepends a small header (`RULING <id> | <title> | HTS Chapter[s]: ...`) so the model has explicit anchors.

## Verified corpus stats (Qwen3.6-27B tokenizer)

```
Total rows:         270,791
Skipped:             11,503  (10,620 errored fetches + 883 empty textContent)
Usable docs:        259,288
Total tokens:   253,430,158
Mean tokens/doc:        977   (p50=704, p95=2,431, max=46,592)

processed=True:  122,862 docs, 131.3M tokens (51.81% of corpus)
processed=False: 136,426 docs, 122.1M tokens (48.19% of corpus)

Top 8 buckets by tokens:
  NO_CHAP        43,557 docs   59.4M  (23.43%)   no HTS classification
  MULTI_CHAP     25,458 docs   34.8M  (13.75%)   spans 2+ HTS chapters
  CHAP_61        29,994 docs   23.6M  ( 9.30%)   knit apparel
  CHAP_62        23,565 docs   19.9M  ( 7.87%)   woven apparel
  CHAP_85         8,927 docs    8.5M  ( 3.34%)   electrical
  CHAP_95        11,425 docs    8.4M  ( 3.30%)   toys/sport
  CHAP_42        10,853 docs    8.0M  ( 3.15%)   leather goods
  CHAP_84         7,138 docs    7.0M  ( 2.76%)   machinery
```

98 distinct single-HTS-chapter buckets in total, plus `MULTI_CHAP` (multi-chapter) and `NO_CHAP` (no classification — origin / valuation / drawback / 337 etc.).

Token-to-parameter ratio: 253M / 27.78B = **0.009** (light-CPT regime).

## Why no chapter rebalancing by default

37% of usable rulings sit outside a single HTS chapter (20% have no classification, 21% span multiple chapters). Trying to balance chapters would force you to drop or arbitrarily remap the largest two buckets (`NO_CHAP` and `MULTI_CHAP`). `train.py` therefore defaults to `--alpha 1.0` (raw distribution). The temperature-scaled balancing knob is still there if you want to dampen the CHAP_61/62 apparel skew within the single-chapter rulings — just pass `--alpha 0.5` or `--alpha 0.3`.

## Step 1 — Local: prepare and upload the dataset (one-time)

```bash
pip install transformers datasets huggingface_hub
huggingface-cli login                                  # or: export HF_TOKEN=hf_...
python prepare_dataset.py --repo-id <yourname>/customs-rulings-cpt
# add --private if you don't want it public
```

End-to-end takes ~75 seconds (3 s read + 70 s tokenize + a couple of minutes to upload depending on bandwidth).

## Step 2 — VM: install + train

On a fresh GPU VM:

```bash
uv pip install -r requirements.txt
huggingface-cli login                                  # or: export HF_TOKEN=hf_...
python train.py \
    --hf-dataset   <yourname>/customs-rulings-cpt \
    --push-to-hub  <yourname>/qwen36-27b-cpt-customs
```

That single command:
1. Loads `unsloth/Qwen3.6-27B` (16-bit LoRA by default).
2. Pulls the prepared dataset from HF.
3. Tokenizes the corpus and EOS-packs into 4096-token sequences (~62,000 sequences).
4. Trains for 2 epochs with the CPT-tuned recipe:
   - LoRA `r=64, alpha=16`, rsLoRA, targets include `lm_head` + `embed_tokens`
   - `lr=2e-5` for non-embedding, `embedding_lr=2e-6` (decoupled, per Unsloth blog)
   - Cosine schedule, 5% warmup, 0.01 weight decay
   - `bf16`, `adamw_8bit`, gradient checkpointing = `"unsloth"`
5. Saves the LoRA adapter locally and pushes to HF Hub.

Expected throughput on H100-80GB: ~30 min/epoch, ~1 hr total (the corpus is 38% larger than before). On A100-80GB: ~1.5 hr/epoch.

## Smaller GPUs (24-48 GB)

```bash
python train.py \
    --hf-dataset   <yourname>/customs-rulings-cpt \
    --push-to-hub  <yourname>/qwen36-27b-cpt-customs \
    --load-in-4bit \
    --max-seq-length 2048 \
    --no-embedding-tuning            # optional, saves ~10GB more, hurts CPT quality
```

## Optional: enable chapter rebalancing

If you want to dampen the single-chapter skew (CHAP_61 + CHAP_62 ≈ 17% of tokens):

```bash
python train.py --hf-dataset ... --alpha 0.5     # mild
python train.py --hf-dataset ... --alpha 0.3     # standard mT5-style flattening
```

`train.py` will print the effective per-bucket token table at startup so you can confirm the balance before committing.

## What it actually outputs

```
outputs/qwen36-27b-cpt-customs/
├── adapter_model.safetensors      # the LoRA weights
├── adapter_config.json
├── tokenizer.json + special_tokens_map.json
└── checkpoint-*/                  # latest 3 saves
```

To use the adapter at inference time:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "<yourname>/qwen36-27b-cpt-customs",   # auto-resolves base + adapter
    max_seq_length = 8192,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)
```

## Why these defaults

- **Token regime**: 253M tokens / 27.78B params = **0.009 tokens per parameter** — light-CPT territory.
- **LoRA, not full FT**: Unsloth's docs say 16-bit LoRA on 27B = ~56 GB VRAM and full FT = ~4× that ≈ 224 GB. That doesn't fit on a single H200-141 GB. We do LoRA with `r=128, rsLoRA, on every linear layer + lm_head + embed_tokens`, which is the closest you get to FFT on a single GPU. Per *LoRA Learns Less and Forgets Less*, this configuration matches FFT loss curves on domain CPT.
- **Decoupled embedding LR**: per Unsloth's CPT blog, blindly training `lm_head`/`embed_tokens` at the same LR as adapter layers degrades the model. Use 10× smaller LR (`5e-6` vs `5e-5`).
- **`raw.textContent` as canonical body**: the CBP ruling letter already contains the legal reasoning. Using it for both processed and unprocessed rows gives a uniform format the model can learn, instead of mixing two text styles.
- **Bucket vocabulary**: 98 single-chapter buckets + `MULTI_CHAP` + `NO_CHAP`. Empty `classification` is **not** an error — it marks legitimate non-classification rulings (origin, valuation, drawback, 337 enforcement, etc.).
