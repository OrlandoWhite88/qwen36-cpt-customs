#!/usr/bin/env python3
"""Prepare the US customs rulings CPT dataset and push to Hugging Face Hub.

Run this ONCE locally before kicking off training on a VM:

    python prepare_dataset.py --repo-id <yourname>/customs-rulings-cpt

Source: us-rulings-merged.ndjson (270,791 rulings, both processed and raw).

The resulting dataset has columns:
    - ruling_id  : str
    - bucket     : str   (CHAP_NN if exactly one HTS chapter, MULTI_CHAP if more, NO_CHAP if none)
    - chapters   : list  (all distinct HTS chapters seen on the ruling, in order)
    - processed  : bool  (True if the source row has LLM-distilled processed_data)
    - n_tokens   : int   (token count under the Qwen3.6-27B tokenizer)
    - text       : str   (formatted document used for CPT)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter


# ---------------- chapter / bucket extraction --------------------------------

def chapter_from_hts(hts: str) -> str:
    """Extract a 2-digit HTS chapter, stripping leading-letter special-program prefixes."""
    if not hts:
        return ""
    p = hts[:2]
    if p and p[0].isalpha():  # e.g. "A2905..." -> chapter 29
        p = hts[1:3]
    return p if p.isdigit() else ""


def chapters_from_classification(cls: list[str] | None) -> list[str]:
    """Distinct two-digit HTS chapters from a list of HTS codes (preserves order of first sight)."""
    seen: list[str] = []
    for c in cls or []:
        ch = chapter_from_hts(c or "")
        if ch and ch not in seen:
            seen.append(ch)
    return seen


def bucket_for(chapters: list[str]) -> str:
    if not chapters:
        return "NO_CHAP"
    if len(chapters) == 1:
        return f"CHAP_{chapters[0]}"
    return "MULTI_CHAP"


# ---------------- formatting -------------------------------------------------

def format_doc(d: dict) -> str | None:
    """Return the formatted document text, or None if the row is unusable.

    Uses raw.textContent as the body (the original CBP ruling letter) for both
    processed and unprocessed rows, so the format is uniform across the corpus.
    Adds a small header with ruling_id, title, and HTS chapter(s) so the model
    has explicit anchors to learn the chapter→content mapping.
    """
    raw = d.get("raw") or {}
    if "error" in raw:
        return None
    text_content = raw.get("textContent") or ""
    if not text_content.strip():
        return None

    chapters = chapters_from_classification(raw.get("classification"))
    title = (raw.get("title") or "").strip()

    head = f"RULING {d.get('ruling_id', '')}"
    if title:
        head += f" | {title}"
    if len(chapters) == 1:
        head += f" | HTS Chapter: {chapters[0]}"
    elif len(chapters) > 1:
        head += f" | HTS Chapters: {', '.join(chapters)}"

    return head + "\n\n" + text_content


# ---------------- main -------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="us-rulings-merged.ndjson",
                    help="Path to the source NDJSON file (default: us-rulings-merged.ndjson).")
    ap.add_argument("--repo-id", required=True,
                    help="HF dataset repo id, e.g. yourname/customs-rulings-cpt")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B",
                    help="Tokenizer used to count n_tokens.")
    ap.add_argument("--private", action="store_true",
                    help="Push to HF as a private dataset.")
    ap.add_argument("--no-push", action="store_true",
                    help="Build the dataset locally only (skip upload).")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    from transformers import AutoTokenizer
    from datasets import Dataset

    print(f"Loading tokenizer {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  vocab_size={tok.vocab_size}, eos={tok.eos_token!r}, pad={tok.pad_token!r}")

    print(f"\nReading + formatting {args.input} ...")
    t0 = time.time()
    rows: list[dict] = []
    skipped_error = 0
    skipped_empty = 0
    with open(args.input) as f:
        for line in f:
            d = json.loads(line)
            raw = d.get("raw") or {}
            if "error" in raw:
                skipped_error += 1
                continue
            text = format_doc(d)
            if text is None:
                skipped_empty += 1
                continue
            chapters = chapters_from_classification(raw.get("classification"))
            rows.append({
                "ruling_id": d.get("ruling_id", ""),
                "bucket": bucket_for(chapters),
                "chapters": chapters,
                "processed": bool(d.get("processed", False)),
                "text": text,
            })
    elapsed = time.time() - t0
    print(f"  {len(rows):,} usable rows in {elapsed:.1f}s "
          f"(skipped {skipped_error:,} errored + {skipped_empty:,} empty)")

    print(f"\nTokenizing for n_tokens (corpus-wide, batched) ...")
    t0 = time.time()
    BATCH = 1000
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        enc = tok(
            [r["text"] for r in chunk],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for r, ids in zip(chunk, enc["input_ids"]):
            r["n_tokens"] = len(ids)
    elapsed = time.time() - t0
    total_tokens = sum(r["n_tokens"] for r in rows)
    print(f"  done in {elapsed:.1f}s")

    bucket_counts = Counter(r["bucket"] for r in rows)
    bucket_tokens: Counter = Counter()
    for r in rows:
        bucket_tokens[r["bucket"]] += r["n_tokens"]

    n_processed = sum(1 for r in rows if r["processed"])
    proc_tok = sum(r["n_tokens"] for r in rows if r["processed"])

    print(f"\n=== Dataset stats ===")
    print(f"  Total documents: {len(rows):,}  "
          f"(processed={n_processed:,} / raw-only={len(rows) - n_processed:,})")
    print(f"  Total tokens:    {total_tokens:,}  "
          f"(processed={proc_tok:,} / raw-only={total_tokens - proc_tok:,})")
    print(f"  Mean tokens/doc: {total_tokens / max(len(rows), 1):.0f}")
    print(f"  Distinct chapters in CHAP_NN buckets: "
          f"{sum(1 for b in bucket_counts if b.startswith('CHAP_'))}")

    print(f"\n  Top 8 buckets by token count:")
    for b, t in sorted(bucket_tokens.items(), key=lambda x: -x[1])[:8]:
        print(f"    {b:<14} docs={bucket_counts[b]:>7,}  tokens={t:>13,}  ({t / total_tokens * 100:5.2f}%)")

    if args.no_push:
        print(f"\n[--no-push] Skipping upload.")
        return

    print(f"\nUploading to https://huggingface.co/datasets/{args.repo_id} (private={args.private}) ...")
    print(f"  (set HF_TOKEN env var or run `huggingface-cli login` if you haven't already)")
    ds = Dataset.from_list(rows)
    ds.push_to_hub(
        args.repo_id,
        private=args.private,
        commit_message=(
            f"prepare_dataset.py: {len(rows):,} rulings, {total_tokens:,} tokens, "
            f"{len(bucket_counts)} buckets, processed={n_processed:,}, "
            f"tokenizer={args.tokenizer}"
        ),
    )
    print("Done.")


if __name__ == "__main__":
    main()
