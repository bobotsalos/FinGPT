#!/usr/bin/env python3
"""LoRA fine-tuning for crypto-news sentiment (1..5) from JSON files.

This script is designed to mirror the FinGPT instruction-tuning approach, but
for chat/instruct models (e.g., Llama-3.1-8B-Instruct).

It:
  1) Loads many JSON files (each can be {"articles": [...]} or a list of objects)
  2) Extracts news text from fields like title/body/summary
  3) Extracts (or derives) a sentiment label in the 1..5 ordinal scale
     - If a label field exists (e.g., sentiment_1_5), it uses it
     - Else it can derive from `fieldSentiments` (sentiment + score)
  4) Builds chat-style training examples and applies loss masking so only the
     assistant answer contributes to the loss
  5) Fine-tunes the base model with PEFT LoRA

Example:
  python train_lora_crypto_news_json.py \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --input_dir /path/to/crypto_news_json \
    --glob '**/*.json' \
    --label_source fieldSentiments_score \
    --max_train_samples 50000 \
    --num_train_epochs 1 \
    --output_dir ./fingpt_lora_crypto

Notes:
  - For supervised fine-tuning, you must have labels (directly, or derivable).
  - If your JSON uses different field names, use the CLI options (e.g.
    --text_fields, --label_field, --symbol_field, --timestamp_field).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model

@dataclass(frozen=True)
class ScoreThresholds:
    """Four cut-points that map a continuous score into 5 ordinal buckets."""

    t1: float
    t2: float
    t3: float
    t4: float

    @staticmethod
    def parse(text: str) -> "ScoreThresholds":
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError("--score_thresholds must have 4 comma-separated floats, e.g. -0.6,-0.2,0.2,0.6")
        t = [float(x) for x in parts]
        return ScoreThresholds(t[0], t[1], t[2], t[3])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune Llama-3.1-8B-Instruct for crypto news sentiment (1..5)")

    p.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--input_dir", required=True, help="Directory containing JSON files.")
    p.add_argument("--glob", default="**/*.json", help="Glob pattern under input_dir.")

    # Text extraction
    p.add_argument(
        "--text_fields",
        default="title,body,summary",
        help="Comma-separated fields (or dotted paths) to concatenate as input text.",
    )
    p.add_argument(
        "--symbol_field",
        default="",
        help="Optional dotted path for symbol/ticker (e.g., customFields.symbol).",
    )
    p.add_argument(
        "--timestamp_field",
        default="publishedAt",
        help="Optional dotted path for timestamp (default: publishedAt).",
    )

    # Label extraction
    p.add_argument(
        "--label_source",
        default="auto",
        choices=[
            "auto",
            "label_field",
            "fieldSentiments_score",
            "fieldSentiments_sentiment",
        ],
        help=(
            "How to get a 1..5 label. 'auto' tries label_field then fieldSentiments. "
            "'fieldSentiments_score' bins the score into 1..5. "
            "'fieldSentiments_sentiment' maps NEGATIVE/NEUTRAL/POSITIVE -> 1/3/5."
        ),
    )
    p.add_argument(
        "--label_field",
        default="sentiment_1_5",
        help="Dotted path to an existing 1..5 label (used when label_source is label_field/auto).",
    )
    p.add_argument(
        "--field_sentiments_path",
        default="fieldSentiments",
        help="Dotted path to field sentiment list (default: fieldSentiments).",
    )
    p.add_argument(
        "--field_sentiments_priority",
        default="BODY,SUMMARY,TITLE",
        help="Priority order for choosing an entry inside fieldSentiments by fieldId.",
    )
    p.add_argument(
        "--score_thresholds",
        default="-0.6,-0.2,0.2,0.6",
        help="Four thresholds to map sentiment score into 1..5.",
    )
    p.add_argument("--drop_unlabeled", action="store_true", help="Drop records where a 1..5 label cannot be derived.")
    p.add_argument(
        "--fallback_label",
        type=int,
        default=3,
        help="If not dropping unlabeled, use this label when missing (default: 3).",
    )

    # Dataset sizing
    p.add_argument("--eval_ratio", type=float, default=0.05)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=2000)

    # LoRA
    p.add_argument("--output_dir", default="./fingpt_lora_crypto")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target module names for LoRA.",
    )

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory (slower, but helps avoid OOM).",
    )
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # Quantization
    p.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (bnb nf4).")
    p.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit.")

    # deepspeed
    p.add_argument('--ds_config', default=None, help='Path to deepspeed config json')

    return p.parse_args()


def _get_by_path(obj: Any, path: str) -> Any:
    if not path:
        return None
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _as_int_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        v = int(value)
    except Exception:
        return None
    if 1 <= v <= 5:
        return v
    return None


def score_to_1_5(score: float, thresholds: ScoreThresholds) -> int:
    if score <= thresholds.t1:
        return 1
    if score <= thresholds.t2:
        return 2
    if score <= thresholds.t3:
        return 3
    if score <= thresholds.t4:
        return 4
    return 5


def sentiment_label_to_1_5(label: str) -> Optional[int]:
    if not label:
        return None
    s = label.strip().upper()
    if s in {"NEGATIVE", "BEARISH"}:
        return 1
    if s in {"NEUTRAL", "MIXED"}:
        return 3
    if s in {"POSITIVE", "BULLISH"}:
        return 5
    return None


def extract_best_field_sentiment(
    record: Dict[str, Any],
    field_sentiments_path: str,
    priority: Sequence[str],
) -> Optional[Dict[str, Any]]:
    fs = _get_by_path(record, field_sentiments_path)
    if not isinstance(fs, list) or not fs:
        return None

    def norm(x: Any) -> str:
        return str(x).strip().upper()

    # Prefer entries where fieldId matches priority order
    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for item in fs:
        if isinstance(item, dict):
            by_field.setdefault(norm(item.get("fieldId")), []).append(item)

    for field_id in priority:
        items = by_field.get(norm(field_id), [])
        if items:
            # pick first (could also pick max abs(score), but keep deterministic)
            return items[0]

    # fallback: first dict element
    for item in fs:
        if isinstance(item, dict):
            return item
    return None


def extract_label_1_5(record: Dict[str, Any], args: argparse.Namespace, thresholds: ScoreThresholds) -> Optional[int]:
    # 1) explicit label field
    if args.label_source in {"auto", "label_field"}:
        v = _get_by_path(record, args.label_field)
        lab = _as_int_label(v)
        if lab is not None:
            return lab
        if args.label_source == "label_field":
            return None

    # 2) fieldSentiments -> score or sentiment
    priority = [p.strip() for p in args.field_sentiments_priority.split(",") if p.strip()]
    best = extract_best_field_sentiment(record, args.field_sentiments_path, priority)
    if not best:
        return None

    if args.label_source in {"auto", "fieldSentiments_score"}:
        score = best.get("score")
        try:
            if score is not None:
                return score_to_1_5(float(score), thresholds)
        except Exception:
            pass
        if args.label_source == "fieldSentiments_score":
            return None

    if args.label_source in {"auto", "fieldSentiments_sentiment"}:
        return sentiment_label_to_1_5(str(best.get("sentiment") or ""))

    return None


def extract_text(record: Dict[str, Any], field_paths: Sequence[str]) -> str:
    parts: List[str] = []
    for p in field_paths:
        v = _get_by_path(record, p)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                parts.append(s)
        else:
            s = str(v).strip()
            if s and s.lower() != "none":
                parts.append(s)
    return "\n".join(parts).strip()


def iter_json_records(input_dir: str, pattern: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    root = os.path.abspath(input_dir)
    paths = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    print(f"JSON Files number: {len(paths)}")

    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        if isinstance(obj, dict) and isinstance(obj.get("articles"), list):
            for rec in obj["articles"]:
                if isinstance(rec, dict):
                    yield path, rec
            continue

        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield path, rec
            continue

        if isinstance(obj, dict):
            yield path, obj


def build_messages(
    text: str,
    label_1_5: int,
    symbol: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> List[Dict[str, str]]:
    system = (
        "You are a seasoned crypto market analyst. "
        "Given crypto news, output a single integer sentiment score from 1 to 5, where "
        "1 is strongly bearish, 2 is bearish, 3 is neutral, 4 is bullish, 5 is strongly bullish. "
        "Return the integer only."
    )

    header_bits = []
    if symbol:
        header_bits.append(f"Symbol: {symbol}")
    if timestamp:
        header_bits.append(f"Time: {timestamp}")
    header = ("\n".join(header_bits) + "\n\n") if header_bits else ""

    user = header + "News:\n" + text

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(int(label_1_5))},
    ]


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    # Keep only minimal columns.
    rows = [
        {
            "messages": r["messages"],
        }
        for _, r in df.iterrows()
    ]
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(ex):
        messages = ex["messages"]
        full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        prompt_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=True)

        input_ids = full_ids[:max_length]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        labels = labels[:max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    tokenized = dataset.map(_tokenize, remove_columns=["messages"])
    return tokenized


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    quant_config = None
    load_kwargs: Dict[str, Any] = {"device_map": "auto"}

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["quantization_config"] = quant_config
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    if args.gradient_checkpointing:
        # Reduces activation memory; required to disable KV-cache during training.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    thresholds = ScoreThresholds.parse(args.score_thresholds)
    text_fields = [p.strip() for p in args.text_fields.split(",") if p.strip()]

    rows: List[Dict[str, Any]] = []
    for path, rec in iter_json_records(args.input_dir, args.glob):
        text = extract_text(rec, text_fields)
        if not text:
            continue

        label = extract_label_1_5(rec, args, thresholds)
        if label is None:
            if args.drop_unlabeled:
                continue
            label = int(args.fallback_label)

        symbol = _get_by_path(rec, args.symbol_field) if args.symbol_field else None
        if symbol is not None:
            symbol = str(symbol)

        timestamp = _get_by_path(rec, args.timestamp_field) if args.timestamp_field else None
        if timestamp is not None:
            timestamp = str(timestamp)

        rows.append(
            {
                "messages": build_messages(text=text, label_1_5=int(label), symbol=symbol, timestamp=timestamp),
                "_source_file": path,
            }
        )

    if not rows:
        raise SystemExit("No usable records found. Check --input_dir/--glob and your label extraction settings.")

    df = pd.DataFrame(rows)

    dataset = dataframe_to_dataset(df)

    # Split train/eval
    eval_size = max(1, int(len(dataset) * float(args.eval_ratio)))
    if args.max_eval_samples is not None:
        eval_size = min(eval_size, int(args.max_eval_samples))
    eval_size = min(eval_size, max(1, len(dataset) - 1))

    dataset = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=args.seed)

    if args.max_train_samples:
        n = min(len(dataset["train"]), int(args.max_train_samples))
        dataset["train"] = dataset["train"].select(range(n))

    model, tokenizer = load_model_and_tokenizer(args)

    train_tok = tokenize_dataset(dataset["train"], tokenizer, args.max_length)
    eval_tok = tokenize_dataset(dataset["test"], tokenizer, args.max_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="steps",
        eval_steps=max(1, args.logging_steps * 10),
        save_total_limit=3,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        deepspeed=args.ds_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
