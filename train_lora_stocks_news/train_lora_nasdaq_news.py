"""LoRA training script for NASDAQ news sentiment scoring.

This version aligns preprocessing with the output produced by
`sentiment_deepseek_deepinfra.py`, which adds a column (default: sentiment_deepseek)
containing integer scores 1..5. We build chat-style examples mirroring the
few-shot prompt pattern used for inference labeling.

Key features:
  * Load either a prepared CSV with sentiment labels or fall back to HF dataset
  * Optional grouping of multiple summaries per sample (sliding or per-symbol)
  * Few-shot in-context examples replicating the DeepSeek prompt pattern
  * Proper loss masking: only assistant answer tokens contribute to loss
  * LoRA configuration via CLI

Example:
  python scripts/train_lora_nasdaq_news.py \\
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \\
    --input_csv ./DATASETS/sentiment_deepseek_./DATASETS/filtered_nasdaq_news.csv \\
    --sentiment_column sentiment_deepseek \\
    --group_size 1 \\
    --max_train_samples 5000 \\
    --num_train_epochs 1
"""

import argparse
import os
from typing import List, Dict, Iterable

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    p.add_argument('--input_csv', type=str, help='Path to CSV with sentiment labels (output of sentiment_deepseek_deepinfra).')
    p.add_argument('--sentiment_column', default='sentiment_deepseek', help='Name of column with integer sentiment 1..5.')
    p.add_argument('--fallback_default', type=int, default=3, help='Default label if sentiment column missing/NaN.')
    p.add_argument('--group_size', type=int, default=1, help='Number of consecutive summaries to bundle into one training example.')
    p.add_argument('--group_by_symbol', action='store_true', help='Group only within the same Stock_symbol.')
    p.add_argument('--max_train_samples', type=int, default=20000)
    p.add_argument('--max_eval_samples', type=int, default=2000)
    p.add_argument('--eval_ratio', type=float, default=0.05)
    p.add_argument('--output_dir', default='./fingpt_lora_nasdaq')
    p.add_argument('--r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--target_modules', default='q_proj,v_proj', help='Comma-separated target module names.')
    p.add_argument('--per_device_train_batch_size', type=int, default=2)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--num_train_epochs', type=float, default=3)
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--save_strategy', default='epoch')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


FEWSHOT_EXAMPLES = [
    # (user_content, assistant_answer)
    ("News to Stock Symbol -- AAPL: Apple (AAPL) increase 22% ### News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30% ### News to Stock Symbol -- MSFT: Microsoft (MSTF) price has no change", "5, 1, 3"),
    ("News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15 ### News to Stock Symbol -- AAPL: Apple (AAPL) will release VisonPro on Feb 2, 2024", "4, 4"),
]


def build_fewshot_messages() -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [
        {
            'role': 'system',
            'content': (
                "Forget all your previous instructions. You are a financial expert with stock recommendation experience. "
                "Based on a specific stock, output integer scores from 1 to 5, where 1 is negative, 2 somewhat negative, 3 neutral, "
                "4 somewhat positive, 5 positive. Return comma-separated numbers only."
            )
        }
    ]
    for u, a in FEWSHOT_EXAMPLES:
        msgs.append({'role': 'user', 'content': u})
        msgs.append({'role': 'assistant', 'content': a})
    return msgs


def iter_group_rows(df: pd.DataFrame, group_size: int, by_symbol: bool) -> Iterable[List[pd.Series]]:
    if by_symbol:
        for symbol, sub in df.groupby('Stock_symbol'):
            rows = list(sub.itertuples(index=False))
            for i in range(0, len(rows), group_size):
                yield rows[i:i+group_size]
    else:
        rows = list(df.itertuples(index=False))
        for i in range(0, len(rows), group_size):
            yield rows[i:i+group_size]


def build_conversation(batch_rows: List[pd.Series], sentiment_col: str, default_label: int) -> List[Dict[str, str]]:
    # Determine symbol (use first row symbol; batch may have mixed if not grouping by symbol)
    symbol = getattr(batch_rows[0], 'Stock_symbol', 'UNK')
    texts = []
    labels = []
    for r in batch_rows:
        summary = getattr(r, 'Lsa_summary', '') or ''
        label = getattr(r, sentiment_col, None)
        if pd.isna(label):
            label = default_label
        labels.append(int(label))
        texts.append(f"News to Stock Symbol -- {symbol}: {summary}")
    user_content = " ### ".join(texts)
    label_str = ", ".join(str(x) for x in labels)

    msgs = build_fewshot_messages()

    msgs.append({'role': 'user', 'content': user_content})
    msgs.append({'role': 'assistant', 'content': label_str})
    return msgs


def dataframe_to_hf(df: pd.DataFrame, args) -> Dataset:
    records = []
    for batch in iter_group_rows(df, args.group_size, args.group_by_symbol):
        conv = build_conversation(batch, args.sentiment_column, args.fallback_default)
        records.append({'messages': conv})
    return Dataset.from_list(records)


def load_data(args):
    raw = load_dataset('benstaf/FNSPID-nasdaq-100-1news-per-row-random')['train']
    df = raw.to_pandas()
    dataset = dataframe_to_hf(df, args)
    # train/eval split
    eval_size = max(1, int(len(dataset)*args.eval_ratio))
    if args.max_eval_samples:
        eval_size = min(eval_size, args.max_eval_samples)
    train_size = len(dataset) - eval_size
    print(f"Dataset size: {len(dataset)} train {train_size} eval {eval_size}")
    dataset = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=args.seed)
    return dataset


def tokenize_dataset(dataset, tokenizer, max_length: int):
    # We produce input_ids, labels with masking of prompt tokens.
    def _tokenize(example):
        messages = example['messages']
        # Full with answer
        full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        # Prompt only (drop last assistant)
        prompt_messages = messages[:-1]
        prompt_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=False, tokenize=True)
        input_ids = full_ids[:max_length]
        # Mask labels for prompt part
        labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:max_length]
        attention_mask = [1]*len(input_ids)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    return dataset.map(_tokenize)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    data = load_data(args)
    tokenized = {}
    tokenized['train'] = tokenize_dataset(data['train'], tokenizer, args.max_length)
    tokenized['test'] = tokenize_dataset(data['test'], tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map='auto',
        load_in_8bit=True
    )
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.target_modules.split(',') if m.strip()],
        lora_dropout=args.lora_dropout,
        bias='none'
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_strategy='steps',
        eval_steps=max(1, args.logging_steps*5),
        save_total_limit=3,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test']
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
