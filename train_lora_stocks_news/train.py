#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict, Iterable
import torch

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
    LlamaForCausalLM
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', default='meta-llama/Meta-Llama-3-8B')
    p.add_argument('--input_csv', type=str, help='(optional) Path to CSV with sentiment labels, otherwise HF dataset is used.')
    p.add_argument('--sentiment_column', default='sentiment_deepseek', help='Name of column with integer sentiment 1..5.')
    p.add_argument('--fallback_default', type=int, default=3, help='Default label if sentiment column missing/NaN.')
    p.add_argument('--group_size', type=int, default=1, help='Number of consecutive summaries to bundle into one training example.')
    p.add_argument('--group_by_symbol', action='store_true', help='Group only within the same Stock_symbol.')
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--max_eval_samples', type=int, default=None)
    p.add_argument('--output_dir', default='./fingpt_lora_nasdaq')
    p.add_argument('--r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--target_modules', default='q_proj,v_proj', help='Comma-separated target module names.')
    p.add_argument('--per_device_train_batch_size', type=int, default=2)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--num_train_epochs', type=float, default=1)
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--save_strategy', default='epoch')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# --- Few-shot examples (same pattern as original) ---
FEWSHOT_EXAMPLES = [
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
    # If user provides a CSV (already labeled), use it; otherwise use HF raw dataset
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        raw = load_dataset('benstaf/FNSPID-nasdaq-100-1news-per-row-random')['train']
        df = raw.to_pandas()

    dataset = dataframe_to_hf(df, args)

    print("Dataset head: ", dataset[20:22])

    # ---- explicit requested split: 20% test, shuffled, seed=42 ----
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    # optional sample-limits for quick runs
    if args.max_train_samples:
        n = min(len(dataset['train']), args.max_train_samples)
        dataset['train'] = dataset['train'].select(range(n))
    if args.max_eval_samples:
        n = min(len(dataset['test']), args.max_eval_samples)
        dataset['test'] = dataset['test'].select(range(n))

    print(f"After split: train={len(dataset['train'])}, test={len(dataset['test'])}")
    return dataset


def tokenize_dataset(dataset, tokenizer, max_length: int):
    # Tokenize and produce fixed-length input_ids, attention_mask, labels (labels use -100 for prompt tokens and padding)
    def _tokenize(example):
        messages = example['messages']

        # prefer specialized chat-template if available; else fallback
        try:
            full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)
        except Exception:
            # basic fallback join
            def msgs_to_str(msgs):
                pieces = []
                for m in msgs:
                    pieces.append(f"{m['role'].upper()}: {m['content']}")
                return "\n".join(pieces)
            full_text = msgs_to_str(messages)
            prompt_text = msgs_to_str(messages[:-1])

        enc_full = tokenizer(full_text, truncation=True, max_length=max_length, padding='max_length', return_attention_mask=True)
        enc_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)
        input_ids = enc_full['input_ids']
        attention_mask = enc_full['attention_mask']
        prompt_len = len(enc_prompt['input_ids'])

        # Build labels: hide prompt tokens (-100) and keep assistant tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        # pad/truncate labels to max_length; pad with -100
        if len(labels) < max_length:
            labels = labels + [-100] * (max_length - len(labels))
        else:
            labels = labels[:max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # map over both train and test splits
    tokenized = {}
    tokenized['train'] = dataset['train'].map(_tokenize)
    tokenized['test'] = dataset['test'].map(_tokenize)
    # remove the original messages column to avoid Trainer complaints
    tokenized['train'] = tokenized['train'].remove_columns([c for c in tokenized['train'].column_names if c == 'messages'])
    tokenized['test'] = tokenized['test'].remove_columns([c for c in tokenized['test'].column_names if c == 'messages'])
    return tokenized


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    data = load_data(args)
    tokenized = tokenize_dataset(data, tokenizer, args.max_length)

    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model,
    #     device_map='auto',
    #     load_in_8bit=True,
    # )
    from transformers.utils import is_bitsandbytes_available
    is_bitsandbytes_available()

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        quantization_config = q_config,
        trust_remote_code=True,
        device_map='auto'
    )

    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']  # Modules for the Llama model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.target_modules.split(',') if m.strip()],
        lora_dropout=args.lora_dropout,
        bias='none'
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    

    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     learning_rate=args.learning_rate,
    #     num_train_epochs=args.num_train_epochs,
    #     logging_steps=args.logging_steps,
    #     save_strategy=args.save_strategy,
    #     bf16=args.bf16,
    #     fp16=args.fp16,
    #     eval_strategy='steps',
    #     eval_steps=max(1, args.logging_steps*5),
    #     save_total_limit=3,
    #     report_to='none'
    # )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_steps = 500,               # Log every 500 steps
        # max_steps=10000,                 # Maximum number of training steps (commented out, can be enabled)
        num_train_epochs = args.num_train_epochs,   # Number of training epochs (train for 2 epochs)
        per_device_train_batch_size=args.per_device_train_batch_size,     # Batch size of 4 for training on each device (GPU/CPU)
        gradient_accumulation_steps=args.gradient_accumulation_steps,     # Accumulate gradients for 8 steps before updating weights
        learning_rate=args.learning_rate,                # Learning rate set to 1e-4
        weight_decay=0.01,                 # Weight decay (L2 regularization) set to 0.01
        warmup_steps=1000,                 # Warm up the learning rate for the first 1000 steps
        save_steps=500,                    # Save the model every 500 steps
        fp16=False,                         # Enable FP16 mixed precision training to save memory and speed up training
        # bf16=False,                       # Enable BF16 mixed precision training (commented out)
        torch_compile = False,             # Whether to enable Torch compile (`False` means not enabled)
        load_best_model_at_end = True,     # Load the best-performing model at the end of training
        eval_strategy="steps",       # Evaluation strategy is set to evaluate every few steps
        remove_unused_columns=False,       # Whether to remove unused columns during training (keep all columns)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished, model saved to", args.output_dir)


if __name__ == '__main__':
    main()
