# NASDAQ News LoRA Demo

This demo shows how to fine-tune a causal language model with LoRA adapters on the `benstaf/nasdaq_news` dataset.

## Quick Start

Create / activate an environment with the required packages (add any CUDA-specific wheels you need):

```bash
pip install torch transformers datasets peft accelerate sentencepiece
```

(You may add `bitsandbytes` for 8-bit loading if desired.)

Run a small test training (few samples, 1 epoch):

```bash
python scripts/train_lora_nasdaq_news.py \
  --base_model chatglm2 \
  --from_remote \
  --task summarize \
  --max_train_samples 400 \
  --max_eval_samples 80 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --output_dir outputs/nasdaq_news_lora_test
```

If VRAM is limited, try a smaller model:

```bash
python scripts/train_lora_nasdaq_news.py \
  --hf_model gpt2 \
  --task summarize \
  --max_length 384 \
  --max_train_samples 1000 \
  --output_dir outputs/gpt2_nasdaq
```

## Script Highlights

-   Builds simple instruction/target pairs from each news article (title, date, ticker). The target defaults to the original title (placeholder). Adjust `format_example` to implement a better supervised objective (e.g., summarization by using the body as input and the title as label, or generating a true summary via external tool first).
-   LoRA configuration is configurable via CLI flags (`--lora_r`, `--lora_alpha`, `--lora_dropout`).
-   Saves adapter weights to `output_dir/adapter`.

## Custom Tasks

Use `--task headline` or `--task sentiment` for alternative instructions, or pass a custom text (e.g., `--task "Classify the market impact"`).

## Improving Quality

1. Replace placeholder target with human or algorithmically generated summaries.
2. Add filtering to drop extremely long bodies or low-information articles.
3. Use a validation metric (e.g., ROUGE, accuracy for sentiment) by writing a custom `compute_metrics` function.
4. Consider packing multiple short samples per sequence for efficiency.

## Inference (after training)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
model_name = 'THUDM/chatglm2-6b'  # or your selected base
base = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
base_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
adapter_path = 'outputs/nasdaq_news_lora_test/adapter'
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()
prompt = "Title: Apple Earnings\nDate: 2024-01-01\nTicker: AAPL\n\nInstruction: Provide a concise 1-2 sentence summary."
inputs = base_tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
print(base_tokenizer.decode(out[0], skip_special_tokens=True))
```

## License / Data

Ensure you comply with the dataset licensing and the base model's license (e.g., Llama 2 has specific usage terms).

---

This demo is minimal and intended as a starting point.
