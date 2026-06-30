"""Train LoRA on Qwen2.5-0.5B for tool-calling (CPU-safe)."""

import os, json, torch

# ── CPU 安全：限制线程数，防止电脑卡死 ──
N_THREADS = 2
torch.set_num_threads(N_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ── Config ──
MODEL_PATH = "../models/Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "../output/lora_tool_calling"
LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
MAX_LENGTH = 256
MAX_STEPS = 200  # 安全上限，防止无限训练

# ── Load model ──
print(f"Loading model (CPU, {N_THREADS} threads)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── LoRA ──
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load data ──
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

train_raw = load_jsonl("../data/data_train.json")
eval_raw = load_jsonl("../data/data_eval.json")

SYSTEM = """你是工具调用助手。只输出 JSON：{"tool":"<name>","params":{...}}，不要解释。"""

def format_sample(sample):
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)

def tokenize(sample):
    text = format_sample(sample)
    return tokenizer(text, truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_list(train_raw).map(tokenize, remove_columns=["instruction", "output"])
eval_dataset = Dataset.from_list(eval_raw).map(tokenize, remove_columns=["instruction", "output"])

# ── Train ──
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    gradient_accumulation_steps=4,
    warmup_steps=10,
    save_total_limit=2,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"\nTraining on CPU ({N_THREADS} threads): {len(train_dataset)} train / {len(eval_dataset)} eval")
print(f"Max length: {MAX_LENGTH} | Epochs: {EPOCHS} | Max steps: {MAX_STEPS}")
print(f"Steps per epoch: ~{len(train_dataset) // (BATCH_SIZE * 4)}")
print("Starting...\n")

trainer.train()

# ── Save ──
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved to {OUTPUT_DIR}")
print(f"  adapter_model.safetensors  — LoRA weights")
print(f"  adapter_config.json        — LoRA config")
