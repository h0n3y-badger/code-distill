"""Python-focused QLoRA of Qwen2.5-Coder-1.5B-Instruct (the smallest fast Qwen coder
that's still genuinely useful; runs on CPU on the ThinkPad W541). Same recipe as
train.py but 1.5B base, Python-only data (python_clean.jsonl), 3 epochs (smaller set).
Merges to qwen-coder-1.5b-py/ for GGUF conversion."""
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import config as C

import os
STUDENT = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
OUT_DIR = os.environ.get("OUT_DIR", "qwen-coder-1.5b-py")
DATA    = os.environ.get("TRAIN_DATA", "python_clean.jsonl")

model, tok = FastLanguageModel.from_pretrained(
    model_name=STUDENT, max_seq_length=C.MAX_SEQ_LEN, load_in_4bit=True)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)

texts, skipped = [], 0
with open(DATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        if (not isinstance(msgs, list) or len(msgs) < 2 or
                not all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs)):
            skipped += 1
            continue
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
print(f"loaded {len(texts)} Python examples (skipped {skipped})")
ds = Dataset.from_dict({"text": texts})

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    max_seq_length=C.MAX_SEQ_LEN,
    args=SFTConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        warmup_steps=10, num_train_epochs=3, learning_rate=2e-4,
        logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", bf16=True, seed=3407,
        output_dir="out_1b", report_to="none"))
trainer.train()

model.save_pretrained_merged(OUT_DIR, tok, save_method="merged_16bit")
print(f"Merged model written to ./{OUT_DIR}")
