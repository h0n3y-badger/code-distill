"""Phase 3 — QLoRA SFT of the student with Unsloth.

Run AFTER unloading the teacher in LM Studio (both won't fit in 16GB).
Trains a LoRA, then merges to a 16-bit HF model in MERGED_DIR for GGUF conversion.
"""
import os, json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import config as C
from datalib import is_valid_messages as _valid

TRAIN_DATA = os.environ.get("TRAIN_DATA", C.CLEAN_FILE)
OUT_DIR    = os.environ.get("OUT_DIR", C.MERGED_DIR)

model, tok = FastLanguageModel.from_pretrained(
    model_name=C.STUDENT_MODEL,
    max_seq_length=C.MAX_SEQ_LEN,
    load_in_4bit=True,          # QLoRA: 4-bit base weights
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",   # big VRAM saver, key on 16GB
    random_state=3407,
)

# Load JSONL ourselves and render each row to a FLAT `text` string up front.
# We hand pyarrow only strings — never the nested `messages`/`tests` structures,
# whose per-row shape varies (teacher wrote `tests` as string vs array, etc.),
# which is what made both load_dataset() and Dataset.from_list(rows) fail.
# _valid == datalib.is_valid_messages (shared with the regression tests).
texts, skipped = [], 0
with open(TRAIN_DATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        if not _valid(msgs):
            skipped += 1
            continue
        texts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False))
print(f"loaded {len(texts)} training examples from {TRAIN_DATA} "
      f"(skipped {skipped} with empty/invalid content)")
ds = Dataset.from_dict({"text": texts})

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    max_seq_length=C.MAX_SEQ_LEN,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,      # effective batch 16
        warmup_steps=10,
        num_train_epochs=2,                 # 1-3; watch for overfit on small sets
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=True,
        seed=3407,
        output_dir="out",
        report_to="none",
    ),
)

trainer.train()

# The end-of-training fp16 merge is a ~15GB RAM cliff (optimizer/grad buffers are
# still resident here). On a tight-RAM box set SKIP_MERGE=1 to just save the LoRA
# adapter and merge later in a fresh process via merge_adapter.py.
if os.environ.get("SKIP_MERGE") == "1":
    adapter_dir = os.environ.get("ADAPTER_DIR", "out_adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Adapter saved to ./{adapter_dir} (merge skipped; run merge_adapter.py)")
else:
    # Merge LoRA into base -> 16-bit HF model, ready for llama.cpp conversion.
    model.save_pretrained_merged(OUT_DIR, tok, save_method="merged_16bit")
    print(f"Merged model written to ./{OUT_DIR}")
