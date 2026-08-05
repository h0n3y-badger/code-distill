"""Universal QLoRA SFT — CONTINUE from the merged v3 coder and add web + chat +
tools + replayed coding. Separate from train.py (which the coding pipeline still
uses) so that pipeline stays intact.

Renders each row with the tokenizer's chat template; TOOL rows are rendered with
their `tools_spec` so the training text carries the canonical <tools> system
block + <tool_call>/<tool_response> turns (matches inference). Run AFTER
unloading the teacher (both won't fit 16GB). SKIP_MERGE=1 to save adapter only
and merge later via merge_adapter.py (the fp16 merge is a RAM cliff).
"""
import os, json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import config_uni as C
from datalib import is_valid_messages, has_tool_turn

TRAIN_DATA = os.environ.get("TRAIN_DATA", C.UNI_CLEAN)
OUT_DIR    = os.environ.get("OUT_DIR", C.MERGED_DIR)
BASE       = os.environ.get("BASE_MODEL", C.STUDENT_MODEL)
# When embed_tokens is trainable, CE materializes a (seq x 152064-vocab) logit
# tensor that scales with sequence length -> capping MAX_SEQ bounds peak VRAM.
# Data is short (p99=975; only ~2/2854 rows exceed 2048), so a 2048 cap is ~free.
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ", str(C.MAX_SEQ_LEN)))

model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
)

# TRANSPLANT_EMBED=<dir with *.safetensors>: our Coder base ships the tool tokens
# <tool_call>(151657)/</tool_call>(151658) with ZERO embeddings (dead: a tied
# zero row => zero output logit => the token can NEVER be emitted). Copy healthy
# vectors for them from a general Qwen2.5-Instruct (same base, compatible space)
# so the model can actually route probability there once trained.
_TP = os.environ.get("TRANSPLANT_EMBED")
if _TP:
    import glob, torch
    from safetensors import safe_open
    TOOL_TOKS = [151657, 151658]
    emb = model.get_input_embeddings().weight
    src_rows = {}
    for f in sorted(glob.glob(os.path.join(_TP, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith("embed_tokens.weight"):
                    w = sf.get_slice(k)
                    for t in TOOL_TOKS:
                        src_rows[t] = w[t, :]
    scale = float(os.environ.get("TRANSPLANT_SCALE", "1.0"))
    with torch.no_grad():
        for t, vec in src_rows.items():
            emb[t, :] = (vec.to(emb.dtype).to(emb.device)) * scale
    print(f"transplanted tool-token embeddings from {_TP} "
          f"(scale={scale}): "
          f"{ {t: round(float(emb[t,:].norm()),4) for t in TOOL_TOKS} }")
# r/alpha/epochs are env-tunable so iterations can strengthen a weak skill
# (v2: r=32 alpha=64 epochs=3 to revive the under-learned <tool_call> token that
# v3's coding-only fine-tune had suppressed) without editing code each round.
LORA_R     = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
EPOCHS     = float(os.environ.get("EPOCHS", "2"))
# TRAIN_EMBED=1 also LoRA-trains embed_tokens + lm_head. Needed to teach a token
# the base suppresses: v3 (coding-only) left the <tool_call> token (151657)
# unreachable, and with the head FROZEN no amount of data/epochs could raise its
# probability — the model spilled to garbage tokens in that slot. Unfreezing the
# head lets training route probability to it.
_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]
if os.environ.get("TRAIN_EMBED") == "1":
    # Qwen2.5-Coder-7B has TIED embeddings (tie_word_embeddings=True) -> embed_tokens
    # IS lm_head. Train only embed_tokens; adding lm_head too triggers a PEFT
    # tied-layer warning and can corrupt the adapter merge / GGUF conversion.
    _targets += ["embed_tokens"]
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
    target_modules=_targets,
    use_gradient_checkpointing="unsloth", random_state=3407,
)


def render(obj):
    """Row -> flat training string. Any row carrying a tools_spec (tools rows and
    multi subtask/recall rows) is rendered WITH it so the template emits the
    <tools> system block + <tool_call>/<tool_response> turns (matches inference)."""
    msgs = obj["messages"]
    if obj.get("tools_spec"):
        return tok.apply_chat_template(
            msgs, tools=obj["tools_spec"],
            tokenize=False, add_generation_prompt=False)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def valid(obj):
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    # tool-call turns have empty content by design -> skip the plain gate
    if obj.get("kind") == "tools" or has_tool_turn(msgs):
        return isinstance(msgs[-1].get("content"), str) and msgs[-1]["content"].strip()
    return is_valid_messages(msgs)


texts, skipped, by_kind = [], 0, {}
with open(TRAIN_DATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not valid(obj):
            skipped += 1
            continue
        try:
            texts.append(render(obj))
        except Exception as e:
            skipped += 1
            continue
        k = obj.get("kind", "?")
        by_kind[k] = by_kind.get(k, 0) + 1
print(f"loaded {len(texts)} examples from {TRAIN_DATA} (skipped {skipped}); "
      f"by kind: {by_kind}")
ds = Dataset.from_dict({"text": texts})

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=SFTConfig(
        per_device_train_batch_size=int(os.environ.get("BATCH", "2")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
        warmup_steps=10, num_train_epochs=EPOCHS, learning_rate=2e-4,
        logging_steps=5, optim=os.environ.get("OPTIM", "adamw_8bit"), weight_decay=0.01,
        lr_scheduler_type="linear", bf16=True, seed=3407,
        output_dir="out_uni", report_to="none",
    ),
)
trainer.train()

if os.environ.get("SKIP_MERGE") == "1":
    adapter_dir = os.environ.get("ADAPTER_DIR", "out_uni_adapter")
    model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
    print(f"Adapter saved to ./{adapter_dir} (merge later via merge_adapter.py)")
else:
    model.save_pretrained_merged(OUT_DIR, tok, save_method="merged_16bit")
    print(f"Merged model written to ./{OUT_DIR}")
