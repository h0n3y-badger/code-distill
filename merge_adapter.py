"""Merge a trained LoRA adapter checkpoint into a 16-bit model in a FRESH process
(no optimizer/gradient state), so the ~15GB fp16 merge fits in RAM where the
end-of-training in-process merge OOM'd."""
import os
from unsloth import FastLanguageModel
CKPT = os.environ.get("CKPT", "out/checkpoint-152")
OUT  = os.environ.get("OUT_DIR", "qwen-coder-7b-mine-v2")
model, tok = FastLanguageModel.from_pretrained(model_name=CKPT, max_seq_length=4096, load_in_4bit=True)
model.save_pretrained_merged(OUT, tok, save_method="merged_16bit")
print(f"Merged model written to ./{OUT}")
