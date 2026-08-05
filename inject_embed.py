"""Post-merge fix: inject healthy, scaled <tool_call>/</tool_call> embeddings into
a MERGED model dir. The merge reloads the original (dead-embed) coder base, so the
train-time transplant is lost; the trained attn/MLP LoRA learned to aim at the
scaled tool embedding, but that embedding must be present in the final artifact.
This writes it into the merged model so it survives to the GGUF.

  MODEL_DIR=qwen-uni-7b-v5 TRANSPLANT_EMBED=<general snapshot dir> \
  TRANSPLANT_SCALE=50 python inject_embed.py
"""
import os, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

MODEL = os.environ["MODEL_DIR"]
SRC   = os.environ["TRANSPLANT_EMBED"]
SCALE = float(os.environ.get("TRANSPLANT_SCALE", "50"))
TOKS  = [151657, 151658]

src = {}
for f in sorted(glob.glob(os.path.join(SRC, "*.safetensors"))):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.endswith("embed_tokens.weight"):
                w = sf.get_slice(k)
                for t in TOKS:
                    src[t] = torch.tensor(w[t, :])
assert src, "no embed_tokens.weight in source"

print(f"loading merged model {MODEL} (cpu, fp16) ...")
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
emb = m.get_input_embeddings().weight
print("before:", {t: round(float(emb[t, :].norm()), 4) for t in TOKS})
with torch.no_grad():
    for t, v in src.items():
        emb[t, :] = v.to(emb.dtype) * SCALE
# tied embeddings => output head shares this tensor; if untied, sync lm_head too
out = m.get_output_embeddings()
if out is not None and out.weight.data_ptr() != emb.data_ptr():
    with torch.no_grad():
        for t, v in src.items():
            out.weight[t, :] = v.to(out.weight.dtype) * SCALE
print("after :", {t: round(float(emb[t, :].norm()), 4) for t in TOKS})
m.save_pretrained(MODEL, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL).save_pretrained(MODEL)
print(f"injected + saved -> {MODEL}")
