#!/usr/bin/env bash
# Serve the quantized UNIVERSAL student on :8091 with --jinja so the model's own
# chat template drives tool-calling (the OpenAI `tools` param -> <tools> block ->
# parsed <tool_call> back out). eval_universal.py points here. Teacher must be
# stopped (same port + VRAM). 7B Q5 ~5-6GB, fully on GPU.
set -euo pipefail

R="$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2"
V="$HOME/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
F="${STUDENT_GGUF:-$HOME/code-distill/gguf/qwen-uni-7b-Q5_K_M.gguf}"

PORT="${PORT:-8091}"; CTX="${CTX:-4096}"; NGL="${NGL:-99}"
if [ ! -f "$F" ]; then echo "!! student GGUF not found: $F (run quantize)"; exit 1; fi
echo ">> serving student $(basename "$F") on :$PORT  (--jinja, ngl=$NGL ctx=$CTX)"
exec env LD_LIBRARY_PATH="$R:$V" "$R/llama-server" \
  -m "$F" -ngl "$NGL" -c "$CTX" --jinja --port "$PORT" --host 127.0.0.1
