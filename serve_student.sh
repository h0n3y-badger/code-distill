#!/usr/bin/env bash
# Serve the QUANTIZED STUDENT via llama.cpp's llama-server, for `make eval`.
# Student-side twin of serve_teacher.sh — same runtime, same port (8091), so
# eval.py (which points at :8091) hits it with no config change. Run this AFTER
# `make quantize`, with the teacher NOT running (they'd collide on 8091 + VRAM).
set -euo pipefail

R="$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2"
V="$HOME/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
# Output of quantize.sh: gguf/<name>-<quant>.gguf . Override with STUDENT_GGUF=... if needed.
F="${STUDENT_GGUF:-$HOME/code-distill/gguf/qwen-coder-7b-mine-Q5_K_M.gguf}"

PORT="${PORT:-8091}"           # same port eval.py uses; teacher must be stopped
CTX="${CTX:-4096}"
NGL="${NGL:-99}"               # 7B Q5 ~5-6GB, easily fully on GPU

if [ ! -f "$F" ]; then
  echo "!! student GGUF not found: $F"
  echo "   run 'make quantize' first, or set STUDENT_GGUF=/path/to/model.gguf"
  exit 1
fi

echo ">> serving student $(basename "$F") on http://127.0.0.1:$PORT  (ngl=$NGL ctx=$CTX)"
exec env LD_LIBRARY_PATH="$R:$V" "$R/llama-server" \
  -m "$F" -ngl "$NGL" -c "$CTX" --port "$PORT" --host 127.0.0.1
