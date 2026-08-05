#!/usr/bin/env bash
# Serve the Qwen2.5-32B-Instruct teacher via llama.cpp's llama-server (reusing
# LM Studio's bundled CUDA runtime, per serve_teacher.sh). 32B Q4_K_M is ~20GB —
# too big for 16GB VRAM — so we offload as many layers as fit (NGL) and spill the
# rest to system RAM. Slower than the 14B coder, fine for overnight generation.
# OpenAI-compatible API at http://localhost:8091/v1  (config_uni.py points here).
set -euo pipefail

R="$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2"
V="$HOME/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
F="${TEACHER_GGUF:-$HOME/code-distill/teacher/Qwen2.5-32B-Instruct-Q4_K_M.gguf}"

PORT="${PORT:-8091}"
CTX="${CTX:-4096}"
NGL="${NGL:-42}"          # layers on GPU; tune down if it OOMs, up if VRAM spare
THREADS="${THREADS:-12}"  # CPU threads for the spilled layers

if [ ! -f "$F" ]; then echo "!! teacher GGUF not found: $F"; exit 1; fi
echo ">> serving $(basename "$F") on :$PORT  (ngl=$NGL ctx=$CTX threads=$THREADS)"
exec env LD_LIBRARY_PATH="$R:$V" "$R/llama-server" \
  -m "$F" -ngl "$NGL" -c "$CTX" -t "$THREADS" --port "$PORT" --host 127.0.0.1
