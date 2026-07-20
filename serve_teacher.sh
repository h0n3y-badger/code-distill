#!/usr/bin/env bash
# Serve the teacher GGUF via llama.cpp's own llama-server, reusing LM Studio's
# bundled CUDA runtime + vendor libs. We do this because LM Studio's *load
# orchestration* SIGSEGV'd on this model, while the bare runtime loads it fine.
#
# Exposes an OpenAI-compatible API at http://localhost:8091/v1  (config.py points here).
# Leave this running in its own terminal while `make generate` works.
set -euo pipefail

# Newest CUDA runtime LM Studio shipped + its vendored CUDA libs (libcudart.so.12 etc.)
R="$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2"
V="$HOME/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
F="$HOME/.lmstudio/models/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf"

PORT="${PORT:-8091}"           # 8080=SearXNG, 1234=LM Studio, so 8091
CTX="${CTX:-4096}"
NGL="${NGL:-99}"               # offload all layers to GPU (14B Q5 ~11GB, fits 16GB)

echo ">> serving $(basename "$F") on http://127.0.0.1:$PORT  (ngl=$NGL ctx=$CTX)"
exec env LD_LIBRARY_PATH="$R:$V" "$R/llama-server" \
  -m "$F" -ngl "$NGL" -c "$CTX" --port "$PORT" --host 127.0.0.1
