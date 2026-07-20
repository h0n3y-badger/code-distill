#!/usr/bin/env bash
# Phase 4 — convert merged HF model -> GGUF, then quantize to Q5_K_M.
# Clones llama.cpp on first run. No pip deps beyond what llama.cpp's convert needs.
set -euo pipefail

MERGED_DIR="${MERGED_DIR:-qwen-coder-7b-mine}"
GGUF_DIR="${GGUF_DIR:-gguf}"
QUANT="${QUANT:-Q5_K_M}"          # Q5_K_M sweet spot; Q6_K if you want more headroom
OUT_NAME="${OUT_NAME:-qwen-coder-7b-mine}"
PY="${VENV:-.venv}/bin/python"   # use the project venv (set by the Makefile)

mkdir -p "$GGUF_DIR"

if [ ! -d llama.cpp ]; then
  echo ">> cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

echo ">> installing convert deps (uv; only the extras — NOT llama.cpp's"
echo "   requirements.txt, which pins a CPU torch that would clobber our CUDA one)"
# uv-created venvs have no pip, so use `uv pip`. We already have torch/transformers/
# numpy/safetensors from the training install; convert_hf_to_gguf.py just also needs:
VIRTUAL_ENV="${VENV:-.venv}" uv pip install -q gguf sentencepiece protobuf

echo ">> converting to f16 GGUF"
"$PY" llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" \
  --outfile "$GGUF_DIR/${OUT_NAME}-f16.gguf" --outtype f16

echo ">> building llama-quantize (cmake)"
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF >/dev/null
cmake --build llama.cpp/build --target llama-quantize -j >/dev/null

echo ">> quantizing to ${QUANT}"
./llama.cpp/build/bin/llama-quantize \
  "$GGUF_DIR/${OUT_NAME}-f16.gguf" \
  "$GGUF_DIR/${OUT_NAME}-${QUANT}.gguf" "$QUANT"

echo ">> done: $GGUF_DIR/${OUT_NAME}-${QUANT}.gguf"
echo "   Load it in LM Studio (drop into ~/.lmstudio/models) or: llama.cpp/build/bin/llama-cli -m <file>"
