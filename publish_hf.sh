#!/usr/bin/env bash
# Upload v12 (fp16 model dir + README/card + Q5_K_M GGUF) to HuggingFace.
# v12 supersedes v6/v8/v10/v11 as the canonical model at the repo root.
set -uo pipefail
cd "$HOME/code-distill"
HF=".venv/bin/hf"; REPO="h0ney-badger/qwen2.5-7b-universal-distill"; LOG="publish_hf.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "uploading fp16 model dir (README + config + safetensors + tokenizer) ..."
$HF upload "$REPO" ./qwen-uni-7b-v12 . --repo-type model >>"$LOG" 2>&1 || { say "PUBLISH-FAILED: model dir"; exit 1; }
say "uploading GGUF (Q5_K_M) ..."
$HF upload "$REPO" ./gguf/qwen-uni-7b-v12-Q5_K_M.gguf qwen-uni-7b-v12-Q5_K_M.gguf --repo-type model >>"$LOG" 2>&1 \
  || { say "PUBLISH-FAILED: gguf"; exit 1; }
say "PUBLISH DONE -> https://huggingface.co/$REPO"
