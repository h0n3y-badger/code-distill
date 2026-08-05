#!/usr/bin/env bash
# v-uni-3: teach the suppressed <tool_call> token by ALSO LoRA-training
# embed_tokens + lm_head (TRAIN_EMBED=1). Epochs back to 2 (3 dinged chat),
# r=32, tools upweighted 3x. LOUD phone ring at each milestone (user request).
# Detached-safe: nohup bash run_v3_pipeline.sh &
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni3.log"; : > "$LOG"
B="llama.cpp/build/bin"; DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }
fail(){ say "FAILED: $1"; ring "v-uni-3 FAILED at: $1 — see build_uni3.log"; exit 1; }

say "rebuild mix (TOOL_UPWEIGHT=3) ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"

pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5
say "training v-uni-3 (TRAIN_EMBED=1, r=32, epochs=2, batch=1 to fit embed logits) ..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v3_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL=qwen-coder-7b-mine-v3 LORA_R=32 LORA_ALPHA=64 EPOCHS=2 TRAIN_EMBED=1 \
  BATCH=1 GRAD_ACCUM=16 MAX_SEQ=2048 OPTIM=paged_adamw_8bit \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"
ring "v-uni-3 step: training done, merging next"

say "merging -> qwen-uni-7b-v3 ..."
CKPT=out_uni_v3_adapter OUT_DIR=qwen-uni-7b-v3 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v3/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v3 OUT_NAME=qwen-uni-7b-v3 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v3-f16.gguf
[ -f gguf/qwen-uni-7b-v3-Q5_K_M.gguf ] || fail "gguf-missing"
ring "v-uni-3 step: quantized, running constant test next"

say "serving on CURRENT llama.cpp (CPU) + constant test (VER=uni-3) ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v3-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni3.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"
VER=uni-3 STUDENT_ID=qwen-uni-7b-v3 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

SCORES=$(tail -1 UNI_RESULTS.md)
say "V-UNI-3 DONE. $SCORES"
ring "v-uni-3 DONE. $SCORES (base-v3: C42% web100% tools25%). Check if tools jumped."
