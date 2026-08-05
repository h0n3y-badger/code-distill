#!/usr/bin/env bash
# v-uni-2: strengthen the under-learned <tool_call> emission. Rebuild mix with
# tool rows upweighted 3x, train with r=32/alpha=64/3 epochs, merge, quantize,
# then eval on the CURRENT llama.cpp (CPU) which renders/parses tool calls
# correctly. Rings phone on done/fail. Detached-safe: setsid bash run_v2_pipeline.sh &
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni2.log"; : > "$LOG"
B="llama.cpp/build/bin"; DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }
fail(){ say "FAILED: $1"; ring "v-uni-2 FAILED at: $1 — see build_uni2.log"; exit 1; }

say "rebuild mix with TOOL_UPWEIGHT=3 ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"

say "freeing GPU ..."
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5

say "training v-uni-2 (r=32 alpha=64 epochs=3, tools upweighted) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v2_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL=qwen-coder-7b-mine-v3 LORA_R=32 LORA_ALPHA=64 EPOCHS=3 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v2 ..."
CKPT=out_uni_v2_adapter OUT_DIR=qwen-uni-7b-v2 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v2/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v2 OUT_NAME=qwen-uni-7b-v2 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v2-f16.gguf
[ -f gguf/qwen-uni-7b-v2-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving v-uni-2 on CURRENT llama.cpp (CPU) for correct tool parsing ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v2-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni2.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"

say "constant test (VER=uni-2) ..."
VER=uni-2 STUDENT_ID=qwen-uni-7b-v2 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

SCORES=$(tail -1 UNI_RESULTS.md)
say "V-UNI-2 DONE. $SCORES"
ring "v-uni-2 DONE. $SCORES (base-v3: C42% web100% tools25%)."
