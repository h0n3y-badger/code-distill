#!/usr/bin/env bash
# v-uni-7: confirm-the-ceiling / strict-dominance attempt. General base (tools
# solved in v6), rebalanced: TOOL_UPWEIGHT=2 (tools native, don't over-weight) +
# CODE_UPWEIGHT=2 to try to recover C to >= base-v3's 5/12. If C doesn't move,
# that confirms the small-model C ceiling and v6/v7 is the win. Silent; ring only
# on blocker.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni7.log"; : > "$LOG"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }
fail(){ say "FAILED: $1"; ring "v-uni-7 BLOCKER: failed at $1 — see build_uni7.log"; exit 1; }

say "rebuild mix (TOOL_UPWEIGHT=2, CODE_UPWEIGHT=2) ..."
TOOL_UPWEIGHT=2 CODE_UPWEIGHT=2 $PY mix.py >>"$LOG" 2>&1 || fail "mix"

pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5
say "training v-uni-7 from general base (r=32, epochs=2, code-boosted) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v7_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ=2048 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v7 ..."
CKPT=out_uni_v7_adapter OUT_DIR=qwen-uni-7b-v7 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v7/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v7 OUT_NAME=qwen-uni-7b-v7 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v7-f16.gguf
[ -f gguf/qwen-uni-7b-v7-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving + constant test VER=uni-7 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v7-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni7.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"
VER=uni-7 STUDENT_ID=qwen-uni-7b-v7 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

say "V-UNI-7 DONE. $(tail -1 UNI_RESULTS.md)"
