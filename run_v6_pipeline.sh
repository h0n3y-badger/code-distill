#!/usr/bin/env bash
# v-uni-6: STRATEGIC PIVOT. Train from the GENERAL Qwen2.5-7B-Instruct base
# (ships WORKING tool tokens + emits <tool_call> natively) instead of the coder
# base (tool tokens dead/zero — unfixable in 6 tries). Coding is rebuilt from the
# REPLAY data already in the mix (all verified C/HTML/Java + downsampled Python).
# Plain attn/MLP LoRA — no transplant/embed surgery needed. Silent (win/blocker
# policy); rings only on failure (blocker).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni6.log"; : > "$LOG"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }   # blockers ring anytime
fail(){ say "FAILED: $1"; ring "v-uni-6 BLOCKER: failed at $1 — need you. see build_uni6.log"; exit 1; }

say "rebuild mix (TOOL_UPWEIGHT=3) ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"

pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5
say "training v-uni-6 from GENERAL base Qwen2.5-7B-Instruct (r=32, epochs=2) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v6_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ=2048 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v6 ..."
CKPT=out_uni_v6_adapter OUT_DIR=qwen-uni-7b-v6 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v6/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v6 OUT_NAME=qwen-uni-7b-v6 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v6-f16.gguf
[ -f gguf/qwen-uni-7b-v6-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving + constant test VER=uni-6 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v6-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni6.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"
VER=uni-6 STUDENT_ID=qwen-uni-7b-v6 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

say "V-UNI-6 DONE. $(tail -1 UNI_RESULTS.md)"
