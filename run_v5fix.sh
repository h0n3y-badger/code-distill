#!/usr/bin/env bash
# Quantize the embed-injected v5 model + constant test (VER=uni-5fix). No retrain
# (v5 adapter already trained to aim at the scaled tool embedding; we only fixed
# the merged model's embedding). Silent (overnight win/blocker policy).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni5fix.log"; : > "$LOG"; B="llama.cpp/build/bin"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "quantizing embed-injected qwen-uni-7b-v5 -> uni-5fix Q5_K_M ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
MERGED_DIR=qwen-uni-7b-v5 OUT_NAME=qwen-uni-7b-v5fix QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || { say "FINISH-FAILED: quantize"; exit 1; }
rm -f gguf/qwen-uni-7b-v5fix-f16.gguf
[ -f gguf/qwen-uni-7b-v5fix-Q5_K_M.gguf ] || { say "FINISH-FAILED: gguf"; exit 1; }
say "serving + constant test VER=uni-5fix ..."
"$B/llama-server" -m gguf/qwen-uni-7b-v5fix-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni5fix.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || { say "FINISH-FAILED: serve"; exit 1; }
VER=uni-5fix STUDENT_ID=qwen-uni-7b-v5fix STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 \
  || { say "FINISH-FAILED: eval"; exit 1; }
say "V-UNI-5FIX DONE. $(tail -1 UNI_RESULTS.md)"
