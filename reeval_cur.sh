#!/usr/bin/env bash
# Re-run the constant test for BOTH base-v3 and uni-1 on the CURRENT llama.cpp
# runtime (correct <tool_call> rendering + --jinja tool parsing), so the
# comparison is apples-to-apples. Appends rows to UNI_RESULTS.md.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }

run_one(){  # $1=label  $2=gguf  $3=model-id
  local label="$1" gguf="$2" mid="$3"
  echo "[$(date +%H:%M:%S)] serving $label ($gguf) on current runtime..."
  pkill -f "llama-server" 2>/dev/null || true; sleep 3
  "$B/llama-server" -m "$gguf" -ngl 99 -c 4096 --jinja --port 8091 --host 127.0.0.1 \
    > "serve_${label}.log" 2>&1 &
  for i in $(seq 1 60); do
    curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2
  done
  curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || { echo "serve FAILED $label"; return 1; }
  echo "[$(date +%H:%M:%S)] eval $label..."
  VER="$label" STUDENT_ID="$mid" STAMP="$(date +%F)" $PY eval_universal.py 2>&1 | grep -E 'pass@1|valid|correct|kw-hit|VER='
}

run_one "base-v3-cur" "gguf/qwen-coder-7b-mine-v3-Q5_K_M.gguf" "v3"
run_one "uni-1-cur"   "gguf/qwen-uni-7b-Q5_K_M.gguf"           "uni1"
pkill -f "llama-server" 2>/dev/null || true
echo "=== RESULTS ==="; tail -5 UNI_RESULTS.md
ring "re-eval on correct runtime done: $(tail -1 UNI_RESULTS.md)"
