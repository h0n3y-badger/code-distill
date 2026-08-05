#!/usr/bin/env bash
# Smoke-test all four models on realistic "write a program" prompts.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="smoke_run.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ # $1 tag  $2 gguf
  pkill -f "[l]lama-server" 2>/dev/null; sleep 2
  STUDENT_GGUF="$PWD/$2" nohup ./serve_student.sh > serve_tmp.log 2>&1 &
  for i in $(seq 1 40); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break; sleep 2; done
  say "smoke: $1"
  $PY smoke_test.py "$1" 2>&1 | tee -a "$LOG"
}
run student-1.5b gguf/qwen-coder-1.5b-py-Q4_K_M.gguf
run base-1.5b    gguf/base-1.5b-Q4_K_M.gguf
run student-7b   gguf/qwen-coder-7b-mine-Q5_K_M.gguf
run base-7b      gguf/base-qwen7b-Q5_K_M.gguf
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
say "SMOKE ALL DONE"
