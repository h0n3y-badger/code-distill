#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/code-distill"; PY=".venv/bin/python"; LOG="eval_v2.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
srv(){ pkill -f "[l]lama-server" 2>/dev/null; sleep 2
  STUDENT_GGUF="$PWD/$1" nohup ./serve_student.sh > serve_tmp.log 2>&1 &
  for i in $(seq 1 40); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break; sleep 2; done; }
say "eval 1.5B v2 (Python function-eval)..."
srv gguf/qwen-coder-1.5b-py-v2-Q4_K_M.gguf
$PY eval_py.py > eval_1b_v2.log 2>&1; say "1.5B v2: $(grep PY_PASS eval_1b_v2.log | tail -1)"
say "eval 7B v2 (Python+C function-eval)..."
srv gguf/qwen-coder-7b-mine-v2-Q5_K_M.gguf
$PY eval.py > eval_7b_v2.log 2>&1
say "7B v2:"; grep -E "Python|^\s*C |TOTAL" eval_7b_v2.log | tail -4 | tee -a "$LOG"
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
say "EVAL_V2 DONE"
