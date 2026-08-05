#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/code-distill"; PY=".venv/bin/python"; LOG="build_7b_v2.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
say "training 7B v2 (standalone)..."
TRAIN_DATA=clean_v2.jsonl OUT_DIR=qwen-coder-7b-mine-v2 $PY train.py >>"$LOG" 2>&1 || { say "FAILED train"; exit 1; }
ls qwen-coder-7b-mine-v2/model-00004-of-00004.safetensors >/dev/null 2>&1 || { say "FAILED merge incomplete"; exit 1; }
say "quantizing 7B v2 -> Q5_K_M"
MERGED_DIR=qwen-coder-7b-mine-v2 OUT_NAME=qwen-coder-7b-mine-v2 QUANT=Q5_K_M VENV=.venv ./quantize.sh >>"$LOG" 2>&1 || { say "FAILED quant"; exit 1; }
rm -f gguf/qwen-coder-7b-mine-v2-f16.gguf
say "smoke 7B v2..."
STUDENT_GGUF="$PWD/gguf/qwen-coder-7b-mine-v2-Q5_K_M.gguf" nohup ./serve_student.sh > serve_tmp.log 2>&1 &
for i in $(seq 1 40); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break; sleep 2; done
$PY smoke_test.py student-7b-v2 >>"$LOG" 2>&1
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
say "BUILD_7B_V2 DONE"
