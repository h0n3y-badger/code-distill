#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/code-distill"; PY=".venv/bin/python"; LOG="build_7b_v2b.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
pkill -f "[l]lama-server" 2>/dev/null; sync; sleep 3
say "merging 7B v2 adapter in fresh process (RAM: $(free -g | awk '/Mem:/{print $7}')G avail)..."
CKPT=out/checkpoint-152 OUT_DIR=qwen-coder-7b-mine-v2 $PY merge_adapter.py >>"$LOG" 2>&1 || { say "FAILED merge"; exit 1; }
ls qwen-coder-7b-mine-v2/model-00004-of-00004.safetensors >/dev/null 2>&1 || { say "FAILED merge incomplete"; exit 1; }
say "merge OK; quantizing -> Q5_K_M"
MERGED_DIR=qwen-coder-7b-mine-v2 OUT_NAME=qwen-coder-7b-mine-v2 QUANT=Q5_K_M VENV=.venv ./quantize.sh >>"$LOG" 2>&1 || { say "FAILED quant"; exit 1; }
rm -f gguf/qwen-coder-7b-mine-v2-f16.gguf
say "smoke 7B v2..."
STUDENT_GGUF="$PWD/gguf/qwen-coder-7b-mine-v2-Q5_K_M.gguf" nohup ./serve_student.sh > serve_tmp.log 2>&1 &
for i in $(seq 1 40); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break; sleep 2; done
$PY smoke_test.py student-7b-v2 >>"$LOG" 2>&1
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
say "7B_V2B DONE"
