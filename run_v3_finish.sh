#!/usr/bin/env bash
# Finish v-uni-3 from the already-trained adapter (training done, embed included).
# NO phone rings (user watching a show until 21:00) — writes result to log only;
# the alert is sent manually after 21:00.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni3.log"; B="llama.cpp/build/bin"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

pkill -f "llama-server" 2>/dev/null || true; sleep 3
say "FINISH: merging tied-embed adapter -> qwen-uni-7b-v3 ..."
CKPT=out_uni_v3_adapter OUT_DIR=qwen-uni-7b-v3 $PY merge_adapter.py >>"$LOG" 2>&1 \
  || { say "FINISH-FAILED: merge"; exit 1; }
ls qwen-uni-7b-v3/*.safetensors >/dev/null 2>&1 || { say "FINISH-FAILED: merge-incomplete"; exit 1; }

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v3 OUT_NAME=qwen-uni-7b-v3 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || { say "FINISH-FAILED: quantize"; exit 1; }
rm -f gguf/qwen-uni-7b-v3-f16.gguf
[ -f gguf/qwen-uni-7b-v3-Q5_K_M.gguf ] || { say "FINISH-FAILED: gguf-missing"; exit 1; }

say "serving (current llama.cpp CPU) + constant test VER=uni-3 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v3-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni3.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || { say "FINISH-FAILED: serve"; exit 1; }
VER=uni-3 STUDENT_ID=qwen-uni-7b-v3 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 \
  || { say "FINISH-FAILED: eval"; exit 1; }

say "V-UNI-3-FINISH DONE (NO RING). $(tail -1 UNI_RESULTS.md)"
