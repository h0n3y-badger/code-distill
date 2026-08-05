#!/usr/bin/env bash
# RESUME v-uni-1 from the trained adapter (training already succeeded; the prior
# orchestrator run was reaped during merge). Merge -> GGUF Q5_K_M -> serve ->
# constant test -> RING PHONE. Run detached: setsid bash run_v1_resume.sh &
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni.log"          # append to the same log
DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }
fail(){ say "FAILED: $1"; ring "v-uni-1 build FAILED at: $1 — see build_uni.log"; exit 1; }

# ensure nothing is holding the GPU
pkill -f "[l]lama-server" 2>/dev/null || true; sleep 3

say "RESUME: merging adapter (fresh process) -> qwen-uni-7b ..."
CKPT=out_uni_adapter OUT_DIR=qwen-uni-7b $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b OUT_NAME=qwen-uni-7b QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-f16.gguf
[ -f gguf/qwen-uni-7b-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving student (--jinja) for the constant test ..."
STUDENT_GGUF="$HOME/code-distill/gguf/qwen-uni-7b-Q5_K_M.gguf" \
  setsid bash serve_uni.sh > serve_uni.log 2>&1 < /dev/null &
for i in $(seq 1 40); do
  curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 3
done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"

say "running the constant test (VER=uni-1) ..."
VER=uni-1 STUDENT_ID=qwen-uni-7b STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

SCORES=$(tail -1 UNI_RESULTS.md)
say "V-UNI-1 DONE. $SCORES"
ring "v-uni-1 DONE. $SCORES (baseline C42% web100% tools25%). Live on :8091."
say "done; student serving on :8091."
