#!/usr/bin/env bash
# v-uni-4: TRANSPLANT healthy <tool_call>/</tool_call> embeddings from general
# Qwen2.5-7B-Instruct into the coder base (its own are dead/zero), THEN train
# (embed refine on, from the good init) with tools upweighted. This is the fix
# for the root cause found in v1-v3. Rings are TIME-GATED: only fire after 21:00
# (user watching a show until then).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni4.log"; : > "$LOG"
B="llama.cpp/build/bin"; DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ if [ "$(date +%H%M)" -ge 2100 ]; then kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; fi; }
fail(){ say "FAILED: $1"; ring "v-uni-4 FAILED at: $1 — see build_uni4.log"; exit 1; }

# resolve the general-instruct snapshot dir (transplant source)
GEN=$(ls -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
[ -n "$GEN" ] || fail "no-general-model-for-transplant"
say "transplant source: $GEN"

say "rebuild mix (TOOL_UPWEIGHT=3) ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"

pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5
say "training v-uni-4 (transplant + embed refine, r=32, epochs=2) ..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v4_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL=qwen-coder-7b-mine-v3 LORA_R=32 LORA_ALPHA=64 EPOCHS=2 TRAIN_EMBED=1 \
  BATCH=1 GRAD_ACCUM=16 MAX_SEQ=2048 OPTIM=paged_adamw_8bit \
  TRANSPLANT_EMBED="$GEN" \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"
ring "v-uni-4: training done, building"

say "merging -> qwen-uni-7b-v4 ..."
CKPT=out_uni_v4_adapter OUT_DIR=qwen-uni-7b-v4 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v4/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v4 OUT_NAME=qwen-uni-7b-v4 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v4-f16.gguf
[ -f gguf/qwen-uni-7b-v4-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving (current llama.cpp CPU) + constant test VER=uni-4 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v4-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni4.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"
VER=uni-4 STUDENT_ID=qwen-uni-7b-v4 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

SCORES=$(tail -1 UNI_RESULTS.md)
say "V-UNI-4 DONE. $SCORES"
ring "v-uni-4 DONE. $SCORES (base-v3: C42% web100% tools25%). transplant fix — did tools move?"
