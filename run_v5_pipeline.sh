#!/usr/bin/env bash
# v-uni-5: transplant the tool-token embeddings AND SCALE them up to ~normal-token
# magnitude (TRANSPLANT_SCALE=50 -> norm ~0.85), FROZEN (no embed training). At
# tiny norm (0.017-0.057) a normal token like 'gMaps' always out-scored <tool_call>;
# at ~normal magnitude a well-aligned hidden state can win, and attn/MLP LoRA only
# has to aim the hidden state. Frozen embed => no full-vocab-logit OOM => batch=2.
# Rings time-gated after 21:00.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni5.log"; : > "$LOG"
B="llama.cpp/build/bin"; DEV="eb32d797b3f0467da06e042599fed67a"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ if [ "$(date +%H%M)" -ge 2100 ]; then kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; fi; }
fail(){ say "FAILED: $1"; ring "v-uni-5 FAILED at: $1 — see build_uni5.log"; exit 1; }

GEN=$(ls -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"*/ 2>/dev/null | head -1)
[ -n "$GEN" ] || fail "no-general-model"

say "rebuild mix (TOOL_UPWEIGHT=3) ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"

pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5
say "training v-uni-5 (transplant SCALE=50, embed FROZEN, r=32, batch=2, epochs=2) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v5_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL=qwen-coder-7b-mine-v3 LORA_R=32 LORA_ALPHA=64 EPOCHS=2 \
  MAX_SEQ=2048 TRANSPLANT_EMBED="$GEN" TRANSPLANT_SCALE=50 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v5 ..."
CKPT=out_uni_v5_adapter OUT_DIR=qwen-uni-7b-v5 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v5/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v5 OUT_NAME=qwen-uni-7b-v5 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v5-f16.gguf
[ -f gguf/qwen-uni-7b-v5-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving (current llama.cpp CPU) + constant test VER=uni-5 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v5-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni5.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "serve"
VER=uni-5 STUDENT_ID=qwen-uni-7b-v5 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

SCORES=$(tail -1 UNI_RESULTS.md)
say "V-UNI-5 DONE. $SCORES"
# no auto-ring: overnight policy is win/blocker only; Claude rings conditionally.
