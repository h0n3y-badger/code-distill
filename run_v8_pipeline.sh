#!/usr/bin/env bash
# v-uni-8: fix the v6 FIELD-TEST failures — mode-lock (can't re-call a tool /
# answer a plain follow-up / revise once it's emitted an artifact) and code
# presentation (bare text, no ```code box```, minified). Adds a multi-turn
# mode-switch data axis (gen_multi.py), regenerates web multi-line-formatted,
# fences all artifact/code replies at mix time, and measures three NEW eval axes
# (fenced / modeswitch / tool-subtask) alongside the frozen constant test.
#
# Same winning recipe as v6: GENERAL Qwen2.5-7B-Instruct base, r=32 a=64 ep=2.
# Fully self-contained + detached-safe. Rings the phone ONLY on a blocker
# (per the win/blocker policy); success is silent — a DONE sentinel lets Claude
# pick it up, judge win-vs-iterate, and ring the WIN itself.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni8.log"; : > "$LOG"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
rm -f DONE_V8 FAIL_V8
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }
fail(){ say "FAILED: $1"; echo "$1" > FAIL_V8; \
        ring "v-uni-8 BLOCKER: failed at $1 — need you. see build_uni8.log"; \
        pkill -f "llama-server" 2>/dev/null || true; exit 1; }

GEN_TIMEOUT="${GEN_TIMEOUT:-28800}"   # 8h cap on generation; mix proceeds w/ what exists

# --- 0. free the GPU: unload v6 from LM Studio ------------------------------
say "unloading LM Studio models to free VRAM ..."
lms unload --all >>"$LOG" 2>&1 || true
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5

# --- 1. bring up the 32B teacher on :8091 -----------------------------------
say "starting 32B teacher ..."
setsid bash serve_teacher_32b.sh > serve_teacher_32b.log 2>&1 &
for i in $(seq 1 120); do
  curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "teacher-serve"
say "teacher up."

# --- 2. generate: fresh multi-line web + new multi-turn axis (concurrent) ---
# retire the old minified web set so ALL web data is formatted going forward
[ -f web_raw.jsonl ] && ! [ -f web_raw.v6minified.jsonl ] && mv web_raw.jsonl web_raw.v6minified.jsonl
: > web_raw.jsonl
say "generating web (fresh, formatted) + multi (mode-switch) up to ${GEN_TIMEOUT}s ..."
timeout "$GEN_TIMEOUT" $PY gen_web.py   >> gen_web.log   2>&1 &
WPID=$!
timeout "$GEN_TIMEOUT" $PY gen_multi.py >> gen_multi.log 2>&1 &
MPID=$!
wait $WPID; wait $MPID
say "gen done: web=$(wc -l < web_raw.jsonl) multi=$(wc -l < multi_raw.jsonl 2>/dev/null || echo 0) \
chat=$(wc -l < chat_raw.jsonl) tools=$(wc -l < tools_raw.jsonl)"

# --- 3. teacher down, free VRAM for training --------------------------------
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 8

# --- 4. build the mix (fence-normalized; tool-upweight kept from v6) --------
say "building mix ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"
# 3072 headroom for multi-turn rows (v6 used 2048 for single-turn); no
# TRAIN_EMBED needed off the general base, so the full-vocab logit tensor stays
# cheap and 3072 fits 16GB. p99 measured after mix (see build log).
MAXSEQ=3072
$PY - <<'PYEOF' >>"$LOG" 2>&1 || true
import json
lens=[sum(len(m.get("content") or "") for m in json.loads(l)["messages"]) for l in open("clean_universal.jsonl")]
lens.sort(); n=len(lens)
print(f"[mix] {n} rows; char-len p50={lens[n//2]} p95={lens[int(n*0.95)]} max={lens[-1]}")
PYEOF
say "training v-uni-8 (general base, r=32 a=64 ep=2, MAX_SEQ=$MAXSEQ) ..."

# --- 5. train from the GENERAL base (v6's winning pivot) --------------------
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v8_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ="$MAXSEQ" \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v8 ..."
CKPT=out_uni_v8_adapter OUT_DIR=qwen-uni-7b-v8 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v8/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v8 OUT_NAME=qwen-uni-7b-v8 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v8-f16.gguf
[ -f gguf/qwen-uni-7b-v8-Q5_K_M.gguf ] || fail "gguf-missing"

# --- 6. serve + constant test (frozen axes + new v8 axes) -------------------
say "serving + constant test VER=uni-8 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v8-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni8.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "student-serve"
VER=uni-8 STUDENT_ID=qwen-uni-7b-v8 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

say "V-UNI-8 DONE."
say "frozen: $(tail -1 UNI_RESULTS.md)"
say "v8axes: $(tail -1 UNI_RESULTS_V8.md)"
echo done > DONE_V8
