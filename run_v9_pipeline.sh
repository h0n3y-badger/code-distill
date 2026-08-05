#!/usr/bin/env bash
# v-uni-9: fix the CONFIRMED weights mode-lock (isolation probe: v8 fires tools in
# a fresh chat but NEVER once an assistant artifact is in history — it fabricates
# instead). Adds two multi-turn types that put a <tool_call> AFTER an existing
# artifact:  enrich (search -> revise the artifact)  and  research (search ->
# answer in PROSE, no doc).  Also folds in the placeholder-image gold.
#
# CHEAP run: web/chat/tools data is REUSED from v8 (already banked) — only the two
# new multi types are generated, then retrain on the same winning recipe.
# Detached-safe; rings the phone only on a blocker; success writes DONE_V9.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni9.log"; : > "$LOG"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
rm -f DONE_V9 FAIL_V9
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# NOTE: no phone rings here — ALL notifications are time-gated by morning_gate.sh
# (user policy: silent before 4:00 ET, breakthroughs 4:00-5:30, report at 5:30).
ring(){ :; }
fail(){ say "FAILED: $1"; echo "$1" > FAIL_V9; \
        pkill -f "llama-server" 2>/dev/null || true; exit 1; }
GEN_TIMEOUT="${GEN_TIMEOUT:-10800}"   # 3h cap on the (smaller) generation

# --- 0/1. free GPU, bring up the 32B teacher -------------------------------
say "unloading LM Studio + starting 32B teacher ..."
lms unload --all >>"$LOG" 2>&1 || true
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 5
setsid bash serve_teacher_32b.sh > serve_teacher_32b.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "teacher-serve"
say "teacher up."

# --- 2. generate ONLY the new multi types (top up multi_raw) + image gold ---
CUR=$(wc -l < multi_raw.jsonl 2>/dev/null || echo 0)
TARGET=$((CUR + 300))
say "topping up multi_raw with enrich+research: $CUR -> $TARGET ..."
MULTI_ONLY="enrich,research" MULTI_TARGET="$TARGET" \
  timeout "$GEN_TIMEOUT" $PY gen_multi.py >> gen_multi.log 2>&1 || true
$PY gold_images.py >>"$LOG" 2>&1 || fail "gold_images"
say "multi now $(wc -l < multi_raw.jsonl) ($(
  $PY -c "import json,collections;print(dict(collections.Counter(json.loads(l).get('mtype') for l in open('multi_raw.jsonl'))))" 2>/dev/null))"

# --- 3. teacher down, free VRAM --------------------------------------------
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 8

# --- 4. mix (image + toollock upweights on by default in config) -----------
say "building mix ..."
TOOL_UPWEIGHT=3 $PY mix.py >>"$LOG" 2>&1 || fail "mix"
$PY - <<'PYEOF' >>"$LOG" 2>&1 || true
import json
lens=[sum(len(m.get("content") or "") for m in json.loads(l)["messages"]) for l in open("clean_universal.jsonl")]
lens.sort(); n=len(lens); print(f"[mix] {n} rows; char-len p50={lens[n//2]} p95={lens[int(n*0.95)]} max={lens[-1]}")
PYEOF

# --- 5. train from the GENERAL base (v6/v8 winning recipe) ------------------
say "training v-uni-9 (general base, r=32 a=64 ep=2, MAX_SEQ=3072) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v9_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ=3072 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v9 ..."
CKPT=out_uni_v9_adapter OUT_DIR=qwen-uni-7b-v9 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v9/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v9 OUT_NAME=qwen-uni-7b-v9 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v9-f16.gguf
[ -f gguf/qwen-uni-7b-v9-Q5_K_M.gguf ] || fail "gguf-missing"

# --- 6. serve + constant test (frozen + v8/v9 axes incl. artifact_search) ---
say "serving + constant test VER=uni-9 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v9-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni9.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "student-serve"
VER=uni-9 STUDENT_ID=qwen-uni-7b-v9 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

say "V-UNI-9 DONE."
say "frozen: $(tail -1 UNI_RESULTS.md)"
say "v8axes: $(tail -1 UNI_RESULTS_V8.md)"
echo done > DONE_V9