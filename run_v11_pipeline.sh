#!/usr/bin/env bash
# v-uni-11: fix the v10 field-test residual — when an artifact is in history and
# the user says "no html, just search & tell me the <profile/rundown>", v10
# INTERMITTENTLY wraps the plain answer in a ```html/```markdown doc instead of
# prose. New eval axis research_prose (multi-sample, messy searxng-style blob)
# measures it: v10 baseline = 12/18 (67%), with the SpaceX "launch profile" case
# failing 0/6. Fix = (a) more research (search->PROSE) data with terse "no html"
# phrasing, (b) research gets its OWN upweight (x5) > enrich's toollock x3, so
# prose-after-search out-masses fence-after-search. enrich (search->revise-doc,
# which WORKED in the field) is untouched. NO change to the winning recipe.
# Detached-safe. Rings the phone ONCE on completion (user: "ping me when anything
# major happens") with the research_prose delta + frozen axes.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni11.log"; : > "$LOG"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
rm -f DONE_V11 FAIL_V11
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" >/dev/null 2>&1 || true; }
fail(){ say "FAILED: $1"; echo "$1" > FAIL_V11; pkill -f "llama-server" 2>/dev/null || true; \
        ring "code-distill v11 FAILED at: $1"; exit 1; }
GEN_TIMEOUT="${GEN_TIMEOUT:-7200}"   # 2h cap on the small research top-up

# --- 0/1. free GPU, bring up the 32B teacher -------------------------------
say "unloading LM Studio + freeing GPU + starting 32B teacher ..."
lms unload --all >>"$LOG" 2>&1 || true
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 6
setsid bash serve_teacher_32b.sh > serve_teacher_32b.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "teacher-serve"
say "teacher up."

# --- 2. top up ONLY research (terse 'no html' phrasing, prose answers) ------
CUR=$(wc -l < multi_raw.jsonl 2>/dev/null || echo 0)
TARGET=$((CUR + 60))
say "topping up research: multi $CUR -> $TARGET ..."
MULTI_ONLY="research" MULTI_TARGET="$TARGET" \
  timeout "$GEN_TIMEOUT" $PY gen_multi.py >> gen_multi.log 2>&1 || true
say "multi now $(wc -l < multi_raw.jsonl) ($(
  $PY -c "import json,collections;print(dict(collections.Counter(json.loads(l).get('mtype') for l in open('multi_raw.jsonl'))))" 2>/dev/null))"

# --- 3. teacher down, free VRAM --------------------------------------------
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 8

# --- 4. mix: TOOL_UPWEIGHT=1 (v10 winner), RESEARCH_UPWEIGHT=5 (new) --------
say "building mix (TOOL_UPWEIGHT=1 RESEARCH_UPWEIGHT=5) ..."
TOOL_UPWEIGHT=1 RESEARCH_UPWEIGHT=5 $PY mix.py >>"$LOG" 2>&1 || fail "mix"
$PY - <<'PYEOF' >>"$LOG" 2>&1 || true
import json,collections
kinds=collections.Counter(); mt=collections.Counter()
for l in open("clean_universal.jsonl"):
    o=json.loads(l); kinds[o.get("kind","?")]+=1
    if o.get("mtype"): mt[o["mtype"]]+=1
print(f"[mix] {sum(kinds.values())} rows by kind {dict(kinds)} | multi types {dict(mt)}")
PYEOF

# --- 5. train from the GENERAL base (v6/v8/v10 winning recipe) --------------
say "training v-uni-11 (general base, r=32 a=64 ep=2, MAX_SEQ=3072) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v11_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ=3072 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v11 ..."
CKPT=out_uni_v11_adapter OUT_DIR=qwen-uni-7b-v11 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v11/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v11 OUT_NAME=qwen-uni-7b-v11 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v11-f16.gguf
[ -f gguf/qwen-uni-7b-v11-Q5_K_M.gguf ] || fail "gguf-missing"

# --- 6. serve + constant test (frozen + v8 axes + NEW research_prose) -------
say "serving + constant test VER=uni-11 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v11-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni11.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "student-serve"
VER=uni-11 STUDENT_ID=qwen-uni-7b-v11 STAMP="$(date +%F)" RP_SAMPLES=6 \
  $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

# pull the research_prose line + frozen row for the ping
RP=$(grep "research-prose" "$LOG" | tail -1)
FROZEN=$(tail -1 UNI_RESULTS.md)
say "V-UNI-11 DONE.  $RP"
say "frozen: $FROZEN"
echo "done $RP" > DONE_V11
ring "code-distill v11 DONE. $RP (v10 was 12/18). frozen: $FROZEN"
