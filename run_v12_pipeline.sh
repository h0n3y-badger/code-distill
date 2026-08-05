#!/usr/bin/env bash
# v-uni-12: fix the REAL v10/v11 field failure — after a web_search returns a
# LARGE, NOISY searxng blob, a plain "give me the profile" ask intermittently gets
# an HTML/markdown DOCUMENT instead of prose (v11 measured 15% @temp0.8 / 10% @0.4
# on the REAL 11KB blob, via LM Studio :1234 — my :8091 harness + small clean blobs
# HID it, which is why v10/v11 both "passed" and failed the user).
#
# Root cause: every research/recall/enrich/subtask row trained on a SMALL CLEAN
# result; reality is a big messy blob. Fix (teacher-FREE, augment_tools.py):
#   (1) INFLATE every tool result to a ~18-hit searxng-style blob (incl. enrich/
#       subtask, whose correct output stays a DOC -> discriminator = INTENT, not
#       blob size; protects the working Florida update case).
#   (2) DERIVE `freshresearch` (plain Q -> search -> PROSE, NO artifact) — the
#       field failure was a fresh chat; only 44 recall rows covered that before.
# No teacher, no generation: reshape + retrain on the v10/v11 winning recipe.
# Validates in LM STUDIO's own runtime (:1234), not the :8091 harness.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni12.log"; : > "$LOG"; B="llama.cpp/build/bin"
DEV="eb32d797b3f0467da06e042599fed67a"
LMID="h0ney-badger/qwen2.5-7b-universal-distill/qwen-uni-7b-v12-q5_k_m.gguf"
LMDIR="$HOME/.lmstudio/models/h0ney-badger/qwen2.5-7b-universal-distill"
rm -f DONE_V12 FAIL_V12
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" >/dev/null 2>&1 || true; }
fail(){ say "FAILED: $1"; echo "$1" > FAIL_V12; pkill -f "llama-server" 2>/dev/null || true; \
        ring "code-distill v12 FAILED at: $1"; exit 1; }

# --- 0. free GPU (unload LM Studio v11 + any llama-server) ------------------
say "freeing GPU (unload LM Studio + llama-server) ..."
lms unload --all >>"$LOG" 2>&1 || true
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 6

# --- 1. augment: big noisy blobs + freshresearch (teacher-free) -------------
say "augmenting tool data (inflate blobs + derive freshresearch) ..."
$PY augment_tools.py >>"$LOG" 2>&1 || fail "augment"
grep -q "freshresearch" multi_aug.jsonl || fail "augment-empty"

# --- 2. mix: v12 source + winning upweights --------------------------------
say "building mix (MULTI_RAW_FILE=multi_aug.jsonl TOOL_UPWEIGHT=1 RESEARCH_UPWEIGHT=5) ..."
MULTI_RAW_FILE=multi_aug.jsonl TOOL_UPWEIGHT=1 RESEARCH_UPWEIGHT=5 $PY mix.py >>"$LOG" 2>&1 || fail "mix"
$PY -c "import json,collections;mt=collections.Counter();[mt.update([json.loads(l).get('mtype')]) for l in open('clean_universal.jsonl') if json.loads(l).get('mtype')];print('[mix] multi types:',dict(mt))" >>"$LOG" 2>&1 || true

# --- 3. train (v10/v11 winning recipe), merge, quantize --------------------
say "training v-uni-12 (general base, r=32 a=64 ep=2, MAX_SEQ=3072) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v12_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ=3072 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"
say "merging -> qwen-uni-7b-v12 ..."
CKPT=out_uni_v12_adapter OUT_DIR=qwen-uni-7b-v12 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v12/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"
say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v12 OUT_NAME=qwen-uni-7b-v12 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v12-f16.gguf
[ -f gguf/qwen-uni-7b-v12-Q5_K_M.gguf ] || fail "gguf-missing"

# --- 4. stage into LM Studio + byte-verify ---------------------------------
cp gguf/qwen-uni-7b-v12-Q5_K_M.gguf "$LMDIR/" && sync
s1=$(stat -c %s gguf/qwen-uni-7b-v12-Q5_K_M.gguf); s2=$(stat -c %s "$LMDIR/qwen-uni-7b-v12-Q5_K_M.gguf")
[ "$s1" = "$s2" ] || fail "stage-byte-mismatch"
say "staged v12 in LM Studio ($s2 bytes)."

# --- 5. constant test on the :8091 harness (regression guard) --------------
say "constant test VER=uni-12 (:8091 regression guard) ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v12-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni12.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "student-serve"
VER=uni-12 STUDENT_ID=qwen-uni-7b-v12 STAMP="$(date +%F)" RP_SAMPLES=6 \
  $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"
FROZEN=$(tail -1 UNI_RESULTS.md)
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 4

# --- 6. FAITHFUL validation in LM Studio's own runtime (:1234) --------------
say "loading v12 into LM Studio for faithful :1234 validation ..."
lms load "$LMID" --context-length 4096 --gpu max -y >>"$LOG" 2>&1 || fail "lms-load"
for i in $(seq 1 30); do curl -s --max-time 3 http://localhost:1234/v1/models >/dev/null 2>&1 && break; sleep 2; done
say "validating big-blob prose (real 11KB blob, temp0.8, N=60) ..."
V12_08=$(MODEL_ID="$LMID" N=60 TEMP=0.8 $PY validate_bigblob.py 2>>"$LOG")
V12_04=$(MODEL_ID="$LMID" N=60 TEMP=0.4 $PY validate_bigblob.py 2>>"$LOG")
say "V-UNI-12 DONE."
say "  big-blob validation  $V12_08   (v11 was 15%)"
say "  big-blob validation  $V12_04   (v11 was 10%)"
say "  frozen: $FROZEN"
echo "done | v12 $V12_08 (v11 15%) | $V12_04 (v11 10%) | $FROZEN" > DONE_V12
ring "code-distill v12 DONE. big-blob HTML rate: $V12_08 (v11 was 15%). $FROZEN"
