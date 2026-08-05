#!/usr/bin/env bash
# v-uni-10: keep v9's tool-call-mode-lock FIX (enrich/research data) but RECOVER
# the coding/web regression it caused. v9's mix was tool/multi-dominated because
# TOOL_UPWEIGHT=3 (a legacy from when tools was the WEAK skill) bloated tools to
# 1158 rows vs code 838. Tools is saturated (100%) and no longer needs it, so
# drop TOOL_UPWEIGHT to 1 — code/web regain their share, the enrich/research
# toollock upweight (x3) stays to hold the fix. NO new data: pure re-mix +
# retrain on the banked v9 data. Detached-safe; NO rings (morning_gate handles).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_uni10.log"; : > "$LOG"; B="llama.cpp/build/bin"
rm -f DONE_V10 FAIL_V10
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ say "FAILED: $1"; echo "$1" > FAIL_V10; pkill -f "llama-server" 2>/dev/null || true; exit 1; }

# free the GPU (v9 eval server is still serving on :8091)
say "freeing GPU ..."
pkill -f "llama-server" 2>/dev/null || true; sync; sleep 6

# rebalanced mix: TOOL_UPWEIGHT=1 (was 3); IMG/TOOLLOCK upweights keep config defaults
say "building rebalanced mix (TOOL_UPWEIGHT=1) ..."
TOOL_UPWEIGHT=1 $PY mix.py >>"$LOG" 2>&1 || fail "mix"
$PY - <<'PYEOF' >>"$LOG" 2>&1 || true
import json,collections
kinds=collections.Counter(); mt=collections.Counter()
lens=[]
for l in open("clean_universal.jsonl"):
    o=json.loads(l); kinds[o.get("kind","?")]+=1
    if o.get("mtype"): mt[o["mtype"]]+=1
    lens.append(sum(len(m.get("content") or "") for m in o["messages"]))
lens.sort(); n=len(lens)
print(f"[mix] {n} rows by kind {dict(kinds)} | multi types {dict(mt)} | char p95={lens[int(n*0.95)]} max={lens[-1]}")
PYEOF

say "training v-uni-10 (general base, r=32 a=64 ep=2, MAX_SEQ=3072) ..."
SKIP_MERGE=1 ADAPTER_DIR=out_uni_v10_adapter TRAIN_DATA=clean_universal.jsonl \
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" LORA_R=32 LORA_ALPHA=64 EPOCHS=2 MAX_SEQ=3072 \
  $PY train_uni.py >>"$LOG" 2>&1 || fail "train"

say "merging -> qwen-uni-7b-v10 ..."
CKPT=out_uni_v10_adapter OUT_DIR=qwen-uni-7b-v10 $PY merge_adapter.py >>"$LOG" 2>&1 || fail "merge"
ls qwen-uni-7b-v10/*.safetensors >/dev/null 2>&1 || fail "merge-incomplete"

say "quantizing -> Q5_K_M ..."
MERGED_DIR=qwen-uni-7b-v10 OUT_NAME=qwen-uni-7b-v10 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || fail "quantize"
rm -f gguf/qwen-uni-7b-v10-f16.gguf
[ -f gguf/qwen-uni-7b-v10-Q5_K_M.gguf ] || fail "gguf-missing"

say "serving + constant test VER=uni-10 ..."
pkill -f "llama-server" 2>/dev/null || true; sleep 3
"$B/llama-server" -m gguf/qwen-uni-7b-v10-Q5_K_M.gguf -ngl 0 -c 4096 -t "$(nproc)" \
  --jinja --port 8091 --host 127.0.0.1 > serve_uni10.log 2>&1 &
for i in $(seq 1 60); do curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s --max-time 3 http://localhost:8091/v1/models >/dev/null 2>&1 || fail "student-serve"
VER=uni-10 STUDENT_ID=qwen-uni-7b-v10 STAMP="$(date +%F)" $PY eval_universal.py >>"$LOG" 2>&1 || fail "eval"

say "V-UNI-10 DONE."
say "frozen: $(tail -1 UNI_RESULTS.md)"
echo done > DONE_V10