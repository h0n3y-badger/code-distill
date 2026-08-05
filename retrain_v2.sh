#!/usr/bin/env bash
# v2 retrain with complete-program data added. Trains into *-v2 dirs so the current
# published models stay intact until v2 is proven better. Builds combined datasets,
# retrains 1.5B + 7B, quantizes, and smoke-tests each. ~30-35 min (under the cap).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="retrain_v2.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
smoke(){ # $1 tag  $2 gguf
  pkill -f "[l]lama-server" 2>/dev/null; sleep 2
  STUDENT_GGUF="$PWD/$2" nohup ./serve_student.sh > serve_tmp.log 2>&1 &
  for i in $(seq 1 40); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break; sleep 2; done
  $PY smoke_test.py "$1" 2>&1 | tee -a "$LOG"
  pkill -f "[l]lama-server" 2>/dev/null; sleep 2
}

pkill -f "[l]lama-server" 2>/dev/null; sleep 2   # free VRAM (teacher may still be up)

# --- build combined datasets ---
say "building v2 datasets (functions + complete-programs + gold)..."
$PY - <<'PYEOF'
import json
def load(p):
    out=[]
    try:
        for line in open(p):
            line=line.strip()
            if line: out.append(line)
    except FileNotFoundError: pass
    return out
programs = load("programs.jsonl")
gold     = load("gold_python.jsonl")
pyfunc   = load("python_clean.jsonl")
allfunc  = load("clean.jsonl")
open("python_v2.jsonl","w").write("\n".join(pyfunc+programs+gold)+"\n")
open("clean_v2.jsonl","w").write("\n".join(allfunc+programs+gold)+"\n")
print(f"programs={len(programs)} gold={len(gold)} py_func={len(pyfunc)} all_func={len(allfunc)}")
print(f"-> python_v2.jsonl={len(pyfunc)+len(programs)+len(gold)}  clean_v2.jsonl={len(allfunc)+len(programs)+len(gold)}")
PYEOF
say "python_v2: $(wc -l < python_v2.jsonl) | clean_v2: $(wc -l < clean_v2.jsonl)"

# --- 1.5B v2 (priority: the W541 model) ---
say "training 1.5B v2..."
TRAIN_DATA=python_v2.jsonl OUT_DIR=qwen-coder-1.5b-py-v2 $PY train_1b.py >>"$LOG" 2>&1 || { say "FAILED 1.5B train"; exit 1; }
say "quantizing 1.5B v2 -> Q4_K_M"
MERGED_DIR=qwen-coder-1.5b-py-v2 OUT_NAME=qwen-coder-1.5b-py-v2 QUANT=Q4_K_M VENV=.venv ./quantize.sh >>"$LOG" 2>&1 || { say "FAILED 1.5B quant"; exit 1; }
say "smoke 1.5B v2:"
smoke student-1.5b-v2 gguf/qwen-coder-1.5b-py-v2-Q4_K_M.gguf

# --- 7B v2 ---
say "training 7B v2..."
TRAIN_DATA=clean_v2.jsonl OUT_DIR=qwen-coder-7b-mine-v2 $PY train.py >>"$LOG" 2>&1 || { say "FAILED 7B train"; exit 1; }
say "quantizing 7B v2 -> Q5_K_M"
MERGED_DIR=qwen-coder-7b-mine-v2 OUT_NAME=qwen-coder-7b-mine-v2 QUANT=Q5_K_M VENV=.venv ./quantize.sh >>"$LOG" 2>&1 || { say "FAILED 7B quant"; exit 1; }
say "smoke 7B v2:"
smoke student-7b-v2 gguf/qwen-coder-7b-mine-v2-Q5_K_M.gguf

say "RETRAIN_V2 DONE"
