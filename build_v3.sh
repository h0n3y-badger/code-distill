#!/usr/bin/env bash
# v3: retrain the 7B on C data augmented with header-targeted samples
# (c_hard.jsonl teacher-generated + gold_c.jsonl hand-authored), to fix the
# missing_include failure mode diag_c.py found. Trains into *-v3 so v2 stays
# intact until v3 is proven better. OOM-safe merge (SKIP_MERGE + merge_adapter).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"; LOG="build_v3.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- build clean_v3 = clean_v2 + header-targeted C -------------------------
say "building clean_v3.jsonl (clean_v2 + c_hard + gold_c)..."
$PY - <<'PYEOF'
import datalib
def load(p):
    try: return [l for l in open(p) if l.strip()]
    except FileNotFoundError: return []
base = load("clean_v2.jsonl")
chard = load("c_hard.jsonl")
goldc = load("gold_c.jsonl")
with open("clean_v3.jsonl","w") as f:
    f.write("".join(base))
    for r in chard+goldc:
        f.write(r if r.endswith("\n") else r+"\n")
print(f"clean_v2={len(base)} c_hard={len(chard)} gold_c={len(goldc)} "
      f"-> clean_v3={len(base)+len(chard)+len(goldc)}")
PYEOF
say "clean_v3: $(wc -l < clean_v3.jsonl) rows"

# --- free the GPU (teacher must be down before training) -------------------
pkill -f "[l]lama-server" 2>/dev/null; sync; sleep 4
say "GPU freed; RAM avail $(free -g | awk '/Mem:/{print $7}')G"

# --- train (adapter only; merge separately to dodge the OOM cliff) ---------
say "training 7B v3 (adapter only)..."
SKIP_MERGE=1 ADAPTER_DIR=out_v3 TRAIN_DATA=clean_v3.jsonl $PY train.py >>"$LOG" 2>&1 \
  || { say "FAILED train"; exit 1; }

say "merging adapter in fresh process..."
CKPT=out_v3 OUT_DIR=qwen-coder-7b-mine-v3 $PY merge_adapter.py >>"$LOG" 2>&1 \
  || { say "FAILED merge"; exit 1; }
ls qwen-coder-7b-mine-v3/model-00004-of-00004.safetensors >/dev/null 2>&1 \
  || { say "FAILED merge incomplete"; exit 1; }

say "quantizing -> Q5_K_M"
MERGED_DIR=qwen-coder-7b-mine-v3 OUT_NAME=qwen-coder-7b-mine-v3 QUANT=Q5_K_M VENV=.venv \
  ./quantize.sh >>"$LOG" 2>&1 || { say "FAILED quant"; exit 1; }
rm -f gguf/qwen-coder-7b-mine-v3-f16.gguf

say "V3 BUILD DONE -> gguf/qwen-coder-7b-mine-v3-Q5_K_M.gguf"
