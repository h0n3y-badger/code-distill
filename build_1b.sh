#!/usr/bin/env bash
# Build a Python-focused 1.5B model for the ThinkPad W541: train -> quantize to
# Q4_K_M (~1GB, CPU-friendly) -> eval vs stock 1.5B. Fully autonomous; writes
# RESULTS_1B.md. Base eval is non-fatal (student is the deliverable).
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"
LOG="build_1b.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
serve_wait(){ pkill -f "[l]lama-server" 2>/dev/null; sleep 2
  STUDENT_GGUF="$1" nohup ./serve_student.sh > serve_tmp.log 2>&1 &
  for i in $(seq 1 45); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && return 0; sleep 2; done
  return 1; }

pkill -f "[l]lama-server" 2>/dev/null; sleep 2

# 1) train the Python 1.5B student (downloads base on first run)
say "training 1.5B Python student (376 samples, 3 epochs)..."
$PY train_1b.py >>"$LOG" 2>&1 || { say "FAILED: train"; exit 1; }
grep -q "Merged model written" "$LOG" || { say "FAILED: train merge marker"; exit 1; }
say "trained + merged -> qwen-coder-1.5b-py"

# 2) quantize student -> Q4_K_M (the W541 artifact)
say "quantizing student -> Q4_K_M"
MERGED_DIR=qwen-coder-1.5b-py OUT_NAME=qwen-coder-1.5b-py QUANT=Q4_K_M VENV=.venv ./quantize.sh >>"$LOG" 2>&1 \
  || { say "FAILED: quantize student"; exit 1; }
STU="gguf/qwen-coder-1.5b-py-Q4_K_M.gguf"
[ -f "$STU" ] || { say "FAILED: no student gguf"; exit 1; }
say "student gguf: $(du -h "$STU" | cut -f1)"

# 3) eval student (Python-only)
say "eval student (Python)..."
STU_SCORE="(eval failed)"
if serve_wait "$PWD/$STU"; then
  $PY eval_py.py > eval_1b_student.log 2>&1 && STU_SCORE=$(grep PY_PASS eval_1b_student.log | tail -1)
fi
say "STUDENT: $STU_SCORE"

# 4) base 1.5B comparison (non-fatal)
BASE_SCORE="(skipped)"
SNAP=$(find "$HOME/.cache/huggingface/hub/models--unsloth--qwen2.5-coder-1.5b-instruct/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -n "${SNAP:-}" ] && ls "$SNAP"/*.safetensors >/dev/null 2>&1; then
  say "staging + quantizing base 1.5B..."
  rm -rf base_1b_hf; mkdir base_1b_hf
  ln -sf "$SNAP"/*.safetensors base_1b_hf/
  cp "$SNAP"/config.json base_1b_hf/ 2>/dev/null
  cp "$SNAP"/model.safetensors.index.json base_1b_hf/ 2>/dev/null || true
  cp qwen-coder-1.5b-py/tokenizer.json qwen-coder-1.5b-py/tokenizer_config.json \
     qwen-coder-1.5b-py/generation_config.json base_1b_hf/ 2>/dev/null
  if MERGED_DIR=base_1b_hf OUT_NAME=base-1.5b QUANT=Q4_K_M VENV=.venv ./quantize.sh >>"$LOG" 2>&1 \
     && serve_wait "$PWD/gguf/base-1.5b-Q4_K_M.gguf"; then
    $PY eval_py.py > eval_1b_base.log 2>&1 && BASE_SCORE=$(grep PY_PASS eval_1b_base.log | tail -1)
  else
    say "WARN: base build/serve failed — skipping base comparison"
  fi
else
  say "WARN: base 1.5B full weights not in cache — skipping base comparison"
fi
say "BASE: $BASE_SCORE"

pkill -f "[l]lama-server" 2>/dev/null; sleep 2
{
  echo "# 1.5B Python model (for ThinkPad W541) — results"
  echo
  echo "| model | Python pass@1 |"
  echo "|---|---|"
  echo "| stock Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) | ${BASE_SCORE#PY_PASS } |"
  echo "| distilled Python student (Q4_K_M)          | ${STU_SCORE#PY_PASS } |"
  echo
  echo "Artifact: \`$STU\` ($(du -h "$STU" | cut -f1)) — copy to the W541, run with llama.cpp CPU."
} > RESULTS_1B.md
say "DONE -> RESULTS_1B.md"
