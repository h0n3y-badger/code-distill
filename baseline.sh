#!/usr/bin/env bash
# Baseline: run the STOCK Qwen2.5-Coder-7B-Instruct (the exact weights our student
# was trained from) through the SAME eval, at the SAME Q5_K_M quant. Isolates what
# the QLoRA distillation actually did. Uses the full weights already in the HF cache
# (no download) and our existing llama.cpp build.
set -uo pipefail
cd "$HOME/code-distill"
PY=".venv/bin/python"
LOG="baseline.log"; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

pkill -f "[l]lama-server" 2>/dev/null; sleep 2
SNAP=$(find "$HOME/.cache/huggingface/hub/models--unsloth--qwen2.5-coder-7b-instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
say "base weights: $SNAP"

# The unsloth cache snapshot has weights+config but NO tokenizer files, so the
# GGUF converter can't ID the tokenizer. Stage a complete HF dir = base weights +
# tokenizer files from the student merged dir (LoRA doesn't touch the tokenizer, so
# they're byte-identical to the base's).
STAGE="base_hf"
say "staging complete HF dir at $STAGE"
rm -rf "$STAGE"; mkdir -p "$STAGE"
ln -sf "$SNAP"/*.safetensors "$STAGE"/
cp "$SNAP"/config.json "$SNAP"/model.safetensors.index.json "$STAGE"/
cp qwen-coder-7b-mine/tokenizer.json qwen-coder-7b-mine/tokenizer_config.json \
   qwen-coder-7b-mine/generation_config.json "$STAGE"/

if [ ! -f gguf/base-qwen7b-Q5_K_M.gguf ]; then
  say "converting base -> f16 gguf"
  $PY llama.cpp/convert_hf_to_gguf.py "$STAGE" --outfile gguf/base-qwen7b-f16.gguf --outtype f16 >>"$LOG" 2>&1 \
    || { say "FAILED: convert"; exit 1; }
  say "quantizing base -> Q5_K_M"
  ./llama.cpp/build/bin/llama-quantize gguf/base-qwen7b-f16.gguf gguf/base-qwen7b-Q5_K_M.gguf Q5_K_M >>"$LOG" 2>&1 \
    || { say "FAILED: quantize"; exit 1; }
  rm -f gguf/base-qwen7b-f16.gguf   # done with the f16, keep only the Q5
fi
say "base gguf: $(du -h gguf/base-qwen7b-Q5_K_M.gguf | cut -f1)"

say "serving base on :8091"
STUDENT_GGUF="$PWD/gguf/base-qwen7b-Q5_K_M.gguf" nohup ./serve_student.sh > serve_base.log 2>&1 &
for i in $(seq 1 40); do
  curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && { say "base healthy"; break; }
  sleep 2; [ "$i" = 40 ] && { say "FAILED: base never healthy"; exit 1; }
done

say "evaluating base on the same eval_set.jsonl..."
$PY eval.py > baseline_eval.log 2>&1 || { say "FAILED: eval"; exit 1; }
pkill -f "[l]lama-server" 2>/dev/null; sleep 2

say "=== BASE (stock Qwen2.5-Coder-7B-Instruct) results ==="
sed -n '/=== pass@1 by language ===/,$p' baseline_eval.log | tee -a "$LOG"
say "BASELINE COMPLETE"
