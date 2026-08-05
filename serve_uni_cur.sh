#!/usr/bin/env bash
# Serve the universal student with the CURRENT (freshly-built) llama.cpp instead
# of LM Studio's bundled 2.25.2 — which mis-renders the USER_DEFINED <tool_call>
# token (151657) as garbage, breaking tool-call parsing. Current build renders it
# correctly and parses tool calls under --jinja.
set -euo pipefail
B="$HOME/code-distill/llama.cpp/build/bin"
F="${STUDENT_GGUF:-$HOME/code-distill/gguf/qwen-uni-7b-Q5_K_M.gguf}"
PORT="${PORT:-8091}"; CTX="${CTX:-4096}"; NGL="${NGL:-99}"
[ -f "$F" ] || { echo "!! GGUF not found: $F"; exit 1; }
[ -x "$B/llama-server" ] || { echo "!! current llama-server not built: $B"; exit 1; }
echo ">> serving $(basename "$F") on :$PORT via CURRENT llama.cpp (--jinja ngl=$NGL)"
exec "$B/llama-server" -m "$F" -ngl "$NGL" -c "$CTX" --jinja --port "$PORT" --host 127.0.0.1
