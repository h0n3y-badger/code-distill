#!/usr/bin/env bash
# Launch the three universal-data generators CONCURRENTLY so the 32B teacher's
# llama-server batches them across its slots (~2x aggregate throughput vs one at
# a time). Each is resumable (appends to its own *_raw.jsonl, stops at target).
# The teacher (serve_teacher_32b.sh) must already be listening on :8091.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
echo ">> launching gen_web / gen_chat / gen_tools concurrently"
setsid $PY gen_web.py   > gen_web.log   2>&1 < /dev/null &
setsid $PY gen_chat.py  > gen_chat.log  2>&1 < /dev/null &
setsid $PY gen_tools.py > gen_tools.log 2>&1 < /dev/null &
echo ">> started. tail the *.log files, or: wc -l web_raw.jsonl chat_raw.jsonl tools_raw.jsonl"
wait
echo ">> ALL GENERATORS DONE"
