#!/usr/bin/env bash
# Serve the teacher, generate complete-program samples, stop teacher.
set -uo pipefail
cd "$HOME/code-distill"
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
nohup ./serve_teacher.sh > serve_teacher.log 2>&1 &
for i in $(seq 1 45); do curl -s http://127.0.0.1:8091/health 2>/dev/null | grep -q ok && break; sleep 2; done
echo "teacher up; generating complete programs..."
.venv/bin/python gen_programs.py
pkill -f "[l]lama-server" 2>/dev/null; sleep 2
echo "GEN_PROGRAMS_DONE"
