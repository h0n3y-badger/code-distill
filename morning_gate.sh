#!/usr/bin/env bash
# Time-gated overnight notifier (detached, survives session death). User policy
# for this run:
#   - SILENT before 04:00 America/New_York (no ping at all, blockers included)
#   - 04:00-05:30 ET: ring ONLY for a breakthrough (BREAKTHROUGH.txt exists)
#   - 05:30 ET: ring the full status (OVERNIGHT_STATUS.txt), then exit
# Claude writes BREAKTHROUGH.txt when it confirms the tool-call mode-lock is
# fixed, and keeps OVERNIGHT_STATUS.txt current after each iteration.
cd "$HOME/code-distill"
DEV=eb32d797b3f0467da06e042599fed67a
ring(){ kdeconnect-cli -d "$DEV" --ring --ping-msg "$1" 2>/dev/null || true; }
now=$(date +%s)
t400=$(TZ=America/New_York date -d 'today 04:00' +%s); [ "$t400" -le "$now" ] && t400=$(TZ=America/New_York date -d 'tomorrow 04:00' +%s)
t530=$(TZ=America/New_York date -d 'today 05:30' +%s); [ "$t530" -le "$now" ] && t530=$(TZ=America/New_York date -d 'tomorrow 05:30' +%s)
bt=0
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$t400" ] && [ "$bt" = 0 ] && [ -f BREAKTHROUGH.txt ]; then
    ring "code-distill BREAKTHROUGH: $(head -c 400 BREAKTHROUGH.txt)"
    bt=1
  fi
  if [ "$now" -ge "$t530" ]; then
    ring "code-distill 5:30 report: $(head -c 400 OVERNIGHT_STATUS.txt 2>/dev/null)"
    exit 0
  fi
  sleep 60
done
