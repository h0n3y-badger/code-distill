#!/usr/bin/env bash
# Watchdog for the overnight generation run. Samples total kept-sample count
# every 15 min. Exits 0 when all three targets are reached (done), exits 2 if
# progress stalls (no growth for 30 min => teacher hung/crashed), exits 3 if the
# teacher endpoint goes away. Either way the harness notifies on exit.
cd "$(dirname "$0")"
TARGET_TOTAL=1550          # 450 web + 550 chat + 550 tools
STALL_LIMIT=2              # consecutive no-growth samples (2 x 15min = 30min)
last=-1; stall=0
while true; do
  sleep 900
  if ! curl -s --max-time 5 http://localhost:8091/v1/models >/dev/null 2>&1; then
    echo "WATCHDOG: teacher endpoint DOWN"; exit 3
  fi
  w=$(wc -l < web_raw.jsonl 2>/dev/null || echo 0)
  c=$(wc -l < chat_raw.jsonl 2>/dev/null || echo 0)
  t=$(wc -l < tools_raw.jsonl 2>/dev/null || echo 0)
  tot=$((w + c + t))
  echo "WATCHDOG $(cat /proc/uptime | cut -d' ' -f1)s: web=$w chat=$c tools=$t total=$tot"
  if [ "$tot" -ge "$TARGET_TOTAL" ]; then echo "WATCHDOG: targets reached"; exit 0; fi
  if [ "$tot" -le "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$tot
  if [ "$stall" -ge "$STALL_LIMIT" ]; then echo "WATCHDOG: STALLED at $tot"; exit 2; fi
done
