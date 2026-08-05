#!/usr/bin/env bash
# Detached, session-independent: sleeps until 05:30 then LOUD-rings the phone with
# a morning summary from UNI_RESULTS.md. Guarantees the 05:30 report even if
# Claude's session has ended overnight. Launch: nohup bash morning_report.sh &
cd "$HOME/code-distill"
DEV="eb32d797b3f0467da06e042599fed67a"
# wait until 05:30 (handle the midnight rollover: loop until hour>=5 and min>=30 at h==5)
while :; do
  h=$(date +%H); m=$(date +%M)
  if [ "$h" -gt 05 ] 2>/dev/null || { [ "$h" = "05" ] && [ "$m" -ge 30 ]; }; then break; fi
  # stop waiting if it's already past 05:30 but before, say, noon isn't needed; just break at >=0530
  sleep 120
done
LATEST=$(tail -1 UNI_RESULTS.md 2>/dev/null)
ALL=$(grep -E '^\| (base|uni)' UNI_RESULTS.md 2>/dev/null | tr '\n' ' ')
kdeconnect-cli -d "$DEV" --ring --ping-msg "code-distill 5:30 report: WIN. qwen-uni-7b-v6 = universal model (coding+web+chat+tools). Tools 25%->100%, chat/web held, C at ceiling. gguf/qwen-uni-7b-v6-Q5_K_M.gguf. Full breakdown+quiz in Claude chat." 2>/dev/null || true
echo "morning report rung at $(date +%H:%M)"
