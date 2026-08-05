#!/usr/bin/env bash
# Loud phone alert via KDE Connect "find my phone" (--ring: loud, bypasses DND)
# plus a message. Used for milestone/step/iteration alerts in the universal build.
#   ./ring.sh "v-uni-2 DONE: tools 7/8"
DEV="eb32d797b3f0467da06e042599fed67a"   # Pixel 8a
kdeconnect-cli -d "$DEV" --ring --ping-msg "${1:-code-distill milestone}" 2>/dev/null || true
