#!/bin/bash
# RAM watchdog for the two full-size jobs; interrupts both if the box starts thrashing.
S=/tmp/claude-1000/-home-dushyant/4f721a6b-64a7-419b-8a33-2e611d728844/scratchpad
s0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
while true; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo); s=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  busy=0; for p in 8188 8189; do n=$(curl -s -m 3 127.0.0.1:$p/queue | python3 -c "import json,sys;print(len(json.load(sys.stdin)['queue_running']))" 2>/dev/null); busy=$((busy + ${n:-0})); done
  echo "$(date +%T) avail=$((a/1024))M swap+=$(((s-s0)/1024))M running=$busy"
  if [ $a -lt 600000 ] || [ $((s - s0)) -gt 6291456 ]; then
    for p in 8188 8189; do curl -s -X POST 127.0.0.1:$p/interrupt; done; echo "TRIPPED"; break; fi
  [ $busy -eq 0 ] && { echo "BOTH DONE"; break; }
  sleep 10
done
