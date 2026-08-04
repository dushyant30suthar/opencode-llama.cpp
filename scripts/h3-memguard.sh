#!/usr/bin/env bash
# Memory watchdog: run any command under it; the command is killed before the
# box can thrash.
#
#   bench/memguard.sh -- <command...>
#
# Exists because of 2026-08-04 ~07:15: a --novram benchmark run put ~37 GB of
# model weights into 31 GB of host RAM, kswapd went into a stall loop
# (journalctl -b -1: folio_alloc_swap / kswapd traces, swap filling), the
# desktop froze and the machine needed a hard reset. dual.py already had this
# guard inline; the sweep runs did not. Now nothing heavy runs without it.
#
# Kill rules (checked every 2s):
#   MemAvailable < 2 GB        — the thrash point on this box
#   swap growth  > 1.5 GB      — over the run's own baseline
# On trip: SIGKILL the whole process group. A dead benchmark row is nothing;
# a frozen machine loses every unsaved thing on it.
set -uo pipefail

AVAIL_MIN_MIB=2048
SWAP_GROW_MAX_MIB=4096   # staging blips of ~1.6 GB into zram are normal for the 20 GB pin stage and self-stabilize; the freeze signature was +4 GB in 20s AND still climbing

[[ "${1:-}" == "--" ]] && shift
[[ $# -ge 1 ]] || { echo "usage: memguard.sh -- <command...>"; exit 64; }

mem_avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
swap_used() { awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo; }

swap0=$(swap_used)
setsid "$@" &
child=$!
pgid=$(ps -o pgid= "$child" | tr -d ' ')

trap 'kill -TERM -- "-$pgid" 2>/dev/null' INT TERM

while kill -0 "$child" 2>/dev/null; do
  sleep 2
  avail=$(mem_avail); sw=$(swap_used); grown=$((sw - swap0))
  if (( avail < AVAIL_MIN_MIB || grown > SWAP_GROW_MAX_MIB )); then
    echo "[memguard] TRIPPED: MemAvailable=${avail}MiB swap+${grown}MiB — killing pgid $pgid" >&2
    kill -KILL -- "-$pgid" 2>/dev/null
    # also take down any comfy servers the command spawned detached
    for p in $(pgrep -f "ma""in.py --po""rt"); do kill -KILL "$p" 2>/dev/null; done
    exit 75
  fi
done
wait "$child"
