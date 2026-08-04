#!/usr/bin/env bash
# Complete the matrix. M runs first: it is the only phase testing a STRUCTURAL
# problem (idle GPU1 + ~6 GB/step streaming from host RAM) rather than a tuning
# knob, so its outcome changes what the later phases are measured against.
#
# pkill/pgrep patterns are assembled at runtime. `pgrep -f 'main.py --port'`
# matches the very shell running it — that self-kill produced every mysterious
# exit-144 during this session before it was diagnosed.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S="--use-sage-attention"
LEN="--length 124 --steps 10"
PAT="ma""in.py --po""rt"

reap() { for p in $(pgrep -f "$PAT"); do [ "$p" = "$$" ] && continue; kill "$p" 2>/dev/null; done; sleep 5; }
step() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date +%H:%M)"; }

reap
step "M — DisTorch2 sharding: DiT spill onto idle GPU1 instead of host RAM"
$PY -u bench/sweep.py M --base-flags="$S" $LEN

step "K — remaining cache rows + sage3 FP4"
$PY -u bench/sweep.py K --base-flags="$S" $LEN

step "L — resolution scaling"
$PY -u bench/sweep.py L --base-flags="$S" $LEN

step "D — device placement"
$PY -u bench/sweep.py D --base-flags="$S" $LEN

step "E — sampler/scheduler"
$PY -u bench/sweep.py E --base-flags="$S" $LEN

step "H — encoder/VAE precision"
$PY -u bench/sweep.py H --base-flags="$S" $LEN

step "C — offload and memory"
$PY -u bench/sweep.py C --base-flags="$S" $LEN

step "G — sigma shift"
$PY -u bench/sweep.py G --base-flags="$S" $LEN

step "F — cfg and steps"
$PY -u bench/sweep.py F --base-flags="$S" $LEN

step "J — geometry scaling (longest, last)"
$PY -u bench/sweep.py J --base-flags="$S" $LEN

step "ANALYSE"
$PY bench/analyse.py
reap
echo "MATRIX COMPLETE"
