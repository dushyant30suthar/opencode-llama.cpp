#!/usr/bin/env bash
# Optimisation-focused driver. Everything runs at 124 frames / 10 steps — a real
# clip length (so attention's share is realistic) but half the steps, which keeps
# a row near ~6 min instead of ~12.
#
# Ordered by expected value, not by phase letter: caching and --fast already
# proved they move double digits, placement and memory are cheap to test, and
# the sampler phase is last because its rows change output quality rather than
# just speed, so it needs eyes on the result anyway.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S="--use-sage-attention"
L="--length 124 --steps 10"

wait_free() { while pgrep -f '[s]weep.py' >/dev/null; do sleep 20; done; }

step() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date +%H:%M)"; }

wait_free
step "K extended — cache thresholds 0.5/0.6, full-trajectory, torch.compile"
$PY -u bench/sweep.py K --base-flags="$S" $L

step "D — device placement (free: no restart)"
$PY -u bench/sweep.py D --base-flags="$S" $L

step "H — encoder/VAE precision"
$PY -u bench/sweep.py H --base-flags="$S" $L

step "C — offload and memory"
$PY -u bench/sweep.py C --base-flags="$S" $L

step "L — resolution scaling (largest untested lever)"
$PY -u bench/sweep.py L --base-flags="$S" $L

step "E — sampler/scheduler (res_multistep is 2nd order: fewer steps for equal quality)"
$PY -u bench/sweep.py E --base-flags="$S" $L

step "G — sigma shift"
$PY -u bench/sweep.py G --base-flags="$S" $L

step "ANALYSE"
$PY bench/analyse.py
pkill -f '[m]ain\.py --port 8191' 2>/dev/null
echo "optimisation sweep complete"
