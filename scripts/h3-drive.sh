#!/usr/bin/env bash
# Autonomous driver: run the matrix in dependency order, unattended.
#
# Order matters. Attention is the dominant term, so phase A settles first and
# every later phase inherits its winner as base flags — tuning memory flags
# against a slower attention backend would rank them on the wrong baseline.
#
# NOTE the pkill pattern: '[m]ain\.py' not 'main.py'. pkill -f matches against
# full command lines INCLUDING the shell running the pkill, so the literal
# pattern kills its own parent. The bracket makes the regex not match itself.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BASE_ATTN="${BASE_ATTN:---use-sage-attention}"

step() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date +%H:%M:%S)"; }

step "A confirm — does the attention win grow with sequence length?"
$PY -u bench/sweep.py A --confirm

step "F — cfg and steps (biggest wall-clock lever: cfg=1.0 skips the negative pass)"
$PY -u bench/sweep.py F --base-flags="$BASE_ATTN"

step "D — device placement"
$PY -u bench/sweep.py D --base-flags="$BASE_ATTN"

step "C — offload and memory"
$PY -u bench/sweep.py C --base-flags="$BASE_ATTN"

step "B — --fast optimizations"
$PY -u bench/sweep.py B --base-flags="$BASE_ATTN"

step "H — encoder/VAE precision"
$PY -u bench/sweep.py H --base-flags="$BASE_ATTN"

step "E — sampler/scheduler"
$PY -u bench/sweep.py E --base-flags="$BASE_ATTN"

step "G — sigma shift"
$PY -u bench/sweep.py G --base-flags="$BASE_ATTN"

step "J — geometry scaling (longest; validates the quadratic model)"
$PY -u bench/sweep.py J --base-flags="$BASE_ATTN"

step "DONE"
pkill -f '[m]ain\.py --port 8191' 2>/dev/null
echo "all phases complete"
