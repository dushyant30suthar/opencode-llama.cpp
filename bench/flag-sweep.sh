#!/usr/bin/env bash
# Flag sweep on the daily-driver MoE: split-mode x ubatch, pp2048 + tg128.
# Waits for the ctx experiment to finish first (both need exclusive VRAM).
set -u
BENCH=~/Projects/llama/llama.cpp/build/bin/llama-bench
MODEL=~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
OUT=~/.local/state/llamastack/sweep-results.txt

while pgrep -f ctx-experiment.sh >/dev/null; do sleep 20; done
: > "$OUT"

run() { # run <label> <extra bench args...>
  local label="$1"; shift
  local line
  line=$(timeout 600 "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -p 2048 -n 128 "$@" 2>/dev/null \
    | grep -E "pp2048|tg128" | awk -F'|' '{gsub(/ /,"",$(NF-1)); printf "%s ", $(NF-1)}')
  echo "$label: ${line:-FAILED}" >> "$OUT"
}

run "baseline(layer,ub512)"
run "ub1024" -ub 1024
run "ub2048" -ub 2048
run "sm-row" -sm row
run "sm-tensor" -sm tensor
run "sm-tensor-ub2048" -sm tensor -ub 2048
echo "SWEEP DONE" >> "$OUT"
