#!/usr/bin/env bash
# Find max loadable ctx-size per model on 2x16GB (ngl 99, fa on, kv q8_0).
# Binary search: a config "fits" if llama-cli loads and generates 1 token.
set -u
BIN="$(dirname "$0")/../llama.cpp/build/bin"/llama-cli
OUT=~/.local/state/llamastack/ctx-results.txt
LOG=~/.local/state/llamastack/ctx-experiment.log
: > "$OUT"; : > "$LOG"

try() { # try <model.gguf> <ctx> -> 0 if loads+generates
  timeout 180 "$BIN" -m "$1" -c "$2" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    --no-warmup -n 1 -p "hi" -no-cnv -st >>"$LOG" 2>&1
}

search() { # search <name> <model.gguf> <train_ctx>
  local name="$1" file="$2" hi="$3" lo=8192 best=0
  # quick sanity: does 32k (current default) even fit?
  if try "$file" 32768; then best=32768; lo=32768; else hi=32768; fi
  echo "$(date +%T) $name: start lo=$lo hi=$hi best=$best" >>"$LOG"
  while (( hi - lo > 4096 )); do
    local mid=$(( (lo + hi) / 2 / 4096 * 4096 ))
    (( mid == lo )) && break
    if try "$file" "$mid"; then best=$mid; lo=$mid; else hi=$mid; fi
    echo "$(date +%T) $name: tried $mid -> best=$best" >>"$LOG"
  done
  echo "$name $best" >>"$OUT"
}

# free VRAM: unload whatever the router holds (router itself keeps running)
for m in $(curl -s http://127.0.0.1:9337/v1/models 2>/dev/null | jq -r '.data[].id' 2>/dev/null); do
  curl -s -X POST http://127.0.0.1:9337/models/unload -H "Content-Type: application/json" -d "{\"model\":\"$m\"}" >/dev/null 2>&1
done
sleep 3

M=~/.lmstudio/models/lmstudio-community
# train-context ceilings: qwen3.6 = 262144, all three models = 262144 (read from GGUF headers)
search "Qwen3.6-27B-GGUF"      "$M/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"           262144
search "Qwen3.6-35B-A3B-GGUF"  "$M/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"   262144
search "gemma-4-31B-it-QAT-GGUF" "$M/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf" 262144
echo "DONE" >>"$OUT"
