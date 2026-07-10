#!/usr/bin/env bash
# MTP benchmark: baseline vs draft-mtp (n-max 2, 3) x KV cache (q8_0, f16).
# Server-based (production-faithful), port 9444. Requires idle GPUs.
set -u
SRV=~/Projects/llama/llama.cpp/build/bin/llama-server
MODEL=~/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf
OUT=~/.local/state/llamastack/mtp-bench-results.txt
LOG=~/.local/state/llamastack/mtp-bench.log
PORT=9444
: > "$OUT"; : > "$LOG"

# ~600 tokens of code-flavored generation, measured via server timings
measure() { # measure <label> <server args...>
  local label="$1"; shift
  "$SRV" -m "$MODEL" -ngl 99 -fa on -sm tensor -ub 2048 -c 32768 --jinja \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 60); do
    sleep 3
    kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return; fi
  # warmup
  curl -s --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}' >/dev/null
  # measured run: ask for code so speculation has realistic material
  local t
  t=$(curl -s --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Write a Python class implementing an LRU cache with get, put, delete, plus unit tests."}],"max_tokens":600,"temperature":0}' \
    | jq -r '.timings.predicted_per_second // empty')
  echo "$label: ${t:-NO_TIMING} t/s" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
}

measure "baseline-q8kv"      -ctk q8_0 -ctv q8_0
measure "mtp2-q8kv"          -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2
measure "mtp3-q8kv"          -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
measure "mtp2-f16kv"         --spec-type draft-mtp --spec-draft-n-max 2
echo "MTP BENCH DONE" >>"$OUT"
