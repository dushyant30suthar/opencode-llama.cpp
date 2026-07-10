#!/usr/bin/env bash
# NVFP4 vs Q4_K_XL champion — same binaries, same flags, same method as
# bench/mtp-bench.sh (server timings) + llama-bench for raw pp/tg.
set -u
BIN=~/Projects/llama/llama.cpp/build/bin
NV=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
Q4=~/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf
SCRATCH="$(dirname "$0")"
OUT="$SCRATCH/nvfp4-bench-results.txt"
LOG="$SCRATCH/nvfp4-bench.log"
PORT=9444
: > "$OUT"; : > "$LOG"

echo "== llama-bench raw (no MTP): pp2048 / tg128, tensor split, ub2048, q8_0 KV" >> "$OUT"
for entry in "Q4_K_XL|$Q4" "NVFP4|$NV"; do
  label="${entry%%|*}"; path="${entry#*|}"
  "$BIN/llama-bench" -m "$path" -ngl 99 -fa 1 -sm tensor -ub 2048 \
    -ctk q8_0 -ctv q8_0 -p 2048 -n 128 2>>"$LOG" \
    | grep -E "pp2048|tg128" | sed "s/^/$label /" >> "$OUT"
done

echo "== server-based MTP ladder on NVFP4 (mtp-bench.sh method)" >> "$OUT"
measure() { # measure <label> <model> <extra server args...>
  local label="$1" model="$2"; shift 2
  "$BIN/llama-server" -m "$model" -ngl 99 -fa on -sm tensor -ub 2048 -c 32768 --jinja \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 60); do
    sleep 3
    kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return; fi
  curl -s --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}' >/dev/null
  local t
  t=$(curl -s --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Write a Python class implementing an LRU cache with get, put, delete, plus unit tests."}],"max_tokens":600,"temperature":0}' \
    | jq -r '.timings.predicted_per_second // empty')
  echo "$label: ${t:-NO_TIMING} t/s" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
}

measure "nvfp4-baseline-q8kv" "$NV" -ctk q8_0 -ctv q8_0
measure "nvfp4-mtp3-q8kv"     "$NV" -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3
measure "nvfp4-mtp4-q8kv"     "$NV" -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 4
measure "nvfp4-mtp5-q8kv"     "$NV" -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 5
echo "NVFP4 BENCH DONE" >> "$OUT"
cat "$OUT"
