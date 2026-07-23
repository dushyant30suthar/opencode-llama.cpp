#!/usr/bin/env bash
# q27-ngram-sweep.sh — round 4. Tests a hypothesis DERIVED from the model
# rather than guessed.
#
# Currency on this rig is "inter-GPU sync crossings per emitted token", because
# decode is sync-bound (mem-controller util 60-68%, --poll makes it worse, deeper
# MTP drafting hurts even when acceptance improves).
#
#   MTP     : drafting n tokens costs n sequential forward passes -> n+1 crossings
#   n-gram  : drafting is a table lookup over history -> 0 passes -> 1 crossing
#
# So n-gram speculation should beat MTP here, and its n-max should be cheap to
# raise (a wrong n-gram draft costs batched verify compute, not sequential syncs).
# Code is the ideal workload for it: highly repetitive token history.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
PORT=9481
OUT="$(dirname "$0")/q27-ngram-sweep-results.txt"
LOG="$(dirname "$0")/q27-ngram-sweep.log"
: > "$OUT"; : > "$LOG"

read -r -d '' PROMPT <<'EOF'
Refactor this Python module for clarity and correctness, then explain each change:

class Cache:
    def __init__(self, cap):
        self.cap = cap
        self.cache = {}
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        self.cache[key] = value
        if len(self.cache) >= self.cap:
            self.cache.popitem(last=False)

Provide the corrected implementation with type hints, docstrings, and a short test suite.
EOF
JPROMPT=$(printf '%s' "$PROMPT" | jq -Rs .)

measure() { # measure <label> <extra flags...>
  local label="$1"; shift
  echo "=== $label ===" >>"$LOG"
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null; sleep 1
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 2048 --jinja --no-warmup \
    -ctk f16 -ctv f16 -sm tensor --cache-ram 0 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then printf '%-26s LOAD_FAILED\n' "$label" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2; return 1; fi
  local tg vram
  tg=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$JPROMPT}],\"max_tokens\":400}" \
    | jq -r '.timings.predicted_per_second // 0')
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  local mlen; mlen=$(grep -o "mean len = *[0-9.]*" "$LOG" | tail -1)
  printf '%-26s tg=%6.2f t/s  vram=%s  %s\n' "$label" "$tg" "$vram" "${mlen:-}" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}

echo "== MTP vs n-gram speculation (tensor, f16 KV, ctx 32768, code prompt) ==" >>"$OUT"
measure "mtp-n3-control"     --spec-type draft-mtp --spec-draft-n-max 3
measure "no-spec-control"
echo "-- n-gram variants at default depth --" >>"$OUT"
for t in ngram-mod ngram-cache ngram-simple ngram-map-k ngram-map-k4v; do
  measure "$t" --spec-type "$t"
done
echo "-- if drafting is free, depth should be cheap: ngram-mod depth ladder --" >>"$OUT"
for n in 4 8 16; do
  measure "ngram-mod-n$n" --spec-type ngram-mod --spec-ngram-mod-n-max $n
done
echo "-- stacking (runs independently per llama.cpp#23184; one data point) --" >>"$OUT"
measure "mtp3+ngram-mod" --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
echo "DONE $(date +%T)" >>"$OUT"
cat "$OUT"
