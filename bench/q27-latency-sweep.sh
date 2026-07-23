#!/usr/bin/env bash
# q27-latency-sweep.sh — round 3, driven by the round-1/2 diagnosis.
#
# Decode on this rig is sync/latency-bound, not bandwidth-bound:
#   * memory-controller utilisation only 60-68% under load
#   * deeper drafting hurts even when acceptance improves (each draft token is
#     another forward pass across the inter-GPU exchange)
#   * topology is PHB (via host, no P2P), GPU1 on Gen3 x4 vs GPU0 x8
# If the cost is waiting on syncs, then how the CPU waits should matter:
#   --poll  busy-wait vs sleep on the completion of GPU work
#   --prio  scheduler priority / jitter while waiting
# These are untested in every previous bench on this box.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
PORT=9480
OUT="$(dirname "$0")/q27-latency-sweep-results.txt"
LOG="$(dirname "$0")/q27-latency-sweep.log"
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
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 2048 --jinja --no-warmup \
    -ctk q8_0 -ctv q8_0 -sm tensor --cache-ram 0 \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then printf '%-22s LOAD_FAILED\n' "$label" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2; return 1; fi
  # two runs; report the better, these deltas are small and noisy
  local best=0 tg
  for _ in 1 2; do
    tg=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":$JPROMPT}],\"max_tokens\":300}" \
      | jq -r '.timings.predicted_per_second // 0')
    awk "BEGIN{exit !($tg > $best)}" && best=$tg
  done
  printf '%-22s tg=%6.2f t/s\n' "$label" "$best" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}

echo "== latency/wait-behaviour sweep (tensor, n3, ctx 32768; best of 2) ==" >>"$OUT"
measure "control"
measure "poll-100"         --poll 100
measure "poll-0"           --poll 0
measure "poll100-prio2"    --poll 100 --prio 2
measure "poll100-prio3"    --poll 100 --prio 3
measure "no-op-offload"    --no-op-offload
measure "poll100-t6"       --poll 100 -t 6
echo "DONE $(date +%T)" >>"$OUT"
cat "$OUT"
