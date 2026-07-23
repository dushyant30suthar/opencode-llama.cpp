#!/usr/bin/env bash
# q27-spec-sweep.sh — maximise token generation on the 27B NVFP4+MTP.
#
# Hypothesis (llama.cpp discussion #25198, same model family): the current
# n-max=3 / p-min=0.0 config pins mean draft length at the ceiling (measured
# 2.71 of 3, acceptance 0.573). p-min lets the draft head stop early when
# unsure, so n-max can be raised far higher: long runs when confident, no
# wasted verification when not. Reported +19.6% on Qwen3.6-27B.
#
# Records tg AND the draft-acceptance diagnostics for every combination,
# because mean-len is what explains the throughput, not tg alone.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
PORT=9478
OUT="$(dirname "$0")/q27-spec-sweep-results.txt"
LOG="$(dirname "$0")/q27-spec-sweep.log"
: > "$OUT"; : > "$LOG"

# code-heavy prompt: the workload this rig actually runs (agentic coding)
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
  local mark="=== $label ==="
  echo "$mark" >>"$LOG"
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 2048 --jinja --no-warmup \
    -ctk q8_0 -ctv q8_0 -sm tensor --cache-ram 0 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2; return 1; fi
  local tg
  tg=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$JPROMPT}],\"max_tokens\":400}" \
    | jq -r '.timings.predicted_per_second // 0')
  # draft diagnostics explain the tg number
  local acc
  acc=$(awk "/$label/,0" "$LOG" | grep -oE "draft acceptance = [0-9.]+ \([0-9]+ accepted / +[0-9]+ generated\), mean len = +[0-9.]+" | tail -1)
  printf '%-24s tg=%6.2f t/s   %s\n' "$label" "$tg" "${acc:-no-draft-stats}" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}

echo "== 27B NVFP4+MTP: p-min x n-max sweep (ctx 32768, code prompt) ==" >>"$OUT"
# control: today's production setting
measure "base-n3-p0.0"   --spec-type draft-mtp --spec-draft-n-max 3
# does raising n-max alone still degrade (reproduce the known plateau)?
measure "n8-p0.0"        --spec-type draft-mtp --spec-draft-n-max 8
# the hypothesis: high n-max unlocked by early-stop
for pm in 0.6 0.75 0.8 0.9; do
  measure "n8-p$pm"      --spec-type draft-mtp --spec-draft-n-max 8  --spec-draft-p-min $pm
done
for pm in 0.75 0.8 0.9; do
  measure "n16-p$pm"     --spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-p-min $pm
done
echo "DONE $(date +%T)" >>"$OUT"
cat "$OUT"
