#!/usr/bin/env bash
# q27-sync-sweep.sh — attack the REAL limiter on this rig.
#
# Round 1 (q27-spec-sweep) showed deeper drafting hurts even when acceptance
# improves (n8-p0.6: acceptance 0.82, waste gone, yet tg 44 vs 70 at n3).
# Diagnosis: MTP drafts sequentially, so each draft token = one forward pass, and
# every forward pass crosses the inter-GPU exchange. Topology is PHB (through
# host, no P2P) with GPU1 on PCIe Gen3 x4 — so sync, not bandwidth, is the
# limiter (memory-controller util only 60-68%).
#
# So: hold spec shallow (n3) and vary how work is split across the GPUs.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
PORT=9479
OUT="$(dirname "$0")/q27-sync-sweep-results.txt"
LOG="$(dirname "$0")/q27-sync-sweep.log"
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

measure() { # measure <label> <ctx> <extra flags...>
  local label="$1" ctx="$2"; shift 2
  echo "=== $label ===" >>"$LOG"
  # a previous config that failed to start can leave the port held, which then
  # fails every later config — clear it before each run
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null; sleep 1
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c "$ctx" -ub 2048 --jinja --no-warmup \
    -ctk q8_0 -ctv q8_0 --cache-ram 0 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then printf '%-26s LOAD_FAILED (likely OOM)\n' "$label" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2; return 1; fi
  local vram tg
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  tg=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$JPROMPT}],\"max_tokens\":400}" \
    | jq -r '.timings.predicted_per_second // 0')
  local mlen
  mlen=$(grep -o "mean len = *[0-9.]*" "$LOG" | tail -1)
  printf '%-26s tg=%6.2f t/s  vram=%s  %s\n' "$label" "$tg" "$vram" "${mlen:-}" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}

echo "== split-mode / balance sweep, spec held at n3 (ctx 32768) ==" >>"$OUT"
measure "tensor-n3"      32768 -sm tensor --spec-type draft-mtp --spec-draft-n-max 3
measure "row-n3"         32768 -sm row    --spec-type draft-mtp --spec-draft-n-max 3
measure "layer-n3"       32768 -sm layer  --spec-type draft-mtp --spec-draft-n-max 3
# GPU1 sits on a Gen3 x4 link vs GPU0's x8 — bias work toward the wider link
measure "tensor-ts6040"  32768 -sm tensor -ts 0.60,0.40 --spec-type draft-mtp --spec-draft-n-max 3
measure "tensor-ts7030"  32768 -sm tensor -ts 0.70,0.30 --spec-type draft-mtp --spec-draft-n-max 3

echo "== single-GPU: zero inter-GPU sync (needs weights+KV under 16.3G) ==" >>"$OUT"
# NVFP4 weights are 15.07 GiB; only a small ctx can fit alongside them
measure "single-gpu-c4096"  4096 -sm none -mg 0 --spec-type draft-mtp --spec-draft-n-max 3
measure "single-gpu-c8192"  8192 -sm none -mg 0 --spec-type draft-mtp --spec-draft-n-max 3
# if single-GPU removes the sync tax, deeper drafting should finally pay off
measure "single-gpu-n8-p0.75" 4096 -sm none -mg 0 --spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.75

echo "== no-spec controls (isolate the draft cost) ==" >>"$OUT"
measure "tensor-nospec"  32768 -sm tensor
measure "single-nospec"   4096 -sm none -mg 0
echo "DONE $(date +%T)" >>"$OUT"
cat "$OUT"
