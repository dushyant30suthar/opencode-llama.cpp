#!/usr/bin/env bash
# q27-backend-sampling.sh — round 5, the decisive test.
#
# ROOT CAUSE (src/llama-context.cpp:1208):
#     if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
#         "backend sampling not supported with SPLIT_MODE_TENSOR; using CPU"; return false; }
#
# So with `-sm tensor` (the production setting) GPU draft sampling is silently
# disabled and every drafted token costs a GPU->CPU->GPU round trip. That is why
# raising n-max collapses throughput here while it gains ~20% on a single 5090.
#
# PREDICTION: `-sm row` also splits weights across both GPUs (keeps parallel
# bandwidth) but is NOT SPLIT_MODE_TENSOR, so backend sampling should survive and
# deep drafting should finally pay. Find the crossover vs tensor's shallow optimum.
#
# Every run asserts whether the CPU-sampler fallback actually occurred, so the
# throughput number is always interpreted next to the mechanism.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF/Qwen3.6-27B-NVFP4-MTP-GGUF.gguf
PORT=9482
OUT="$(dirname "$0")/q27-backend-sampling-results.txt"
LOG="$(dirname "$0")/q27-backend-sampling.log"
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
  local mark="@@@ $label @@@"
  echo "$mark" >>"$LOG"
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null
  # wait for VRAM to actually drain, else the next load OOMs and cascades
  for _ in $(seq 1 20); do
    free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free0:-0}" -gt 14000 ] && break; sleep 2
  done
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 2048 --jinja --no-warmup \
    -ctk q8_0 -ctv q8_0 --cache-ram 0 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 130); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then printf '%-24s LOAD_FAILED\n' "$label" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1; fi
  local tg
  tg=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$JPROMPT}],\"max_tokens\":400}" \
    | jq -r '.timings.predicted_per_second // 0')
  # did GPU draft sampling survive for THIS config?
  local seg smp mlen
  seg=$(awk "/^@@@ $label @@@/,0" "$LOG")
  if grep -q "using CPU sampler" <<<"$seg"; then smp="CPU-sampler"; else smp="GPU-sampler"; fi
  mlen=$(grep -o "mean len = *[0-9.]*" <<<"$seg" | tail -1)
  printf '%-24s tg=%6.2f t/s  %-12s %s\n' "$label" "$tg" "$smp" "${mlen:-}" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
}

echo "== does split-mode decide whether draft sampling runs on GPU? ==" >>"$OUT"
measure "tensor-n3"   -sm tensor --spec-type draft-mtp --spec-draft-n-max 3
measure "row-n3"      -sm row    --spec-type draft-mtp --spec-draft-n-max 3
measure "layer-n3"    -sm layer  --spec-type draft-mtp --spec-draft-n-max 3

echo "== if GPU sampling survives, deep drafting should now pay off ==" >>"$OUT"
for n in 8 16; do
  measure "row-n$n-p0.75"   -sm row   --spec-type draft-mtp --spec-draft-n-max $n --spec-draft-p-min 0.75
  measure "layer-n$n-p0.75" -sm layer --spec-type draft-mtp --spec-draft-n-max $n --spec-draft-p-min 0.75
done
# the published recipe (#25198), now on a split mode that permits GPU sampling
measure "row-n16-p0.8"    -sm row   --spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-p-min 0.8
echo "DONE $(date +%T)" >>"$OUT"
cat "$OUT"
