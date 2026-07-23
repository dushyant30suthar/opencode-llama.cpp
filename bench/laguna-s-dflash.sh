#!/usr/bin/env bash
# laguna-s-dflash.sh — finish the S 2.1 tuning: DFlash speculative decoding.
#
# The first ladder (laguna-s-tune-results.txt) found:
#   K9 8.47 | K13 13.27 | K15 9.71 | K17 LOAD_FAILED
# so throughput PEAKS at K=13, not at the largest K that loads — past that,
# expert layers crowd out the KV/compute buffers and it slows down. The original
# script picked "last K that loaded" (15) and then OOMed trying to add the 2.1GB
# DFlash draft on top, so DFlash was never actually measured.
#
# Here we hold K at values that leave room for the draft model and sweep depth.
# DFlash drafts a whole block per forward pass (unlike MTP's sequential drafting),
# so depth should be much cheaper here than it was on the 27B.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/unsloth/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ3_XXS.gguf
MD=~/.lmstudio/models/poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-DFlash-BF16.gguf
OUT="$(dirname "$0")/laguna-s-dflash-results.txt"
LOG="$(dirname "$0")/laguna-s-dflash.log"
PORT=9467
: > "$OUT"; : > "$LOG"

cpu_regex() {
  python3 -c "
K=$1
cpu=[l for l in range(48) if not (24-K<=l<=23 or 48-K<=l<=47)]
print('blk\\\\.(' + '|'.join(map(str,cpu)) + ')\\\\.ffn_.*_exps\\\\.=CPU')"
}

PROMPT='Review this Python diff for bugs and respond with the corrected function and a one-line verdict:\n```\n def get(self, key):\n-    if key in self.cache:\n-        self.cache.move_to_end(key)\n-    return self.cache[key]\n+    if key not in self.cache:\n+        return None\n+    self.cache.move_to_end(key)\n+    return self.cache[key]\n def put(self, key, value):\n     self.cache[key] = value\n     self.cache.move_to_end(key)\n-    if len(self.cache) >= self.cap:\n+    if len(self.cache) > self.cap:\n         self.cache.popitem(last=False)\n```'

measure() { # measure <label> <server flags...>
  local label="$1"; shift
  echo "=== $label ===" >>"$LOG"
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 512 --jinja --no-warmup \
    --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0 --cache-ram 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 200); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then printf '%-22s LOAD_FAILED\n' "$label" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3; return 1; fi
  local vram t
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  t=$(curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":200}" \
    | jq -r '.timings.predicted_per_second // 0')
  local mlen; mlen=$(grep -o "mean len = *[0-9.]*" "$LOG" | tail -1)
  printf '%-22s gen=%6.2f t/s  vram=%s  %s\n' "$label" "$t" "$vram" "${mlen:-}" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
}

echo "== S 2.1: DFlash at K values that leave room for the 2.1GB draft ==" >>"$OUT"
echo "-- baselines (no draft) --" >>"$OUT"
measure "K13-nospec" -ot "$(cpu_regex 13)" -ctk q8_0 -ctv q8_0
measure "K11-nospec" -ot "$(cpu_regex 11)" -ctk q8_0 -ctv q8_0

echo "-- DFlash depth sweep at K11 (most headroom) --" >>"$OUT"
for N in 3 7 15; do
  measure "K11-dflash-n$N" -ot "$(cpu_regex 11)" -ctk q8_0 -ctv q8_0 \
    -md "$MD" --spec-type draft-dflash --spec-draft-n-max $N
done
echo "-- DFlash at K13 if the draft still fits --" >>"$OUT"
for N in 7 15; do
  measure "K13-dflash-n$N" -ot "$(cpu_regex 13)" -ctk q8_0 -ctv q8_0 \
    -md "$MD" --spec-type draft-dflash --spec-draft-n-max $N
done
echo "DONE $(date +%T)" >>"$OUT"
cat "$OUT"
