#!/usr/bin/env bash
# laguna-xs-tune.sh — Laguna XS 2.1 (33B/3B-active MoE, Q4_K_M ~19GiB) on 2x16GB.
# Fits fully in VRAM. NOTE: -sm row is NOT usable here — the 5060 Ti reports
# "device CUDA0 does not support split buffers" for this model, so every config
# uses layer split (the default). Sweeps KV type / ubatch / ngram, then probes
# max context. Mirrors bench/m25-tune.sh conventions.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/poolside/Laguna-XS-2.1-GGUF/Laguna-XS-2.1-Q4_K_M.gguf
OUT="$(dirname "$0")/laguna-xs-tune-results.txt"
LOG="$(dirname "$0")/laguna-xs-tune.log"
PORT=9455
: > "$OUT"; : > "$LOG"

PROMPT='Review this Python diff for bugs and respond with the corrected function and a one-line verdict:\n```\n def get(self, key):\n-    if key in self.cache:\n-        self.cache.move_to_end(key)\n-    return self.cache[key]\n+    if key not in self.cache:\n+        return None\n+    self.cache.move_to_end(key)\n+    return self.cache[key]\n def put(self, key, value):\n     self.cache[key] = value\n     self.cache.move_to_end(key)\n-    if len(self.cache) >= self.cap:\n+    if len(self.cache) > self.cap:\n         self.cache.popitem(last=False)\n```'

# measure <label> <server flags...> ; official samplers: temp1.0 topk20 topp1 minp0
measure() {
  local label="$1"; shift
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on --jinja --no-warmup \
    --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0 --cache-ram 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2; return 1; fi
  local vram t pp
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  read -r t pp < <(curl -s --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":300}" \
    | jq -r '"\(.timings.predicted_per_second // 0) \(.timings.prompt_per_second // 0)"')
  printf '%s: gen=%.1f t/s prompt=%.1f t/s vram=%s MiB\n' "$label" "${t:-0}" "${pp:-0}" "$vram" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
  return 0
}

echo "== XS Q4_K_M sweep (layer split; row unsupported on this GPU) ==" >>"$OUT"
measure "q8kv-ub512"   -c 32768 -ctk q8_0 -ctv q8_0 -ub 512
measure "q8kv-ub2048"  -c 32768 -ctk q8_0 -ctv q8_0 -ub 2048
measure "f16kv-ub2048" -c 32768 -ctk f16  -ctv f16  -ub 2048
# draft-free ngram speculation (needs f16 KV — conflicts with quantized cache)
measure "f16kv-ngram"  -c 32768 -ctk f16  -ctv f16  -ub 2048 --spec-type ngram-mod

echo "== XS context probe (q8_0 KV, ub2048) ==" >>"$OUT"
for C in 131072 262144 393216 524288 786432; do
  measure "ctx-$C" -c $C -ctk q8_0 -ctv q8_0 -ub 2048 || { echo "ctx-$C: DID NOT FIT — stop" >>"$OUT"; break; }
done
echo "DONE $(date +%T)" >>"$OUT"
