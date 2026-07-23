#!/usr/bin/env bash
# laguna-s-tune.sh — Laguna S 2.1 (118B/8B-active MoE, UD-IQ3_XXS ~44GB) on
# 2x16GB + 31GB RAM. Won't fit in VRAM: keep attention + K expert layers per
# GPU on-device, spill the rest of the routed experts to CPU RAM. Same balanced
# -ot strategy as bench/m25-tune.sh (v2), adapted to 48 layers (split at 24),
# then layer DFlash speculative decoding on top and sweep the draft depth.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/llama.cpp/build/bin"
M=~/.lmstudio/models/unsloth/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ3_XXS.gguf
MD=~/.lmstudio/models/poolside/Laguna-S-2.1-GGUF/laguna-s-2.1-DFlash-BF16.gguf
OUT="$(dirname "$0")/laguna-s-tune-results.txt"
LOG="$(dirname "$0")/laguna-s-tune.log"
PORT=9466
: > "$OUT"; : > "$LOG"

# 48 layers, GPU0 owns 0-23, GPU1 owns 24-47. Keep the last K expert layers of
# each half ON GPU (co-located with that GPU's attention), rest of experts->CPU.
cpu_regex() {
  python3 -c "
K=$1
cpu=[l for l in range(48) if not (24-K<=l<=23 or 48-K<=l<=47)]
print('blk\\\\.(' + '|'.join(map(str,cpu)) + ')\\\\.ffn_.*_exps\\\\.=CPU')"
}

PROMPT='Review this Python diff for bugs and respond with the corrected function and a one-line verdict:\n```\n def get(self, key):\n-    if key in self.cache:\n-        self.cache.move_to_end(key)\n-    return self.cache[key]\n+    if key not in self.cache:\n+        return None\n+    self.cache.move_to_end(key)\n+    return self.cache[key]\n def put(self, key, value):\n     self.cache[key] = value\n     self.cache.move_to_end(key)\n-    if len(self.cache) >= self.cap:\n+    if len(self.cache) > self.cap:\n         self.cache.popitem(last=False)\n```'

# measure <label> <server flags...> ; official S samplers: temp1.0 topk20 topp1 minp0
measure() {
  local label="$1"; shift
  "$BIN/llama-server" -m "$M" -ngl 99 -fa on -c 32768 -ub 512 --jinja --no-warmup \
    --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0 --cache-ram 0 \
    --host 127.0.0.1 --port $PORT "$@" >>"$LOG" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 200); do
    sleep 3; kill -0 $pid 2>/dev/null || break
    curl -s --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && { up=1; break; }
  done
  if [ "$up" != 1 ]; then echo "$label: LOAD_FAILED" >>"$OUT"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3; return 1; fi
  local vram t pp
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/ -)
  read -r t pp < <(curl -s --max-time 1200 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":200}" \
    | jq -r '"\(.timings.predicted_per_second // 0) \(.timings.prompt_per_second // 0)"')
  echo "$label: gen=${t} t/s prompt=${pp} t/s vram=${vram}MiB" >>"$OUT"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 4
  return 0
}

# K = expert layers kept on EACH GPU (2K on GPU total, 48-2K spilled to CPU RAM).
# Push K as high as VRAM allows: experts are ~0.79GB/layer, and every layer we
# keep on GPU is one less fighting for the ~24GB of usable host RAM. Ladder runs
# until a config fails to load, so the last success is the VRAM ceiling.
echo "== S balanced -ot ladder: K expert layers per GPU (q8_0 KV) ==" >>"$OUT"
BESTK=""
for K in 9 13 15 17 19; do
  RE=$(cpu_regex $K)
  if measure "K$K-q8kv" -ot "$RE" -ctk q8_0 -ctv q8_0; then BESTK=$K; else break; fi
done
echo "best K that loaded: ${BESTK:-none}" >>"$OUT"

if [ -n "$BESTK" ]; then
  echo "== DFlash speculative decoding at K=$BESTK, sweep draft depth ==" >>"$OUT"
  RE=$(cpu_regex $BESTK)
  for N in 5 7 10 15; do
    measure "K$BESTK-dflash-n$N" -ot "$RE" -ctk q8_0 -ctv q8_0 \
      -md "$MD" --spec-type draft-dflash --spec-draft-n-max $N
  done
fi
echo "DONE $(date +%T)" >>"$OUT"
